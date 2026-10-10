import argparse
import math
import statistics
import timeit
from contextlib import nullcontext

import torch
import torch.nn.functional as F

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b":    {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

MODES = ("forward", "forward_backward", "full")


def nvtx_range(name, enabled):
    if enabled:
        return torch.cuda.nvtx.range(name)
    return nullcontext()


def add_attention_nvtx_ranges():
    """Add labels around the three main operations in reference attention."""

    def annotated_attention(Q, K, V, mask=None):
        with torch.cuda.nvtx.range("scaled dot product attention"):
            d_k = K.shape[-1]

            with torch.cuda.nvtx.range("attention QK matmul"):
                attention_scores = basics_model.einsum(
                    Q, K, "... query d_k, ... key d_k -> ... query key"
                ) / math.sqrt(d_k)

            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

            with torch.cuda.nvtx.range("attention softmax"):
                attention_weights = basics_model.softmax(attention_scores, dim=-1)

            with torch.cuda.nvtx.range("attention AV matmul"):
                output = basics_model.einsum(
                    attention_weights, V, "... query key, ... key d_v -> ... query d_v"
                )

        return output

    basics_model.scaled_dot_product_attention = annotated_attention


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def precision_context(device, precision):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def run_step(model, optimizer, inputs, targets, mode, device, precision, profile=False):
    if mode != "forward":
        with nvtx_range("zero_grad", profile):
            optimizer.zero_grad(set_to_none=True)

    with precision_context(device, precision):
        with nvtx_range("forward", profile):
            logits = model(inputs)
        if mode != "forward":
            with nvtx_range("loss", profile):
                loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
                )

    if mode == "forward":
        return

    with nvtx_range("backward", profile):
        loss.backward()

    if mode == "full":
        with nvtx_range("optimizer_step", profile):
            optimizer.step()


def benchmark(model, optimizer, inputs, targets, mode, device, precision,
              warmup_steps, measurement_steps, profile=False):
    for _ in range(warmup_steps):
        with nvtx_range("warmup_step", profile):
            run_step(model, optimizer, inputs, targets, mode, device, precision, profile)
            synchronize(device)

    times_ms = []

    for _ in range(measurement_steps):
        synchronize(device)
        start = timeit.default_timer()

        # Nsight captures this range, so setup and warmup stay outside the trace.
        with nvtx_range("measured_step", profile):
            run_step(model, optimizer, inputs, targets, mode, device, precision, profile)
            synchronize(device)
        elapsed_ms = (timeit.default_timer() - start) * 1000
        times_ms.append(elapsed_ms)

    average_ms = statistics.mean(times_ms)
    standard_deviation_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return average_ms, standard_deviation_ms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-sizes",
        nargs="+",
        choices=list(MODEL_CONFIGS),
        default=["small"],
    )
    parser.add_argument(
        "--mode",
        choices=[*MODES, "all"],
        default="all",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "both"],
        default="fp32",
        help="Benchmark full precision, BF16 autocasting, or both",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Add NVTX ranges for Nsight Systems profiling",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.profile and device.type != "cuda":
        parser.error("--profile requires a CUDA device")

    vocab_size = 10000

    torch.manual_seed(0)
    tokens = torch.randint(
        0,
        vocab_size,
        (args.batch_size, args.context_length + 1),
        device=device,
    )
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    if args.mode == "all":
        modes_to_run = MODES
    else:
        modes_to_run = (args.mode,)

    if args.precision == "both":
        precisions_to_run = ("fp32", "bf16")
    else:
        precisions_to_run = (args.precision,)

    if args.profile:
        add_attention_nvtx_ranges()

    for model_size in args.model_sizes:
        config = MODEL_CONFIGS[model_size]

        for precision in precisions_to_run:
            for mode in modes_to_run:
                torch.manual_seed(0)

                model = BasicsTransformerLM(
                    vocab_size=vocab_size,
                    context_length=args.context_length,
                    d_model=config["d_model"],
                    num_layers=config["num_layers"],
                    num_heads=config["num_heads"],
                    d_ff=config["d_ff"]
                ).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                print(f"Running {model_size}, precision={precision}, mode={mode} ...", flush=True)

                average_ms, standard_deviation_ms = benchmark(
                    model,
                    optimizer,
                    inputs,
                    targets,
                    mode,
                    device,
                    precision,
                    args.warmup_steps,
                    args.measurement_steps,
                    args.profile,
                )

                print(
                    f"{model_size:>6} | {precision:>4} | {mode:>16} | "
                    f"mean = {average_ms:.2f} ms | std = {standard_deviation_ms:.2f} ms"
                )

if __name__ == "__main__":
    main()

    
