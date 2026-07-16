from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
import timeit

MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

def benchmark_model(model_size, seq_len, batch_size, device):
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=512,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
    ).to(device)

    optimizer = AdamW(model.parameters())
    input_ids = torch.randint(0, 10000, (batch_size, seq_len)).to(device)

    # Warm-up
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = cross_entropy(input_ids, outputs)
        loss.backward()
        optimizer.step()
