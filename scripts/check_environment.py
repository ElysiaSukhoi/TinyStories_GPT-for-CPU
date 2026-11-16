"""Check that CPU training operations and byte-level BPE work locally."""

import sys

import numpy as np
import tokenizers
import torch
from tokenizers import ByteLevelBPETokenizer


def main():
    if torch.version.cuda is not None or torch.version.hip is not None:
        raise RuntimeError("Expected the CPU build of PyTorch.")

    torch.manual_seed(42)
    layer = torch.nn.Linear(4, 8, device="cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    inputs = torch.randn(2, 4, device="cpu")
    targets = torch.from_numpy(np.array([1, 2], dtype=np.int64))
    original_weight = layer.weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(layer(inputs), targets)
    loss.backward()
    if layer.weight.grad is None or not torch.isfinite(layer.weight.grad).all():
        raise RuntimeError("CPU backward pass produced invalid gradients.")
    optimizer.step()
    if torch.equal(layer.weight, original_weight):
        raise RuntimeError("AdamW did not update the weights.")

    sample = "A little fox found a warm place to sleep."
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator([sample], vocab_size=280, min_frequency=1)
    if tokenizer.decode(tokenizer.encode(sample).ids) != sample:
        raise RuntimeError("Byte-level BPE round trip failed.")

    print(f"Python {sys.version.split()[0]}")
    print(f"PyTorch {torch.__version__} on {inputs.device}")
    print(f"NumPy {np.__version__}, tokenizers {tokenizers.__version__}")
    print("CPU forward, backward, optimizer step and byte-level BPE passed.")


if __name__ == "__main__":
    main()
