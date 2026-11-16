"""Load the project weight files and match them to the training tokenizer."""

import hashlib

import torch

from model import GPT, GPTConfig


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_model(checkpoint, tokenizer_hash):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError(f"{checkpoint} was trained with a different tokenizer.")
    model = GPT(GPTConfig(**saved["model_config"]))
    model.load_state_dict(saved["model_state_dict"])
    return model, saved["step"]
