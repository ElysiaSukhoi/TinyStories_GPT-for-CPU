"""Bundle both inference checkpoints and their tokenizer for distribution."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from checkpoints import sha256
from train_pretrain import PROJECT, write_atomic


def package(pretrain, sft, tokenizer, output, summary=None):
    files = {"pretrain.pt": pretrain, "sft.pt": sft, "tokenizer.json": tokenizer}
    if summary is not None:
        files["training_summary.json"] = summary
    if output.resolve() in {path.resolve() for path in files.values()}:
        raise ValueError("Choose an archive output different from all input files.")
    digests = {name: sha256(path) for name, path in files.items()}
    base = torch.load(pretrain, weights_only=True, map_location="cpu")
    tuned = torch.load(sft, weights_only=True, map_location="cpu")
    if {base["tokenizer_sha256"], tuned["tokenizer_sha256"]} != {digests["tokenizer.json"]}:
        raise ValueError("Both checkpoints must match the bundled tokenizer.")
    if tuned["pretrain_sha256"] != digests["pretrain.pt"] or tuned["model_config"] != base["model_config"]:
        raise ValueError("The supplied SFT and pretrained checkpoints are not a matched pair.")

    manifest = {
        "sha256": digests, "model_config": base["model_config"],
        "pretrain_step": base["step"], "sft_epoch": tuned["epoch"], "sft_step": tuned["step"],
    }

    def write_archive(handle):
        with ZipFile(handle, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in files.items():
                archive.write(path, name)
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(output, write_archive)
    print(f"{output} ({output.stat().st_size:,} bytes)")
    print(f"SHA-256: {sha256(output)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain", type=Path, default=PROJECT / "runs" / "pretrain" / "best.pt")
    parser.add_argument("--sft", type=Path, default=PROJECT / "runs" / "sft" / "best.pt")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT / "data" / "prepared" / "tokenizer.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "dist" / "tiny-gpt-cpu-weights.zip")
    parser.add_argument("--summary", type=Path, help="Include a training and data summary JSON file.")
    args = parser.parse_args()
    package(args.pretrain, args.sft, args.tokenizer, args.output, args.summary)


if __name__ == "__main__":
    main()
