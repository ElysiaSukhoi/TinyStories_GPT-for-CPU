"""Generate an English story continuation with either project checkpoint on CPU."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from tokenizers import Tokenizer

from checkpoints import load_model, sha256
from scripts.story_task import BOS, EOS, INSTRUCTION, encode_prompt

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pretrain", "sft"), default="sft")
    parser.add_argument("--checkpoint", type=Path, help="Override the selected model's default checkpoint.")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT / "data" / "prepared" / "tokenizer.json")
    parser.add_argument("--prompt", required=True, help="English opening text to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="Print text and timing as JSON.")
    args = parser.parse_args()
    if not args.prompt.strip() or min(args.threads, args.max_new_tokens) <= 0:
        parser.error("Provide a nonempty prompt, positive token limit, and positive thread count.")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint = args.checkpoint or PROJECT / "runs" / args.model / "best.pt"
    for path in (checkpoint, args.tokenizer):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    try:
        model, step = load_model(checkpoint, sha256(args.tokenizer))
    except ValueError as exc:
        parser.error(str(exc))
    model.eval()

    opening = args.prompt.strip()
    if args.model == "sft":
        ids = encode_prompt(tokenizer, INSTRUCTION, opening)
    else:
        ids = [tokenizer.token_to_id(BOS), *tokenizer.encode(opening, add_special_tokens=False).ids]
    prompt = torch.tensor(ids, dtype=torch.long)
    if prompt.numel() >= model.config.context_length:
        parser.error(f"The encoded prompt must leave room within {model.config.context_length} tokens.")
    eos_id = tokenizer.token_to_id(EOS)
    started = perf_counter()
    with torch.inference_mode():
        result = model.generate(prompt, args.max_new_tokens, eos_id=eos_id)
    seconds = perf_counter() - started

    new_ids = result[prompt.numel():].tolist()
    stop_reason = "max_new_tokens"
    if new_ids[-1] == eos_id:
        stop_reason = "eos"
    elif result.numel() == model.config.context_length:
        stop_reason = "context_limit"
    response = {
        "model": args.model,
        "checkpoint_step": step,
        "prompt": opening,
        "continuation": tokenizer.decode(new_ids),
        "new_tokens": len(new_ids),
        "generation_seconds": seconds,
        "tokens_per_second": len(new_ids) / seconds,
        "stop_reason": stop_reason,
    }
    if args.json:
        print(json.dumps(response, ensure_ascii=True))
    else:
        print(response["continuation"])
        print(f"CPU | {len(new_ids)} tokens | {seconds:.3f} s | {len(new_ids) / seconds:.1f} token/s | {stop_reason}")


if __name__ == "__main__":
    main()
