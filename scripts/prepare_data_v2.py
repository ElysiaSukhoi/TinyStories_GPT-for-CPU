"""Expand the frozen TinyStories training sets using the four train shards."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from random import Random
from shutil import copyfile
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer

from prepare_data import (
    PROJECT, REVISION, SHARDS, fetch_parquet, file_sha256, first_four_sentences,
    sample_stories, text_id, write_jsonl, write_pretrain,
)
from story_task import CONTEXT_LENGTH, INSTRUCTION, encode_example

TRAIN_SHARDS = (
    SHARDS["train"],
    (
        "train-00001-of-00004-5852b56a2bd28fd9.parquet",
        "e2f173a6dcc2b5895183e8ad376e115bd5c06764eb0cfc01ce1add156b2d017f",
    ),
    (
        "train-00002-of-00004-a26307300439e943.parquet",
        "cac73cd1a0973a7fccde022a5a5c78f8ba58ea6849541835c58e7c5fc747f2ef",
    ),
    (
        "train-00003-of-00004-d243063613e5a057.parquet",
        "862514ae75723ddd339337f1ef7617b88f82290dcddd57445e1d38d257ca7891",
    ),
)
FROZEN = ("tokenizer.json", "pretrain_val.bin", "pretrain_val.jsonl", "sft_val.jsonl", "sft_test.jsonl")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
CAPITALIZED = re.compile(r"\b[A-Z][a-z]+\b")
INTRODUCED_NAME = re.compile(r"\b(?:named|called) ([A-Z][a-z]+)\b")
GENERIC_START = re.compile(r"^(?:once upon a time|once there (?:was|were)|there once (?:was|were)|long long ago)\b")
STOP = set("""
    a an the is am are be was were been have had has do did does can could would should
    i me my we our they them their he him his she her hers it its you your who what
    when there here this that these those and but for from with without about into
    on at up out in of to as so no yes not all one day once time then very just some
    little big good happy said told went wanted decided looked saw looked like
""".split())


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def quality_issue(opening, answer_sentences, name_lexicon):
    phrases = [sentence.casefold() for sentence in answer_sentences]
    if any(GENERIC_START.match(sentence) for sentence in phrases):
        return "generic_restart"
    words = [WORD.findall(sentence) for sentence in phrases]
    combined = words[0] + words[1]
    grams = list(zip(combined, combined[1:], combined[2:], combined[3:]))
    if words[0] == words[1] or len(grams) != len(set(grams)):
        return "repeated_phrase"
    opening_names = name_lexicon.intersection(CAPITALIZED.findall(opening))
    target_names = name_lexicon.intersection(CAPITALIZED.findall(" ".join(answer_sentences)))
    if opening_names and target_names and opening_names.isdisjoint(target_names):
        return "name_discontinuity"
    opening_terms = set(WORD.findall(opening.casefold())) - STOP
    target_terms = set(combined) - STOP
    if opening_terms.isdisjoint(target_terms):
        return "missing_lexical_anchor"
    if not target_terms - opening_terms:
        return "no_new_content_words"
    return None


def select_examples(candidates, count, tokenizer, openings, name_lexicon):
    selected, lengths = [], []
    rejects = Counter()
    for story in candidates:
        sentences = first_four_sentences(story["text"])
        if sentences is None:
            rejects["unusable_sentence_boundaries"] += 1
            continue
        opening = " ".join(sentences[:2])
        opening_id = text_id(opening)
        if opening_id in openings:
            rejects["duplicate_opening"] += 1
            continue
        issue = quality_issue(opening, sentences[2:], name_lexicon)
        if issue:
            rejects[issue] += 1
            continue
        example = {
            "story_id": story["story_id"], "instruction": INSTRUCTION,
            "input": opening, "output": " ".join(sentences[2:]),
        }
        prompt, answer = encode_example(tokenizer, example)
        length = len(prompt) + len(answer)
        if length > CONTEXT_LENGTH:
            rejects["context_overflow"] += 1
            continue
        openings.add(opening_id)
        selected.append(example)
        lengths.append(length)
        if len(selected) == count:
            return selected, {
                "new_examples": count, "new_min_tokens": min(lengths), "new_max_tokens": max(lengths),
                "rejected_by_first_reason": dict(sorted(rejects.items())),
            }
    raise ValueError(f"Found {len(selected)} new SFT examples for a target of {count}.")


def prepare(cache, baseline, output, seed, smoke):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new --output-dir.")
    cache.mkdir(parents=True, exist_ok=True)
    lock = json.loads((PROJECT / "results/optimization_baseline_v1.json").read_text(encoding="utf-8"))
    for name in (*FROZEN, "manifest.json", "sft_train.jsonl"):
        if file_sha256(baseline / name) != lock["files_sha256"][f"data/prepared/{name}"]:
            raise ValueError(f"Baseline digest mismatch: {name}")
    v1 = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(baseline / "tokenizer.json"))
    old_pretrain = read_jsonl(baseline / "pretrain_train.jsonl")
    old_sft = read_jsonl(baseline / "sft_train.jsonl")
    held_out = [record for name in ("pretrain_val", "sft_val", "sft_test")
                for record in read_jsonl(baseline / f"{name}.jsonl")]
    if any(text_id(record["text"]) != record["story_id"] for record in old_pretrain):
        raise ValueError("The baseline training stories have inconsistent IDs.")
    v1_records = old_pretrain + old_sft + held_out
    seen = {record["story_id"] for record in v1_records}
    if len(seen) != len(v1_records):
        raise ValueError("The v1 splits contain duplicate story IDs.")
    openings = {text_id(record["input"]) for record in old_sft + held_out if "input" in record}

    rng = Random(seed)
    official_val, val_stats = sample_stories(fetch_parquet(cache, *SHARDS["validation"]), None, rng, set())
    seen.update(record["story_id"] for record in official_val)
    per_shard = 400 if smoke else 22_000
    candidates, origins, scans = [], {}, {}
    for shard, spec in enumerate(TRAIN_SHARDS):
        rows, stats = sample_stories(fetch_parquet(cache, *spec), per_shard, rng, seen)
        candidates.extend(rows)
        origins.update((row["story_id"], shard) for row in rows)
        scans[str(shard)] = stats
        print(f"Shard {shard}: {stats['unique_stories']:,} new candidates", flush=True)
    rng.shuffle(candidates)
    names = {name for record in candidates for name in INTRODUCED_NAME.findall(record["text"])}

    added_sft, sft_scan = select_examples(candidates, 32 if smoke else 4_000, tokenizer, openings, names)
    sft_ids = {record["story_id"] for record in added_sft}
    added_pretrain = [record for record in candidates if record["story_id"] not in sft_ids]
    pretrain_count = 256 if smoke else 48_000
    if len(added_pretrain) < pretrain_count:
        raise ValueError("Insufficient new pretraining stories for the requested size.")
    added_pretrain = added_pretrain[:pretrain_count]
    pretrain = (old_pretrain[:128] if smoke else old_pretrain) + added_pretrain
    sft = (old_sft[:32] if smoke else old_sft) + added_sft
    rng.shuffle(pretrain)
    rng.shuffle(sft)
    lengths = [sum(map(len, encode_example(tokenizer, row))) for row in sft]
    if max(lengths) > CONTEXT_LENGTH:
        raise ValueError("An SFT example exceeds the model context.")

    with TemporaryDirectory(dir=output.parent, prefix=".prepare-v2-") as temp:
        stage = Path(temp) / "prepared"
        stage.mkdir()
        for name in FROZEN:
            copyfile(baseline / name, stage / name)
        splits = dict(v1["splits"])
        splits["pretrain_train"] = write_pretrain(stage, "pretrain_train", pretrain, tokenizer)
        write_jsonl(stage / "sft_train.jsonl", sft)
        splits["sft_train"] = {"stories": len(sft), "min_tokens": min(lengths), "max_tokens": max(lengths)}
        manifest = {
            **v1, "data_version": 2, "smoke": smoke, "seed": seed,
            "shards": {
                "train": [{"file": name, "sha256": digest} for name, digest in TRAIN_SHARDS],
                "validation": v1["shards"]["validation"],
            },
            "frozen_files_sha256": {name: lock["files_sha256"][f"data/prepared/{name}"] for name in FROZEN},
            "scan": {"train": scans, "validation": val_stats},
            "new_train_by_shard": {
                "pretrain": dict(sorted(Counter(origins[row["story_id"]] for row in added_pretrain).items())),
                "sft": dict(sorted(Counter(origins[row["story_id"]] for row in added_sft).items())),
            },
            "sft_selection": {
                **sft_scan,
                "rules": "Adjacent narrative sentences with lexical anchoring and new content; reject resets, "
                         "exact sentence or four-word repeats, and detectable names that lose all opening names.",
            },
            "splits": splits,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    print(json.dumps(splits, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--baseline-dir", type=Path, default=PROJECT / "data" / "prepared")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.output_dir is None:
        parser.error("--smoke needs a separate --output-dir.")
    output = args.output_dir or PROJECT / "data" / "prepared-v2"
    prepare(args.cache_dir.resolve(), args.baseline_dir.resolve(), output.resolve(), args.seed, args.smoke)


if __name__ == "__main__":
    main()
