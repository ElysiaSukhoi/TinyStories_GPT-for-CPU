"""Prepare TinyStories splits, a byte-level BPE tokenizer and training data."""

import argparse
import hashlib
import json
import unicodedata
from itertools import islice
from pathlib import Path
from random import Random
from shutil import copyfileobj
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from story_task import BOS, CONTEXT_LENGTH, EOS, INSTRUCTION, PAD, encode_example, narrative_sentences

PROJECT = Path(__file__).resolve().parents[1]
DATASET = "roneneldan/TinyStories"
REVISION = "6d02f92396f7e3e60c2251070b6e0b168568074b"
SHARDS = {
    "train": (
        "train-00000-of-00004-2d5a1467fff1081b.parquet",
        "77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b",
    ),
    "validation": (
        "validation-00000-of-00001-869c898b519ad725.parquet",
        "33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab",
    ),
}
TRAIN_CANDIDATES = 65000
PRETRAIN_TRAIN = 32000
PRETRAIN_VALIDATION = 1500
SFT_TRAIN = 4000
SFT_EVAL = 400
VOCAB_SIZE = 4096


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_shard(cache, split):
    name, expected_hash = SHARDS[split]
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{name}"
    path = cache / name
    if not path.exists():
        print(f"Downloading {name}", flush=True)
        with NamedTemporaryFile(dir=cache, suffix=".part", delete_on_close=False) as temp:
            with urlopen(url + "?download=1", timeout=60) as response:
                copyfileobj(response, temp, length=1024 * 1024)
            temp.close()
            downloaded = Path(temp.name)
            if file_sha256(downloaded) != expected_hash:
                raise ValueError(f"SHA-256 mismatch for download {url}")
            downloaded.replace(path)
    elif file_sha256(path) != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}. Replace it with a fresh copy from {url}")
    return path


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def text_id(text):
    return hashlib.sha256(normalize(text).casefold().encode("utf-8")).hexdigest()


def sample_stories(path, limit, rng, seen):
    stories = []
    empty = duplicates = offset = 0
    with path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        rows = parquet.metadata.num_rows
        selected = set(rng.sample(range(rows), min(limit, rows))) if limit else None
        for batch in parquet.iter_batches(batch_size=4096, columns=["text"]):
            for row, text in enumerate(batch.column(0).to_pylist(), offset):
                if selected is not None and row not in selected:
                    continue
                text = normalize(text or "")
                if not text:
                    empty += 1
                    continue
                story_id = text_id(text)
                if story_id in seen:
                    duplicates += 1
                    continue
                seen.add(story_id)
                stories.append({"story_id": story_id, "text": text})
            offset += batch.num_rows
    rng.shuffle(stories)
    return stories, {
        "shard_rows": rows,
        "sampled_rows": len(stories) + empty + duplicates,
        "unique_stories": len(stories),
        "empty_rows": empty,
        "duplicate_rows": duplicates,
    }


def train_tokenizer(stories, path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=[BOS, EOS, PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator((story["text"] for story in stories), trainer, length=len(stories))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError(f"Expected a vocabulary of {VOCAB_SIZE} tokens")
    tokenizer.save(str(path))
    return Tokenizer.from_file(str(path))


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_pretrain(directory, name, stories, tokenizer):
    write_jsonl(directory / f"{name}.jsonl", stories)
    bos_id, eos_id = (tokenizer.token_to_id(token) for token in (BOS, EOS))
    token_count = 0
    with (directory / f"{name}.bin").open("wb") as handle:
        for start in range(0, len(stories), 512):
            batch = [story["text"] for story in stories[start : start + 512]]
            for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
                ids = np.asarray([bos_id, *encoding.ids, eos_id], dtype="<u2")
                ids.tofile(handle)
                token_count += len(ids)
    return {"stories": len(stories), "tokens": token_count}


def first_four_sentences(text):
    sentences = [sentence for sentence, _ in islice(narrative_sentences(text), 4)]
    return sentences if len(sentences) == 4 else None


def write_sft(directory, name, stories, count, tokenizer, openings):
    examples, lengths = [], []
    for story in stories:
        sentences = first_four_sentences(story["text"])
        if sentences is None:
            continue
        opening = " ".join(sentences[:2])
        opening_id = text_id(opening)
        if opening_id in openings:
            continue
        example = {
            "story_id": story["story_id"],
            "instruction": INSTRUCTION,
            "input": opening,
            "output": " ".join(sentences[2:]),
        }
        prompt_ids, answer_ids = encode_example(tokenizer, example)
        length = len(prompt_ids) + len(answer_ids)
        if length > CONTEXT_LENGTH:
            continue
        openings.add(opening_id)
        examples.append(example)
        lengths.append(length)
        if len(examples) == count:
            break
    if len(examples) != count:
        raise ValueError(f"{name}: {len(examples)} qualifying examples for a target of {count}")
    write_jsonl(directory / f"{name}.jsonl", examples)
    return {"stories": len(examples), "min_tokens": min(lengths), "max_tokens": max(lengths)}


def prepare(cache, output, seed):
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new --output-dir to rebuild.")

    # Reserve official validation stories before selecting any training examples.
    rng = Random(seed)
    seen = set()
    valid, valid_scan = sample_stories(fetch_shard(cache, "validation"), None, rng, seen)
    train, train_scan = sample_stories(fetch_shard(cache, "train"), TRAIN_CANDIDATES, rng, seen)
    print(f"Found {len(train)} unique train candidates and {len(valid)} validation stories.", flush=True)
    pretrain_train = train[:PRETRAIN_TRAIN]
    pretrain_val = valid[:PRETRAIN_VALIDATION]
    remaining_val = valid[PRETRAIN_VALIDATION:]
    val_midpoint = len(remaining_val) // 2
    with TemporaryDirectory(dir=output.parent, prefix=".prepare-") as temp:
        directory = Path(temp) / "prepared"
        directory.mkdir()
        tokenizer = train_tokenizer(pretrain_train, directory / "tokenizer.json")
        for story in pretrain_val[:16]:
            text = story["text"]
            if tokenizer.decode(tokenizer.encode(text).ids) != text:
                raise ValueError(f"BPE round trip failed for story {story['story_id']}")
        special_tokens = {
            "bos": tokenizer.token_to_id(BOS),
            "eos": tokenizer.token_to_id(EOS),
            "pad": tokenizer.token_to_id(PAD),
        }
        print("Trained and reloaded the 4,096-token BPE. Writing datasets.", flush=True)
        splits = {
            "pretrain_train": write_pretrain(directory, "pretrain_train", pretrain_train, tokenizer),
            "pretrain_val": write_pretrain(directory, "pretrain_val", pretrain_val, tokenizer),
        }
        openings = set()
        splits["sft_test"] = write_sft(
            directory, "sft_test", remaining_val[val_midpoint:], SFT_EVAL, tokenizer, openings
        )
        splits["sft_val"] = write_sft(
            directory, "sft_val", remaining_val[:val_midpoint], SFT_EVAL, tokenizer, openings
        )
        splits["sft_train"] = write_sft(
            directory, "sft_train", train[PRETRAIN_TRAIN:], SFT_TRAIN, tokenizer, openings
        )
        manifest = {
            "dataset": DATASET,
            "dataset_revision": REVISION,
            "license": "CDLA-Sharing-1.0",
            "shards": {
                split: {"file": name, "sha256": digest} for split, (name, digest) in SHARDS.items()
            },
            "seed": seed,
            "scan": {"train": train_scan, "validation": valid_scan},
            "vocab_size": tokenizer.get_vocab_size(),
            "special_token_ids": special_tokens,
            "context_length": CONTEXT_LENGTH,
            "pretrain_bin_dtype": "<u2",
            "splits": splits,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        directory.rename(output)
    print(json.dumps(splits, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "data" / "prepared")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.cache_dir.resolve(), args.output_dir.resolve(), args.seed)


if __name__ == "__main__":
    main()
