"""Shared prompt format and token boundaries for the story continuation task."""

import re

CONTEXT_LENGTH = 128
INSTRUCTION = "Continue the story in exactly two sentences."
BOS, EOS, PAD = "<bos>", "<eos>", "<pad>"
SENTENCE = re.compile(r'([A-Z][^.!?"“”]*[.!?])(?:\s+|$)')
ABBREVIATION = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|[A-Z])\.$")


def narrative_sentences(text):
    """Yield consecutive plain narrative sentences and their end offsets."""
    offset = 0
    for match in iter(lambda: SENTENCE.match(text, offset), None):
        if ABBREVIATION.search(match[1]):
            break
        offset = match.end()
        yield match[1], offset


def format_prompt(instruction, opening):
    return f"{instruction}\nOpening: {opening}\nContinuation:\n"


def encode_prompt(tokenizer, instruction, opening):
    prompt = format_prompt(instruction, opening)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    return [tokenizer.token_to_id(BOS), *prompt_ids]


def encode_example(tokenizer, example):
    prompt_ids = encode_prompt(tokenizer, example["instruction"], example["input"])
    answer_ids = tokenizer.encode(example["output"], add_special_tokens=False).ids
    return prompt_ids, [*answer_ids, tokenizer.token_to_id(EOS)]
