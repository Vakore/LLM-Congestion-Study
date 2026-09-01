#!/usr/bin/env python3
"""
download_dataset.py
Downloads the ShareGPT dataset from HuggingFace and produces a cleaned,
filtered version suitable for driving inference experiments.

Usage:
    python3 scripts/download_dataset.py
    python3 scripts/download_dataset.py --local ./data/ShareGPT_V3_unfiltered_cleaned_split.json
    python3 scripts/download_dataset.py --max-samples 2000
"""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

console = Console()

DATA_DIR = Path(__file__).parent.parent / "data"
HF_DATASET_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
OUTPUT_FILE = DATA_DIR / "sharegpt_clean.json"

# Prompt length bins (in characters, approximate — token count varies by model)
SHORT_MAX_CHARS = 500
MEDIUM_MAX_CHARS = 2000
# Long: anything above MEDIUM_MAX_CHARS up to filter limit
FILTER_MAX_CHARS = 6000   # discard extremely long prompts that would OOM


def load_from_hf() -> list[dict]:
    console.print("[blue]Downloading ShareGPT dataset from HuggingFace...[/blue]")
    console.print(f"  Source: https://huggingface.co/datasets/{HF_DATASET_ID}")
    ds = load_dataset(HF_DATASET_ID, data_files="ShareGPT_V3_unfiltered_cleaned_split.json", split="train")
    return list(ds)


def load_from_local(path: str) -> list[dict]:
    console.print(f"[blue]Loading local dataset from: {path}[/blue]")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    console.print(f"  Loaded {len(data):,} raw conversations.")
    return data


def extract_turns(raw_data: list[dict]) -> list[dict]:
    """
    Extract (prompt, expected_response) pairs from ShareGPT conversations.
    Keeps only the first human turn and the following assistant turn for
    each conversation, to produce clean single-exchange request samples.
    """
    samples = []
    for convo in tqdm(raw_data, desc="Extracting turns", ncols=80):
        turns = convo.get("conversations", [])
        prompt_text = None
        for i, turn in enumerate(turns):
            role = turn.get("from", "").lower()
            value = turn.get("value", "").strip()

            if role in ("human", "user") and prompt_text is None:
                prompt_text = value
            elif role in ("gpt", "assistant") and prompt_text is not None:
                response_text = value
                samples.append({
                    "prompt": prompt_text,
                    "response": response_text,
                    "prompt_chars": len(prompt_text),
                    "response_chars": len(response_text),
                })
                break  # one exchange per conversation

    return samples


def classify_length(chars: int) -> str:
    if chars <= SHORT_MAX_CHARS:
        return "short"
    elif chars <= MEDIUM_MAX_CHARS:
        return "medium"
    else:
        return "long"


def clean_and_filter(samples: list[dict], max_samples: int) -> list[dict]:
    """Remove empty, malformed, and excessively long samples. Add length class."""
    cleaned = []
    skipped_empty = 0
    skipped_long = 0

    for s in tqdm(samples, desc="Filtering", ncols=80):
        prompt = s["prompt"].strip()
        response = s["response"].strip()

        if not prompt or not response:
            skipped_empty += 1
            continue
        if s["prompt_chars"] > FILTER_MAX_CHARS:
            skipped_long += 1
            continue

        s["length_class"] = classify_length(s["prompt_chars"])
        cleaned.append(s)

    console.print(f"  Skipped {skipped_empty:,} empty turns, {skipped_long:,} oversized prompts.")

    # Balance classes and cap total samples
    per_class = max_samples // 3
    by_class: dict[str, list] = {"short": [], "medium": [], "long": []}
    for s in cleaned:
        by_class[s["length_class"]].append(s)

    balanced = []
    for cls, items in by_class.items():
        np.random.shuffle(items)
        balanced.extend(items[:per_class])

    np.random.shuffle(balanced)
    return balanced


def print_summary(samples: list[dict]):
    table = Table(title="Dataset Summary", border_style="blue")
    table.add_column("Class", style="bold")
    table.add_column("Count")
    table.add_column("Prompt chars (median)")
    table.add_column("Response chars (median)")

    by_class: dict[str, list] = {"short": [], "medium": [], "long": []}
    for s in samples:
        by_class[s["length_class"]].append(s)

    for cls, items in by_class.items():
        if items:
            p_med = int(np.median([x["prompt_chars"] for x in items]))
            r_med = int(np.median([x["response_chars"] for x in items]))
            table.add_row(cls, str(len(items)), str(p_med), str(r_med))

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Download and clean the ShareGPT dataset.")
    parser.add_argument("--local", default=None, help="Path to a locally downloaded JSON file.")
    parser.add_argument("--max-samples", type=int, default=3000, help="Max samples to keep (default: 3000, balanced across classes).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        "[bold blue]LLM Inference Traffic Research[/bold blue]\n"
        "ShareGPT Dataset Downloader",
        border_style="blue"
    ))

    if OUTPUT_FILE.exists():
        console.print(f"[green]Cleaned dataset already exists at {OUTPUT_FILE}.[/green]")
        console.print("Delete it and re-run to regenerate.")
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        print_summary(existing)
        return

    if args.local:
        raw = load_from_local(args.local)
    else:
        raw = load_from_hf()

    console.print(f"\n[blue]Extracting conversation turns...[/blue]")
    samples = extract_turns(raw)
    console.print(f"  Extracted {len(samples):,} prompt/response pairs.")

    console.print(f"\n[blue]Cleaning and filtering (max {args.max_samples:,} samples)...[/blue]")
    cleaned = clean_and_filter(samples, args.max_samples)
    console.print(f"  Final dataset size: {len(cleaned):,} samples.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Saved to: {OUTPUT_FILE}[/green]\n")
    print_summary(cleaned)

    console.print(
        "\n[bold]Next step:[/bold] Download the model with:\n"
        "  python3 scripts/download_model.py"
    )


if __name__ == "__main__":
    main()
