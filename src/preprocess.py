# src/preprocess.py
"""Load raw jsonl, clean, and write train/val jsonl ready for Hugging Face datasets.
This version is more robust to different key names and encodings and adds a CLI.
"""
import json
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional
import argparse

from sklearn.model_selection import train_test_split

DEFAULT_INPUT = Path("data/cleaned_for_training.jsonl")
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# keys we will try (in order) to extract the main text & the summary
POSSIBLE_TEXT_KEYS = ["text", "email", "body", "content", "message", "raw_text", "prompt"]
POSSIBLE_SUMMARY_KEYS = ["summary", "label_summary", "summary_text", "short_summary", "completion"]



def normalize_text(s: Optional[str]) -> str:
    """Normalize text: replace newlines with spaces and strip whitespace."""
    if s is None:
        return ""
    return s.replace("\n", " ").strip()


def read_jsonl(path: Path) -> Generator[Dict, None, None]:
    """Yield parsed json objects from a .jsonl file. Uses utf-8-sig to handle BOM."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # Print a helpful error for debugging, then continue
                print(f"Warning: JSON parse error on line {line_no}: {e}")
                continue


def to_pair(example: Dict) -> Dict:
    """
    Convert an input example to the desired pair format.
    Try several possible keys for text and summary. Handle prompt/completion format.
    """
    # find first available text key
    text_val = ""
    for k in POSSIBLE_TEXT_KEYS:
        v = example.get(k)
        if v:
            text_val = normalize_text(v)
            break

    # If prompt contains an instruction like "summarize: ..." try to remove a short prefix
    # Heuristic: if there's a colon in the first 50 chars and the rest is much longer, remove prefix
    if text_val:
        first_200 = text_val[:200]
        if ":" in first_200:
            colon_idx = first_200.find(":")
            # ensure there's meaningful text after the colon
            if colon_idx < 60 and len(first_200[colon_idx+1:].strip()) > 20:
                text_val = text_val[colon_idx+1:].strip()

    # find first available summary key
    summary_val = ""
    for k in POSSIBLE_SUMMARY_KEYS:
        v = example.get(k)
        if v:
            summary_val = normalize_text(v)
            break

    # If completion looks like it has a leading space or start token, trim it
    if summary_val.startswith(" "):
        summary_val = summary_val.lstrip()

    # fallback: sometimes summary may be inside nested fields like "data"
    if not text_val and isinstance(example.get("data"), dict):
        for k in POSSIBLE_TEXT_KEYS:
            v = example["data"].get(k)
            if v:
                text_val = normalize_text(v)
                break

    if not summary_val and isinstance(example.get("data"), dict):
        for k in POSSIBLE_SUMMARY_KEYS:
            v = example["data"].get(k)
            if v:
                summary_val = normalize_text(v)
                break

    label = example.get("label")
    return {"text": text_val, "summary": summary_val, "label": label}


def write_jsonl(path: Path, data: Iterable[Dict]) -> None:
    """Write an iterable of dicts to a jsonl file."""
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")


def main(input_path: Path) -> None:
    # Basic diagnostics & parse
    print(f"Reading from: {input_path}")
    items = list(read_jsonl(input_path))
    print(f"Total non-empty lines read: {len(items)}")

    # convert to pairs
    all_items: List[Dict] = [to_pair(x) for x in items]

    # debug: inspect first items to show what keys exist & a sample
    if len(items) > 0:
        sample_keys = set()
        for i, it in enumerate(items[:5]):
            sample_keys.update(it.keys())
        print(f"Keys seen in first lines: {sorted(sample_keys)}")

    # filter out examples with no text
    before = len(all_items)
    all_items = [x for x in all_items if x["text"]]
    after = len(all_items)
    print(f"Examples with text: {after} / {before}")

    if len(all_items) == 0:
        raise ValueError(
            "No examples found after filtering — check your input file and field names. "
            "You can edit POSSIBLE_TEXT_KEYS / POSSIBLE_SUMMARY_KEYS in src/preprocess.py if needed."
        )

    if len(all_items) < 2:
        # train_test_split requires at least 2 samples
        write_jsonl(OUT_DIR / "train.jsonl", all_items)
        write_jsonl(OUT_DIR / "validation.jsonl", [])
        print(
            f"Wrote {len(all_items)} train and 0 validation (too few samples for split)."
        )
        return

    train, val = train_test_split(all_items, test_size=0.08, random_state=42)

    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "validation.jsonl", val)
    print(f"Wrote {len(train)} train and {len(val)} validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess jsonl into train/validation jsonl")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Path to input jsonl file (default: data/cleaned_for_training.jsonl)")
    args = parser.parse_args()
    main(Path(args.input))
