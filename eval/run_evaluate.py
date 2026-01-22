# eval/run_evaluate.py
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
import torch

MODEL_DIR = Path("models") / "my-checkpoint"

def score():
    ds = load_dataset(
        "json",
        data_files={"validation": "data/validation.jsonl"}
    )["validation"]

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    rouge = load("rouge")
    bert = load("bertscore")

    preds, refs = [], []

    for ex in ds:
        text = (
            "Write a detailed 2 to 3 sentence summary of the following email. "
            "Include key actions, timelines, and responsibilities.\n\n"
            + ex["text"]
        )

        ref = ex.get("summary") or ""

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                min_length=60,
                max_length=180,
                num_beams=4,
                length_penalty=1.5,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append(pred)
        refs.append(ref)

    r = rouge.compute(predictions=preds, references=refs)
    b = bert.compute(predictions=preds, references=refs, lang="en")

    print("ROUGE:", r)
    print("BERTScore (F1 mean):", sum(b["f1"]) / len(b["f1"]))

if __name__ == "__main__":
    score()
