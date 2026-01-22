# src/inference.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "google/flan-t5-base"


class Summarizer:
    def __init__(self):
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

        # normal move (now works because accelerate is gone)
        self.model.to(self.device)
        self.model.eval()

    def summarize(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        clean_text = re.sub(
            r"^Subject:.*\n?",
            "",
            text,
            flags=re.IGNORECASE
        ).strip()

        prompt = (
            "Write a concise executive summary of the following email in 2 to 3 sentences. "
            "Remove any repeated ideas and combine all points into one clear paragraph. "
            "Use your own words, not the original phrasing.\n\n"
            f"{clean_text}"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            min_length=50,
            max_length=90,     # 🔥 this is the key
            num_beams=4,
            no_repeat_ngram_size=4,
            length_penalty=2.0,
            early_stopping=True,
        )

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True)
