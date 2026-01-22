# src/train.py
import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = os.getenv("BASE_MODEL", "google/flan-t5-small")

OUT = Path("models/my-checkpoint")
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

max_input_length = 512
max_target_length = 256

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "validation": "data/validation.jsonl",
    }
)

# OPTIONAL: remove very short summaries
dataset["train"] = dataset["train"].filter(lambda x: len(x["summary"].split()) >= 20)
dataset["validation"] = dataset["validation"].filter(lambda x: len(x["summary"].split()) >= 20)

def preprocess(batch):
    inputs = [
        "Summarize the following email into 2 to 3 sentences:\n\n" + t
        for t in batch["text"]
    ]

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
    )

    labels = tokenizer(
        text_target=batch["summary"],
        truncation=True,
        padding="max_length",
        max_length=max_target_length,
    )

    labels_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs


encoded = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUT),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    logging_steps=20,
    save_total_limit=1,
    predict_with_generate=True,
    save_safetensors=False,   # important for Windows
    fp16=False,
)




trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(str(OUT))
tokenizer.save_pretrained(str(OUT))

print("Training complete. Model saved to:", OUT)
