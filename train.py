import os
import json
import numpy as np
from dotenv import load_dotenv

from sklearn.metrics import f1_score
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # :)
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback

load_dotenv()

training_dir = os.getenv("TRAIN_DIR")
# This train dir has jsonl files for created for finetuning -- filename representing request classification (determined by gpt-oss), and contents of prompt completion pairs
#ex.
# chit_chat.jsonl, quote_request.jsonl, spam.jsonl,
# troubleshooting.jsonl, unknown.jsonl, warranty_claim.jsonl

def load_datset():
    per_label = {}
    for fname in sorted(os.listdir(training_dir)):
        if not fname.endswith(".jsonl"):
            continue
        label = fname.removesuffix(".jsonl")
        with open(os.path.join(training_dir, fname)) as f:
            per_label[label] = [
                {"text": json.loads(line)["prompt"], "label": label}
                for line in f
                if line.strip()
            ]

    min_count = min(len(v) for v in per_label.values())
    print(f"Balancing dataset to {min_count} examples per label:")
    for label, examples in sorted(per_label.items()):
        print(f"  {label}: {len(examples)} -> {min_count}")

    records = [ex for examples in per_label.values() for ex in examples[:min_count]]

    dataset = Dataset.from_list(records)
    dataset = dataset.class_encode_column("label")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    return dataset, split_dataset


def tokenize_data(split_dataset):
 
    model_id = "answerdotai/ModernBERT-base"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=512)

    if "label" in split_dataset["train"].features.keys():
        split_dataset = split_dataset.rename_column("label", "labels") 
    tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["text"])

    tokenized_dataset["train"].features.keys()
    # dict_keys(['labels', 'input_ids', 'attention_mask'])

    return tokenizer, tokenized_dataset

def load_model(tokenized_dataset):
 
    model_id = "answerdotai/ModernBERT-base"
    
    labels = tokenized_dataset["train"].features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label,
    )

    return model

 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
    return {"f1": float(score) if score == 1 else score}


if __name__ == "__main__":
    dataset, split_dataset = load_datset()
    tokenizer, tokenized_dataset = tokenize_data(split_dataset)
    model = load_model(tokenized_dataset)




    training_args = TrainingArguments(
        output_dir= "ModernBERT-domain-classifier",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        num_train_epochs=5,
        optim="adafactor",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        dataloader_pin_memory=False,
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=True,
        hub_strategy="every_save",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()




