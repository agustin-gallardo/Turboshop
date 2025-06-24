from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import numpy as np
import evaluate
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ====== Paso 1: Cargar dataset ======
dataset = load_dataset("json", data_files="../generate_data/dataset_ner.jsonl", split="train")

# ====== Paso 2: Etiquetas ======
# Asegura que ner_tags estén en texto
unique_labels = set(tag for sample in dataset for tag in sample["ner_tags"])
label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
id2label = {i: label for label, i in label2id.items()}

# ====== Paso 3: Tokenización y alineación ======
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize_and_align(example):
    tokenized = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    word_ids = tokenized.word_ids()

    aligned_labels = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != prev_word_idx:
            tag = example["ner_tags"][word_idx]
            aligned_labels.append(label2id[tag])
        else:
            # Puedes dejarlo igual o marcar como "I-" si prefieres
            tag = example["ner_tags"][word_idx]
            aligned_labels.append(label2id[tag])
        prev_word_idx = word_idx

    tokenized["labels"] = aligned_labels
    return tokenized

# ====== Paso 4: División y mapeo ======
dataset = dataset.train_test_split(test_size=0.1, seed=42)
tokenized_datasets = dataset.map(tokenize_and_align, batched=False)

# ====== Paso 5: Modelo ======
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ====== Paso 6: Métrica ======
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    preds = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    true_preds = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    return metric.compute(predictions=true_preds, references=true_labels)

# ====== Paso 7: Entrenamiento ======
training_args = TrainingArguments(
    output_dir="./ner-transformer-output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
#    evaluation_strategy="epoch",
#    save_strategy="epoch",
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

# ====== Paso 8: Guardar modelo ======
trainer.save_model("modelo_ner_repuestos")
tokenizer.save_pretrained("modelo_ner_repuestos")
