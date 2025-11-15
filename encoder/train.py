import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from datasets import load_dataset
from transformers import AutoTokenizer
import wandb

from src.model import build_transformer_classifier

# -------------------------
# Config
# -------------------------
BATCH_SIZE = 64
EPOCHS = 5
MAX_LEN = 128

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
FF_DIM = 512
DROPOUT = 0.1

# Enable activation checkpointing via env var, e.g.:
# ENABLE_AC=True python train.py
USE_CHECKPOINT = os.environ.get("ENABLE_AC", "False").lower() == "true"

wandb.init(
    project="dabble-single-gpu-encoder",
    group="checkpointing",  # Group checkpointing related experiments
    config={
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "max_len": MAX_LEN,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "num_layers": N_LAYERS,
        "dim_ff": FF_DIM,
        "dropout": DROPOUT,
        "use_checkpoint": USE_CHECKPOINT,
    },
)

# -------------------------
# Device & AMP
# -------------------------
assert (
    torch.cuda.is_available()
), "CUDA not available, but this script assumes a GPU."
device = "cuda"
print(f"Activation checkpointing: {USE_CHECKPOINT}")

scaler = GradScaler()
amp_ctx = lambda: autocast(device_type="cuda", dtype=torch.bfloat16)


# -------------------------
# Dataset: AG News
# -------------------------
print("Loading AG News dataset...")
raw_ds = load_dataset("ag_news")  # train/test splits

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
num_classes = 4  # AG News labels: 0..3


def encode_batch(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    enc["labels"] = examples["label"]
    return enc


print("Tokenizing...")
# AG News gives data like
# {
#     "text": "Wall St. Bears Claw Back Into The Black.",
#     "label": 2
# }
# After encoding, we get
# "input_ids": [...],
# "attention_mask": [...],
# "labels": ...
#
# We no longer need 'text'
encoded_ds = raw_ds.map(encode_batch, batched=True, remove_columns=["text"])

encoded_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

train_ds = encoded_ds["train"]
test_ds = encoded_ds["test"]

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)


# -------------------------
# Model, optimizer, loss
# -------------------------
model = build_transformer_classifier(
    vocab_size=vocab_size,
    num_classes=num_classes,
    max_len=MAX_LEN,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    num_layers=N_LAYERS,
    dim_feedforward=FF_DIM,
    dropout=DROPOUT,
    use_checkpoint=USE_CHECKPOINT,
).to(device)

wandb.watch(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.NLLLoss()

print(model)


# -------------------------
# Eval loop
# -------------------------
def evaluate():
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # HF attention_mask: 1 = real token, 0 = pad
            # src_key_padding_mask: True = pad
            src_key_padding_mask = attn_mask == 0

            with amp_ctx():
                log_probs = model(
                    input_ids,
                    src_key_padding_mask=src_key_padding_mask,
                    train=False,
                )
                loss = criterion(log_probs, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = log_probs.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total += input_ids.size(0)

    return total_loss / total, total_correct / total


# -------------------------
# Training loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    torch.cuda.reset_peak_memory_stats(device)

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        src_key_padding_mask = attn_mask == 0

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx():
            log_probs = model(
                input_ids,
                src_key_padding_mask=src_key_padding_mask,
                train=True,
            )
            loss = criterion(log_probs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * input_ids.size(0)
        preds = log_probs.argmax(dim=-1)
        running_correct += (preds == labels).sum().item()
        running_total += input_ids.size(0)

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total
    val_loss, val_acc = evaluate()

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    curr_mem_mb = torch.cuda.memory_allocated(device) / 1024**2

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
        f"val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | "
        f"peak_mem={peak_mem_mb:.1f} MB | curr_mem={curr_mem_mb:.1f} MB"
    )

    wandb.log(
        {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "peak_mem_MB": peak_mem_mb,
            "curr_mem_MB": curr_mem_mb,
        }
    )

wandb.finish()
