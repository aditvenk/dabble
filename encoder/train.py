import argparse
from contextlib import nullcontext
from pathlib import Path
import os
import sys
import time

import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

try:
    import torch._dynamo as torch_dynamo
except ImportError:
    torch_dynamo = None

from datasets import load_dataset
from transformers import AutoTokenizer
import wandb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.model import (
    build_transformer_classifier,
    estimate_transformer_flops,
)
from utils.flops import get_peak_flops_per_second


# -------------------------
# Config helpers
# -------------------------


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "baseline.yml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small AG News transformer classifier."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=_default_config_path(),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _get_config_sections(cfg: dict) -> tuple[dict, dict, dict]:
    return (
        cfg.get("training", {}),
        cfg.get("model", {}),
        cfg.get("wandb", {}),
    )


args = _parse_args()
training_cfg, model_cfg, wandb_cfg = _get_config_sections(
    _load_config(args.config)
)

# -------------------------
# Config defaults
# -------------------------
BATCH_SIZE = training_cfg.get("batch_size", 64)
EPOCHS = training_cfg.get("epochs", 5)
MAX_LEN = training_cfg.get("max_len", model_cfg.get("max_len", 128))
LEARNING_RATE = float(training_cfg.get("learning_rate", 3e-4))
WEIGHT_DECAY = training_cfg.get("weight_decay", 0.01)
WARMUP_STEPS = training_cfg.get("warmup_steps", 1000)
USE_COMPILE = training_cfg.get("use_compile", False)
CLOCK_RATE_GHZ = float(training_cfg.get("clock_rate_ghz", 1.5))
USE_AMP = training_cfg.get("use_amp", True)
USE_GRAD_SCALING = training_cfg.get("use_grad_scaling", True)
USE_WANDB = training_cfg.get("use_wandb", True)
GRADIENT_ACCUMULATION_STEPS = max(
    1, int(training_cfg.get("gradient_accumulation_steps", 4))
)
assert BATCH_SIZE % GRADIENT_ACCUMULATION_STEPS == 0, (
    "batch_size must be divisible by gradient_accumulation_steps"
)
MICRO_BATCH_SIZE = BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS


D_MODEL = model_cfg.get("d_model", 256)
N_HEADS = model_cfg.get("n_heads", 4)
N_LAYERS = model_cfg.get("num_layers", 4)
FF_DIM = model_cfg.get("dim_feedforward", 512)
DROPOUT = model_cfg.get("dropout", 0.1)

# Enable activation checkpointing via env var, e.g.:
# ENABLE_AC=True python train.py
USE_CHECKPOINT = (
    os.environ.get("ENABLE_AC", "False").lower() == "true"
    or bool(model_cfg.get("use_checkpoint", False))
    and not USE_COMPILE  # disable AC if compile is enabled.
)

# -------------------------
# Device & AMP
# -------------------------
assert (
    torch.cuda.is_available()
), "CUDA not available, but this script assumes a GPU."
device = torch.device("cuda")
print(f"Activation checkpointing: {USE_CHECKPOINT}")

if USE_AMP:
    amp_ctx = lambda: autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    amp_ctx = nullcontext

scaler = GradScaler() if USE_GRAD_SCALING else None

peak_flops_per_sec = get_peak_flops_per_second(
    device, clock_rate_hz=CLOCK_RATE_GHZ * 1e9
)
wandb_config = {
    "batch_size": BATCH_SIZE,
    "micro_batch_size": MICRO_BATCH_SIZE,
    "epochs": EPOCHS,
    "max_len": MAX_LEN,
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "num_layers": N_LAYERS,
    "dim_ff": FF_DIM,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "use_checkpoint": USE_CHECKPOINT,
    "use_amp": USE_AMP,
    "use_grad_scaling": USE_GRAD_SCALING,
    "warmup_steps": WARMUP_STEPS,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "use_wandb": USE_WANDB,
    "use_compile": USE_COMPILE,
    "clock_rate_ghz": CLOCK_RATE_GHZ,
    "peak_flops_per_sec": peak_flops_per_sec,
}
if USE_WANDB:
    wandb.init(
        project=wandb_cfg.get("project", "dabble-single-gpu-encoder"),
        group=wandb_cfg.get("group", "checkpointing"),
        config=wandb_config,
    )


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
    train_ds, batch_size=MICRO_BATCH_SIZE, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    test_ds, batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=2
)
train_steps = len(train_loader)

per_step_flops = estimate_transformer_flops(
    MICRO_BATCH_SIZE, MAX_LEN, D_MODEL, N_LAYERS, N_HEADS, FF_DIM, num_classes
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
)

model = model.to(device)
if USE_COMPILE:
    model = torch.compile(model)

if USE_WANDB:
    wandb.watch(model)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
if WARMUP_STEPS > 0:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / max(1, WARMUP_STEPS), 1.0),
    )
else:
    scheduler = None
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
    step_durations = []
    epoch_flops = 0.0

    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(train_loader):
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        src_key_padding_mask = attn_mask == 0

        with amp_ctx():
            log_probs = model(
                input_ids,
                src_key_padding_mask=src_key_padding_mask,
                train=True,
            )
            loss = criterion(log_probs, labels)
        loss_value = loss.item()
        loss_for_backprop = loss / GRADIENT_ACCUMULATION_STEPS

        if scaler is not None:
            scaler.scale(loss_for_backprop).backward()
        else:
            loss_for_backprop.backward()

        should_step = (
            (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (batch_idx + 1 == train_steps)
        )
        if should_step:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        epoch_flops += per_step_flops
        step_durations.append(time.perf_counter() - step_start)

        running_loss += loss_value * input_ids.size(0)
        preds = log_probs.argmax(dim=-1)
        running_correct += (preds == labels).sum().item()
        running_total += input_ids.size(0)

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total
    val_loss, val_acc = evaluate()

    epoch_duration = sum(step_durations)
    avg_step_time = (
        epoch_duration / len(step_durations) if step_durations else 0.0
    )
    mfu = (
        epoch_flops / (peak_flops_per_sec * epoch_duration)
        if epoch_duration > 0 and peak_flops_per_sec > 0
        else 0.0
    )

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    curr_mem_mb = torch.cuda.memory_allocated(device) / 1024**2

    curr_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | train_acc={train_acc*100:.2f}% | "
        f"val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | "
        f"peak_mem={peak_mem_mb:.1f} MB | curr_mem={curr_mem_mb:.1f} MB | "
        f"avg_step_time={avg_step_time:.4f}s | mfu={mfu*100:.1f}%"
    )

    if USE_WANDB:
        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "peak_mem_MB": peak_mem_mb,
                "curr_mem_MB": curr_mem_mb,
                "avg_step_time_secs": avg_step_time,
                "current_lr": curr_lr,
                "mfu": mfu,
            }
        )

if USE_WANDB:
    wandb.finish()
