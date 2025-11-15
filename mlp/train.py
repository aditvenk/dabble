import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
import os

ENABLE_AC = os.environ.get("ENABLE_AC", False)


# -------------------------------
# 1. MLP definition
# -------------------------------
class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim=128,
        hidden_size=256,
        num_classes=10,
        dropout_rate=0.1,
        enable_ac=False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.enable_ac = enable_ac

    def forward(self, x, train: bool = True):
        def _inner(x, train=True):
            x = self.fc1(x)
            x = self.ln1(x)
            x = F.silu(x)
            if train:
                x = self.dropout(x)
            return x

        if self.enable_ac:
            x = torch.utils.checkpoint.checkpoint(
                _inner, x, use_reentrant=False
            )
        else:
            x = _inner(x)
        x = self.fc2(x)
        # keep numerically sensitive part in fp32
        return F.log_softmax(x.float(), dim=-1)


# -------------------------------
# 2. Setup
# -------------------------------
device = "cuda"

torch.manual_seed(0)

input_dim = 128
num_classes = 10
num_samples = 10_000
batch_size = 256
epochs = 5

model = MLPClassifier(
    input_dim=input_dim,
    hidden_size=256,
    num_classes=num_classes,
    dropout_rate=0.1,
    enable_ac=ENABLE_AC,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.NLLLoss()

scaler = GradScaler()
amp_ctx = lambda: autocast(device_type="cuda", dtype=torch.bfloat16)

# -------------------------------
# 3. Create fixed synthetic dataset (10k samples)
#    y = argmax(W_true x + b_true)
# -------------------------------
W_true = torch.randn(input_dim, num_classes, device=device)
b_true = torch.randn(num_classes, device=device)

with torch.no_grad():
    X = torch.randn(num_samples, input_dim, device=device)
    logits = X @ W_true + b_true
    y = torch.argmax(logits, dim=-1)

# move to CPU for DataLoader (optional)
X_cpu = X.cpu()
y_cpu = y.cpu()

dataset = TensorDataset(X_cpu, y_cpu)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

# -------------------------------
# 4. Training loop with peak GPU memory per epoch
# -------------------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    torch.cuda.reset_peak_memory_stats(device)

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx():
            log_probs = model(batch_x, train=True)
            loss = criterion(log_probs, batch_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_x.size(0)

        with torch.no_grad():
            preds = log_probs.argmax(dim=-1)
            running_correct += (preds == batch_y).sum().item()
            running_total += batch_x.size(0)

    avg_loss = running_loss / running_total
    avg_acc = running_correct / running_total

    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    curr_mem_bytes = torch.cuda.memory_allocated(device)
    peak_mem_mb = peak_mem_bytes / (1024**2)
    curr_mem_mb = curr_mem_bytes / (1024**2)
    mem_str = f"peak_mem={peak_mem_mb:.1f} MB | curr_mem={curr_mem_mb:.1f} MB"

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"loss={avg_loss:.4f} | acc={avg_acc*100:.2f}% | {mem_str}"
    )
