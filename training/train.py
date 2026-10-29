"""Reproducible EfficientNet-B0 transfer learning; test data is never loaded."""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app_core import build_model, preprocess_array, MEAN, STD


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Fundus(Dataset):
    def __init__(self, csv: Path, augment: bool = False):
        self.df = pd.read_csv(csv); self.augment = augment
        if not {"id_code", "diagnosis"}.issubset(self.df.columns):
            raise ValueError(f"Invalid split CSV: {csv}")
        if not self.df.diagnosis.between(0, 4).all():
            raise ValueError(f"Invalid labels in {csv}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = ROOT / "data" / "train_images" / f"{row.id_code}.png"
        with Image.open(path) as image:
            arr = preprocess_array(image)
        if self.augment:
            if random.random() < .5: arr = np.fliplr(arr).copy()
            if random.random() < .5: arr = np.flipud(arr).copy()
            if random.random() < .35:
                arr = np.clip(arr.astype(np.float32) * random.uniform(.85, 1.15), 0, 255).astype(np.uint8)
        x = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float().div_(255.)
        x = (x - MEAN) / STD
        return x, int(row.diagnosis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", action="store_true", help="Time five representative data batches and exit; no checkpoint is written.")
    parser.add_argument("--max-seconds", type=float, default=3600, help="Refuse full CPU training if the measured estimate exceeds this time budget.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1: parser.error("epochs and batch-size must be positive")
    seed_everything(args.seed)
    splits = ROOT / "data" / "splits"
    if not (splits / "train.csv").exists() or not (splits / "val.csv").exists():
        raise SystemExit("Missing train/val split CSVs under data/splits.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep pretrained weights in this workspace so cache access is predictable.
    torch.hub.set_dir(str(ROOT / ".torch_cache"))
    train_ds = Fundus(splits / "train.csv", augment=True)
    val_ds = Fundus(splits / "val.csv")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    labels = train_ds.df.diagnosis.to_numpy()
    counts = np.bincount(labels, minlength=5)
    weights = torch.tensor(len(labels) / (5 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model = build_model(pretrained=True).to(device)
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    # Profile several real batches with all backbone gradients enabled. This is a
    # conservative cost estimate for the later fine-tuning epochs and includes I/O.
    batches_to_measure = min(5, len(loader))
    for param in model.parameters(): param.requires_grad = True
    model.train()
    started = time.perf_counter()
    measured = 0
    for x, y in loader:
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x.to(device)), y.to(device)); loss.backward()
        measured += 1
        if measured >= batches_to_measure: break
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    seconds_per_batch = elapsed / max(1, measured)
    estimated = seconds_per_batch * len(loader) * args.epochs
    # Validation cost is not in this training estimate; include a 25% margin.
    estimated *= 1.25
    print(json.dumps({"device": str(device), "benchmark_batches": measured,
                      "benchmark_seconds_per_batch": seconds_per_batch,
                      "estimated_training_and_validation_seconds": estimated, "train_rows": len(train_ds),
                      "val_rows": len(val_ds), "epochs": args.epochs}))
    if args.benchmark: return
    if device.type == "cpu" and estimated > args.max_seconds:
        raise SystemExit(f"Estimated training time {estimated:.0f}s exceeds --max-seconds {args.max_seconds:.0f}; use GPU or adjust settings.")

    model.load_state_dict(initial_state)
    for param in model.features.parameters(): param.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    out_dir = ROOT / "models"; out_dir.mkdir(exist_ok=True)
    best = -float("inf")
    for epoch in range(args.epochs):
        if epoch == min(2, args.epochs - 1):
            for param in model.features.parameters(): param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        model.train(); total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y); loss.backward(); optimizer.step()
            total += loss.item() * len(y)
        model.eval(); targets, preds, val_loss = [], [], 0.0
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for x, y in val_loader:
                logits = model(x.to(device)); val_loss += criterion(logits, y.to(device)).item() * len(y)
                targets.extend(y.tolist()); preds.extend(logits.argmax(1).cpu().tolist())
        qwk = float(cohen_kappa_score(targets, preds, weights="quadratic"))
        row = {"epoch": epoch + 1, "train_loss": total / len(train_ds), "val_loss": val_loss / len(val_ds),
               "val_qwk": qwk, "seed": args.seed, "batch_size": args.batch_size,
               "device": str(device)}
        print(json.dumps(row), flush=True)
        with (out_dir / "training_log.jsonl").open("a", encoding="utf8") as log: log.write(json.dumps(row) + "\n")
        if np.isfinite(qwk) and qwk > best:
            best = qwk
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_qwk": qwk,
                        "seed": args.seed, "classes": 5, "architecture": "efficientnet_b0"}, out_dir / "best.pt")
    print("Training complete. Checkpoint selected by validation QWK; test split was not accessed.")


if __name__ == "__main__": main()
