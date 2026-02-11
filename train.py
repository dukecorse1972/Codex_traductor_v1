#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SWLLSEDataset, load_label_map
from src.data.features import FeatureSpec
from src.models.tcn import TCNClassifier, GRUBaseline


def compute_mask(x: torch.Tensor) -> torch.Tensor:
    # x [B,T,D], mask [B,T]
    return (x.abs().sum(dim=-1) > 0).float()


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mask = compute_mask(x)
            logits = model(x, mask=mask)
            loss = ce(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    acc = float((y_true_arr == y_pred_arr).mean()) if len(y_true_arr) else 0.0
    macro_f1 = f1_score(y_true_arr, y_pred_arr, average="macro") if len(y_true_arr) else 0.0
    avg_loss = total_loss / max(1, len(y_true_arr))
    return avg_loss, acc, macro_f1, y_true_arr, y_pred_arr


def save_confusion_matrix(y_true, y_pred, out_path: Path, num_classes: int = 300):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--mediapipe_dir", type=str, default=None)
    parser.add_argument("--splits_dir", type=str, default=None)
    parser.add_argument("--annotations_csv", type=str, default=None)
    parser.add_argument("--feature_spec", type=str, default="artifacts/feature_spec.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--t_fixed", type=int, default=None)
    parser.add_argument("--model", type=str, default="tcn", choices=["tcn", "gru"])
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_csv", type=str, default="artifacts/train_log.csv")
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    splits_dir = Path(args.splits_dir) if args.splits_dir else data_root
    mediapipe_dir = Path(args.mediapipe_dir) if args.mediapipe_dir else (data_root / "MEDIAPIPE")
    annotations_csv = Path(args.annotations_csv) if args.annotations_csv else (data_root / "videos_ref_annotations.csv")

    feature_spec_path = Path(args.feature_spec)
    if feature_spec_path.exists():
        feature_spec = FeatureSpec.from_json(str(feature_spec_path))
    else:
        feature_spec = FeatureSpec()
    if args.t_fixed is not None:
        feature_spec.t_fixed = args.t_fixed

    feature_spec_path.parent.mkdir(parents=True, exist_ok=True)
    feature_spec.to_json(str(feature_spec_path))

    label_map = load_label_map(str(annotations_csv))
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/label_map.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)

    train_ds = SWLLSEDataset(str(splits_dir / "train.csv"), str(mediapipe_dir), feature_spec, augment=True)
    val_ds = SWLLSEDataset(str(splits_dir / "val.csv"), str(mediapipe_dir), feature_spec, augment=False)
    test_ds = SWLLSEDataset(str(splits_dir / "test.csv"), str(mediapipe_dir), feature_spec, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "tcn":
        model = TCNClassifier(input_dim=feature_spec.d_frame, num_classes=300)
    else:
        model = GRUBaseline(input_dim=feature_spec.d_frame, num_classes=300)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="artifacts/tensorboard")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    wait = 0

    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "lr"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            mask = compute_mask(x)
            opt.zero_grad()
            logits = model(x, mask=mask)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        train_loss = running_loss / len(train_ds)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_acc, val_f1, opt.param_groups[0]["lr"]])

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metric/val_acc", val_acc, epoch)
            writer.add_scalar("metric/val_macro_f1", val_f1, epoch)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            wait = 0
            torch.save({"model_state_dict": model.state_dict(), "feature_spec": feature_spec.__dict__}, save_dir / "best.pt")
            print(f"Saved best checkpoint with val_acc={best_acc:.4f}")
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping triggered")
                break

    ckpt = torch.load(save_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_f1:.4f}")

    save_confusion_matrix(y_true, y_pred, Path("artifacts/confusion_matrix.png"), num_classes=300)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
