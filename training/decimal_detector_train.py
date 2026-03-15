"""
training/decimal_detector_train.py
Train a MobileNetV3-Small binary classifier to detect decimal points in meter
digit crops. Uses Focal Loss to handle class imbalance.

Usage:
  python training/decimal_detector_train.py \\
      --manifest data/prepared/train_manifest.json \\
      --val_manifest data/prepared/val_manifest.json \\
      --out models/weights/decimal_cnn_best.pt \\
      [--epochs 40] [--batch 64] [--lr 1e-3]
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        pt   = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class DecimalCropDataset(Dataset):
    """
    Each sample: (32×32 crop tensor, label)
    Label: 1 = has decimal point, 0 = integer only.
    Labels are inferred from the 'kwh' field (contains '.' → 1).
    Also generates synthetic negatives from integer-only readings.
    """

    def __init__(self, manifest_path: str, augmentor=None, size=(64, 64)):
        with open(manifest_path) as f:
            self.records = json.load(f)
        self.augmentor = augmentor
        self.size      = size
        self.transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec["image_path"])
        if img is None:
            img = np.zeros((64, 64, 3), np.uint8)
        # Label: 1 if any reading contains '.'
        label = int(any('.' in str(rec.get(f, '')) for f in ('kwh','kvah','md_kw','demand_kva')))

        if self.augmentor and img is not None:
            img = self.augmentor(img)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x   = self.transform(pil)
        return x, torch.tensor(label, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=2):
    model = models.mobilenet_v3_small(weights=None)
    in_f  = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    if not TORCH_OK:
        print("ERROR: PyTorch not found. Install with: pip install torch torchvision")
        sys.exit(1)

    from training.augmentation import MeterAugmentor
    aug = MeterAugmentor(p_augment=0.80)

    train_ds = DecimalCropDataset(args.manifest,     augmentor=aug)
    val_ds   = DecimalCropDataset(args.val_manifest,  augmentor=None)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training decimal detector on {device}. Train={len(train_ds)} Val={len(val_ds)}")

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()          # Focal optional if heavy imbalance
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

        # ── val ──
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                v_correct += (preds == y).sum().item()
                v_total   += len(y)

        train_acc = correct / max(total, 1)
        val_acc   = v_correct / max(v_total, 1)
        scheduler.step(1 - val_acc)

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={total_loss/max(total,1):.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  ✓ Best model saved → {args.out}")

    print(f"\nTraining done. Best val_acc={best_val_acc:.4f}")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",     required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--out",          default="models/weights/decimal_cnn_best.pt")
    parser.add_argument("--epochs",       type=int,   default=40)
    parser.add_argument("--batch",        type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
