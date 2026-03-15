"""
training/trocr_finetune.py
Fine-tune Microsoft TrOCR on meter digit / nameplate crops.

Usage:
  python training/trocr_finetune.py \\
      --manifest data/prepared/train_manifest.json \\
      --val_manifest data/prepared/val_manifest.json \\
      --out models/trocr_finetuned \\
      [--model microsoft/trocr-base-printed] \\
      [--epochs 15] [--batch 16] [--lr 5e-5]
"""
import os
import sys
import json
import argparse

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import cv2
    import numpy as np
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ── Dataset ───────────────────────────────────────────────────────────────────

class MeterDigitDataset(Dataset):
    """Loads meter crops + ground-truth kWh (or other) text labels."""

    def __init__(self, manifest_path: str, processor, field: str = "kwh", augmentor=None):
        with open(manifest_path) as f:
            records = json.load(f)
        # Filter: only records that have a non-empty label for the target field
        self.records   = [r for r in records if r.get(field, "").strip()]
        self.processor = processor
        self.field     = field
        self.augmentor = augmentor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        img  = cv2.imread(rec["image_path"])
        if img is None:
            img = np.zeros((64, 128, 3), np.uint8)

        if self.augmentor:
            img = self.augmentor(img)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB")
        label   = str(rec[self.field]).strip()

        encoding = self.processor(images=pil_img, return_tensors="pt")
        labels   = self.processor.tokenizer(
            label, return_tensors="pt", padding="max_length", max_length=32
        ).input_ids

        # Replace padding token id with -100 (ignore in loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels":       labels.squeeze(0),
            "label_str":    label,
        }


# ── Training ──────────────────────────────────────────────────────────────────

def cer(pred: str, gt: str) -> float:
    if not gt:
        return float(len(pred))
    import Levenshtein
    return Levenshtein.distance(pred, gt) / len(gt)


def train(args):
    if not TORCH_OK:
        print("ERROR: PyTorch / Transformers not available.")
        sys.exit(1)

    print(f"Loading TrOCR model: {args.model}")
    processor = TrOCRProcessor.from_pretrained(args.model)
    model     = VisionEncoderDecoderModel.from_pretrained(args.model)

    # Ensure pad/eos tokens configured
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    from training.augmentation import MeterAugmentor
    aug = MeterAugmentor(p_augment=0.70)

    train_ds = MeterDigitDataset(args.manifest,     processor, field=args.field, augmentor=aug)
    val_ds   = MeterDigitDataset(args.val_manifest, processor, field=args.field, augmentor=None)

    print(f"Device: {device}. Train={len(train_ds)} Val={len(val_ds)}")

    if len(train_ds) == 0:
        print("ERROR: No labelled training samples found. Check --manifest and field name.")
        sys.exit(1)

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels":       torch.stack([b["labels"]       for b in batch]),
            "label_strs":   [b["label_str"] for b in batch],
        }

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=0, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_dl)
    )

    scaler = None
    if device == "cuda":
        try:
            scaler = torch.cuda.amp.GradScaler()
        except Exception:
            pass

    os.makedirs(args.out, exist_ok=True)
    best_cer = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            pv     = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=pv, labels=labels)
                scaler.scale(outputs.loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values=pv, labels=labels)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += outputs.loss.item()

        # ── Val CER ──
        model.eval()
        val_cers = []
        with torch.no_grad():
            for batch in val_dl:
                pv   = batch["pixel_values"].to(device)
                gts  = batch["label_strs"]
                ids  = model.generate(pv, max_new_tokens=30, num_beams=4)
                preds = processor.batch_decode(ids, skip_special_tokens=True)
                for p, g in zip(preds, gts):
                    val_cers.append(cer(p.strip(), g.strip()))

        mean_cer = float(np.mean(val_cers)) if val_cers else 1.0
        mean_loss = total_loss / max(len(train_dl), 1)
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={mean_loss:.4f} val_CER={mean_cer:.4f}")

        if mean_cer < best_cer:
            best_cer = mean_cer
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            print(f"  ✓ Best model saved → {args.out} (CER={best_cer:.4f})")

    print(f"\nFine-tuning done. Best val CER={best_cer:.4f}")
    return best_cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",     required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--out",          default="models/trocr_finetuned")
    parser.add_argument("--model",        default="microsoft/trocr-base-printed")
    parser.add_argument("--field",        default="kwh",
                        choices=["kwh","kvah","md_kw","demand_kva","meter_serial"])
    parser.add_argument("--epochs",       type=int,   default=15)
    parser.add_argument("--batch",        type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=5e-5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
