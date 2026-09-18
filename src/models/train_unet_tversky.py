"""
FASE 3 - U-NET ARCADE STENOSIS
BCE + Tversky Loss para reducir falsos positivos.

Parte del mejor checkpoint de la fase anterior:
entrenamiento_continuado/mejor_unet_arcade_continuado.pt

Mantiene exactamente la arquitectura U-Net anterior.
Primera prueba: 256x256, batch 2, LR 1e-4, épocas 16-20.
"""

import json
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ROOT = os.path.join(PROJECT_ROOT, "arcade")

CHECKPOINT = os.path.join(
    PROJECT_ROOT, "entrenamiento_continuado", "mejor_unet_arcade_continuado.pt"
)

OUT_DIR = os.path.join(PROJECT_ROOT, "entrenamiento_tversky")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE = 256
BATCH_SIZE = 2
NUM_WORKERS = 0

START_EPOCH = 16
END_EPOCH = 20

LEARNING_RATE = 1e-4
THRESHOLD = 0.5

# Alpha > beta => se penalizan más los falsos positivos.
TV_ALPHA = 0.60
TV_BETA = 0.40
TV_SMOOTH = 1e-6

BCE_WEIGHT = 0.50
TV_WEIGHT = 0.50

SEED = 42


# ============================================================
# DATASET
# ============================================================


class ArcadeStenosisDataset(Dataset):
    def __init__(self, root, split, image_size=256):
        split_dir = os.path.join(root, "stenosis", split)
        self.images_dir = os.path.join(split_dir, "images")
        ann_file = os.path.join(split_dir, "annotations", f"{split}.json")

        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)

        self.images_info = [
            x
            for x in data["images"]
            if os.path.exists(os.path.join(self.images_dir, x["file_name"]))
        ]

        self.masks_by_image = {x["id"]: [] for x in self.images_info}

        for ann in data["annotations"]:
            if ann.get("category_id") == 26 and ann.get("image_id") in self.masks_by_image:
                self.masks_by_image[ann["image_id"]].append(ann.get("segmentation", []))

        self.image_size = image_size

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        info = self.images_info[idx]

        image = cv2.imread(os.path.join(self.images_dir, info["file_name"]), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"No se pudo abrir: {info['file_name']}")

        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for polygons in self.masks_by_image[info["id"]]:
            for poly in polygons:
                if not poly or len(poly) < 6:
                    continue

                points = np.array(
                    [
                        [int(round(poly[i])), int(round(poly[i + 1]))]
                        for i in range(0, len(poly) - 1, 2)
                    ],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))

                cv2.fillPoly(mask, [points], 1)

        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return image, mask, info["file_name"]


# ============================================================
# U-NET - MISMA ARQUITECTURA DEL CHECKPOINT
# ============================================================


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


# ============================================================
# BCE + TVERSKY LOSS
# ============================================================


class BCETverskyLoss(nn.Module):
    def __init__(self, bce_weight=0.5, tv_weight=0.5, alpha=0.60, beta=0.40, smooth=1e-6):
        super().__init__()

        self.bce_weight = bce_weight
        self.tv_weight = tv_weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):

        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        probs = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        tversky_loss = 1.0 - tversky.mean()

        return self.bce_weight * bce + self.tv_weight * tversky_loss


# ============================================================
# MÉTRICAS
# ============================================================


def calculate_metrics(logits, targets, threshold=0.5):

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)

    union = pred_sum + target_sum - intersection

    dice = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)

    iou = (intersection + 1e-6) / (union + 1e-6)

    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    precision = (intersection + 1e-6) / (intersection + fp + 1e-6)

    recall = (intersection + 1e-6) / (intersection + fn + 1e-6)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


# ============================================================
# ENTRENAMIENTO / VALIDACIÓN
# ============================================================


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    losses = []
    metric_values = []

    for bi, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        metric_values.append(calculate_metrics(logits.detach(), y))

        if bi == 1 or bi % 100 == 0 or bi == len(loader):
            print(f"  batch {bi:3d}/{len(loader)} | loss {loss.item():.4f}")

    return (
        float(np.mean(losses)),
        {key: float(np.mean([m[key] for m in metric_values])) for key in metric_values[0]},
    )


@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()
    losses = []
    metric_values = []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        losses.append(loss.item())
        metric_values.append(calculate_metrics(logits, y))

    return (
        float(np.mean(losses)),
        {key: float(np.mean([m[key] for m in metric_values])) for key in metric_values[0]},
    )


# ============================================================
# VISUALIZACIÓN: AÑADIMOS SUPERPOSICIÓN
# ============================================================


@torch.no_grad()
def save_predictions(model, dataset, device, epoch, n=2):

    model.eval()

    idxs = np.linspace(0, len(dataset) - 1, min(n, len(dataset)), dtype=int)

    fig, ax = plt.subplots(len(idxs), 5, figsize=(17, 7 * len(idxs)))

    ax = np.atleast_2d(ax)

    for r, i in enumerate(idxs):
        x, y, name = dataset[i]

        logits = model(x.unsqueeze(0).to(device))

        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred = (prob >= THRESHOLD).astype(np.float32)

        image = x[0].numpy()
        gt = y[0].numpy()

        ax[r, 0].imshow(image, cmap="gray")
        ax[r, 0].set_title(f"Original\n{name}")

        ax[r, 1].imshow(gt, cmap="gray")
        ax[r, 1].set_title("Ground Truth")

        ax[r, 2].imshow(prob, cmap="gray", vmin=0, vmax=1)
        ax[r, 2].set_title("Probabilidad")

        ax[r, 3].imshow(pred, cmap="gray", vmin=0, vmax=1)
        ax[r, 3].set_title("Predicción")

        # Superposición:
        # GT y pred se muestran sobre la angiografía.
        ax[r, 4].imshow(image, cmap="gray")

        ax[r, 4].imshow(np.ma.masked_where(gt < 0.5, gt), cmap="autumn", alpha=0.65)

        ax[r, 4].imshow(np.ma.masked_where(pred < 0.5, pred), cmap="winter", alpha=0.45)

        ax[r, 4].set_title("Superposición\nGT + Predicción")

        for c in range(5):
            ax[r, c].axis("off")

    plt.tight_layout()

    path = os.path.join(OUT_DIR, f"predicciones_epoch_{epoch:02d}.png")

    plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()

    return path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print(" FASE 3 - U-NET ARCADE STENOSIS")
    print(" BCE + TVERSKY | REDUCCIÓN DE FALSOS POSITIVOS")
    print("=" * 72)

    print(f"Dispositivo: {device}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Resolución: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Épocas: {START_EPOCH}-{END_EPOCH}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Tversky alpha: {TV_ALPHA}")
    print(f"Tversky beta:  {TV_BETA}")

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"No se encontró el checkpoint:\n{CHECKPOINT}")

    train_ds = ArcadeStenosisDataset(ROOT, "train", IMAGE_SIZE)
    val_ds = ArcadeStenosisDataset(ROOT, "val", IMAGE_SIZE)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train: {len(train_ds)} imágenes")
    print(f"Val:   {len(val_ds)} imágenes")

    model = UNet().to(device)

    checkpoint = torch.load(CHECKPOINT, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        previous_best = float(checkpoint.get("val_dice", 0.4404))
    else:
        model.load_state_dict(checkpoint)
        previous_best = 0.4404

    print("✓ Modelo de época 15 cargado")
    print(f"✓ Val Dice anterior: {previous_best:.4f}")

    criterion = BCETverskyLoss(
        bce_weight=BCE_WEIGHT, tv_weight=TV_WEIGHT, alpha=TV_ALPHA, beta=TV_BETA, smooth=TV_SMOOTH
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_dice = previous_best

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
    }

    for epoch in range(START_EPOCH, END_EPOCH + 1):
        start = time.time()

        print()
        print("=" * 72)
        print(f"ÉPOCA {epoch}/{END_EPOCH}")
        print("=" * 72)

        train_loss, train_metrics = train_one_epoch(model, train_dl, optimizer, criterion, device)

        val_loss, val_metrics = validate(model, val_dl, criterion, device)

        minutes = (time.time() - start) / 60.0

        print()
        print(f"  Train Loss:      {train_loss:.4f}")
        print(f"  Val Loss:        {val_loss:.4f}")
        print(f"  Train Dice:      {train_metrics['dice']:.4f}")
        print(f"  Val Dice:        {val_metrics['dice']:.4f}")
        print(f"  Val IoU:         {val_metrics['iou']:.4f}")
        print(f"  Val Precision:   {val_metrics['precision']:.4f}")
        print(f"  Val Recall:      {val_metrics['recall']:.4f}")
        print(f"  Tiempo época:    {minutes:.2f} min")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])

        with open(os.path.join(OUT_DIR, "historial_tversky.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "learning_rate": LEARNING_RATE,
                    "loss": "BCE + Tversky",
                    "tversky_alpha": TV_ALPHA,
                    "tversky_beta": TV_BETA,
                },
                os.path.join(OUT_DIR, "mejor_unet_arcade_tversky.pt"),
            )

            print(f"  ✓ Nuevo mejor modelo (Val Dice={best_val_dice:.4f})")

        save_predictions(model, val_ds, device, epoch, n=2)

    print()
    print("=" * 72)
    print(" FASE 3 TERMINADA")
    print("=" * 72)
    print(f"Mejor Val Dice: {best_val_dice:.4f}")
    print("Modelo:", os.path.join(OUT_DIR, "mejor_unet_arcade_tversky.pt"))
    print()
    print("Si no supera 0.4404, conservamos el modelo anterior.")
