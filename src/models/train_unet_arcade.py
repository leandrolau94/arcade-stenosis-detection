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

BASE_DIR = "arcade/stenosis"

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

TRAIN_JSON = os.path.join(BASE_DIR, TRAIN_SPLIT, "annotations", f"{TRAIN_SPLIT}.json")
VAL_JSON = os.path.join(BASE_DIR, VAL_SPLIT, "annotations", f"{VAL_SPLIT}.json")

TRAIN_IMAGES = os.path.join(BASE_DIR, TRAIN_SPLIT, "images")
VAL_IMAGES = os.path.join(BASE_DIR, VAL_SPLIT, "images")

OUTPUT_DIR = "entrenamiento_unet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "mejor_unet_arcade.pt")

STENOSIS_CATEGORY_ID = 26

IMAGE_SIZE = 256
BATCH_SIZE = 2
NUM_WORKERS = 0

EPOCHS = 5
LEARNING_RATE = 1e-3

SEED = 42


# ============================================================
# DATASET
# ============================================================


class ARCADEDataset(Dataset):
    def __init__(self, annotations_file, images_dir, image_size=256, stenosis_category_id=26):

        self.images_dir = images_dir
        self.image_size = image_size

        with open(annotations_file, encoding="utf-8") as f:
            data = json.load(f)

        self.images = data["images"]

        self.images_by_id = {img["id"]: img for img in self.images}

        self.annotations_by_image = {}

        for ann in data["annotations"]:
            if ann.get("category_id") != stenosis_category_id:
                continue

            image_id = ann["image_id"]

            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []

            self.annotations_by_image[image_id].append(ann)

        self.image_ids = [img["id"] for img in self.images]

    def __len__(self):
        return len(self.image_ids)

    def _create_mask(self, height, width, annotations):

        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in annotations:
            for polygon in ann.get("segmentation", []):
                if not polygon or len(polygon) < 6:
                    continue

                points = np.array(
                    [
                        [int(round(polygon[i])), int(round(polygon[i + 1]))]
                        for i in range(0, len(polygon), 2)
                    ],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))

                cv2.fillPoly(mask, [points], 1)

        return mask

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image_info = self.images_by_id[image_id]

        file_name = image_info["file_name"]

        image_path = os.path.join(self.images_dir, file_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"No se pudo abrir: {image_path}")

        h, w = image.shape

        annotations = self.annotations_by_image.get(image_id, [])

        mask = self._create_mask(h, w, annotations)

        # Imagen -> bilinear
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )

        # Máscara -> nearest neighbor
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0

        mask = mask.astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)

        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask, file_name


# ============================================================
# U-NET PEQUEÑA
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

        d3 = self.up3(b)

        d3 = torch.cat([d3, e3], dim=1)

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat([d2, e2], dim=1)

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat([d1, e1], dim=1)

        d1 = self.dec1(d1)

        return self.out(d1)


# ============================================================
# LOSS
# ============================================================


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)

        probs = probs.reshape(probs.shape[0], -1)

        targets_flat = targets.reshape(targets.shape[0], -1)

        intersection = (probs * targets_flat).sum(dim=1)

        dice = (2 * intersection + 1e-6) / (probs.sum(dim=1) + targets_flat.sum(dim=1) + 1e-6)

        dice_loss = 1 - dice.mean()

        return bce_loss + dice_loss


# ============================================================
# MÉTRICAS
# ============================================================


def calculate_metrics(logits, targets, threshold=0.5):

    probs = torch.sigmoid(logits)

    preds = (probs > threshold).float()

    preds = preds.reshape(preds.shape[0], -1)

    targets = targets.reshape(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)

    pred_sum = preds.sum(dim=1)

    target_sum = targets.sum(dim=1)

    union = pred_sum + target_sum - intersection

    dice = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)

    iou = (intersection + 1e-6) / (union + 1e-6)

    precision = (intersection + 1e-6) / (pred_sum + 1e-6)

    recall = (intersection + 1e-6) / (target_sum + 1e-6)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


# ============================================================
# ENTRENAMIENTO DE UNA ÉPOCA
# ============================================================


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0.0

    metric_sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    batches = 0

    for batch_idx, (images, masks, _) in enumerate(loader, start=1):
        images = images.to(device)

        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        metrics = calculate_metrics(logits.detach(), masks)

        for key in metric_sums:
            metric_sums[key] += metrics[key]

        batches += 1

        if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == len(loader):
            print(f"    batch {batch_idx:>3}/{len(loader)} | loss {loss.item():.4f}")

    result = {"loss": running_loss / batches}

    for key in metric_sums:
        result[key] = metric_sums[key] / batches

    return result


# ============================================================
# VALIDACIÓN
# ============================================================


def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0

    metric_sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    batches = 0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)

            masks = masks.to(device)

            logits = model(images)

            loss = criterion(logits, masks)

            running_loss += loss.item()

            metrics = calculate_metrics(logits, masks)

            for key in metric_sums:
                metric_sums[key] += metrics[key]

            batches += 1

    result = {"loss": running_loss / batches}

    for key in metric_sums:
        result[key] = metric_sums[key] / batches

    return result


# ============================================================
# VISUALIZAR PREDICCIONES
# ============================================================


def save_prediction_examples(model, loader, device, epoch, output_dir, max_examples=4):

    model.eval()

    images, masks, names = next(iter(loader))

    images = images.to(device)

    with torch.no_grad():
        logits = model(images)

        probs = torch.sigmoid(logits)

        preds = (probs > 0.5).float()

    n = min(max_examples, images.shape[0])

    fig, axes = plt.subplots(n, 4, figsize=(14, 4 * n))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        image = images[i, 0].cpu().numpy()

        mask = masks[i, 0].numpy()

        prob = probs[i, 0].cpu().numpy()

        pred = preds[i, 0].cpu().numpy()

        axes[i, 0].imshow(image, cmap="gray")

        axes[i, 0].set_title(f"Original\n{names[i]}")

        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)

        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(prob, cmap="gray", vmin=0, vmax=1)

        axes[i, 2].set_title("Probabilidad")

        axes[i, 3].imshow(pred, cmap="gray", vmin=0, vmax=1)

        axes[i, 3].set_title("Predicción")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()

    path = os.path.join(output_dir, f"predicciones_epoch_{epoch:02d}.png")

    plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()

    return path


# ============================================================
# GRÁFICAS DEL HISTORIAL
# ============================================================


def save_history_plots(history, output_dir):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, history["train_loss"], marker="o", label="Train")

    plt.plot(epochs, history["val_loss"], marker="o", label="Validation")

    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss de entrenamiento y validación")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "loss.png"), dpi=150)

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, history["train_dice"], marker="o", label="Train Dice")

    plt.plot(epochs, history["val_dice"], marker="o", label="Validation Dice")

    plt.xlabel("Época")
    plt.ylabel("Dice")
    plt.title("Dice de entrenamiento y validación")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "dice.png"), dpi=150)

    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print(" ENTRENAMIENTO U-NET - ARCADE STENOSIS")
    print("=" * 72)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")

    print(f"Dispositivo: {device}")
    print(f"Resolución: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Épocas: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = ARCADEDataset(TRAIN_JSON, TRAIN_IMAGES, IMAGE_SIZE, STENOSIS_CATEGORY_ID)

    val_dataset = ARCADEDataset(VAL_JSON, VAL_IMAGES, IMAGE_SIZE, STENOSIS_CATEGORY_ID)

    print()
    print(f"Train: {len(train_dataset)} imágenes")

    print(f"Val:   {len(val_dataset)} imágenes")

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # MODELO
    # --------------------------------------------------------

    model = UNet().to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parámetros entrenables: {params:,}")

    # --------------------------------------------------------
    # LOSS Y OPTIMIZER
    # --------------------------------------------------------

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --------------------------------------------------------
    # HISTORIAL
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
    }

    best_val_dice = -1.0

    total_start = time.time()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        print()
        print(f"ÉPOCA {epoch}/{EPOCHS}")

        print("  Entrenando...")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)

        print("  Validando...")

        val_metrics = validate(model, val_loader, criterion, device)

        epoch_seconds = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])

        history["val_loss"].append(val_metrics["loss"])

        history["train_dice"].append(train_metrics["dice"])

        history["val_dice"].append(val_metrics["dice"])

        history["train_iou"].append(train_metrics["iou"])

        history["val_iou"].append(val_metrics["iou"])

        print()
        print(f"  Train Loss:      {train_metrics['loss']:.4f}")

        print(f"  Val Loss:        {val_metrics['loss']:.4f}")

        print(f"  Train Dice:      {train_metrics['dice']:.4f}")

        print(f"  Val Dice:        {val_metrics['dice']:.4f}")

        print(f"  Train IoU:       {train_metrics['iou']:.4f}")

        print(f"  Val IoU:         {val_metrics['iou']:.4f}")

        print(f"  Val Precision:   {val_metrics['precision']:.4f}")

        print(f"  Val Recall:      {val_metrics['recall']:.4f}")

        print(f"  Tiempo época:    {epoch_seconds / 60:.2f} min")

        # ----------------------------------------------------
        # GUARDAR MEJOR MODELO
        # ----------------------------------------------------

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_val_dice,
                    "image_size": IMAGE_SIZE,
                },
                BEST_MODEL_PATH,
            )

            print(f"  ✓ Nuevo mejor modelo (Val Dice={best_val_dice:.4f})")

        # ----------------------------------------------------
        # GUARDAR EJEMPLOS
        # ----------------------------------------------------

        example_path = save_prediction_examples(
            model, val_loader, device, epoch, OUTPUT_DIR, max_examples=4
        )

        print(f"  Ejemplos: {example_path}")

    # ========================================================
    # FINAL
    # ========================================================

    total_seconds = time.time() - total_start

    save_history_plots(history, OUTPUT_DIR)

    print()
    print("=" * 72)
    print(" ENTRENAMIENTO TERMINADO")
    print("=" * 72)

    print(f"Mejor Val Dice: {best_val_dice:.4f}")

    print(f"Tiempo total: {total_seconds / 60:.2f} min")

    print()
    print("Mejor modelo guardado en:")

    print(BEST_MODEL_PATH)

    print()
    print("Resultados y gráficas en:")

    print(OUTPUT_DIR)

    print()
    print("Archivos esperados:")

    print("  mejor_unet_arcade.pt")

    print("  loss.png")

    print("  dice.png")

    print("  predicciones_epoch_01.png")

    print("  ...")
