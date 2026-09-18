import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ROOT = os.path.join(PROJECT_ROOT, "arcade")
CHECKPOINT = os.path.join(PROJECT_ROOT, "entrenamiento_unet", "mejor_unet_arcade.pt")
OUT_DIR = os.path.join(PROJECT_ROOT, "entrenamiento_continuado")

IMAGE_SIZE = 256
BATCH_SIZE = 2
START_EPOCH = 6
END_EPOCH = 15
LEARNING_RATE = 0.0002
THRESHOLD = 0.5
os.makedirs(OUT_DIR, exist_ok=True)


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
            if ann["category_id"] == 26 and ann["image_id"] in self.masks_by_image:
                self.masks_by_image[ann["image_id"]].append(ann["segmentation"])
        self.image_size = image_size

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        info = self.images_info[idx]
        image = Image.open(os.path.join(self.images_dir, info["file_name"])).convert("L")
        mask = Image.new("L", (info["width"], info["height"]), 0)
        draw = ImageDraw.Draw(mask)
        for polygons in self.masks_by_image[info["id"]]:
            for poly in polygons:
                if len(poly) >= 6:
                    pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
                    draw.polygon(pts, fill=255)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
        mask = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        return image, mask, info["file_name"]


class DoubleConv(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b),
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
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        s = 1e-6
        inter = (p * targets).sum((1, 2, 3))
        dice = (2 * inter + s) / (p.sum((1, 2, 3)) + targets.sum((1, 2, 3)) + s)
        return bce + 1 - dice.mean()


def metrics(logits, targets):
    p = torch.sigmoid(logits)
    pred = (p >= THRESHOLD).float()
    s = 1e-6
    inter = (pred * targets).sum((1, 2, 3))
    ps = pred.sum((1, 2, 3))
    ts = targets.sum((1, 2, 3))
    dice = ((2 * inter + s) / (ps + ts + s)).mean()
    iou = ((inter + s) / (ps + ts - inter + s)).mean()
    fp = (pred * (1 - targets)).sum((1, 2, 3))
    fn = ((1 - pred) * targets).sum((1, 2, 3))
    precision = ((inter + s) / (inter + fp + s)).mean()
    recall = ((inter + s) / (inter + fn + s)).mean()
    return dice.item(), iou.item(), precision.item(), recall.item()


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    losses = []
    vals = []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        losses.append(criterion(z, y).item())
        vals.append(metrics(z, y))
    a = np.array(vals)
    return np.mean(losses), *a.mean(axis=0).tolist()


@torch.no_grad()
def save_predictions(model, dataset, device, epoch, n=2):
    model.eval()
    idxs = np.linspace(0, len(dataset) - 1, min(n, len(dataset)), dtype=int)
    fig, ax = plt.subplots(len(idxs), 4, figsize=(14, 7 * len(idxs)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(idxs):
        x, y, name = dataset[i]
        z = model(x.unsqueeze(0).to(device))
        p = torch.sigmoid(z)[0, 0].cpu().numpy()
        pred = p >= THRESHOLD
        ax[r, 0].imshow(x[0], cmap="gray")
        ax[r, 0].set_title(f"Original\n{name}")
        ax[r, 1].imshow(y[0], cmap="gray")
        ax[r, 1].set_title("Ground Truth")
        ax[r, 2].imshow(p, cmap="gray", vmin=0, vmax=1)
        ax[r, 2].set_title("Probabilidad")
        ax[r, 3].imshow(pred, cmap="gray")
        ax[r, 3].set_title("Predicción")
        for c in range(4):
            ax[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"predicciones_epoch_{epoch:02d}.png"), dpi=150)
    plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 72)
print(" CONTINUACIÓN DEL ENTRENAMIENTO U-NET - ARCADE STENOSIS")
print("=" * 72)
print(f"Dispositivo: {device}")
print(f"Checkpoint inicial: {CHECKPOINT}")
print(f"Resolución: {IMAGE_SIZE}x{IMAGE_SIZE} | Batch: {BATCH_SIZE}")
print(f"Épocas: {START_EPOCH}-{END_EPOCH} | Learning rate: {LEARNING_RATE}")

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"No se encontró: {CHECKPOINT}")

train_ds = ArcadeStenosisDataset(ROOT, "train", IMAGE_SIZE)
val_ds = ArcadeStenosisDataset(ROOT, "val", IMAGE_SIZE)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)} imágenes | Val: {len(val_ds)} imágenes")

model = UNet().to(device)
ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(
    ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
)
print("✓ Mejor modelo cargado correctamente")

criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_val_dice = -1.0
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
    t = time.time()
    model.train()
    losses = []
    dices = []
    print("\n" + "=" * 72)
    print(f"ÉPOCA {epoch}/{END_EPOCH}")
    print("=" * 72)
    for bi, (x, y, _) in enumerate(train_dl, 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        dices.append(metrics(z.detach(), y)[0])
        if bi == 1 or bi % 100 == 0 or bi == len(train_dl):
            print(f"  batch {bi:3d}/{len(train_dl)} | loss {loss.item():.4f}")
    train_loss = np.mean(losses)
    train_dice = np.mean(dices)
    val_loss, val_dice, val_iou, val_precision, val_recall = validate(
        model, val_dl, device, criterion
    )
    mins = (time.time() - t) / 60
    print(f"\n  Train Loss:      {train_loss:.4f}")
    print(f"  Val Loss:        {val_loss:.4f}")
    print(f"  Train Dice:      {train_dice:.4f}")
    print(f"  Val Dice:        {val_dice:.4f}")
    print(f"  Val IoU:         {val_iou:.4f}")
    print(f"  Val Precision:   {val_precision:.4f}")
    print(f"  Val Recall:      {val_recall:.4f}")
    print(f"  Tiempo época:    {mins:.2f} min")
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_dice"].append(train_dice)
    history["val_dice"].append(val_dice)
    history["val_iou"].append(val_iou)
    history["val_precision"].append(val_precision)
    history["val_recall"].append(val_recall)
    with open(os.path.join(OUT_DIR, "historial_continuado.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "learning_rate": LEARNING_RATE,
            },
            os.path.join(OUT_DIR, "mejor_unet_arcade_continuado.pt"),
        )
        print(f"  ✓ Nuevo mejor modelo (Val Dice={val_dice:.4f})")
    save_predictions(model, val_ds, device, epoch, 2)

print("\n" + "=" * 72)
print(" ENTRENAMIENTO CONTINUADO TERMINADO")
print("=" * 72)
print(f"Mejor Val Dice: {best_val_dice:.4f}")
print(f"Modelo: {os.path.join(OUT_DIR, 'mejor_unet_arcade_continuado.pt')}")
print(f"Resultados: {OUT_DIR}")
