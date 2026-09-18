import os
import json
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = "arcade/stenosis"
SPLIT = "train"

ANNOTATIONS_FILE = os.path.join(
    BASE_DIR, SPLIT, "annotations", f"{SPLIT}.json"
)
IMAGES_DIR = os.path.join(BASE_DIR, SPLIT, "images")

STENOSIS_CATEGORY_ID = 26

IMAGE_SIZE = 256
BATCH_SIZE = 2
NUM_WORKERS = 0

SEED = 42


# ============================================================
# DATASET
# ============================================================

class ARCADEDataset(Dataset):

    def __init__(
        self,
        annotations_file,
        images_dir,
        image_size=256,
        stenosis_category_id=26
    ):

        self.images_dir = images_dir
        self.image_size = image_size
        self.stenosis_category_id = stenosis_category_id

        with open(
            annotations_file,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        self.images = data["images"]

        self.images_by_id = {
            image["id"]: image
            for image in self.images
        }

        self.annotations_by_image = {}

        for ann in data["annotations"]:

            if ann.get("category_id") != stenosis_category_id:
                continue

            image_id = ann["image_id"]

            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []

            self.annotations_by_image[image_id].append(ann)

        self.image_ids = [
            image["id"]
            for image in self.images
        ]


    def __len__(self):
        return len(self.image_ids)


    def create_mask(self, height, width, annotations):

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        for ann in annotations:

            for polygon in ann.get("segmentation", []):

                if len(polygon) < 6:
                    continue

                points = np.array(
                    [
                        [
                            int(round(polygon[i])),
                            int(round(polygon[i + 1]))
                        ]
                        for i in range(0, len(polygon), 2)
                    ],
                    dtype=np.int32
                )

                points = points.reshape((-1, 1, 2))

                cv2.fillPoly(
                    mask,
                    [points],
                    1
                )

        return mask


    def __getitem__(self, index):

        image_id = self.image_ids[index]
        info = self.images_by_id[image_id]
        file_name = info["file_name"]

        path = os.path.join(
            self.images_dir,
            file_name
        )

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(path)

        height, width = image.shape

        annotations = self.annotations_by_image.get(
            image_id,
            []
        )

        mask = self.create_mask(
            height,
            width,
            annotations
        )

        # Imagen: interpolación bilineal
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Máscara: SIEMPRE nearest neighbor.
        # No queremos crear valores intermedios en la máscara.
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        mask = torch.from_numpy(
            mask.astype(np.float32)
        ).unsqueeze(0)

        return image, mask, file_name


# ============================================================
# BLOQUE CONVOLUCIONAL
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        # Una salida:
        # 0 = fondo
        # 1 = estenosis
        self.out = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )


    def forward(self, x):

        # -------------------------
        # Encoder
        # -------------------------

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        # -------------------------
        # Bottleneck
        # -------------------------

        b = self.bottleneck(
            self.pool(e3)
        )

        # -------------------------
        # Decoder
        # -------------------------

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.out(d1)


# ============================================================
# UTILIDADES
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def dice_score_from_logits(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > 0.5
    ).float()

    predictions = predictions.reshape(
        predictions.shape[0],
        -1
    )

    targets = targets.reshape(
        targets.shape[0],
        -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2 * intersection + 1e-6
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + 1e-6
    )

    return dice.mean().item()


def visualize_predictions(
    images,
    masks,
    logits,
    names
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities > 0.5
    ).float()

    n = len(images)

    fig, axes = plt.subplots(
        n,
        4,
        figsize=(14, 4 * n)
    )

    if n == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    for i in range(n):

        image = images[i, 0].cpu().numpy()
        mask = masks[i, 0].cpu().numpy()
        probability = probabilities[i, 0].detach().cpu().numpy()
        prediction = predictions[i, 0].cpu().numpy()

        axes[i, 0].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[i, 0].set_title(
            f"Original\n{names[i]}"
        )

        axes[i, 1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[i, 1].set_title(
            "Ground Truth"
        )

        axes[i, 2].imshow(
            probability,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[i, 2].set_title(
            "Probabilidad predicha"
        )

        axes[i, 3].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[i, 3].set_title(
            "Predicción > 0.5"
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(" PRIMERA U-NET PARA ARCADE")
    print("=" * 70)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")

    print(f"Dispositivo: {device}")
    print(f"Tamaño de entrada: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")

    # -------------------------
    # Dataset
    # -------------------------

    dataset = ARCADEDataset(
        annotations_file=ANNOTATIONS_FILE,
        images_dir=IMAGES_DIR,
        image_size=IMAGE_SIZE,
        stenosis_category_id=STENOSIS_CATEGORY_ID
    )

    print(
        f"Muestras: {len(dataset)}"
    )

    # -------------------------
    # DataLoader
    # -------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # -------------------------
    # Modelo
    # -------------------------

    model = UNet().to(device)

    print(
        f"Parámetros entrenables: "
        f"{count_parameters(model):,}"
    )

    # -------------------------
    # Batch
    # -------------------------

    images, masks, names = next(
        iter(loader)
    )

    images = images.to(device)
    masks = masks.to(device)

    print()
    print("BATCH")
    print(
        f"  imágenes: {tuple(images.shape)}"
    )
    print(
        f"  máscaras: {tuple(masks.shape)}"
    )
    print(
        f"  archivos: {list(names)}"
    )

    # -------------------------
    # Forward pass
    # -------------------------

    model.eval()

    with torch.no_grad():

        logits = model(images)

    print()
    print("FORWARD PASS")
    print(
        f"  entrada: {tuple(images.shape)}"
    )
    print(
        f"  salida:  {tuple(logits.shape)}"
    )
    print(
        f"  logits min: {float(logits.min()):.4f}"
    )
    print(
        f"  logits max: {float(logits.max()):.4f}"
    )

    # -------------------------
    # Probabilidades
    # -------------------------

    probabilities = torch.sigmoid(
        logits
    )

    print(
        f"  probabilidad min: "
        f"{float(probabilities.min()):.4f}"
    )

    print(
        f"  probabilidad max: "
        f"{float(probabilities.max()):.4f}"
    )

    # -------------------------
    # Dice SIN ENTRENAR
    # -------------------------

    dice = dice_score_from_logits(
        logits,
        masks
    )

    print()
    print(
        f"Dice antes de entrenar: {dice:.4f}"
    )

    print()
    print(
        "IMPORTANTE: esta predicción es "
        "ALEATORIA porque la U-Net todavía "
        "NO ha sido entrenada."
    )

    print()
    print(
        "Mostrando original / ground truth / "
        "probabilidad / predicción..."
    )

    visualize_predictions(
        images,
        masks,
        logits,
        names
    )

    print()
    print("=" * 70)
    print(" U-NET Y PIPELINE COMPROBADOS")
    print("=" * 70)
    print()
    print("El siguiente paso será entrenar la red.")