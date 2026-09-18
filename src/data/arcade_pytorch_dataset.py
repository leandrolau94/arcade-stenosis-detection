import json
import os
import random

import cv2
import numpy as np
import torch
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

IMAGES_DIR = os.path.join(
    BASE_DIR, SPLIT, "images"
)

STENOSIS_CATEGORY_ID = 26

BATCH_SIZE = 4
NUM_WORKERS = 0       # 0 funciona en Windows sin problemas
NUM_EJEMPLOS = 4

SEED = 42


# ============================================================
# DATASET
# ============================================================

class ARCADEStenosisDataset(Dataset):
    """
    Dataset de PyTorch para ARCADE/stenosis.

    Cada muestra devuelve:

        image -> tensor [1, H, W], valores [0, 1]
        mask  -> tensor [1, H, W], valores {0, 1}

    La máscara se construye directamente desde segmentation
    del JSON, por lo que no necesitamos guardar 1000 máscaras
    adicionales en disco.
    """

    def __init__(
        self,
        annotations_file,
        images_dir,
        stenosis_category_id=26
    ):

        self.images_dir = images_dir
        self.stenosis_category_id = stenosis_category_id

        with open(
            annotations_file,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        self.images = data["images"]

        # Diccionario: image_id -> información de imagen
        self.images_by_id = {
            image["id"]: image
            for image in self.images
        }

        # Guardamos solamente las anotaciones de estenosis
        self.annotations_by_image = {}

        for ann in data["annotations"]:

            if ann.get("category_id") != stenosis_category_id:
                continue

            image_id = ann["image_id"]

            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []

            self.annotations_by_image[image_id].append(ann)

        # Incluimos también las 3 imágenes sin estenosis.
        self.image_ids = [
            image["id"]
            for image in self.images
        ]


    def __len__(self):
        return len(self.image_ids)


    def _create_mask(self, height, width, annotations):

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        for ann in annotations:

            segmentations = ann.get(
                "segmentation",
                []
            )

            for polygon in segmentations:

                if not polygon or len(polygon) < 6:
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

                points = points.reshape(
                    (-1, 1, 2)
                )

                cv2.fillPoly(
                    mask,
                    [points],
                    1
                )

        return mask


    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image_info = self.images_by_id[image_id]

        file_name = image_info["file_name"]

        image_path = os.path.join(
            self.images_dir,
            file_name
        )

        # --------------------------------------------
        # LEER IMAGEN
        # --------------------------------------------

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"No se pudo abrir: {image_path}"
            )

        height, width = image.shape

        # --------------------------------------------
        # NORMALIZAR IMAGEN
        # --------------------------------------------

        image = image.astype(
            np.float32
        ) / 255.0

        # [H, W] -> [1, H, W]
        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        # --------------------------------------------
        # CREAR MÁSCARA
        # --------------------------------------------

        annotations = self.annotations_by_image.get(
            image_id,
            []
        )

        mask = self._create_mask(
            height,
            width,
            annotations
        )

        # [H, W] -> [1, H, W]
        mask = torch.from_numpy(
            mask.astype(np.float32)
        ).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "image_id": image_id,
            "file_name": file_name,
        }


# ============================================================
# VISUALIZACIÓN
# ============================================================

def visualizar_batch(batch):

    images = batch["image"]
    masks = batch["mask"]
    file_names = batch["file_name"]

    n = min(
        len(images),
        NUM_EJEMPLOS
    )

    fig, axes = plt.subplots(
        n,
        3,
        figsize=(12, 4 * n)
    )

    if n == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    for i in range(n):

        image = images[i, 0].numpy()
        mask = masks[i, 0].numpy()

        # Overlay
        overlay = np.stack(
            [image, image, image],
            axis=-1
        )

        lesion = mask > 0.5

        overlay[lesion] = (
            0.45 * overlay[lesion]
            + 0.55 * np.array([1.0, 0.0, 0.0])
        )

        axes[i, 0].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[i, 0].set_title(
            f"Original\n{file_names[i]}"
        )

        axes[i, 1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[i, 1].set_title(
            "Máscara"
        )

        axes[i, 2].imshow(
            overlay
        )

        axes[i, 2].set_title(
            "Imagen + máscara"
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print(" ARCADE -> PYTORCH DATASET")
    print("=" * 65)

    # --------------------------------------------
    # SEMILLA
    # --------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --------------------------------------------
    # COMPROBAR PYTORCH
    # --------------------------------------------

    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():

        print(
            "GPU disponible:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "GPU disponible: NO"
        )

    # --------------------------------------------
    # CREAR DATASET
    # --------------------------------------------

    dataset = ARCADEStenosisDataset(
        annotations_file=ANNOTATIONS_FILE,
        images_dir=IMAGES_DIR,
        stenosis_category_id=STENOSIS_CATEGORY_ID
    )

    print()
    print(f"Muestras del dataset: {len(dataset)}")

    # --------------------------------------------
    # PRIMERA MUESTRA
    # --------------------------------------------

    sample = dataset[0]

    print()
    print("PRIMERA MUESTRA")
    print(
        "  archivo:",
        sample["file_name"]
    )
    print(
        "  image_id:",
        sample["image_id"]
    )
    print(
        "  image shape:",
        tuple(sample["image"].shape)
    )
    print(
        "  mask shape:",
        tuple(sample["mask"].shape)
    )
    print(
        "  image min/max:",
        float(sample["image"].min()),
        float(sample["image"].max())
    )
    print(
        "  valores máscara:",
        torch.unique(sample["mask"]).tolist()
    )

    # --------------------------------------------
    # DATALOADER
    # --------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    batch = next(iter(loader))

    print()
    print("PRIMER BATCH")
    print(
        "  image batch:",
        tuple(batch["image"].shape)
    )
    print(
        "  mask batch:",
        tuple(batch["mask"].shape)
    )
    print(
        "  image IDs:",
        batch["image_id"].tolist()
    )
    print(
        "  archivos:",
        batch["file_name"]
    )

    # --------------------------------------------
    # PORCENTAJE DE PÍXELES DE LESIÓN
    # --------------------------------------------

    lesion_percentage = (
        batch["mask"].sum(dim=(1, 2, 3))
        / batch["mask"].numel()
        * 100
    )

    print()
    print(
        "Píxeles de lesión por muestra (% del batch):"
    )

    for i, value in enumerate(
        lesion_percentage
    ):
        print(
            f"  {batch['file_name'][i]}: "
            f"{float(value):.4f}%"
        )

    # --------------------------------------------
    # VISUALIZAR
    # --------------------------------------------

    print()
    print(
        "Mostrando el primer batch..."
    )

    visualizar_batch(batch)

    print()
    print("=" * 65)
    print(" DATASET COMPROBADO")
    print("=" * 65)