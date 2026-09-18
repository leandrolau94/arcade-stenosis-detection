import json
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = "arcade/stenosis"

SPLIT = "train"# train, val or test
NUM_IMAGENES = 5
SEED = 42

ANNOTATIONS_FILE = os.path.join(BASE_DIR, SPLIT, "annotations", f"{SPLIT}.json")

IMAGES_DIR = os.path.join(BASE_DIR, SPLIT, "images")

OUTPUT_DIR = os.path.join("mascaras_visualizacion", SPLIT)

os.makedirs(OUTPUT_DIR, exist_ok=True)

STENOSIS_CATEGORY_ID = 26


# ============================================================
# FUNCIONES
# ============================================================

def crear_mascara(image_height, image_width, annotations):
    """
    Crea una máscara binaria de 0/1.

    0 = fondo
    1 = estenosis
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for ann in annotations:
        segmentations = ann.get("segmentation", [])
        for polygon in segmentations:
            # Necesitamos al menos 3 puntos = 6 coordenadas
            if not polygon or len(polygon) < 6:
                continue
            points = np.array([[int(round(polygon[i])), int(round(polygon[i + 1]))] for i in range(0, len(polygon), 2)], dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            # Rellenamos el polígono
            cv2.fillPoly(mask, [points], 1)
    return mask


def crear_overlay(image_rgb, mask):
    """Superpone la máscara sobre la imagen original."""
    overlay = image_rgb.copy()
    # Región de estenosis
    lesion = mask == 1
    # Aplicamos una superposición roja
    overlay[lesion] = (0.45 * overlay[lesion] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    # Contorno de la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    return overlay


# ============================================================
# CARGAR DATASET
# ============================================================

print("=" * 65)
print(" PREPARACIÓN DE MÁSCARAS ARCADE - ESTENOSIS")
print("=" * 65)

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

images_by_id = {image["id"]: image for image in images}

annotations_by_image = {}

for ann in annotations:
    if ann.get("category_id") != STENOSIS_CATEGORY_ID:
        continue
    image_id = ann["image_id"]
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(ann)


image_ids = list(annotations_by_image.keys())

random.seed(SEED)

num = min(NUM_IMAGENES, len(image_ids))

selected_ids = random.sample(image_ids, num)

print(f"Imágenes del split: {len(images)}")
print(f"Estenosis:          {sum(len(v) for v in annotations_by_image.values())}")
print(f"Imágenes a mostrar: {num}")
print()


# ============================================================
# PROCESAR EJEMPLOS
# ============================================================

for counter, image_id in enumerate(selected_ids, start=1):
    image_info = images_by_id[image_id]
    file_name = image_info["file_name"]
    image_path = os.path.join(IMAGES_DIR, file_name)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[AVISO] No se pudo abrir {image_path}")
        continue
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    anns = annotations_by_image[image_id]
    # --------------------------------------------------------
    # CREAR MÁSCARA
    # --------------------------------------------------------
    mask = crear_mascara(height, width, anns)
    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------
    overlay = crear_overlay(image_rgb, mask)
    # --------------------------------------------------------
    # ESTADÍSTICAS DE LA MÁSCARA
    # --------------------------------------------------------
    lesion_pixels = int(np.sum(mask))
    total_pixels = mask.size
    percentage = (100 * lesion_pixels / total_pixels)
    print(f"[{counter}/{num}] {file_name} | "f"estenosis: {len(anns)} | "f"píxeles de lesión: {lesion_pixels:,} | "f"{percentage:.3f}% de la imagen")

    # --------------------------------------------------------
    # GUARDAR MÁSCARA
    # --------------------------------------------------------

    mask_path = os.path.join(OUTPUT_DIR, f"mask_{file_name}")

    cv2.imwrite(mask_path, mask * 255)

    # --------------------------------------------------------
    # GUARDAR OVERLAY
    # --------------------------------------------------------

    overlay_path = os.path.join(OUTPUT_DIR, f"overlay_{file_name}")

    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # --------------------------------------------------------
    # VISUALIZACIÓN
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("1. Imagen original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("2. Máscara de estenosis")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("3. Imagen + máscara")
    axes[2].axis("off")

    fig.suptitle(f"{file_name} | {len(anns)} estenosis")

    plt.tight_layout()

    figure_path = os.path.join(OUTPUT_DIR, f"comparacion_{file_name}")

    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

# ============================================================
# COMPROBAR VALORES DE LA MÁSCARA
# ============================================================

print()
print("=" * 65)
print(" COMPROBACIÓN")
print("=" * 65)

print("Las máscaras deben contener únicamente los valores:")
print("[0, 1]")

print()
print("En el PNG guardado:")
print("  0   = fondo")
print("  255 = estenosis")

print()
print("Resultados guardados en:")
print(OUTPUT_DIR)