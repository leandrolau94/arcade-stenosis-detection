import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = "arcade/stenosis"
SPLIT = "train"
ANNOTATIONS_FILE = os.path.join(BASE_DIR, SPLIT, "annotations", f"{SPLIT}.json")
OUTPUT_DIR = os.path.join("analisis_dataset", SPLIT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STENOSIS_CATEGORY_ID = 26

print("=" * 60)
print(" ANÁLISIS ARCADE - STENOSIS")
print("=" * 60)

with open(ANNOTATIONS_FILE, encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

stenosis = [a for a in annotations if a.get("category_id") == STENOSIS_CATEGORY_ID]

images_by_id = {im["id"]: im for im in images}
annotations_by_image = defaultdict(list)

for ann in stenosis:
    annotations_by_image[ann["image_id"]].append(ann)

# 1) Número de lesiones por imagen
counts = Counter(len(annotations_by_image.get(im["id"], [])) for im in images)

images_with_stenosis = sum(
    n > 0 for n in (len(annotations_by_image.get(im["id"], [])) for im in images)
)

print(f"Imágenes totales: {len(images)}")
print(f"Anotaciones totales: {len(annotations)}")
print(f"Anotaciones de estenosis (cat. 26): {len(stenosis)}")
print(f"Imágenes con estenosis: {images_with_stenosis}")
print(f"Imágenes sin estenosis: {len(images) - images_with_stenosis}")

print("\nESTENOSIS POR IMAGEN")
for n in sorted(counts):
    print(f"  {n} estenosis -> {counts[n]} imágenes")

# 2) Estadísticas de bounding boxes
rows = []
for ann in stenosis:
    x, y, w, h = ann["bbox"]
    rows.append(
        {
            "image_id": ann["image_id"],
            "file_name": images_by_id[ann["image_id"]]["file_name"],
            "annotation_id": ann["id"],
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "bbox_area": w * h,
            "center_x": x + w / 2,
            "center_y": y + h / 2,
            "segmentation_area": ann.get("area", np.nan),
            "occluded": ann.get("attributes", {}).get("occluded", False),
        }
    )

df = pd.DataFrame(rows)

print("\nTAMAÑO DE LAS LESIONES")
for col in ["width", "height", "bbox_area"]:
    print(
        f"  {col}: min={df[col].min():.2f}, "
        f"media={df[col].mean():.2f}, "
        f"mediana={df[col].median():.2f}, "
        f"max={df[col].max():.2f}"
    )

print("\nPOSICIÓN")
print(
    f"  centro X: media={df.center_x.mean():.2f}, "
    f"rango={df.center_x.min():.2f}-{df.center_x.max():.2f}"
)
print(
    f"  centro Y: media={df.center_y.mean():.2f}, "
    f"rango={df.center_y.min():.2f}-{df.center_y.max():.2f}"
)

# 3) Segmentaciones
polygon_points = []
for ann in stenosis:
    for polygon in ann.get("segmentation", []):
        if polygon:
            polygon_points.append(len(polygon) // 2)

print("\nSEGMENTACIONES")
print(f"  polígonos: {len(polygon_points)}")
if polygon_points:
    print(
        f"  puntos por polígono: min={min(polygon_points)}, "
        f"media={np.mean(polygon_points):.2f}, max={max(polygon_points)}"
    )

# 4) Guardar tablas
df.to_csv(os.path.join(OUTPUT_DIR, "estenosis_detalle.csv"), index=False)

summary = pd.DataFrame(
    [
        ["total_images", len(images)],
        ["total_annotations", len(annotations)],
        ["stenosis_annotations", len(stenosis)],
        ["images_with_stenosis", images_with_stenosis],
        ["images_without_stenosis", len(images) - images_with_stenosis],
        [
            "mean_stenoses_per_image",
            np.mean([len(annotations_by_image.get(im["id"], [])) for im in images]),
        ],
        [
            "median_stenoses_per_image",
            np.median([len(annotations_by_image.get(im["id"], [])) for im in images]),
        ],
        ["mean_bbox_width", df.width.mean()],
        ["mean_bbox_height", df.height.mean()],
        ["mean_bbox_area", df.bbox_area.mean()],
    ],
    columns=["metric", "value"],
)

summary.to_csv(os.path.join(OUTPUT_DIR, "resumen.csv"), index=False)

# 5) Gráfico: lesiones por imagen
plt.figure(figsize=(8, 5))
x = sorted(counts)
y = [counts[n] for n in x]
plt.bar(x, y)
plt.xlabel("Número de estenosis por imagen")
plt.ylabel("Número de imágenes")
plt.title("Distribución de estenosis por imagen")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_estenosis_por_imagen.png"), dpi=150)
plt.show()
plt.close()

# 6) Gráfico: ancho y alto
plt.figure(figsize=(8, 5))
plt.hist(df.width, bins=30, alpha=0.7, label="Ancho")
plt.hist(df.height, bins=30, alpha=0.7, label="Alto")
plt.xlabel("Píxeles")
plt.ylabel("Número de lesiones")
plt.title("Tamaño de las bounding boxes")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_tamano_bbox.png"), dpi=150)
plt.show()
plt.close()

# 7) Gráfico: área
plt.figure(figsize=(8, 5))
plt.hist(df.bbox_area, bins=30)
plt.xlabel("Área del bounding box (px²)")
plt.ylabel("Número de lesiones")
plt.title("Distribución del área de las lesiones")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_area_bbox.png"), dpi=150)
plt.show()
plt.close()

# 8) Mapa espacial
plt.figure(figsize=(7, 7))
plt.scatter(df.center_x, df.center_y, alpha=0.5)
plt.xlim(0, 512)
plt.ylim(512, 0)
plt.xlabel("X (píxeles)")
plt.ylabel("Y (píxeles)")
plt.title("Distribución espacial de las estenosis")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_mapa_posiciones.png"), dpi=150)
plt.show()
plt.close()

print("\nArchivos generados en:", OUTPUT_DIR)
print("  resumen.csv")
print("  estenosis_detalle.csv")
print("  01_estenosis_por_imagen.png")
print("  02_tamano_bbox.png")
print("  03_area_bbox.png")
print("  04_mapa_posiciones.png")
