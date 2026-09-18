import json
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Set Up
# ============================================================

BASE_DIR = "arcade/stenosis"

SPLIT = "train"          # train, val or test
NUM_IMAGENES = 10        # how many random images to visualize
SEED = 42                # to repeat same selection

ANNOTATIONS_FILE = os.path.join(BASE_DIR, SPLIT, "annotations", f"{SPLIT}.json")

IMAGES_DIR = os.path.join(BASE_DIR, SPLIT, "images")

OUTPUT_DIR = os.path.join("resultados_visualizer", SPLIT)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Uploading dataset
# ============================================================

print("==========================================")
print(" VISUALIZER ARCADE - STENOSIS")
print("==========================================")
print(f"Split: {SPLIT}")
print(f"JSON:  {ANNOTATIONS_FILE}")
print()

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

print(f"Images:      {len(images)}")
print(f"Annotations:   {len(annotations)}")

# ARCADE use category_id=26 for "stenosis"
stenosis_annotations = [ann for ann in annotations if ann.get("category_id") == 26]

print(f"Estenosis:    {len(stenosis_annotations)}")
print()


# ============================================================
# Creating Index
# ============================================================

images_by_id = {image["id"]: image for image in images}

annotations_by_image = {}

for ann in stenosis_annotations:
    image_id = ann["image_id"]
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(ann)


# Only selecting images that has at least
# one annotation of stenosis
image_ids_with_stenosis = list(annotations_by_image.keys())

random.seed(SEED)

num = min(NUM_IMAGENES, len(image_ids_with_stenosis))

selected_ids = random.sample(image_ids_with_stenosis, num)

# ============================================================
# Processing Images
# ============================================================

for counter, image_id in enumerate(selected_ids, start=1):
    image_info = images_by_id[image_id]
    file_name = image_info["file_name"]
    image_path = os.path.join(IMAGES_DIR, file_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"[AVISO] No se pudo abrir: {image_path}")
        continue
    # OpenCV carga BGR; matplotlib necesita RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Imagen para dibujar
    result = image_rgb.copy()
    anns = annotations_by_image[image_id]
    # --------------------------------------------------------
    # Drawing all STENOSIS in this image
    # --------------------------------------------------------
    for ann_number, ann in enumerate(anns, start=1):
        # ==========================
        # BOUNDING BOX
        # ==========================
        bbox = ann.get("bbox")
        if bbox and len(bbox) >= 4:
            x, y, width, height = bbox
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + width))
            y2 = int(round(y + height))
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2) # rojo en RGB
            # Number of injury
            cv2.putText(result, f"Stenosis {ann_number}", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
        # ==========================
        # Segmentation
        # ==========================
        segmentations = ann.get("segmentation", [])
        for polygon in segmentations:
            if not polygon or len(polygon) < 6:
                continue
            points = np.array([[int(round(polygon[i])), int(round(polygon[i + 1]))] for i in range(0, len(polygon), 2)], dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(result, [points], isClosed=True, color=(0, 255, 0), thickness=2) # verde

    # ========================================================
    # Showing Information
    # ========================================================
    print(f"[{counter}/{num}] "f"{file_name} | "f"estenosis: {len(anns)}")

    # ========================================================
    # Saving
    # ========================================================

    output_path = os.path.join(OUTPUT_DIR, f"resultado_{file_name}")

    # RGB -> BGR para OpenCV
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, result_bgr)

    # ========================================================
    # MOSTRAR
    # ========================================================

    plt.figure(figsize=(7, 7))
    plt.imshow(result)
    plt.title(f"{file_name} | "f"{len(anns)} estenosis")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

print()
print("==========================================")
print("Finished")
print("==========================================")
print(f"Results saved to: {OUTPUT_DIR}")