import json
import cv2
import os

# ============================================================
# Set Up
# ============================================================

BASE_DIR = "arcade/stenosis"

ANNOTATIONS_FILE = os.path.join(BASE_DIR, "train", "annotations", "train.json")

IMAGES_DIR = os.path.join(BASE_DIR, "train", "images")

OUTPUT_DIR = "resultados"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# upload annotations
# ============================================================

print("Uploading annotations...")

with open(ANNOTATIONS_FILE, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

print(f"Available images: {len(images)}")
print(f"Available annotations: {len(annotations)}")


# ============================================================
# Creating index of image
# ============================================================

images_by_id = {image["id"]: image for image in images}


# ============================================================
# Look for stenosis annotations
# category_id = 26
# ============================================================

stenosis_annotations = [annotation for annotation in annotations if annotation["category_id"] == 26]

print(f"Stenosis presented: {len(stenosis_annotations)}")


# ============================================================
# Take first stenosis
# ============================================================

annotation = stenosis_annotations[0]

image_id = annotation["image_id"]

image_info = images_by_id[image_id]

file_name = image_info["file_name"]

print()
print("Processing:")
print(f"  Image: {file_name}")
print(f"  Image ID: {image_id}")
print(f"  BBOX: {annotation['bbox']}")


# ============================================================
# Uploading image
# ============================================================

image_path = os.path.join(IMAGES_DIR, file_name)

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image can not be opened: {image_path}")


# ============================================================
# Drawing Bounding Box
# ============================================================

x, y, width, height = annotation["bbox"]

x1 = int(x)
y1 = int(y)

x2 = int(x + width)
y2 = int(y + height)

cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


# ============================================================
# Drawing Segmentation
# ============================================================

segmentation = annotation["segmentation"]

for polygon in segmentation:
    points = []
    for i in range(0, len(polygon), 2):
        px = int(polygon[i])
        py = int(polygon[i + 1])
        points.append([px, py])
    points = __import__("numpy").array(points, dtype="int32")
    points = points.reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)


# ============================================================
# Saving result
# ============================================================

output_path = os.path.join(OUTPUT_DIR, f"resultado_{file_name}")

cv2.imwrite(output_path, image)

print()
print("Resultado guardado en:")
print(output_path)