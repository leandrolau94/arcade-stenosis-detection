import json

import pytest
import torch
from PIL import Image

from src.data.arcade_pytorch_dataset import ARCADEStenosisDataset


@pytest.fixture
def synthetic_dataset(tmp_path):
    images_dir = tmp_path / "images"
    annotations_dir = tmp_path / "annotations"

    images_dir.mkdir()
    annotations_dir.mkdir()

    image = Image.new("L", (512, 512), color=128)
    image_path = images_dir / "test.png"
    image.save(image_path)

    data = {
        "images": [
            {
                "id": 1,
                "width": 512,
                "height": 512,
                "file_name": "test.png",
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 26,
                "segmentation": [
                    [100, 100, 200, 100, 200, 200, 100, 200]
                ],
                "bbox": [100, 100, 100, 100],
                "area": 10000,
            }
        ],
        "categories": [
            {
                "id": 26,
                "name": "stenosis",
            }
        ],
    }

    annotations_file = annotations_dir / "test.json"

    with open(annotations_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return ARCADEStenosisDataset(
        annotations_file=str(annotations_file),
        images_dir=str(images_dir),
        stenosis_category_id=26,
    )


def test_dataset_length(synthetic_dataset):
    assert len(synthetic_dataset) == 1


def test_dataset_sample_shapes(synthetic_dataset):
    sample = synthetic_dataset[0]

    assert sample["image"].shape == (1, 512, 512)
    assert sample["mask"].shape == (1, 512, 512)


def test_image_is_normalized(synthetic_dataset):
    sample = synthetic_dataset[0]
    image = sample["image"]

    assert image.dtype == torch.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_mask_is_binary(synthetic_dataset):
    sample = synthetic_dataset[0]
    mask = sample["mask"]

    unique_values = torch.unique(mask)

    assert torch.all((unique_values == 0) | (unique_values == 1))