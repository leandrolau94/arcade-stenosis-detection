import os

import torch

from src.data.arcade_pytorch_dataset import ARCADEStenosisDataset


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

ANNOTATIONS_FILE = os.path.join(
    PROJECT_ROOT,
    "arcade",
    "stenosis",
    "train",
    "annotations",
    "train.json",
)

IMAGES_DIR = os.path.join(
    PROJECT_ROOT,
    "arcade",
    "stenosis",
    "train",
    "images",
)


def test_dataset_length():
    dataset = ARCADEStenosisDataset(
        annotations_file=ANNOTATIONS_FILE,
        images_dir=IMAGES_DIR,
        stenosis_category_id=26,
    )

    assert len(dataset) == 1000


def test_dataset_sample_shapes():
    dataset = ARCADEStenosisDataset(
        annotations_file=ANNOTATIONS_FILE,
        images_dir=IMAGES_DIR,
        stenosis_category_id=26,
    )

    sample = dataset[0]

    assert sample["image"].shape == (1, 512, 512)
    assert sample["mask"].shape == (1, 512, 512)


def test_image_is_normalized():
    dataset = ARCADEStenosisDataset(
        annotations_file=ANNOTATIONS_FILE,
        images_dir=IMAGES_DIR,
        stenosis_category_id=26,
    )

    sample = dataset[0]
    image = sample["image"]

    assert image.dtype == torch.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_mask_is_binary():
    dataset = ARCADEStenosisDataset(
        annotations_file=ANNOTATIONS_FILE,
        images_dir=IMAGES_DIR,
        stenosis_category_id=26,
    )

    sample = dataset[0]
    mask = sample["mask"]

    unique_values = torch.unique(mask)

    assert torch.all(
        (unique_values == 0) | (unique_values == 1)
    )