import torch

from src.models.first_unet_arcade import (
    UNet,
    count_parameters,
    dice_score_from_logits,
)


def test_unet_output_shape():
    model = UNet()
    model.eval()

    x = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 1, 256, 256)


def test_unet_output_is_finite():
    model = UNet()
    model.eval()

    x = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        output = model(x)

    assert torch.isfinite(output).all()


def test_unet_parameter_count():
    model = UNet()

    assert count_parameters(model) == 1_927_841


def test_dice_perfect_prediction():
    targets = torch.ones(2, 1, 16, 16)
    logits = torch.full_like(targets, 10.0)

    dice = dice_score_from_logits(logits, targets)

    assert dice > 0.99


def test_dice_zero_prediction():
    targets = torch.ones(2, 1, 16, 16)
    logits = torch.full_like(targets, -10.0)

    dice = dice_score_from_logits(logits, targets)

    assert dice < 0.01