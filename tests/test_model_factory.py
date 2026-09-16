import pytest
import torch

from src.modeling.model_factory import build_segmentation_model, model_provenance


def test_tiny_unet3d_factory_forward_and_backward() -> None:
    config = {
        "model": {
            "name": "tiny_unet3d",
            "in_channels": 1,
            "num_classes": 2,
            "base_channels": 8,
        }
    }
    model = build_segmentation_model(config)
    x = torch.randn(1, 1, 16, 16, 16)
    logits = model(x)
    assert logits.shape == (1, 2, 16, 16, 16)
    logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    provenance = model_provenance(config)
    assert provenance["model_name"] == "tiny_unet3d"
    assert provenance["source"] == "local_sanity_baseline"


def test_model_factory_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="未知 model.name"):
        build_segmentation_model({"model": {"name": "not_a_model"}})
