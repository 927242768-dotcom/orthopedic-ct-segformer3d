from pathlib import Path

import pytest

from src.modeling.segformer3d_adapter import (
    SegFormer3DUpstreamNotFound,
    _extract_model_parameters,
    ensure_upstream_available,
)


def _model_config() -> dict:
    return {
        "model": {
            "in_channels": 2,
            "sr_ratios": [4, 2, 1, 1],
            "embed_dims": [32, 64, 160, 256],
            "patch_kernel_size": [7, 3, 3, 3],
            "patch_stride": [4, 2, 2, 2],
            "patch_padding": [3, 1, 1, 1],
            "mlp_ratios": [4, 4, 4, 4],
            "num_heads": [1, 2, 5, 8],
            "depths": [2, 2, 2, 2],
            "decoder_head_embedding_dim": 256,
            "num_classes": 2,
            "decoder_dropout": 0.1,
        }
    }


def test_extract_model_parameters() -> None:
    params = _extract_model_parameters(_model_config())
    assert params["in_channels"] == 2
    assert params["num_classes"] == 2
    assert params["embed_dims"] == [32, 64, 160, 256]


def test_extract_model_parameters_rejects_missing_field() -> None:
    config = _model_config()
    del config["model"]["depths"]
    with pytest.raises(ValueError, match="depths"):
        _extract_model_parameters(config)


def test_missing_upstream_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(SegFormer3DUpstreamNotFound, match="fetch_segformer3d"):
        ensure_upstream_available(tmp_path / "missing")
