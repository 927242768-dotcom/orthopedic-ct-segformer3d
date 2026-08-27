from __future__ import annotations

import torch

from src.modeling.compare_checkpoint_dynamics import (
    activation_diagnostics_on_patch,
    state_dict_delta,
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segformer_encoder = torch.nn.Module()
        self.segformer_encoder.embed_1 = torch.nn.Module()
        self.segformer_encoder.embed_1.patch_embeddings = torch.nn.Conv3d(1, 2, 1)
        self.segformer_encoder.tf_block1 = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.segformer_encoder.embed_2 = torch.nn.Module()
        self.segformer_encoder.embed_2.patch_embeddings = torch.nn.Conv3d(2, 2, 1)
        self.segformer_encoder.tf_block2 = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.segformer_encoder.embed_3 = torch.nn.Module()
        self.segformer_encoder.embed_3.patch_embeddings = torch.nn.Conv3d(2, 2, 1)
        self.segformer_encoder.tf_block3 = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.segformer_encoder.embed_4 = torch.nn.Module()
        self.segformer_encoder.embed_4.patch_embeddings = torch.nn.Conv3d(2, 2, 1)
        self.segformer_encoder.tf_block4 = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.segformer_decoder = torch.nn.Module()
        self.segformer_decoder.linear_fuse = torch.nn.Sequential(
            torch.nn.Conv3d(2, 2, 1),
            torch.nn.BatchNorm3d(2),
        )
        self.segformer_decoder.linear_pred = torch.nn.Conv3d(2, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.segformer_encoder.embed_1.patch_embeddings(x)
        x = self.segformer_encoder.tf_block1[1](x)
        x = self.segformer_encoder.embed_2.patch_embeddings(x)
        x = self.segformer_encoder.tf_block2[1](x)
        x = self.segformer_encoder.embed_3.patch_embeddings(x)
        x = self.segformer_encoder.tf_block3[1](x)
        x = self.segformer_encoder.embed_4.patch_embeddings(x)
        x = self.segformer_encoder.tf_block4[1](x)
        x = self.segformer_decoder.linear_fuse(x)
        return self.segformer_decoder.linear_pred(x)


def test_activation_diagnostics_captures_head_input_and_output() -> None:
    model = _TinyModel()
    image = torch.randn((1, 1, 4, 4, 4))
    payload = activation_diagnostics_on_patch(model, image)

    assert "segformer_decoder.linear_pred.input" in payload
    assert "segformer_decoder.linear_pred" in payload
    assert payload["segformer_decoder.linear_pred"]["shape"] == [1, 2, 4, 4, 4]
    assert payload["segformer_encoder.embed_1.patch_embeddings"]["sampled_count"] > 0


def test_state_dict_delta_reports_group_and_bn_running_buffers() -> None:
    left_model = _TinyModel()
    right_model = _TinyModel()
    right_model.load_state_dict(left_model.state_dict())
    with torch.no_grad():
        right_model.segformer_decoder.linear_pred.weight.add_(0.25)
        right_model.segformer_decoder.linear_fuse[1].running_mean.add_(1.0)

    payload = state_dict_delta(left_model.state_dict(), right_model.state_dict())

    groups = payload["parameter_groups_by_relative_delta"]
    assert any(row["group"] == "segformer_decoder.linear_pred" for row in groups)
    assert payload["top_parameters_by_relative_delta"][0]["delta_norm"] > 0.0
    buffers = payload["batchnorm_running_buffers_by_relative_delta"]
    assert any("running_mean" in row["name"] and row["delta_norm"] > 0.0 for row in buffers)
