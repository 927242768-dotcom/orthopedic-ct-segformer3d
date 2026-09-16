from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def load_state(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint 必须是 dict: {path}")
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"model_state_dict 必须是 mapping: {path}")
    return payload, state


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="同结构 SegFormer3D checkpoint 线性权重插值")
    parser.add_argument("--base", type=Path, required=True, help="基准 checkpoint，例如 v3")
    parser.add_argument("--adapted", type=Path, required=True, help="微调 checkpoint，例如 bg1.25")
    parser.add_argument("--adapted-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    alpha = float(args.adapted_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("adapted-weight 必须在 [0,1]")

    base_payload, base_state = load_state(args.base)
    adapted_payload, adapted_state = load_state(args.adapted)
    if set(base_state) != set(adapted_state):
        missing = sorted(set(base_state) ^ set(adapted_state))
        raise RuntimeError(f"两个 checkpoint state_dict key 不一致: {missing[:10]}")

    blended: dict[str, torch.Tensor] = {}
    float_count = 0
    copied_count = 0
    for key in base_state:
        a = base_state[key]
        b = adapted_state[key]
        if tuple(a.shape) != tuple(b.shape):
            raise RuntimeError(f"shape 不一致: {key}: {tuple(a.shape)} vs {tuple(b.shape)}")
        if a.dtype.is_floating_point or a.dtype.is_complex:
            blended[key] = (1.0 - alpha) * a + alpha * b
            float_count += 1
        else:
            blended[key] = b.clone()
            copied_count += 1

    output = {
        "model_state_dict": blended,
        "interpolation": {
            "base_checkpoint": str(args.base.resolve()),
            "adapted_checkpoint": str(args.adapted.resolve()),
            "base_weight": 1.0 - alpha,
            "adapted_weight": alpha,
            "floating_tensors_interpolated": float_count,
            "nonfloating_tensors_copied_from_adapted": copied_count,
            "base_epoch": base_payload.get("epoch"),
            "adapted_epoch": adapted_payload.get("epoch"),
            "base_git_commit": base_payload.get("git_commit"),
            "adapted_git_commit": adapted_payload.get("git_commit"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(output["interpolation"])
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
