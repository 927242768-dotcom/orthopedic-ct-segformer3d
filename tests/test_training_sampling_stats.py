import pytest

from src.modeling.train import summarize_foreground_fractions


def test_summarize_foreground_fractions_reports_epoch_distribution() -> None:
    summary = summarize_foreground_fractions([0.0, 0.0, 0.25, 0.5])

    assert summary["patch_count"] == 4
    assert summary["foreground_fraction_mean"] == pytest.approx(0.1875)
    assert summary["foreground_fraction_median"] == pytest.approx(0.125)
    assert summary["foreground_fraction_std"] == pytest.approx(0.2072890494)
    assert summary["foreground_fraction_min"] == 0.0
    assert summary["foreground_fraction_max"] == 0.5
    assert summary["foreground_fraction_q25"] == pytest.approx(0.0)
    assert summary["foreground_fraction_q75"] == pytest.approx(0.3125)
    assert summary["foreground_patch_count"] == 2
    assert summary["background_patch_count"] == 2


def test_summarize_foreground_fractions_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="fractions 不能为空"):
        summarize_foreground_fractions([])
