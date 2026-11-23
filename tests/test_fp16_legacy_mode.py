"""Tests for the fp16 legacy-GPU mode (RTX 20xx / GTX 16xx).

Policy: bf16 stays the default and recommendation everywhere. fp16 is an
explicit opt-in for cards without native bf16, relying on sd-scripts >=
0.10.5 (conditional fp32 upcasting fixed the historic Anima fp16 NaN,
kohya-ss/sd-scripts#2293). Everything that is NOT an explicit "fp16"
must still land as bf16 in the TOML.
"""

import pytest

from utils.config_generator import generate_training_config
from nodes.training_wizard import AnimaTrainingWizard
from nodes.training_params import AnimaMemoryOptimizer
from nodes.training_launcher import AnimaTrainingLauncher


def _gen(**overrides):
    kwargs = dict(output_dir="/o", pretrained_model="m", qwen3_path="q",
                  vae_path="v", train_data_dir="/d")
    kwargs.update(overrides)
    return generate_training_config(**kwargs)


class TestConfigGenerator:
    def test_default_is_bf16(self):
        assert _gen()["mixed_precision"] == "bf16"

    def test_explicit_fp16_is_honored(self):
        assert _gen(mixed_precision="fp16")["mixed_precision"] == "fp16"

    def test_anything_else_falls_back_to_bf16(self):
        for weird in ("fp32", "no", "", None, 0, "FP16"):
            assert _gen(mixed_precision=weird)["mixed_precision"] == "bf16"


class TestWizardToggle:
    def _wizard(self, **overrides):
        kwargs = dict(
            train_data_dir="/d", output_dir="/o", output_name="x",
            what_are_you_training="character / person",
            dataset_size="small (10-30 images)",
            image_resolution="1024x1024 (recommended)", gpu_vram="8 GB (low)",
            quality_goal="balanced (recommended)", network_type="LoRA")
        kwargs.update(overrides)
        return AnimaTrainingWizard().configure(**kwargs)

    def test_off_by_default_bf16(self):
        params, summary = self._wizard()
        assert params["mixed_precision"] == "bf16"
        assert "bf16 — recommended" in summary

    def test_on_sets_fp16_with_summary_note(self):
        params, summary = self._wizard(old_gpu_fp16=True)
        assert params["mixed_precision"] == "fp16"
        assert "old-GPU mode" in summary
        assert "NaN" in summary  # tells the user the escape hatch

    def test_tooltip_targets_the_right_cards(self):
        it = AnimaTrainingWizard.INPUT_TYPES()
        tip = it["optional"]["old_gpu_fp16"][1]["tooltip"]
        assert "RTX 20xx" in tip and "GTX 16xx" in tip
        assert "RTX 30xx" in tip  # explicitly warns modern-card owners off


class TestMemoryOptimizer:
    def test_fp16_choice_available_and_applies(self):
        choices = AnimaMemoryOptimizer.INPUT_TYPES()["optional"]["mixed_precision"][0]
        assert "fp16" in choices and "bf16" in choices
        params, _ = AnimaMemoryOptimizer().optimize_memory(
            {"mixed_precision": "bf16"}, mixed_precision="fp16")
        assert params["mixed_precision"] == "fp16"


class TestEstimatorFp16Awareness:
    """fp16 old-GPU mode INCREASES activation memory (sd-scripts upcasts
    block computations to fp32 for stability) — the estimator must
    reflect that instead of treating fp16 like bf16."""

    def test_fp16_estimates_higher_than_bf16(self):
        from nodes.vram_estimator import estimate_training_vram, FP16_UPCAST_ACT_FACTOR
        base = dict(resolution="1024,1024", train_batch_size=1, network_rank=16)
        bf16 = estimate_training_vram({**base, "mixed_precision": "bf16"})
        fp16 = estimate_training_vram({**base, "mixed_precision": "fp16"})
        assert fp16["activations"] == pytest.approx(
            bf16["activations"] * FP16_UPCAST_ACT_FACTOR, rel=0.01)
        assert fp16["total"] > bf16["total"]
        assert fp16["is_fp16"] and not bf16["is_fp16"]

    def test_report_warns_about_fp16_memory(self):
        from nodes.vram_estimator import build_report
        report, _ = build_report(
            {"resolution": "1024,1024", "mixed_precision": "fp16"}, 12.0)
        assert "MORE of this than bf16" in report


class TestAccelerateConfigSync:
    """A bf16 accelerate config would silently override an fp16 run —
    the yaml must follow the training precision."""

    def test_bf16_default(self, tmp_path):
        p = AnimaTrainingLauncher()._write_accelerate_config(str(tmp_path), "ts")
        assert "mixed_precision: bf16" in open(p).read()

    def test_fp16_threaded_through(self, tmp_path):
        p = AnimaTrainingLauncher()._write_accelerate_config(str(tmp_path), "ts", "fp16")
        assert "mixed_precision: fp16" in open(p).read()

    def test_garbage_precision_falls_back_to_bf16(self, tmp_path):
        p = AnimaTrainingLauncher()._write_accelerate_config(str(tmp_path), "ts", "fp32")
        assert "mixed_precision: bf16" in open(p).read()
