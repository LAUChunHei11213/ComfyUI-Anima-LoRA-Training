"""Tests for the low-VRAM presets, wizard force_* overrides / warn-only
resolution check, and the VRAM Estimator node.

The estimator is a calibrated heuristic — tests assert *behavioural*
properties (monotonicity: more swap -> less VRAM, higher res -> more
VRAM, sane ranges for known configs), never exact byte counts, so
recalibrating constants doesn't break the suite.
"""

import pytest

from nodes.training_wizard import (
    AnimaTrainingWizard,
    VRAM_PRESETS,
    parse_resolution_max_dim,
)
from nodes.vram_estimator import (
    AnimaVRAMEstimator,
    estimate_training_vram,
    DIT_NUM_BLOCKS,
)


def _wizard(**overrides):
    """Run the wizard with sane defaults, returning (params, summary)."""
    kwargs = dict(
        train_data_dir="/data", output_dir="/out", output_name="test",
        what_are_you_training="character / person",
        dataset_size="small (10-30 images)",
        image_resolution="1024x1024 (recommended)",
        gpu_vram="12 GB",
        quality_goal="balanced (recommended)",
        network_type="LoRA",
    )
    kwargs.update(overrides)
    return AnimaTrainingWizard().configure(**kwargs)


# ─── VRAM presets ────────────────────────────────────────────────────
class TestVramPresets:
    def test_new_tiers_exist(self):
        for tier in ("6 GB (minimum)", "8 GB (low)", "10 GB", "12 GB",
                     "16 GB", "24 GB+ (high)"):
            assert tier in VRAM_PRESETS

    def test_legacy_keys_unchanged(self):
        """Published workflows store the dropdown label string — renaming
        a key would break every workflow saved with the old label."""
        for legacy in ("8 GB (low)", "12 GB", "16 GB", "24 GB+ (high)"):
            assert legacy in VRAM_PRESETS

    def test_all_tiers_have_recommended_max_resolution(self):
        for name, preset in VRAM_PRESETS.items():
            assert preset.get("recommended_max_resolution", 0) > 0, name

    def test_swap_decreases_as_vram_grows(self):
        tiers = ["6 GB (minimum)", "8 GB (low)", "10 GB", "12 GB", "16 GB"]
        swaps = [VRAM_PRESETS[t]["blocks_to_swap"] for t in tiers]
        assert swaps == sorted(swaps, reverse=True)

    def test_6gb_preset_is_batch_1(self):
        params, _ = _wizard(gpu_vram="6 GB (minimum)",
                            dataset_size="medium (30-100 images)")
        assert params["train_batch_size"] == 1

    def test_6gb_preset_estimates_under_6gb_at_768(self):
        """The whole point of the tier: ~5 GB peak at 768, per field reports."""
        params, _ = _wizard(gpu_vram="6 GB (minimum)", image_resolution="768x768")
        est = estimate_training_vram(params)
        assert est["total"] < 6.0
        assert est["total"] > 2.0  # sanity: not absurdly low either

    def test_10gb_preset_estimates_under_10gb_at_1024(self):
        params, _ = _wizard(gpu_vram="10 GB")
        est = estimate_training_vram(params)
        assert est["total"] < 10.0


# ─── Warn-only resolution check ──────────────────────────────────────
class TestResolutionWarning:
    def test_warns_at_1024_on_6gb(self):
        params, summary = _wizard(gpu_vram="6 GB (minimum)")
        assert "VRAM WARNING" in summary
        # Warn-only: the resolution must NOT be changed
        assert params["resolution"] == "1024,1024"

    def test_no_warning_at_768_on_6gb(self):
        _, summary = _wizard(gpu_vram="6 GB (minimum)", image_resolution="768x768")
        assert "VRAM WARNING" not in summary

    def test_no_warning_at_1024_on_12gb(self):
        _, summary = _wizard(gpu_vram="12 GB")
        assert "VRAM WARNING" not in summary

    def test_warning_on_forced_batch_above_tier_cap(self):
        _, summary = _wizard(gpu_vram="8 GB (low)", force_batch_size=4)
        assert "VRAM WARNING" in summary

    def test_1536_preset_exists_and_flows_through(self):
        from nodes.training_wizard import RESOLUTION_PRESETS
        assert "1536x1536 (high VRAM)" in RESOLUTION_PRESETS
        params, _ = _wizard(gpu_vram="16 GB",
                            image_resolution="1536x1536 (high VRAM)")
        assert params["resolution"] == "1536,1536"

    def test_no_warning_at_1536_on_16gb(self):
        """16 GB tier's recommended max is 1536 - allowed without warning."""
        _, summary = _wizard(gpu_vram="16 GB",
                             image_resolution="1536x1536 (high VRAM)")
        assert "VRAM WARNING" not in summary

    def test_warns_at_1536_on_12gb(self):
        params, summary = _wizard(gpu_vram="12 GB",
                                  image_resolution="1536x1536 (high VRAM)")
        assert "VRAM WARNING" in summary
        assert params["resolution"] == "1536,1536"  # warn-only, not changed

    def test_estimator_scales_up_at_1536(self):
        low = estimate_training_vram({"resolution": "1024,1024"})
        high = estimate_training_vram({"resolution": "1536,1536"})
        # tokens scale 2.25x, activations should follow proportionally
        assert high["activations"] == pytest.approx(low["activations"] * 2.25, rel=0.01)

    def test_parse_resolution_max_dim(self):
        assert parse_resolution_max_dim("1024,1024") == 1024
        assert parse_resolution_max_dim("832,1216") == 1216
        assert parse_resolution_max_dim("768") == 768
        assert parse_resolution_max_dim("garbage") == 0
        assert parse_resolution_max_dim("") == 0


class TestPresetTransparency:
    """The gpu_vram tiers are tweakable preset bundles — both the
    tooltip and the summary must say so and show their contents."""

    def test_tooltip_is_plain_language_and_lists_every_tier(self):
        from nodes.training_wizard import _vram_preset_tooltip, VRAM_PRESETS
        tip = _vram_preset_tooltip()
        assert "STARTING" in tip and "nothing is locked" in tip
        # No raw jargon in the tier lines — speed + image size instead
        assert "grad-ckpt" not in tip and "split" not in tip
        for name, preset in VRAM_PRESETS.items():
            assert name in tip
            assert f"images up to {preset['recommended_max_resolution']}px" in tip

    def test_tooltip_speed_labels_order(self):
        """Speed wording must track block swap monotonically."""
        from nodes.training_wizard import _speed_label
        assert _speed_label(0) == "full speed"
        assert _speed_label(10) == "near full speed"
        assert _speed_label(14) == "a bit slower"
        assert _speed_label(20) == "slower"
        assert _speed_label(24) == "much slower"

    def test_summary_has_preset_block_with_values(self):
        _, summary = _wizard(gpu_vram="8 GB (low)")
        assert "Memory Settings (from the '8 GB (low)' preset)" in summary
        assert "NOT locked" in summary
        assert "20 of 28 model parts stay in normal RAM" in summary
        assert "memory-saving mode (xformers + split_attn)" in summary
        assert "Memory Optimizer" in summary

    def test_summary_marks_swap_override_inside_preset_block(self):
        _, summary = _wizard(gpu_vram="8 GB (low)", force_blocks_to_swap=5)
        assert "5 of 28 model parts stay in normal RAM  (override)" in summary

    def test_summary_swap_off_wording(self):
        _, summary = _wizard(gpu_vram="16 GB")
        assert "whole model stays on the GPU (fastest)" in summary

    def test_resolution_ownership_is_explicit_when_fitting(self):
        """Users must be able to tell the preset never touches their
        resolution choice — both when it fits and when it doesn't."""
        _, summary = _wizard(gpu_vram="12 GB")  # 1024 on 12 GB: fits
        assert "your choice — presets never change it" in summary
        assert "your 1024px choice fits" in summary

    def test_resolution_ownership_is_explicit_when_exceeding(self):
        params, summary = _wizard(gpu_vram="6 GB (minimum)")  # 1024 on 6 GB
        assert "your 1024px choice EXCEEDS" in summary
        assert "KEPT AS-IS" in summary
        assert "has NOT changed it" in summary  # warning block
        assert params["resolution"] == "1024,1024"

    def test_tooltips_state_resolution_is_never_capped(self):
        from nodes.training_wizard import AnimaTrainingWizard, _vram_preset_tooltip
        it = AnimaTrainingWizard.INPUT_TYPES()
        res_tip = it["required"]["image_resolution"][1]["tooltip"]
        assert "ALWAYS USED EXACTLY AS-IS" in res_tip
        assert "never changed" in _vram_preset_tooltip()


# ─── force_* overrides ───────────────────────────────────────────────
class TestForceOverrides:
    def test_sentinels_are_noops(self):
        base, _ = _wizard()
        forced, _ = _wizard(force_resolution="", force_network_rank=0,
                            force_learning_rate=0.0, force_batch_size=0,
                            force_blocks_to_swap=-1)
        assert forced == base

    def test_force_resolution(self):
        params, summary = _wizard(force_resolution="768,768")
        assert params["resolution"] == "768,768"
        assert "(override)" in summary

    def test_force_resolution_garbage_falls_back_to_preset(self):
        params, _ = _wizard(force_resolution="not a res")
        assert params["resolution"] == "1024,1024"

    def test_force_rank_beats_lycoris_halving_and_rederives_alpha(self):
        params, _ = _wizard(network_type="LoKr", force_network_rank=8)
        assert params["network_rank"] == 8
        assert params["network_alpha"] == 4.0

    def test_force_learning_rate(self):
        params, _ = _wizard(force_learning_rate=5e-5)
        assert params["learning_rate"] == pytest.approx(5e-5)

    def test_force_learning_rate_beats_prodigy_override(self):
        params, _ = _wizard(quality_goal="experimental (adaptive optimizer)",
                            force_learning_rate=5e-5)
        assert params["learning_rate"] == pytest.approx(5e-5)

    def test_force_batch_size_beats_vram_cap(self):
        params, _ = _wizard(gpu_vram="8 GB (low)", force_batch_size=4)
        assert params["train_batch_size"] == 4

    def test_force_blocks_to_swap_zero_disables(self):
        """0 is a meaningful value (disable swapping), hence the -1 sentinel."""
        params, _ = _wizard(gpu_vram="8 GB (low)", force_blocks_to_swap=0)
        assert params["blocks_to_swap"] == 0

    def test_force_blocks_to_swap_positive(self):
        params, _ = _wizard(gpu_vram="16 GB", force_blocks_to_swap=24)
        assert params["blocks_to_swap"] == 24


# ─── VRAM estimator model ────────────────────────────────────────────
def _base_params(**overrides):
    p = dict(
        resolution="1024,1024", train_batch_size=1, network_rank=16,
        network_type="LoRA", optimizer_type="AdamW8bit", attn_mode="torch",
        split_attn=False, gradient_checkpointing=True,
        cache_text_encoder_outputs=True, cache_text_encoder_outputs_to_disk=False,
        blocks_to_swap=0,
    )
    p.update(overrides)
    return p


class TestEstimatorModel:
    def test_more_swap_means_less_vram(self):
        totals = [estimate_training_vram(_base_params(blocks_to_swap=s))["total"]
                  for s in (0, 10, 20, DIT_NUM_BLOCKS)]
        assert totals == sorted(totals, reverse=True)

    def test_higher_resolution_means_more_vram(self):
        low = estimate_training_vram(_base_params(resolution="512,512"))["total"]
        mid = estimate_training_vram(_base_params(resolution="768,768"))["total"]
        high = estimate_training_vram(_base_params(resolution="1024,1024"))["total"]
        assert low < mid < high

    def test_bigger_batch_means_more_vram(self):
        b1 = estimate_training_vram(_base_params(train_batch_size=1))["total"]
        b4 = estimate_training_vram(_base_params(train_batch_size=4))["total"]
        assert b4 > b1

    def test_no_checkpointing_means_much_more_vram(self):
        on = estimate_training_vram(_base_params())["total"]
        off = estimate_training_vram(_base_params(gradient_checkpointing=False))["total"]
        assert off > on * 1.5

    def test_uncached_text_encoder_adds_memory(self):
        cached = estimate_training_vram(_base_params())["total"]
        uncached = estimate_training_vram(
            _base_params(cache_text_encoder_outputs=False))["total"]
        assert uncached > cached

    def test_1024_batch1_lands_in_plausible_range(self):
        """Community reports put this config around 8-11 GB."""
        est = estimate_training_vram(_base_params())
        assert 7.0 < est["total"] < 12.0

    def test_swap_capped_at_block_count(self):
        est = estimate_training_vram(_base_params(blocks_to_swap=999))
        assert est["blocks_to_swap"] == DIT_NUM_BLOCKS

    def test_survives_string_typed_numerics(self):
        """Same workflow-JSON quirk the rest of the pack defends against:
        INT/FLOAT widget values can arrive as strings."""
        est = estimate_training_vram(_base_params(
            train_batch_size="2", network_rank="32", blocks_to_swap="10"))
        assert est["total"] > 0


class TestLiveEstimation:
    """The /anima_lora/estimate_vram endpoint re-derives params from
    browser-collected widget values via evaluate_chain — these tests
    cover that derivation without a running server."""

    WIZARD_WIDGETS = dict(
        train_data_dir="/data", output_dir="/out", output_name="x",
        what_are_you_training="character / person",
        dataset_size="small (10-30 images)",
        image_resolution="1024x1024 (recommended)",
        gpu_vram="8 GB (low)",
        quality_goal="balanced (recommended)",
        network_type="LoRA", training_seed=42, caption_extension=".txt",
    )

    def test_wizard_chain_matches_direct_call(self):
        from nodes.vram_estimator import evaluate_chain
        chain = [{"type": "AnimaTrainingWizard", "widgets": dict(self.WIZARD_WIDGETS)}]
        derived = evaluate_chain(chain)
        direct, _ = AnimaTrainingWizard().configure(**self.WIZARD_WIDGETS)
        assert derived == direct

    def test_manual_chain_applies_memory_overrides(self):
        from nodes.vram_estimator import evaluate_chain
        chain = [
            {"type": "AnimaBasicParams", "widgets": {
                "train_data_dir": "/d", "output_dir": "/o", "output_name": "x",
                "resolution": "1024,1024", "network_type": "LoRA",
                "network_rank": 16, "network_alpha": 8.0,
                "learning_rate": 1e-4, "optimizer_type": "AdamW8bit",
                "lr_scheduler": "cosine", "max_train_epochs": 10,
                "train_batch_size": 1,
            }},
            {"type": "AnimaMemoryOptimizer", "widgets": {"blocks_to_swap": 20}},
        ]
        params = evaluate_chain(chain)
        assert params is not None
        assert params["blocks_to_swap"] == 20

    def test_unknown_nodes_are_skipped(self):
        from nodes.vram_estimator import evaluate_chain
        chain = [
            {"type": "Reroute", "widgets": {}},
            {"type": "AnimaTrainingWizard", "widgets": dict(self.WIZARD_WIDGETS)},
            {"type": "SomeRandomNode", "widgets": {"a": 1}},
        ]
        assert evaluate_chain(chain) is not None

    def test_empty_or_garbage_chain_returns_none(self):
        from nodes.vram_estimator import evaluate_chain
        assert evaluate_chain([]) is None
        assert evaluate_chain(None) is None
        assert evaluate_chain([{"type": "AnimaAdvancedParams", "widgets": {}}]) is None

    def test_broken_widgets_fail_soft(self):
        """A garbage dropdown value must return None, never raise —
        the live endpoint runs inside the ComfyUI server process."""
        from nodes.vram_estimator import evaluate_chain
        w = dict(self.WIZARD_WIDGETS, what_are_you_training="nonsense")
        assert evaluate_chain([{"type": "AnimaTrainingWizard", "widgets": w}]) is None

    def test_live_report_text(self):
        from nodes.vram_estimator import live_report
        chain = [{"type": "AnimaTrainingWizard", "widgets": dict(self.WIZARD_WIDGETS)}]
        report = live_report(chain, 8.0)
        assert "LIVE ESTIMATE" in report
        assert "ESTIMATED PEAK" in report

    def test_live_report_no_chain_message(self):
        from nodes.vram_estimator import live_report
        assert "Connect" in live_report([], 8.0)

    def test_live_report_bad_gpu_value(self):
        from nodes.vram_estimator import live_report
        chain = [{"type": "AnimaTrainingWizard", "widgets": dict(self.WIZARD_WIDGETS)}]
        report = live_report(chain, "not a number")
        assert "ESTIMATED PEAK" in report
        assert "VERDICT" not in report  # falls back to no-verdict mode


class TestChartData:
    """build_chart feeds the JS bar-chart widget — component sums must
    match the total and the payload must survive JSON round-trips."""

    def test_components_sum_to_total(self):
        from nodes.vram_estimator import build_chart
        est = estimate_training_vram(_base_params())
        chart = build_chart(est, 12.0, "fits")
        assert sum(gb for _, gb in chart["components"]) == pytest.approx(chart["total"], abs=0.06)

    def test_chart_is_json_serializable(self):
        import json
        from nodes.vram_estimator import build_chart
        chart = build_chart(estimate_training_vram(_base_params()), 12.0, "tight")
        assert json.loads(json.dumps(chart)) == chart

    def test_component_labels_stable(self):
        """JS COMPONENT_COLORS is keyed on these exact labels."""
        from nodes.vram_estimator import build_chart
        chart = build_chart(estimate_training_vram(_base_params()), 0, "info")
        labels = [l for l, _ in chart["components"]]
        assert labels == ["Model on GPU", "Working memory", "LoRA + trainer",
                          "Text encoder", "System overhead"]

    def test_live_payload_includes_chart(self):
        from nodes.vram_estimator import live_payload
        chain = [{"type": "AnimaTrainingWizard",
                  "widgets": dict(TestLiveEstimation.WIZARD_WIDGETS)}]
        payload = live_payload(chain, 8.0)
        assert payload["chart"] is not None
        assert payload["chart"]["gpu"] == 8.0
        assert payload["chart"]["verdict"] in ("fits", "tight", "oom")

    def test_live_payload_no_chain_has_null_chart(self):
        from nodes.vram_estimator import live_payload
        assert live_payload([], 8.0)["chart"] is None

    def test_node_ui_includes_chart_json(self):
        import json
        result = AnimaVRAMEstimator().estimate(_base_params(), 12.0)
        chart = json.loads(result["ui"]["chart"][0])
        assert chart["total"] > 0
        assert len(chart["components"]) == 5


class TestEstimatorNode:
    def test_passthrough_params_untouched(self):
        params = _base_params()
        result = AnimaVRAMEstimator().estimate(params, gpu_vram_gb=12.0)
        assert result["result"][0] is params
        assert "Estimated Training VRAM" in result["result"][1]

    def test_verdict_fits_on_big_gpu(self):
        result = AnimaVRAMEstimator().estimate(_base_params(blocks_to_swap=20), 24.0)
        assert "FITS" in result["result"][1]

    def test_verdict_oom_with_suggestions_on_small_gpu(self):
        report = AnimaVRAMEstimator().estimate(_base_params(), 6.0)["result"][1]
        assert "OOM" in report
        assert "Suggestions" in report

    def test_no_verdict_when_gpu_unset(self):
        report = AnimaVRAMEstimator().estimate(_base_params(), 0.0)["result"][1]
        assert "VERDICT" not in report

    def test_suggestions_actually_reduce_estimate(self):
        """Every suggestion line advertises a projected saving — verify the
        engine only emits changes that genuinely lower the estimate."""
        from nodes.vram_estimator import _suggestions
        params = _base_params(train_batch_size=2, optimizer_type="AdamW")
        est = estimate_training_vram(params)
        sugg = _suggestions(params, est, 6.0)
        assert len(sugg) >= 3
        for line in sugg:
            assert "saves" in line

    def test_suggestions_say_where_wizard_mode(self):
        """Wizard users can't reach most memory knobs directly — every
        suggestion must point at a wizard field, a preset, or the Memory
        Optimizer node."""
        from nodes.vram_estimator import _suggestions
        params = _base_params(train_batch_size=2, optimizer_type="AdamW",
                              gradient_checkpointing=False, _wizard_mode=True)
        est = estimate_training_vram(params)
        sugg = _suggestions(params, est, 6.0)
        assert len(sugg) >= 4
        for line in sugg:
            assert "where:" in line
            assert ("wizard" in line or "preset" in line
                    or "Memory Optimizer" in line or "quality_goal" in line)

    def test_suggestions_say_where_manual_mode(self):
        """Manual-chain users get pointed at the node that owns the knob."""
        from nodes.vram_estimator import _suggestions
        params = _base_params(train_batch_size=2, gradient_checkpointing=False)
        est = estimate_training_vram(params)
        sugg = _suggestions(params, est, 6.0)
        for line in sugg:
            assert "where:" in line
            assert "Basic Params" in line or "Memory Optimizer" in line
        # and no wizard-specific advice leaks into manual mode
        assert not any("force_" in line for line in sugg)
