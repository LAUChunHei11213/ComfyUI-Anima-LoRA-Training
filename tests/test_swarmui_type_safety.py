"""SwarmUI / non-native-ComfyUI type-safety regression tests.

Why this file exists
--------------------
Native ComfyUI dropdowns serialise as their string label ("LoRA",
"AdamW8bit", ...). SwarmUI and a few older ComfyUI workflow exporters
serialise dropdowns as the numeric index (0, 1, 2, ...) into the
dropdown's choice list. When that workflow re-runs against our nodes,
the widget value arrives as `int` instead of `str`.

Any code path that calls a string method on a widget input — `.lower()`,
`.strip()`, `.split()`, etc. — crashes with
``AttributeError: 'int' object has no attribute '...'`` before training
even begins.

We have multiple defensive coercion layers (`coerce_seed`,
`coerce_network_type`, `coerce_dropdown`, `normalize_training_params`)
to handle this, but each new code path that touches a widget value
needs to either go through one of those layers or have its own local
coerce. This test file enumerates the entry points and verifies they
all survive int-encoded dropdown inputs.

Adding a new node? Add a test here that calls its entry function with
int values for every dropdown widget and asserts no exception. Future
regressions caught immediately.
"""

import os
import pytest

from utils.config_generator import (
    coerce_seed,
    coerce_network_type,
    coerce_dropdown,
    normalize_training_params,
    generate_training_config,
    _NETWORK_TYPE_BY_INDEX,
    _DROPDOWN_PARAM_CHOICES,
)


# ─── coerce_* primitives ────────────────────────────────────────────────
class TestCoerceSeed:
    """Numeric INT widget delivered as string is the original
    workflow-JSON-quirk path. Test it round-trips cleanly."""

    def test_int_passes_through(self):
        assert coerce_seed(42) == 42

    def test_numeric_string_coerces(self):
        assert coerce_seed("1234") == 1234

    def test_garbage_falls_back_to_default(self):
        assert coerce_seed("random") == 42

    def test_none_falls_back_to_default(self):
        assert coerce_seed(None) == 42


class TestCoerceNetworkType:
    """SwarmUI dropdown indices for network_type."""

    @pytest.mark.parametrize("idx,expected", [
        (0, "LoRA"),
        (1, "LoKr"),
        (2, "LoHa"),
    ])
    def test_int_index_maps_to_label(self, idx, expected):
        assert coerce_network_type(idx) == expected

    def test_string_passes_through(self):
        assert coerce_network_type("LoKr") == "LoKr"

    def test_out_of_range_int_falls_back(self):
        assert coerce_network_type(99) == "LoRA"

    def test_bool_is_not_treated_as_int(self):
        # bool is a subclass of int — without the explicit branch True
        # would silently map to choices[1] ("LoKr"). Make sure we reject.
        assert coerce_network_type(True) == "LoRA"
        assert coerce_network_type(False) == "LoRA"


class TestCoerceDropdownGeneric:
    """The generic coerce_dropdown handles every other dropdown widget."""

    def test_each_registered_dropdown_accepts_int_indices(self):
        """Iterate every entry in _DROPDOWN_PARAM_CHOICES and confirm
        coerce_dropdown maps each valid int index to the corresponding
        string label without raising."""
        for key, (choices, default) in _DROPDOWN_PARAM_CHOICES.items():
            for idx in range(len(choices)):
                result = coerce_dropdown(idx, choices, default)
                assert result == choices[idx], (
                    f"key={key} idx={idx} expected {choices[idx]!r} "
                    f"got {result!r}"
                )

    def test_unknown_string_passes_through_with_warning(self):
        # Custom optimizer names that aren't in the official list should
        # be passed through (sd-scripts decides if it's valid).
        result = coerce_dropdown("MyCustomOpt", ("AdamW8bit", "AdamW"), "AdamW8bit")
        assert result == "MyCustomOpt"

    def test_garbage_type_falls_back(self):
        assert coerce_dropdown({"weird": "dict"}, ("a", "b"), "a") == "a"


# ─── normalize_training_params — the dict-level coercion layer ──────────
class TestNormalizeTrainingParams:
    """The launcher calls this on the entire training_params dict before
    anything downstream reads it. Verify every dropdown key in the dict
    gets coerced when it arrives as an int."""

    def test_int_network_type_in_dict_coerces(self):
        params = {"network_type": 1}  # SwarmUI sent index for "LoKr"
        out = normalize_training_params(params)
        assert out["network_type"] == "LoKr"

    def test_int_optimizer_in_dict_coerces(self):
        params = {"optimizer_type": 0}  # index for "AdamW8bit"
        out = normalize_training_params(params)
        assert out["optimizer_type"] == "AdamW8bit"

    def test_int_lr_scheduler_in_dict_coerces(self):
        params = {"lr_scheduler": 0}  # "cosine"
        out = normalize_training_params(params)
        assert out["lr_scheduler"] == "cosine"

    def test_string_seed_in_dict_coerces(self):
        params = {"seed": "1234"}
        out = normalize_training_params(params)
        assert out["seed"] == 1234

    def test_mixed_legitimate_and_int_indices(self):
        # Real-world SwarmUI workflow: some fields strings, some ints,
        # some numeric strings. Should all normalise together.
        params = {
            "network_type": 2,           # int index → "LoHa"
            "optimizer_type": "AdamW",   # string label, pass through
            "seed": "9999",              # numeric string → int
            "network_rank": 32,          # already int
            "lr_scheduler": 1,           # int index → "cosine_with_restarts"
        }
        out = normalize_training_params(params)
        assert out["network_type"] == "LoHa"
        assert out["optimizer_type"] == "AdamW"
        assert out["seed"] == 9999
        assert out["network_rank"] == 32
        assert out["lr_scheduler"] == "cosine_with_restarts"


# ─── generate_training_config — function-level coerce ───────────────────
class TestGenerateTrainingConfigCoerce:
    """Even bypassing normalize_training_params (e.g. a future direct
    caller), generate_training_config itself coerces network_type so the
    `.lower()` on it can't crash."""

    @pytest.mark.parametrize("net_in,expected_module", [
        (0, "networks.lora_anima"),
        (1, "networks.lokr"),
        (2, "networks.loha"),
        ("LoRA", "networks.lora_anima"),
        ("LoKr", "networks.lokr"),
        ("LoHa", "networks.loha"),
    ])
    def test_network_type_int_or_string_picks_correct_module(self, net_in, expected_module):
        config = generate_training_config(
            output_dir="/tmp/out",
            pretrained_model="/tmp/model.safetensors",
            qwen3_path="/tmp/qwen.safetensors",
            vae_path="/tmp/vae.safetensors",
            train_data_dir="/tmp/data",
            network_type=net_in,
        )
        assert config["network_module"] == expected_module


# ─── AnimaTrainingWizard.configure() — the entry point that crashed ────
class TestWizardSwarmUIInputs:
    """The wizard's configure() is called directly with widget values.
    Native ComfyUI sends strings; SwarmUI sends ints for the dropdowns.
    This test class is the regression net for what bit us before:
    `network_type.lower()` crashed when SwarmUI sent int 1.
    """

    @pytest.fixture
    def wizard(self):
        from nodes.training_wizard import AnimaTrainingWizard
        return AnimaTrainingWizard()

    @pytest.fixture
    def base_kwargs(self):
        """Minimum required wizard inputs — all valid string values.
        Tests override individual fields with int variants."""
        return dict(
            train_data_dir="/tmp/data",
            output_dir="/tmp/out",
            output_name="test_lora",
            what_are_you_training="character / person",
            dataset_size="small (10-30 images)",
            image_resolution="1024x1024 (recommended)",
            gpu_vram="24 GB+ (high)",
            quality_goal="balanced (recommended)",
            network_type="LoRA",
        )

    @pytest.mark.parametrize("idx,expected", [
        (0, "LoRA"),
        (1, "LoKr"),
        (2, "LoHa"),
    ])
    def test_int_network_type_does_not_crash(self, wizard, base_kwargs, idx, expected):
        """The original SwarmUI-reported bug. Must not raise
        AttributeError on the `.lower()` call inside the Lycoris
        rank-halving branch."""
        base_kwargs["network_type"] = idx
        params, summary = wizard.configure(**base_kwargs)
        assert params["network_type"] == expected

    def test_int_network_type_lokr_still_halves_rank(self, wizard, base_kwargs):
        """Functional regression: even when network_type comes in as
        int 1 ("LoKr"), the rank-halving logic that depends on
        recognising "lokr" must still fire."""
        base_kwargs["network_type"] = 1  # "LoKr"
        params_lokr, _ = wizard.configure(**base_kwargs)
        base_kwargs["network_type"] = 0  # "LoRA"
        params_lora, _ = wizard.configure(**base_kwargs)
        assert params_lokr["network_rank"] == max(4, params_lora["network_rank"] // 2)

    def test_string_seed_is_handled(self, wizard, base_kwargs):
        """The wizard accepts `training_seed` as an INT widget. SwarmUI
        sometimes sends numeric strings ("42") for INT widgets too —
        coerce_seed at the params-dict assignment catches that."""
        params, _ = wizard.configure(training_seed="42", **base_kwargs)
        assert params["seed"] == 42


# ─── AnimaBasicParams.set_params() ─────────────────────────────────────
class TestBasicParamsSwarmUIInputs:
    """Same coverage for the manual-config path's basic params node."""

    @pytest.fixture
    def basic(self):
        from nodes.training_params import AnimaBasicParams
        return AnimaBasicParams()

    @pytest.fixture
    def base_kwargs(self):
        return dict(
            train_data_dir="/tmp/data",
            output_dir="/tmp/out",
            output_name="test_lora",
            resolution="1024,1024",
            network_type="LoRA",
            network_rank=32,
            network_alpha=16.0,
            learning_rate=1e-4,
            optimizer_type="AdamW8bit",
            lr_scheduler="cosine",
            max_train_epochs=10,
            train_batch_size=1,
        )

    @pytest.mark.parametrize("network_idx", [0, 1, 2])
    def test_int_network_type_does_not_crash(self, basic, base_kwargs, network_idx):
        base_kwargs["network_type"] = network_idx
        params, summary = basic.set_params(**base_kwargs)
        # The basic params node stores values as-given (no coerce until
        # launcher's normalize_training_params runs), but the call itself
        # must not raise. Coerce happens downstream.
        assert "network_type" in params

    def test_string_seed_coerces_via_coerce_seed(self, basic, base_kwargs):
        params, _ = basic.set_params(seed="1234", **base_kwargs)
        assert params["seed"] == 1234


# ─── End-to-end smoke: launcher params dict → generate_training_config ─
class TestEndToEndIntInputs:
    """Full pipeline simulation: launcher receives training_params dict
    with int-encoded dropdowns (as if from SwarmUI), runs through
    normalize_training_params, hands to generate_training_config. No
    crash, sane TOML-ready config out the other end."""

    def test_swarmui_style_params_produce_valid_config(self):
        # Simulate exactly what arrives if SwarmUI serialised every
        # dropdown as an int index.
        swarmui_params = {
            "train_data_dir": "/tmp/data",
            "output_dir": "/tmp/out",
            "output_name": "test_lora",
            "network_type": 1,           # → "LoKr"
            "network_rank": 24,
            "network_alpha": 12.0,
            "learning_rate": 1e-4,
            "optimizer_type": 1,         # → "AdamW"
            "lr_scheduler": 0,           # → "cosine"
            "max_train_epochs": 10,
            "max_train_steps": 0,
            "train_batch_size": 1,
            "resolution": "1024,1024",
            "seed": "9999",              # numeric string
            "timestep_sampling": 0,      # → "sigmoid"
            "attn_mode": 0,              # → "torch"
            "save_model_as": 0,          # → "safetensors"
            "log_with": 0,               # → "tensorboard"
        }

        normalised = normalize_training_params(swarmui_params)

        # All dropdown ints became their string labels.
        assert normalised["network_type"] == "LoKr"
        assert normalised["optimizer_type"] == "AdamW"
        assert normalised["lr_scheduler"] == "cosine"
        assert normalised["timestep_sampling"] == "sigmoid"
        assert normalised["attn_mode"] == "torch"
        assert normalised["save_model_as"] == "safetensors"
        assert normalised["log_with"] == "tensorboard"
        assert normalised["seed"] == 9999

        # Now hand to generate_training_config — must not crash.
        config = generate_training_config(
            output_dir=normalised["output_dir"],
            pretrained_model="/tmp/m.safetensors",
            qwen3_path="/tmp/q.safetensors",
            vae_path="/tmp/v.safetensors",
            train_data_dir=normalised["train_data_dir"],
            network_type=normalised["network_type"],
            network_rank=normalised["network_rank"],
            network_alpha=normalised["network_alpha"],
            learning_rate=normalised["learning_rate"],
            optimizer_type=normalised["optimizer_type"],
            lr_scheduler=normalised["lr_scheduler"],
            max_train_epochs=normalised["max_train_epochs"],
            train_batch_size=normalised["train_batch_size"],
            resolution=normalised["resolution"],
            seed=normalised["seed"],
            timestep_sampling=normalised["timestep_sampling"],
            attn_mode=normalised["attn_mode"],
            save_model_as=normalised["save_model_as"],
            log_with=normalised["log_with"],
        )

        assert config["network_module"] == "networks.lokr"
        assert config["optimizer_type"] == "AdamW"
        assert config["seed"] == 9999
