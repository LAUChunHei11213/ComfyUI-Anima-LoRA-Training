"""Regression tests for the bitsandbytes runtime probe.

Background: bitsandbytes 0.47.0 moved `str2optimizer8bit_blockwise` out of
`bitsandbytes.functional` (into backends/cuda/ops.py). The old probe checked
functional for exactly that symbol, so healthy modern installs were declared
broken and AdamW8bit users got silently downgraded to AdamW (~2x optimizer
VRAM). Reported by a user on bnb >= 0.47.

Each test fabricates a fake `bitsandbytes` package in sys.modules matching a
real era's shape (verified against the published wheels for 0.44.1, 0.45.0,
0.46.0, 0.47.0, and 0.50.0) and asserts the probe's verdict.
"""

import sys
import types

import pytest

from utils.config_generator import (
    _probe_bitsandbytes,
    _bitsandbytes_is_working,
    coerce_optimizer_for_runtime,
)


@pytest.fixture(autouse=True)
def _clean_bnb(monkeypatch):
    """Remove fake modules and the probe cache around every test."""
    if hasattr(_bitsandbytes_is_working, "_cached"):
        del _bitsandbytes_is_working._cached
    yield
    for mod in list(sys.modules):
        if mod == "bitsandbytes" or mod.startswith("bitsandbytes."):
            del sys.modules[mod]
    if hasattr(_bitsandbytes_is_working, "_cached"):
        del _bitsandbytes_is_working._cached


class _RealLib:  # stands in for BNBNativeLibrary
    pass


class ErrorHandlerMockBNBNativeLibrary:  # exact class name bnb 0.45+ uses
    pass


def install_fake_bnb(functional_symbols, lib):
    pkg = types.ModuleType("bitsandbytes")
    functional = types.ModuleType("bitsandbytes.functional")
    for name in functional_symbols:
        setattr(functional, name, object())
    cextension = types.ModuleType("bitsandbytes.cextension")
    cextension.lib = lib
    pkg.functional = functional
    pkg.cextension = cextension
    sys.modules["bitsandbytes"] = pkg
    sys.modules["bitsandbytes.functional"] = functional
    sys.modules["bitsandbytes.cextension"] = cextension


class TestProbeEras:
    def test_old_bnb_healthy(self):
        # <= 0.46: legacy dict present, real native lib
        install_fake_bnb(["str2optimizer8bit_blockwise",
                          "optimizer_update_8bit_blockwise"], _RealLib())
        assert _probe_bitsandbytes() is True

    def test_old_bnb_broken_dll(self):
        # 0.44-era Windows failure: lib is None, guarded dict never defined
        install_fake_bnb([], None)
        assert _probe_bitsandbytes() is False

    def test_modern_bnb_healthy_is_not_downgraded(self):
        # >= 0.47: legacy dict GONE from functional, new fn present.
        # This is the reported bug: must be True, not False.
        install_fake_bnb(["optimizer_update_8bit_blockwise"], _RealLib())
        assert _probe_bitsandbytes() is True

    def test_modern_bnb_broken_native_lib(self):
        # 0.45+: import succeeds but lib is the error-handler mock
        install_fake_bnb(["optimizer_update_8bit_blockwise"],
                         ErrorHandlerMockBNBNativeLibrary())
        assert _probe_bitsandbytes() is False

    def test_not_installed(self):
        sys.modules["bitsandbytes"] = None  # makes import raise
        assert _probe_bitsandbytes() is False

    def test_ancient_bnb_without_cextension_lib_falls_back_to_symbols(self):
        pkg = types.ModuleType("bitsandbytes")
        functional = types.ModuleType("bitsandbytes.functional")
        functional.str2optimizer8bit_blockwise = object()
        pkg.functional = functional
        sys.modules["bitsandbytes"] = pkg
        sys.modules["bitsandbytes.functional"] = functional
        # no bitsandbytes.cextension at all
        assert _probe_bitsandbytes() is True


class TestRuntimeCoercion:
    def test_adamw8bit_kept_on_modern_healthy_bnb(self):
        install_fake_bnb(["optimizer_update_8bit_blockwise"], _RealLib())
        opt, warning = coerce_optimizer_for_runtime("AdamW8bit")
        assert opt == "AdamW8bit" and warning is None

    def test_adamw8bit_downgraded_on_broken_bnb_with_plain_warning(self):
        install_fake_bnb(["optimizer_update_8bit_blockwise"],
                         ErrorHandlerMockBNBNativeLibrary())
        opt, warning = coerce_optimizer_for_runtime("AdamW8bit")
        assert opt == "AdamW"
        assert "native CUDA library" in warning
        # old symbol name must not leak into user-facing text anymore
        assert "str2optimizer8bit_blockwise" not in warning

    def test_result_is_cached(self):
        install_fake_bnb(["optimizer_update_8bit_blockwise"], _RealLib())
        assert _bitsandbytes_is_working() is True
        # break the fake module; cached verdict must survive
        install_fake_bnb([], None)
        assert _bitsandbytes_is_working() is True
