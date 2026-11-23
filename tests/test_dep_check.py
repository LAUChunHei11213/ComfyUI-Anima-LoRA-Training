"""Tests for the shared sd-scripts dependency probe (utils/dep_check.py)
and its integration in the SD-Scripts Manager and Training Launcher.

Field report that drove this (Aug 2026): a user's pip install failed
(the real error was even swallowed by an unguarded `result.stderr[:500]`
-> "'NoneType' object is not subscriptable"), the Manager still reported
is_ready=True, the Launcher's old 3-module probe said "Dependencies: OK",
and training crashed in the terminal window with
`ModuleNotFoundError: No module named 'diffusers'`.
"""

import subprocess
import sys
import types
from unittest import mock

import pytest

from utils.dep_check import (
    REQUIRED_MODULES, PIP_NAMES, pip_name, probe_modules, diagnose_modules,
)
from nodes.sdscripts_manager import AnimaSDScriptsManager
from nodes.training_launcher import AnimaTrainingLauncher


class TestProbe:
    def test_diffusers_is_covered(self):
        """The exact module from the field report must be in the list."""
        for mod in ("diffusers", "torchvision", "safetensors", "cv2",
                    "PIL", "einops", "toml"):
            assert mod in REQUIRED_MODULES, mod

    def test_pip_name_mapping(self):
        assert pip_name("cv2") == "opencv-python"
        assert pip_name("PIL") == "pillow"
        assert pip_name("diffusers") == "diffusers"

    def test_probe_ok_with_this_python_on_stdlib_subset(self):
        """Full probe against a python that has everything -> simulate by
        probing python -c that ignores the code? Instead: mock run."""
        with mock.patch("utils.dep_check.subprocess.run") as m:
            m.return_value = types.SimpleNamespace(returncode=0, stderr="", stdout="")
            ok, missing, detail = probe_modules(sys.executable)
        assert ok is True and missing == []

    def test_probe_parses_missing_module(self):
        with mock.patch("utils.dep_check.subprocess.run") as m:
            m.return_value = types.SimpleNamespace(
                returncode=1, stdout="",
                stderr="Traceback...\nModuleNotFoundError: No module named 'diffusers'")
            ok, missing, detail = probe_modules("python")
        assert ok is False and missing == ["diffusers"]

    def test_probe_survives_none_streams(self):
        """The manager bug class: never subscript a possibly-None stream."""
        with mock.patch("utils.dep_check.subprocess.run") as m:
            m.return_value = types.SimpleNamespace(returncode=1, stderr=None, stdout=None)
            ok, missing, detail = probe_modules("python")
        assert ok is False and missing == [] and detail == ""

    def test_probe_timeout_is_inconclusive(self):
        with mock.patch("utils.dep_check.subprocess.run",
                        side_effect=subprocess.TimeoutExpired(cmd="x", timeout=90)):
            ok, _, detail = probe_modules("python")
        assert ok is None and "timed out" in detail

    def test_real_probe_detects_actually_missing_module(self):
        """End-to-end sanity with the real subprocess: this container
        lacks torch, so the probe must fail and not raise."""
        ok, missing, detail = probe_modules(sys.executable, timeout=60)
        assert ok is False
        assert len(missing) >= 1  # first missing module in import order


class TestDiagnoseModules:
    def _run_result(self, rc, stderr):
        return types.SimpleNamespace(returncode=rc, stderr=stderr, stdout="")

    def test_separates_missing_from_broken(self):
        def fake_run(cmd, **kw):
            code = cmd[2]  # "import <mod>"
            if "voluptuous" in code:
                return self._run_result(1, "ModuleNotFoundError: No module named 'voluptuous'")
            if "cv2" in code:
                return self._run_result(
                    1, "Traceback ...\n  File cv2/__init__.py line 181\n"
                       "ImportError: numpy.core.multiarray failed to import")
            return self._run_result(0, "")
        with mock.patch("utils.dep_check.subprocess.run", side_effect=fake_run):
            missing, broken = diagnose_modules("python")
        assert missing == ["voluptuous"]
        assert list(broken) == ["cv2"]
        assert "numpy.core.multiarray" in broken["cv2"]

    def test_transitive_missing_counts_as_broken_not_missing(self):
        """cv2 failing because NUMPY is gone is a cv2-environment problem,
        not 'pip install cv2'."""
        def fake_run(cmd, **kw):
            if "cv2" in cmd[2]:
                return self._run_result(1, "ModuleNotFoundError: No module named 'numpy'")
            return self._run_result(0, "")
        with mock.patch("utils.dep_check.subprocess.run", side_effect=fake_run):
            missing, broken = diagnose_modules("python")
        assert missing == [] and "cv2" in broken


class TestManagerIntegration:
    def _manage(self, pip_result, probe_result, tmp_path):
        """Run manage_sdscripts with git/pip/probe mocked."""
        mgr = AnimaSDScriptsManager()
        (tmp_path / ".git").mkdir(parents=True)
        (tmp_path / "train_network.py").write_text("# anima stuff")
        (tmp_path / "requirements.txt").write_text("diffusers")
        with mock.patch.object(mgr, "_update_sdscripts", return_value=(True, "Update successful")), \
             mock.patch.object(mgr, "_install_requirements", return_value=pip_result), \
             mock.patch("nodes.sdscripts_manager.probe_modules", return_value=probe_result), \
             mock.patch.object(mgr, "_get_version", return_value="test"), \
             mock.patch.object(mgr, "_check_anima_support", return_value=True):
            return mgr.manage_sdscripts("install_or_update", "main",
                                        custom_sdscripts_path=str(tmp_path),
                                        install_requirements=True)

    def _manage_full(self, mgr, tmp_path, **mocks):
        (tmp_path / ".git").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train_network.py").write_text("#")
        defaults = dict(
            _update_sdscripts=(True, "ok"),
            _get_version="v",
            _check_anima_support=True,
        )
        ctx = []
        for name, val in defaults.items():
            ctx.append(mock.patch.object(mgr, name, return_value=val))
        for c in ctx:
            c.start()
        try:
            return mgr.manage_sdscripts("install_or_update", "main",
                                        custom_sdscripts_path=str(tmp_path),
                                        install_requirements=True)
        finally:
            for c in ctx:
                c.stop()

    def test_pip_fail_selfheals_by_installing_missing_unpinned(self, tmp_path):
        """Field case (Python 3.13 + safetensors==0.4.5 pin): -r fails on
        an unbuildable pin, only voluptuous is missing. The manager must
        install just that package and end up ready."""
        mgr = AnimaSDScriptsManager()
        probe_results = iter([
            (False, ["voluptuous"], "No module named 'voluptuous'"),  # after -r fail
            (True, [], ""),                                            # after retry
        ])
        with mock.patch.object(mgr, "_install_requirements",
                               return_value=(False, "pip install failed: safetensors build")), \
             mock.patch.object(mgr, "_install_packages",
                               return_value=(True, "Installed: voluptuous")) as inst, \
             mock.patch("nodes.sdscripts_manager.probe_modules",
                        side_effect=lambda *a, **k: next(probe_results)), \
             mock.patch("nodes.sdscripts_manager.diagnose_modules",
                        return_value=(["voluptuous"], {})):
            *_, status, is_ready = self._manage_full(mgr, tmp_path)
        inst.assert_called_once_with(["voluptuous"])
        assert is_ready is True
        assert "installing the missing package(s) directly: voluptuous" in status

    def test_pip_fail_and_retry_fail_is_not_ready_with_targeted_fix(self, tmp_path):
        """If self-healing also fails, is_ready=False and the message
        leads with the TARGETED pip command."""
        mgr = AnimaSDScriptsManager()
        with mock.patch.object(mgr, "_install_requirements", return_value=(False, "boom")), \
             mock.patch.object(mgr, "_install_packages", return_value=(False, "no network")), \
             mock.patch("nodes.sdscripts_manager.probe_modules",
                        return_value=(False, ["diffusers"], "No module named 'diffusers'")), \
             mock.patch("nodes.sdscripts_manager.diagnose_modules",
                        return_value=(["diffusers"], {})):
            *_, status, is_ready = self._manage_full(mgr, tmp_path)
        assert is_ready is False
        assert "DEPENDENCY CHECK FAILED" in status
        assert "pip install diffusers" in status  # targeted command

    def test_broken_module_reported_with_reinstall_command(self, tmp_path):
        """Field case #2: cv2 installed but crashing on import (built for
        another numpy/Python). Not installable - must be reported as
        BROKEN with a force-reinstall command and block readiness."""
        mgr = AnimaSDScriptsManager()
        with mock.patch.object(mgr, "_install_requirements", return_value=(False, "boom")), \
             mock.patch.object(mgr, "_install_packages") as inst, \
             mock.patch("nodes.sdscripts_manager.probe_modules",
                        return_value=(False, [], "ImportError deep in cv2 bootstrap")), \
             mock.patch("nodes.sdscripts_manager.diagnose_modules",
                        return_value=([], {"cv2": "ImportError: numpy.core.multiarray failed to import"})):
            *_, status, is_ready = self._manage_full(mgr, tmp_path)
        inst.assert_not_called()  # nothing missing -> nothing to install
        assert is_ready is False
        assert "BROKEN" in status
        assert "--upgrade --force-reinstall opencv-python" in status
        assert "numpy.core.multiarray" in status

    def test_missing_and_broken_together(self, tmp_path):
        """Mixed case: install the missing one, still block on the broken
        one with its reinstall command."""
        mgr = AnimaSDScriptsManager()
        with mock.patch.object(mgr, "_install_requirements", return_value=(False, "boom")), \
             mock.patch.object(mgr, "_install_packages",
                               return_value=(True, "Installed: voluptuous")) as inst, \
             mock.patch("nodes.sdscripts_manager.probe_modules",
                        return_value=(False, [], "x")), \
             mock.patch("nodes.sdscripts_manager.diagnose_modules",
                        return_value=(["voluptuous"], {"cv2": "ImportError: boom"})):
            *_, status, is_ready = self._manage_full(mgr, tmp_path)
        inst.assert_called_once_with(["voluptuous"])
        assert is_ready is False
        assert "Still missing" not in status  # the missing one was handled
        assert "--upgrade --force-reinstall opencv-python" in status

    def test_pip_fail_but_deps_present_stays_ready(self, tmp_path):
        """pip can fail for irrelevant reasons when everything is already
        installed - must not block the user then."""
        *_, status, is_ready = self._manage(
            (False, "pip install failed (exit code 1): network"),
            (True, [], ""), tmp_path)
        assert is_ready is True
        assert "packages are present" in status

    def test_pip_success_skips_probe(self, tmp_path):
        """Healthy path must not pay the torch-import probe every queue."""
        with mock.patch("nodes.sdscripts_manager.probe_modules") as probe:
            mgr = AnimaSDScriptsManager()
            (tmp_path / ".git").mkdir(parents=True)
            (tmp_path / "train_network.py").write_text("#")
            with mock.patch.object(mgr, "_update_sdscripts", return_value=(True, "ok")), \
                 mock.patch.object(mgr, "_install_requirements", return_value=(True, "Requirements installed")), \
                 mock.patch.object(mgr, "_get_version", return_value="v"), \
                 mock.patch.object(mgr, "_check_anima_support", return_value=True):
                *_, is_ready = mgr.manage_sdscripts(
                    "install_or_update", "main",
                    custom_sdscripts_path=str(tmp_path), install_requirements=True)
        assert is_ready is True
        probe.assert_not_called()

    def test_pip_error_message_shows_real_detail(self, tmp_path):
        """No more swallowed errors: _install_requirements must show exit
        code + captured output even when stderr is None."""
        mgr = AnimaSDScriptsManager()
        req = tmp_path / "requirements.txt"
        req.write_text("x")
        with mock.patch("nodes.sdscripts_manager.subprocess.run") as m:
            m.return_value = types.SimpleNamespace(returncode=1, stderr=None,
                                                   stdout="some pip output")
            ok, msg = mgr._install_requirements(str(tmp_path))
        assert ok is False
        assert "exit code 1" in msg and "some pip output" in msg
        with mock.patch("nodes.sdscripts_manager.subprocess.run") as m:
            m.return_value = types.SimpleNamespace(returncode=2, stderr=None, stdout=None)
            ok, msg = mgr._install_requirements(str(tmp_path))
        assert ok is False and "no output captured" in msg


class TestAttentionBackendFallback:
    """Field report: presets request xformers, user's venv doesn't have
    it, sd-scripts crashes at step 1 with `'NoneType' object has no
    attribute 'memory_efficient_attention'`. The launcher must detect
    this pre-launch and fall back to torch attention."""

    def _launch(self, tmp_path, attn_available):
        launcher = AnimaTrainingLauncher()
        sds = tmp_path / "sd"; sds.mkdir()
        (sds / "anima_train_network.py").write_text("#")
        ds = tmp_path / "data" / "10_x"; ds.mkdir(parents=True)
        (ds / "a.png").write_bytes(b"x"); (ds / "a.txt").write_text("c")
        with mock.patch.object(launcher, "_attention_backend_available",
                               return_value=attn_available), \
             mock.patch.object(launcher, "_check_sdscripts_dependencies",
                               return_value=(True, [], "")), \
             mock.patch.object(launcher, "_launch_in_new_window",
                               return_value=(True, "launched")):
            ret = launcher.launch_training(
                sdscripts_path=str(sds), is_ready=True,
                model_config={"is_valid": True, "pretrained_model": "m",
                              "qwen3_path": "q", "vae_path": "v"},
                training_params={"train_data_dir": str(tmp_path / "data"),
                                 "output_dir": str(tmp_path / "out"),
                                 "attn_mode": "xformers", "split_attn": True},
                dry_run=False)
        config_path = ret["result"][0]
        return ret["ui"]["text"][0], open(config_path).read() if config_path else ""

    def test_missing_xformers_falls_back_to_torch(self, tmp_path):
        status, toml = self._launch(tmp_path, attn_available=False)
        assert 'attn_mode = "torch"' in toml
        assert "not installed in the training Python" in status
        assert "pip install xformers" in status
        # split_attn survives the fallback (works with torch SDPA too)
        assert "split_attn = true" in toml

    def test_available_xformers_is_kept(self, tmp_path):
        status, toml = self._launch(tmp_path, attn_available=True)
        assert 'attn_mode = "xformers"' in toml

    def test_inconclusive_probe_keeps_user_choice(self, tmp_path):
        status, toml = self._launch(tmp_path, attn_available=None)
        assert 'attn_mode = "xformers"' in toml

    def test_module_mapping(self):
        """flash -> flash_attn, sageattn -> sageattention: the probe must
        import the right module for each mode."""
        launcher = AnimaTrainingLauncher()
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd[2])
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")
        with mock.patch("nodes.training_launcher.subprocess.run", side_effect=fake_run):
            launcher._attention_backend_available("py", "flash")
            launcher._attention_backend_available("py", "sageattn")
            assert launcher._attention_backend_available("py", "torch") is True
        assert calls == ["import flash_attn", "import sageattention"]


class TestLauncherUsesFullList:
    def test_launcher_probe_covers_diffusers(self):
        """The launcher pre-flight must delegate to the shared full list."""
        with mock.patch("nodes.training_launcher.probe_modules",
                        return_value=(False, ["diffusers"], "x")) as probe:
            ok, missing, _ = AnimaTrainingLauncher()._check_sdscripts_dependencies("py")
        probe.assert_called_once()
        assert ok is False and missing == ["diffusers"]
