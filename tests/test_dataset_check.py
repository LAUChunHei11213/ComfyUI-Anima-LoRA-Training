"""Tests for the train_data_dir structure check (utils/dataset_check.py)
and its integration points (wizard summary, Basic Params summary,
Training Launcher gate)."""

import pytest

from utils.dataset_check import check_dataset, format_dataset_report
from nodes.training_wizard import AnimaTrainingWizard
from nodes.training_params import AnimaBasicParams
from nodes.training_launcher import AnimaTrainingLauncher


def make_dataset(root, folder="10_mychar", images=3, captions=None, ext=".txt"):
    """Create a dataset folder. captions=None -> caption per image."""
    d = root / folder
    d.mkdir(parents=True)
    n_caps = images if captions is None else captions
    for i in range(images):
        (d / f"img{i}.png").write_bytes(b"fake")
    for i in range(n_caps):
        (d / f"img{i}{ext}").write_text("a caption")
    return d


# ─── core checks ─────────────────────────────────────────────────────
class TestCheckDataset:
    def test_valid_dataset_ok(self, tmp_path):
        make_dataset(tmp_path)
        r = check_dataset(str(tmp_path))
        assert r["ok"] and not r["errors"] and not r["warnings"]
        assert r["total_images"] == 3
        assert r["total_steps_per_epoch"] == 30  # 3 images x 10 repeats
        assert r["concepts"][0]["concept"] == "mychar"

    def test_empty_path_errors(self):
        r = check_dataset("")
        assert not r["ok"] and "empty" in r["errors"][0]

    def test_missing_folder_errors(self):
        r = check_dataset("/does/not/exist/anywhere")
        assert not r["ok"] and "not found" in r["errors"][0]

    def test_file_instead_of_folder(self, tmp_path):
        f = tmp_path / "data.zip"
        f.write_bytes(b"x")
        r = check_dataset(str(f))
        assert not r["ok"] and "file, not a folder" in r["errors"][0]

    def test_images_in_root_is_the_classic_mistake(self, tmp_path):
        (tmp_path / "img0.png").write_bytes(b"fake")
        (tmp_path / "img0.txt").write_text("cap")
        r = check_dataset(str(tmp_path))
        assert not r["ok"]
        assert "IGNORE" in r["errors"][0]
        assert "10_mychar" in r["errors"][0]  # tells them the fix

    def test_subfolder_without_repeats_prefix_warns(self, tmp_path):
        make_dataset(tmp_path, folder="mychar")  # no "10_" prefix
        r = check_dataset(str(tmp_path))
        assert not r["ok"]  # nothing usable at all
        assert any("SKIP" in w and "10_mychar" in w for w in r["warnings"])

    def test_missing_captions_warn_with_names(self, tmp_path):
        make_dataset(tmp_path, images=4, captions=2)
        r = check_dataset(str(tmp_path))
        assert r["ok"]  # trains, just badly
        assert any("2 image(s) have no matching" in w for w in r["warnings"])
        assert any("EMPTY caption" in w for w in r["warnings"])

    def test_orphan_captions_mild_warning(self, tmp_path):
        d = make_dataset(tmp_path, images=2)
        (d / "extra.txt").write_text("caption without image")
        r = check_dataset(str(tmp_path))
        assert r["ok"]
        assert any("no matching image" in w for w in r["warnings"])

    def test_zero_repeats_warns(self, tmp_path):
        make_dataset(tmp_path, folder="0_mychar")
        r = check_dataset(str(tmp_path))
        assert any("repeats is 0" in w for w in r["warnings"])

    def test_huge_repeats_warns(self, tmp_path):
        make_dataset(tmp_path, folder="500_mychar")
        r = check_dataset(str(tmp_path))
        assert any("unusually high" in w for w in r["warnings"])

    def test_empty_root_errors(self, tmp_path):
        r = check_dataset(str(tmp_path))
        assert not r["ok"] and "empty" in r["errors"][0]

    def test_custom_caption_extension(self, tmp_path):
        make_dataset(tmp_path, ext=".caption")
        r = check_dataset(str(tmp_path), caption_extension=".caption")
        assert r["ok"] and not r["warnings"]
        # same folder checked with .txt -> captions all "missing"
        r2 = check_dataset(str(tmp_path), caption_extension=".txt")
        assert any("no matching" in w for w in r2["warnings"])

    def test_extension_without_dot_normalized(self, tmp_path):
        make_dataset(tmp_path)
        assert check_dataset(str(tmp_path), caption_extension="txt")["ok"]

    def test_multiple_concepts_summed(self, tmp_path):
        make_dataset(tmp_path, folder="10_alpha", images=2)
        make_dataset(tmp_path, folder="5_beta", images=4)
        r = check_dataset(str(tmp_path))
        assert r["total_images"] == 6
        assert r["total_steps_per_epoch"] == 2 * 10 + 4 * 5

    def test_uppercase_image_extensions_counted(self, tmp_path):
        d = tmp_path / "10_x"
        d.mkdir()
        (d / "IMG.PNG").write_bytes(b"f")
        (d / "IMG.txt").write_text("c")
        assert check_dataset(str(tmp_path))["ok"]


# ─── integration ─────────────────────────────────────────────────────
class TestIntegration:
    def _wizard(self, data_dir):
        return AnimaTrainingWizard().configure(
            train_data_dir=str(data_dir), output_dir="/out", output_name="x",
            what_are_you_training="character / person",
            dataset_size="small (10-30 images)",
            image_resolution="1024x1024 (recommended)", gpu_vram="12 GB",
            quality_goal="balanced (recommended)", network_type="LoRA")

    def test_wizard_summary_shows_good_dataset(self, tmp_path):
        make_dataset(tmp_path)
        _, summary = self._wizard(tmp_path)
        assert "Dataset Check" in summary
        assert "3 image(s) x 10 repeats = 30 steps/epoch" in summary
        assert "looks good" in summary

    def test_wizard_summary_shows_problems(self, tmp_path):
        (tmp_path / "img.png").write_bytes(b"f")
        _, summary = self._wizard(tmp_path)
        assert "PROBLEM" in summary and "IGNORE" in summary

    def test_basic_params_summary_includes_check(self, tmp_path):
        make_dataset(tmp_path)
        _, summary = AnimaBasicParams().set_params(
            train_data_dir=str(tmp_path), output_dir="/out", output_name="x",
            resolution="1024,1024", network_type="LoRA", network_rank=16,
            network_alpha=8.0, learning_rate=1e-4, optimizer_type="AdamW8bit",
            lr_scheduler="cosine", max_train_epochs=10, train_batch_size=1)
        assert "Dataset Check" in summary and "looks good" in summary

    def test_launcher_blocks_broken_dataset(self, tmp_path):
        (tmp_path / "img.png").write_bytes(b"f")  # classic mistake
        ret = AnimaTrainingLauncher().launch_training(
            sdscripts_path="/fake", is_ready=True,
            model_config={"is_valid": True, "pretrained_model": "m",
                          "qwen3_path": "q", "vae_path": "v"},
            training_params={"train_data_dir": str(tmp_path),
                             "output_dir": str(tmp_path / "out")},
            dry_run=True)
        out = ret["result"]
        assert out[0] == "" and out[1] == ""
        assert "ERROR" in out[2] and "0 images" in out[2]
        assert "IGNORE" in out[2]  # the report explains the actual problem
        # the error must ALSO be visible on the node itself
        assert "ERROR" in ret["ui"]["text"][0]
        # and nothing was created on disk
        assert not (tmp_path / "out").exists()

    def test_config_preview_shows_dataset_status(self, tmp_path):
        """Config Preview is the always-visible panel in the wizard example
        workflow — the dataset report must appear there."""
        from nodes.config_preview import AnimaConfigPreview
        (tmp_path / "img.png").write_bytes(b"f")  # classic mistake
        params, _ = self._wizard(tmp_path)
        out = AnimaConfigPreview().preview(
            model_config={"is_valid": True, "pretrained_model": "m",
                          "qwen3_path": "q", "vae_path": "v"},
            training_params=params)
        text = out["ui"]["text"][0]
        assert "Dataset Check" in text and "PROBLEM" in text
        assert "Anima LoRA Training Config" in text  # TOML still there

    def test_estimator_report_shows_dataset_status(self, tmp_path):
        """The estimator (in BOTH example workflows, live + post-run) must
        carry a compact dataset status line."""
        from nodes.vram_estimator import build_report
        make_dataset(tmp_path)
        good, _ = build_report({"train_data_dir": str(tmp_path),
                                "resolution": "1024,1024"}, 12.0)
        assert "Dataset: OK - 3 image(s), 30 steps/epoch" in good

        bad, _ = build_report({"train_data_dir": "/does/not/exist",
                               "resolution": "1024,1024"}, 12.0)
        assert "Dataset PROBLEM" in bad
        assert "Launcher checks it too" in bad

        unset, _ = build_report({"resolution": "1024,1024"}, 12.0)
        assert "no training folder set yet" in unset

    def test_launcher_status_always_reaches_the_node_ui(self, tmp_path):
        """Regression: a user saw 'training doesn't start, no error' because
        the launcher's error lived only in the undisplayed status output.
        Every launch_training call must now carry ui.text."""
        ret = AnimaTrainingLauncher().launch_training(
            sdscripts_path="/fake", is_ready=False,  # earliest error path
            model_config={}, training_params={})
        assert "sd-scripts is not ready" in ret["ui"]["text"][0]

    def test_deps_probe_timeout_is_inconclusive_not_blocking(self):
        """A slow torch import must not block training. None = unknown."""
        import subprocess as sp
        from unittest import mock
        launcher = AnimaTrainingLauncher()
        with mock.patch("subprocess.run",
                        side_effect=sp.TimeoutExpired(cmd="x", timeout=90)):
            ok, missing, err = launcher._check_sdscripts_dependencies("python")
        assert ok is None and "timed out" in err

    def test_headless_linux_falls_back_to_background_launch(self, tmp_path, monkeypatch):
        """Colab regression: no DISPLAY and no terminal emulator used to
        return 'No supported terminal emulator found' (invisibly). Now it
        must run training as a background process logging to a file."""
        import platform, sys, time
        if platform.system() != "Linux":
            pytest.skip("Linux-only behavior")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

        sds = tmp_path / "sd-scripts"
        sds.mkdir()
        script = sds / "anima_train_network.py"
        script.write_text("import sys\nprint('FAKE TRAINING OK', sys.argv[1:])\n")
        out = tmp_path / "out"
        out.mkdir()

        ok, msg = AnimaTrainingLauncher()._launch_in_new_window(
            [sys.executable, str(script), "--config_file", "cfg.toml"],
            str(sds), "title", str(out), timestamp="ts1")

        assert ok is True
        assert "BACKGROUND" in msg and "tail -f" in msg
        log = out / "training_log_ts1.txt"
        assert str(log) in msg
        # background process should finish quickly and have logged output
        for _ in range(50):
            if log.exists() and "FAKE TRAINING OK" in log.read_text():
                break
            time.sleep(0.1)
        assert "FAKE TRAINING OK" in log.read_text()
        assert "--config_file" in log.read_text()

    def test_display_but_no_emulator_also_falls_back(self, tmp_path, monkeypatch):
        """A DISPLAY with no installed emulator (docker with X forward,
        minimal distros) must fall back too, not fail."""
        import platform, sys
        if platform.system() != "Linux":
            pytest.skip("Linux-only behavior")
        monkeypatch.setenv("DISPLAY", ":0")

        sds = tmp_path / "sd"
        sds.mkdir()
        script = sds / "t.py"
        script.write_text("print('x')\n")
        ok, msg = AnimaTrainingLauncher()._launch_in_new_window(
            [sys.executable, str(script)], str(sds), "t", str(tmp_path), timestamp="ts2")
        assert ok is True and "BACKGROUND" in msg

    def test_launcher_allows_valid_dataset_past_the_gate(self, tmp_path):
        """A valid dataset must get PAST the dataset gate. dry_run avoids
        actually launching; we only assert the failure is NOT dataset-related
        if a later step (sd-scripts path etc.) trips."""
        make_dataset(tmp_path)
        ret = AnimaTrainingLauncher().launch_training(
            sdscripts_path="/fake", is_ready=True,
            model_config={"is_valid": True, "pretrained_model": "m",
                          "qwen3_path": "q", "vae_path": "v"},
            training_params={"train_data_dir": str(tmp_path),
                             "output_dir": str(tmp_path / "out")},
            dry_run=True)
        assert "not set up the way" not in ret["result"][2]
