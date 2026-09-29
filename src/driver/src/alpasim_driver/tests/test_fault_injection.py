# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from alpasim_driver import fault_injection as fi


class FaultInjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sync = patch.object(fi.torch.cuda, "synchronize")
        self.sync.start()
        self.addCleanup(self.sync.stop)
        fi._step = 0

    def configure(self, mode):
        os.environ.update(ALPASIM_FI_MODE=mode, ALPASIM_FI_OUTPUT=str(self.directory))

    def event(self):
        return json.loads((self.directory / "inference-000001.json").read_text())

    def test_disabled_has_no_files_or_cuda_calls(self):
        with fi.inference_window(7, 100, "cuda") as record:
            self.assertIsNone(record)
        fi.torch.cuda.synchronize.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_clean_preserves_prediction_and_records_identity(self):
        self.configure("clean")
        with fi.inference_window(7, 100, "cuda") as record:
            pass
        positions = np.ones((1, 3, 3))
        record.save_prediction(positions, positions, 0)
        with np.load(self.directory / "prediction-000001.npz") as saved:
            np.testing.assert_array_equal(saved["candidate_positions"], positions)
        self.assertEqual(self.event()["status"], "completed")
        self.assertEqual(self.event()["seed"], 7)
        with self.assertRaises(FileExistsError):
            record.save_prediction(positions, positions, 0)

    def test_trace_records_nonfinite_module_outputs_for_selected_inference(self):
        self.configure("clean")
        os.environ["ALPASIM_FI_TRACE"] = "2"
        model = fi.torch.nn.Sequential(fi.torch.nn.Identity(), fi.torch.nn.ReLU())
        for _ in range(2):
            with fi.inference_window(7, 100, "cpu", model):
                model(fi.torch.tensor([1.0, float("nan"), -2.0]))
        self.assertFalse((self.directory / "trace-000001.jsonl").exists())
        lines = [
            json.loads(line)
            for line in (self.directory / "trace-000002.jsonl").read_text().splitlines()
        ]
        self.assertEqual([x["module"] for x in lines[:-1]], ["0", "1", "<root>"])
        self.assertEqual(lines[0]["nonfinite"], 1)
        self.assertEqual(lines[0]["finite_max_abs"], 2.0)
        self.assertEqual(lines[-1], {"total_module_calls": 3})
        self.assertEqual(model._forward_hooks, {})

    def test_missing_preload_fails_before_model_runs(self):
        self.configure("bitflip")
        os.environ["TD_MODE"] = "bitflip"
        with patch.object(fi.ctypes, "CDLL", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "Preload"):
                with fi.inference_window(7, 100, "cuda"):
                    self.fail("must not run")

    def test_missed_target_is_an_error(self):
        self.configure("observe")
        os.environ["TD_MODE"] = "observe"
        tool = Mock()
        tool.td_begin_inference.return_value = 0
        tool.td_end_inference.return_value = 0
        with patch.object(fi.ctypes, "CDLL", return_value=tool):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                with fi.inference_window(7, 100, "cuda"):
                    pass
        self.assertEqual(self.event()["status"], "error")
        tool.td_end_inference.assert_called_once()

    def test_model_failure_closes_window_and_releases_lock(self):
        self.configure("bitflip")
        os.environ["TD_MODE"] = "bitflip"
        tool = Mock()
        tool.td_begin_inference.return_value = 0
        tool.td_end_inference.return_value = 1
        with patch.object(fi.ctypes, "CDLL", return_value=tool):
            with self.assertRaisesRegex(ValueError, "model failure"):
                with fi.inference_window(7, 100, "cuda"):
                    raise ValueError("model failure")
        self.assertEqual(self.event()["status"], "error")
        self.assertFalse(fi._lock.locked())
        tool.td_end_inference.assert_called_once()

    def test_nested_inference_is_rejected(self):
        self.configure("clean")
        with fi.inference_window(7, 100, "cuda"):
            with self.assertRaisesRegex(RuntimeError, "serial"):
                with fi.inference_window(7, 100, "cuda"):
                    pass

    def test_experiment_seed_overrides_random_session_seed(self):
        self.configure("clean")
        os.environ["ALPASIM_FI_SEED"] = "42"
        with patch.object(fi.torch, "manual_seed") as cpu_seed:
            with patch.object(fi.torch.cuda, "manual_seed_all") as gpu_seed:
                with fi.inference_window(9999, 100, "cuda"):
                    pass
                with fi.inference_window(1234, 200, "cuda"):
                    pass
        self.assertEqual([c.args[0] for c in cpu_seed.call_args_list], [42, 43])
        self.assertEqual([c.args[0] for c in gpu_seed.call_args_list], [42, 43])
        self.assertEqual(self.event()["seed"], 42)


if __name__ == "__main__":
    unittest.main()
