# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Optional TensorDynamic inference window and immediate trajectory recording.

Set ALPASIM_FI_MODE=clean|observe|bitflip and ALPASIM_FI_OUTPUT to a fresh
directory. Observe/bitflip require the TensorDynamic alpasim_injector preloaded.
Inference numbers are process-wide and one-based; use one driver and one rollout.
"""

from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_lock = threading.Lock()
_step = 0


@dataclass
class InferenceRecord:
    directory: Path
    step: int

    def save_prediction(self, positions, rotations, selected_index: int) -> None:
        """Save all candidates and the selected trajectory without changing them."""
        with (self.directory / f"prediction-{self.step:06d}.npz").open("xb") as f:
            np.savez_compressed(
                f,
                candidate_positions=positions,
                candidate_rotations=rotations,
                selected_index=np.asarray(selected_index),
            )


@contextmanager
def inference_window(seed: int, model_t0_us: int, device):
    """Synchronize and bracket only the model call, excluding model loading.

    Disabled unless explicitly configured. The clean run has the same recording
    and synchronization as the two instrumented runs. A missed injection target
    raises an error instead of silently reporting a successful experiment.
    """
    global _step
    mode = os.environ.get("ALPASIM_FI_MODE")
    if mode is None:
        yield None
        return
    if mode not in {"clean", "observe", "bitflip", "profile", "values"}:
        raise ValueError(
            "ALPASIM_FI_MODE must be clean, observe, bitflip, profile, or values"
        )
    if not _lock.acquire(blocking=False):
        raise RuntimeError("Fault injection requires serial inference")
    try:
        directory = Path(os.environ["ALPASIM_FI_OUTPUT"])
        directory.mkdir(parents=True, exist_ok=True)
        tool = None
        if mode != "clean":
            if os.environ.get("TD_MODE") != mode:
                raise ValueError("TD_MODE and ALPASIM_FI_MODE must match")
            tool = ctypes.CDLL(None)
            try:
                tool.td_begin_inference.argtypes = [ctypes.c_int]
                tool.td_begin_inference.restype = ctypes.c_int
                tool.td_end_inference.argtypes = []
                tool.td_end_inference.restype = ctypes.c_int
            except AttributeError as exc:
                raise RuntimeError(
                    "Preload alpasim_injector.so or alpasim_profiler.so before Python starts"
                ) from exc
        _step += 1
        step = _step
        # ONESHOT runtime sessions otherwise choose fresh random driver seeds.
        # Override only inside explicitly enabled experiments, before the model call.
        if "ALPASIM_FI_SEED" in os.environ:
            seed = int(os.environ["ALPASIM_FI_SEED"]) + step - 1
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        record = InferenceRecord(directory, step)
        # Exclusive creation prevents mixing outputs from separate driver processes.
        with (directory / f"inference-{step:06d}.json").open("x") as metadata:
            event = dict(
                inference=step,
                seed=seed,
                model_t0_us=model_t0_us,
                mode=mode,
                status="started",
            )

            def write_event():
                metadata.seek(0)
                json.dump(event, metadata)
                metadata.truncate()
                metadata.flush()

            write_event()
            torch.cuda.synchronize(device)
            if tool is not None and tool.td_begin_inference(step) != 0:
                raise RuntimeError("TensorDynamic refused the inference window")
            start = time.perf_counter()
            try:
                yield record
                torch.cuda.synchronize(device)
                if tool is not None:
                    reached = tool.td_end_inference()
                    event["reached"] = reached
                    target = int(os.environ.get("TD_TARGET_INFERENCE", "1"))
                    # The profiler returns 1 after writing the target window's
                    # counts, so the same one-target check applies to it.
                    if reached < 0 or (step == target and reached != 1):
                        raise RuntimeError(
                            "TensorDynamic did not reach exactly one target"
                        )
                event["status"] = "completed"
            except BaseException as exc:
                # Best-effort close on a model exception; CUDA failures may prevent it.
                if "reached" not in event and tool is not None:
                    try:
                        torch.cuda.synchronize(device)
                        event["reached"] = tool.td_end_inference()
                    except Exception:
                        pass
                event["status"] = "error"
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                event["elapsed_s"] = time.perf_counter() - start
                write_event()
    finally:
        _lock.release()
