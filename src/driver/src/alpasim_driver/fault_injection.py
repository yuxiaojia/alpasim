# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Optional TensorDynamic inference window and immediate trajectory recording.

Set ALPASIM_FI_MODE=clean|observe|bitflip and ALPASIM_FI_OUTPUT to a fresh
directory. Observe/bitflip require the TensorDynamic alpasim_injector preloaded.
Inference numbers are process-wide and one-based; use one driver and one rollout.
ALPASIM_FI_TRACE=8 (or 8,87) additionally records, for those inferences, every
module output of the model that contains inf/NaN, in call order (trace-*.jsonl).
ALPASIM_FI_MODE=operands needs alpasim_operands.so preloaded. ALPASIM_FI_MODE=layers
needs no tool: for the inferences in ALPASIM_FI_LAYER_STEPS (e.g. 1,15,30) it records,
per nn.Linear and output channel, how many outputs have |y| > 2, the per-channel max |y|,
the 16 largest |y| with their position, and once per run the weight statistics per
output channel (layers-*.npz, layer-weights.npz).
"""

from __future__ import annotations

import ctypes
import json
import os
import threading
import time
import contextlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import torch_fault_injection

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
def _nonfinite_trace(model, path: Path):
    """Log module outputs containing inf/NaN, in call order, flushed per line.

    Each check synchronizes, so this is only for diagnosing a chosen inference;
    lines are written before a later device-side assert can abort the process.
    """
    calls = 0
    handles = []
    with path.open("x") as log:

        def hook(module, _inputs, output):
            nonlocal calls
            calls += 1
            tensors = output if isinstance(output, (tuple, list)) else (output,)
            for index, tensor in enumerate(tensors):
                if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                    continue
                bad = int((~torch.isfinite(tensor)).sum())
                if bad:
                    finite = tensor[torch.isfinite(tensor)]
                    log.write(
                        json.dumps(
                            dict(
                                call=calls,
                                module=module._trace_name,
                                output=index,
                                shape=list(tensor.shape),
                                nonfinite=bad,
                                finite_max_abs=(
                                    float(finite.abs().max())
                                    if finite.numel()
                                    else None
                                ),
                            )
                        )
                        + "\n"
                    )
                    log.flush()

        for name, module in model.named_modules():
            module._trace_name = name or "<root>"
            handles.append(module.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            log.write(json.dumps(dict(total_module_calls=calls)) + "\n")


# Model parts reported to an injector that exports td_set_part (alpasim_multi.so with
# TD_WINDOW_PARTS): 1 vision encoder, 2 language model (reasoning), 3 action expert.
_PARTS = (
    ("vlm.model.visual", 1),
    ("vlm.model.language_model", 2),
    ("vlm.lm_head", 2),
    ("expert", 3),
    ("action_in_proj", 3),
    ("action_out_proj", 3),
)


@contextmanager
def _part_hooks(model, tool):
    """Tell the injector which model part is issuing kernels (pre hook: the part,
    post hook: 0). Kernel launches happen on the calling thread, so the part is
    current when the injector sees each launch."""
    handles = []
    for name, part in _PARTS:
        try:
            module = model.get_submodule(name)
        except AttributeError:
            continue
        handles.append(module.register_forward_pre_hook(lambda m, a, p=part: tool.td_set_part(p)))
        handles.append(module.register_forward_hook(lambda m, a, o: tool.td_set_part(0)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _layer_profile(model, directory: Path, step: int):
    """Per nn.Linear output statistics for one inference, reduced on the GPU."""
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    weights = directory / "layer-weights.npz"
    if not weights.exists():
        out = {}
        for i, (_name, m) in enumerate(linears):
            w = m.weight.detach().float()
            out[f"l2_{i}"] = w.norm(dim=1).cpu().numpy()
            out[f"l1_{i}"] = w.abs().sum(dim=1).cpu().numpy()
            out[f"max_{i}"] = w.abs().amax(dim=1).cpu().numpy()
        np.savez(weights, names=np.array([n for n, _ in linears]), **out)
    stats = {}
    handles = []

    def make_hook(i):
        def hook(_module, inputs, output):
            if not torch.is_tensor(output) or not output.is_floating_point():
                return
            y = output.detach().float().reshape(-1, output.shape[-1]).abs()
            finite = torch.isfinite(y)
            y = torch.where(finite, y, torch.zeros_like(y))
            s = stats.get(i)
            if s is None:
                s = stats[i] = dict(
                    rows=0, calls=0,
                    gt2=torch.zeros(y.shape[1], dtype=torch.int64, device=y.device),
                    nonfinite=torch.zeros(y.shape[1], dtype=torch.int64, device=y.device),
                    max=torch.zeros(y.shape[1], device=y.device),
                    top_v=torch.empty(0, device=y.device),
                    top_idx=torch.empty(0, 3, dtype=torch.int64, device=y.device),
                    x_max=torch.zeros((), device=y.device),
                )
            s["gt2"] += (y > 2).sum(0)
            s["nonfinite"] += (~finite).sum(0)
            s["max"] = torch.maximum(s["max"], y.amax(0))
            x = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
            if x is not None and x.numel():
                xa = x.detach().float().abs()
                s["x_max"] = torch.maximum(s["x_max"], torch.where(torch.isfinite(xa), xa, 0).amax())
            k = min(16, y.numel())
            v, flat = y.flatten().topk(k)
            idx = torch.stack(
                [torch.full_like(flat, s["calls"]), flat // y.shape[1] + 0, flat % y.shape[1]], 1
            )
            v = torch.cat([s["top_v"], v])
            idx = torch.cat([s["top_idx"], idx])
            keep = v.topk(min(16, v.numel())).indices
            s["top_v"], s["top_idx"] = v[keep], idx[keep]
            s["rows"] += y.shape[0]
            s["calls"] += 1

        return hook

    for i, (_name, m) in enumerate(linears):
        handles.append(m.register_forward_hook(make_hook(i)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        out = {}
        for i, s in stats.items():
            out[f"rows_{i}"] = np.array(s["rows"])
            out[f"calls_{i}"] = np.array(s["calls"])
            for key in ("gt2", "nonfinite", "max", "top_v", "top_idx", "x_max"):
                out[f"{key}_{i}"] = s[key].cpu().numpy()
        np.savez(directory / f"layers-{step:06d}.npz", **out)


@contextmanager
def inference_window(seed: int, model_t0_us: int, device, model=None):
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
    if mode not in {"clean", "observe", "bitflip", "profile", "values", "operands", "layers",
                    "torchfi"}:
        raise ValueError(
            "ALPASIM_FI_MODE must be clean, observe, bitflip, profile, values, "
            "operands, layers, or torchfi"
        )
    if not _lock.acquire(blocking=False):
        raise RuntimeError("Fault injection requires serial inference")
    try:
        directory = Path(os.environ["ALPASIM_FI_OUTPUT"])
        directory.mkdir(parents=True, exist_ok=True)
        tool = None
        if mode not in ("clean", "layers", "torchfi"):
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
            traced = {
                int(x) for x in os.environ.get("ALPASIM_FI_TRACE", "").split(",") if x
            }
            layer_steps = {
                int(x) for x in os.environ.get("ALPASIM_FI_LAYER_STEPS", "").split(",") if x
            }
            parts = contextlib.nullcontext()
            if tool is not None and model is not None and hasattr(tool, "td_set_part"):
                tool.td_set_part.argtypes = [ctypes.c_int]
                tool.td_set_part.restype = None
                parts = _part_hooks(model, tool)
            if model is not None and step in traced:
                trace = _nonfinite_trace(model, directory / f"trace-{step:06d}.jsonl")
            elif mode == "layers" and model is not None and step in layer_steps:
                trace = _layer_profile(model, directory, step)
            elif mode == "torchfi" and model is not None and torch_fault_injection.selected(step):
                trace = torch_fault_injection.inject(model, directory, step, _PARTS)
            else:
                trace = contextlib.nullcontext()
            try:
                with trace, parts:
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
