# SPDX-License-Identifier: Apache-2.0
"""PyTorch-level fault injection for Alpamayo (ALPASIM_FI_MODE=torchfi).

Corrupts outputs of nn.Linear layers in chosen model parts (vision encoder, language
model, action expert) with forward hooks; no NVBit tool is needed. Enabled from
fault_injection.inference_window, which passes the module-prefix -> part table.
Settings are documented in inject().
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

_PART_IDS = {"vision": 1, "language": 2, "action": 3}


def selected(step: int) -> bool:
    """ALPASIM_FI_STEPS=a-b (default every decision)."""
    spec = os.environ.get("ALPASIM_FI_STEPS", "1-")
    a, _, b = spec.partition("-")
    return step >= int(a) and (not b or step <= int(b))


def _bitflip_into(y32, exps):
    """Single exponent-bit flip that moves each FP32 value's exponent into `exps`
    (lowest such bit); returns (flipped bits, mask of values that have one)."""
    bits = y32.view(torch.int32)
    e = (bits >> 23) & 0xFF
    out = bits.clone()
    found = torch.zeros_like(bits, dtype=torch.bool)
    for k in range(8):
        ne = e ^ (1 << k)
        hit = torch.zeros_like(found)
        for x in exps:
            hit |= ne == x
        hit &= ~found
        out = torch.where(hit, bits ^ (1 << (23 + k)), out)
        found |= hit
    return out.view(torch.float32), found


def _boundcheck(y32):
    """MPGemmFI BoundCheck on FP32: exponent 1eeexxxx with any e set is corrected by
    clearing bits 29..27; returns (corrected, flagged mask)."""
    bits = y32.view(torch.int32)
    flag = (((bits >> 30) & 1) == 1) & (((bits >> 27) & 7) != 0)
    return torch.where(flag, bits & ~(7 << 27), bits).view(torch.float32), flag


@contextmanager
def inject(model, directory: Path, step: int, parts_table):
    """PyTorch-level fault injection into nn.Linear outputs of chosen model parts.

    ALPASIM_FI_PARTS=vision,language,action (default all three)
    ALPASIM_FI_MODEL=scale | window | replace
      scale:   y * ALPASIM_FI_SCALE (default 1000) for |y| > ALPASIM_FI_ABS_GT (default 2)
      window:  one exponent-bit flip into exponents ALPASIM_FI_EXPS (default 142,143)
      replace: sign(y) * U[lo, hi) from ALPASIM_FI_REPLACE (default 1024-130000), any y
      clamp:   every |y| > ALPASIM_FI_ABS_GT (default 2) becomes sign(y) * ALPASIM_FI_CLAMP_TO
               (default 1.9); set ALPASIM_FI_P=1 to clamp all of them
    ALPASIM_FI_P: probability per qualifying output element
    ALPASIM_FI_BOUNDCHECK=1: apply BoundCheck to the hooked layers' outputs after the fault
    ALPASIM_FI_TORCH_SEED (default 1): faults are a function of seed, decision, call.
    Writes torchfi-<decision>.jsonl: one line per layer call with faults (up to 64 of
    them listed: flat index, before, after) and a summary line.
    """
    parts = {_PART_IDS[x] for x in os.environ.get("ALPASIM_FI_PARTS", "vision,language,action").split(",") if x}
    fault = os.environ.get("ALPASIM_FI_MODEL", "scale")
    if fault not in ("scale", "window", "replace", "clamp"):
        raise ValueError("ALPASIM_FI_MODEL must be scale, window, replace or clamp")
    clamp_to = float(os.environ.get("ALPASIM_FI_CLAMP_TO", "1.9"))
    p = float(os.environ.get("ALPASIM_FI_P", "0"))
    scale = float(os.environ.get("ALPASIM_FI_SCALE", "1000"))
    abs_gt = float(os.environ.get("ALPASIM_FI_ABS_GT", "2"))
    exps = [int(x) for x in os.environ.get("ALPASIM_FI_EXPS", "142,143").split(",")]
    lo, hi = (float(x) for x in os.environ.get("ALPASIM_FI_REPLACE", "1024-130000").split("-"))
    boundcheck = os.environ.get("ALPASIM_FI_BOUNDCHECK", "0") == "1"
    seed = int(os.environ.get("ALPASIM_FI_TORCH_SEED", "1"))
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        part = next((pid for prefix, pid in parts_table if name.startswith(prefix)), 0)
        if part in parts:
            targets.append((name, part, module))
    totals = dict(calls=0, outputs=0, qualifying=0, faults=0, flagged=0)
    handles = []
    log = (directory / f"torchfi-{step:06d}.jsonl").open("x")

    def make_hook(name, part):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output) or not output.is_floating_point():
                return None
            call = totals["calls"]
            totals["calls"] += 1
            y = output.detach().float()
            if fault in ("scale", "clamp"):
                qual = torch.isfinite(y) & (y.abs() > abs_gt)
            elif fault == "window":
                flipped, qual = _bitflip_into(y.contiguous(), exps)
            else:
                qual = torch.isfinite(y)
            gen = torch.Generator(device=y.device)
            gen.manual_seed((seed * 1000003 + step) * 1000003 + call)
            pick = qual if p >= 1 else qual & (
                torch.rand(y.shape, generator=gen, device=y.device, dtype=torch.float64) < p)
            nq = int(qual.sum())
            n = int(pick.sum())
            totals["outputs"] += y.numel()
            totals["qualifying"] += nq
            if n == 0 and not boundcheck:
                return None
            if fault == "scale":
                new = y * scale
            elif fault == "clamp":
                new = torch.sign(y) * clamp_to
            elif fault == "window":
                new = flipped
            else:
                u = torch.rand(y.shape, generator=gen, device=y.device)
                new = torch.sign(y) * (lo + u * (hi - lo))
                new = torch.where(new == 0, torch.full_like(new, lo), new)
            out = torch.where(pick, new, y)
            if boundcheck:
                out, flag = _boundcheck(out.contiguous())
                totals["flagged"] += int(flag.sum())
            totals["faults"] += n
            if n:
                idx = pick.flatten().nonzero().flatten()[:64]
                log.write(json.dumps(dict(
                    layer=name, part=part, call=call, faults=n, qualifying=nq,
                    index=idx.tolist(),
                    before=y.flatten()[idx].tolist(),
                    after=out.flatten()[idx].tolist())) + "\n")
            return out.to(output.dtype)
        return hook

    for name, part, module in targets:
        handles.append(module.register_forward_hook(make_hook(name, part)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        log.write(json.dumps(dict(summary=True, decision=step, model=fault, p=p, parts=sorted(parts),
                                  boundcheck=boundcheck, layers=len(targets), **totals)) + "\n")
        log.close()
