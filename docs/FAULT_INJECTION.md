# TensorDynamic fault injection

The optional hook in `alpasim_driver.fault_injection` brackets the actual Alpamayo
1/1.5 model call and saves immediate candidate trajectories. It has no effect
unless `ALPASIM_FI_MODE` is configured. Use the `alpasim_injector` tool from the
matching TensorDynamic branch. See its README for compilation and GPU smoke tests.

The tool injects one HMMA register bit flip at a chosen inference, kernel,
instruction, block, and thread. It does not modify checkpoint weights. Start with
one driver process, one scene, batch size one, deterministic inference, and a new
output directory for every trial. The preset enforces serial rollouts and disables
retries so a failed injection cannot disappear into a successful retry.
It also fixes the model seed to `defines.fi_seed + inference_index - 1` (default
base seed 42), overriding randomly assigned driver session seeds at the model-call
boundary. Keep traffic simulation disabled for the initial paired experiment;
this hook does not seed other simulator services.

## Four matched runs

Prepare the standard AlpaSim environment, container images, checkpoint, and one
NuRec scene as described in TUTORIAL.md. Build the injector for the **driver
container environment** and GPU, including CUDA disassembler tools in that image.
The host-built library may need rebuilding against the container's glibc/CUDA.

```bash
export TENSORDYNAMIC_ROOT=/absolute/path/to/TensorDynamic_own_version

# Repeat this clean command with wizard.log_dir=$PWD/fi_clean_2.
uv run alpasim_wizard deploy=local topology=1gpu driver=alpamayo1 \
  +exp/presets=tensordynamic 'scenes.scene_ids=[YOUR_SCENE_ID]' \
  wizard.log_dir=$PWD/fi_clean_1

# Instrumentation control: visit and rewrite the register without changing bits.
uv run alpasim_wizard deploy=local topology=1gpu driver=alpamayo1 \
  +exp/presets=tensordynamic 'scenes.scene_ids=[YOUR_SCENE_ID]' \
  defines.fi_mode=observe \
  defines.fi_preload=/mnt/tensordynamic/alpasim_injector/alpasim_injector.so \
  wizard.log_dir=$PWD/fi_observe

# One sign-bit flip at the first HMMA in the first model invocation.
uv run alpasim_wizard deploy=local topology=1gpu driver=alpamayo1 \
  +exp/presets=tensordynamic 'scenes.scene_ids=[YOUR_SCENE_ID]' \
  defines.fi_mode=bitflip \
  defines.fi_preload=/mnt/tensordynamic/alpasim_injector/alpasim_injector.so \
  defines.fi_step=1 defines.fi_kernel=1 defines.fi_instruction=1 defines.fi_bit=31 \
  wizard.log_dir=$PWD/fi_bitflip
```

These commands require downloaded scene/model assets and functioning containers.
The preset preloads only the driver's Python process. Applying LD_PRELOAD to the
outer wizard does not instrument a container's model. Reuse the exact same scene,
seed policy, software and settings in all four runs. Compare the two clean runs
first; determinism settings are not a substitute for measuring repeatability.

## Evaluate

- `<run>/faults.jsonl`: selected SASS/register and original/corrupted bits. Require
  exactly one observed site; bit-flip XOR must equal `1 << defines.fi_bit`.
- `<run>/fault-injection/inference-*.json`: inference number, seed, simulation
  timestamp, status, and duration. A missed target raises an error.
- `<run>/fault-injection/prediction-*.npz`: all position/rotation candidates and
  the selected candidate index, immediately after inference and selection.
- AlpaSim's ordinary ASL/evaluation outputs: compare ego movement, collisions,
  off-road events, and progress over the completed rollout. A fault can alter
  later camera inputs and amplify or disappear through feedback.

Separate confirmed-but-masked faults, changed trajectories, degraded driving,
crashes, and timeouts. Do not infer a collision or a closed-loop effect from a
model-only test. Confirm output files exist and the selected invocation actually
ran; a zero-injection trial is an invalid experiment, not evidence of robustness.

The preset currently supports the shared Alpamayo 1/1.5 inference path. It does
not add hooks to VaVAM, Alpamayo 2, or renderer models. CUDA graph capture/replay
and concurrent inference are unsupported by this first single-fault tool.

`ALPASIM_FI_MODE=profile` with the TensorDynamic `alpasim_profiler.so` counts all HMMA
executions in the window instead of injecting. Use it for BER denominators.
