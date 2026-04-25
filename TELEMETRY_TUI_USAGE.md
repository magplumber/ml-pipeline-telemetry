# Terminal Telemetry Dashboard Usage Guide

## 1. Purpose

`scripts/telemetry_tui.py` is a low-noise, terminal-only telemetry dashboard for Linux systems.

It provides:

1. CPU telemetry (left panel)
2. GPU + Memory telemetry (middle/right panel):
   - RAM and SWAP summary
   - GPU section (per-device metrics)
3. GPU process telemetry (third panel when width allows)

The interface is intentionally plain:

- no colors
- no bars/graphs
- fixed-width aligned text
- refresh-based updates only

## 2. Operational Boundary

In scope:

1. CPU metrics from `psutil`
2. GPU summary from CLI tools:
   - NVIDIA: `nvidia-smi`
   - AMD ROCm: `rocm-smi`
3. GPU process telemetry (NVIDIA implemented)
4. Curses-based TUI rendering
5. Timeout + stale fallback behavior

Out of scope:

1. File logging/export
2. Background threads
3. Async event loops
4. Rich/graphical visualizations
5. NVML Python bindings

## 3. Requirements

1. Python 3
2. `psutil`
3. At least one supported GPU CLI tool:
   - `nvidia-smi` for NVIDIA
   - `rocm-smi` for ROCm
4. Terminal with curses support

## 4. Start Commands

From repository root:

```bash
.venv/bin/python scripts/telemetry_tui.py
```

Force backend:

```bash
.venv/bin/python scripts/telemetry_tui.py --gpu-backend nvidia
.venv/bin/python scripts/telemetry_tui.py --gpu-backend rocm
```

Hide process panel and keep CPU/GPU only:

```bash
.venv/bin/python scripts/telemetry_tui.py --hide-processes
```

## 5. Controls

Runtime controls:

1. `q` to quit
2. `Q` to quit

## 6. Layout Rules

The dashboard adapts by terminal width:

1. Width >= 120 columns and processes enabled:
   - left: CPU
   - middle: GPU summary
   - right: GPU processes
2. Width < 120 columns and processes enabled:
   - left: CPU
   - right: combined GPU summary + process section
3. Processes hidden (`--hide-processes`):
   - two-panel CPU + GPU summary layout

On short terminal height, panels truncate and show:

- `... (N hidden)`

## 7. Telemetry State Semantics

Panel states:

1. `OK`: fresh telemetry sample
2. `STALE`: recent failure, showing last good data
3. `N/A`: no usable data, unsupported backend, or stale threshold exceeded

Timestamps are shown in panel headers (`Updated: HH:MM:SS`).

## 8. Backend Selection

`--gpu-backend auto` chooses in this order:

1. `nvidia-smi` if available
2. `rocm-smi` if available
3. `none` (N/A state)

Panel backend labels are explicit:

1. `nvidia-smi`
2. `rocm-smi`
3. `none`

## 9. Polling Model

Polling is independent per telemetry stream:

1. CPU: every UI refresh loop (`--refresh-interval`)
2. GPU summary: `--gpu-poll-interval`
3. GPU processes: `--process-poll-interval`

A process polling failure does not force GPU summary state changes.

## 10. Timeouts and Fallback

Each collector uses timeout guards:

1. GPU summary: `--gpu-timeout`
2. Process telemetry: `--process-timeout`

On failure:

1. if last-good sample exists -> panel becomes `STALE`
2. if stale/failure thresholds exceed limits -> panel becomes `N/A`

Threshold controls:

1. `--stale-seconds` for GPU summary
2. `--process-stale-seconds` for process panel
3. `--max-fails-before-na` shared failure threshold

Deprecated compatibility alias:

- `--rocm-timeout` maps to `--gpu-timeout`

## 11. CLI Reference

```text
--gpu-backend {auto,nvidia,rocm}
--gpu-timeout <seconds>
--refresh-interval <seconds>
--gpu-poll-interval <seconds>
--process-poll-interval <seconds>
--process-timeout <seconds>
--process-stale-seconds <seconds>
--max-process-name <int>
--hide-processes
--rocm-timeout <seconds>   # deprecated alias for --gpu-timeout
--stale-seconds <seconds>
--max-fails-before-na <int>
```

## 12. Process Panel Behavior

GPU panel begins with RAM/SWAP lines, then a separate GPU section below them:

1. `RAM : <used> / <total> GiB (<percent>)`
2. `SWAP: <used> / <total> GiB`
3. `GPU`

RAM pressure marker:

- `*` is appended to RAM percent when RAM usage is above 85%.

If memory telemetry cannot be read, dashboard falls back to:

1. `RAM : N/A`
2. `SWAP: N/A`

NVIDIA backend (`nvidia-smi`):

- process rows include GPU index, PID, process name, and used GPU memory

ROCm backend (`rocm-smi`):

- process telemetry currently reports N/A by design
- message indicates ROCm process telemetry is not implemented

## 13. Recommended Profiles

Default balanced profile:

```bash
.venv/bin/python scripts/telemetry_tui.py
```

Higher responsiveness:

```bash
.venv/bin/python scripts/telemetry_tui.py \
  --refresh-interval 0.5 \
  --gpu-poll-interval 1.0 \
  --process-poll-interval 1.0
```

More conservative polling:

```bash
.venv/bin/python scripts/telemetry_tui.py \
  --refresh-interval 1.0 \
  --gpu-poll-interval 3.0 \
  --process-poll-interval 3.0
```

Short timeout stress test:

```bash
.venv/bin/python scripts/telemetry_tui.py \
  --gpu-timeout 0.1 \
  --process-timeout 0.1
```

## 14. Troubleshooting

### 14.1 GPU panel shows N/A immediately

Possible causes:

1. no supported CLI tool in PATH
2. forced backend not present (for example `--gpu-backend rocm` on NVIDIA host)

Checks:

```bash
command -v nvidia-smi
command -v rocm-smi
```

### 14.2 NVIDIA process panel empty

If no active compute apps are running, panel shows:

- `No active compute processes`

Manual source check:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits
```

### 14.3 Values frequently STALE

Tune:

1. increase `--gpu-timeout` and/or `--process-timeout`
2. increase `--stale-seconds` and/or `--process-stale-seconds`
3. increase poll intervals

### 14.4 Terminal clipping

1. widen terminal to 120+ columns for 3-panel mode
2. increase terminal height to reduce truncation

## 15. Example End-to-End Verification

```bash
# 1) Verify script help
.venv/bin/python scripts/telemetry_tui.py --help

# 2) Run default
.venv/bin/python scripts/telemetry_tui.py

# 3) Run without process panel
.venv/bin/python scripts/telemetry_tui.py --hide-processes

# 4) Force NVIDIA mode
.venv/bin/python scripts/telemetry_tui.py --gpu-backend nvidia
```
