#!/usr/bin/env python3
"""
Terminal Telemetry Dashboard (Low-Noise TUI)

Purpose:
- Provide a minimal two-panel terminal dashboard for CPU and GPU telemetry.
- Keep output static and text-only with fixed-width fields and no visual effects.

Operational Boundary:
- CPU metrics use psutil only.
- NVIDIA GPU metrics use nvidia-smi CLI.
- AMD GPU metrics use rocm-smi CLI.
- Rendering uses curses only.
- No file writes, no background workers, no async framework.

Inputs:
- System telemetry from psutil.
- nvidia-smi command output:
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader,nounits
- rocm-smi command output:
  rocm-smi --showtemp --showpower --showuse --showmemuse --showclocks

Outputs:
- In-terminal live telemetry view refreshed once per second.
- Exit status 0 on clean quit, non-zero on fatal setup errors.

Fail-Safe Behavior:
- Primes cpu_percent to avoid first-sample noise.
- Uses timeout for GPU telemetry calls.
- Falls back to last good GPU sample on timeout/parse failure.
- Marks GPU panel state as STALE or N/A when fresh data is unavailable.
- Truncates panel output with a hidden-line summary when terminal height is limited.

Usage:
- python scripts/telemetry_tui.py
- python scripts/telemetry_tui.py --refresh-interval 1.0 --gpu-poll-interval 2.0
- python scripts/telemetry_tui.py --gpu-timeout 0.8 --stale-seconds 30
- python scripts/telemetry_tui.py --gpu-backend auto
- python scripts/telemetry_tui.py --process-poll-interval 2.0 --process-timeout 0.8
- python scripts/telemetry_tui.py --hide-processes
"""

from __future__ import annotations

import argparse
import curses
import re
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any

import psutil


DEFAULT_REFRESH_INTERVAL = 1.0
DEFAULT_GPU_POLL_INTERVAL = 2.0
DEFAULT_GPU_TIMEOUT = 0.8
DEFAULT_STALE_SECONDS = 30.0
DEFAULT_MAX_FAILS_BEFORE_NA = 10
DEFAULT_GPU_BACKEND = "auto"
DEFAULT_PROCESS_POLL_INTERVAL = 2.0
DEFAULT_PROCESS_TIMEOUT = 0.8
DEFAULT_PROCESS_STALE_SECONDS = 30.0
DEFAULT_MAX_PROCESS_NAME = 32
RAM_WARN_THRESHOLD_PERCENT = 85.0

ROCM_SMI_CMD = [
    "rocm-smi",
    "--showtemp",
    "--showpower",
    "--showuse",
    "--showmemuse",
    "--showclocks",
]

NVIDIA_SMI_CMD = [
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem",
    "--format=csv,noheader,nounits",
]

NVIDIA_GPU_UUID_MAP_CMD = [
    "nvidia-smi",
    "--query-gpu=index,uuid",
    "--format=csv,noheader,nounits",
]

NVIDIA_COMPUTE_APPS_CMD = [
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
]

GPU_INDEX_RE = re.compile(r"GPU\[(\d+)\]")
FLOAT_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")
MHZ_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*mhz", re.IGNORECASE)


def now_ts() -> str:
    """Return current local time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def safe_float(text: str) -> float | None:
    """Extract first floating-point value from text."""
    match = FLOAT_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def bytes_to_mib(value: float | None) -> float | None:
    """Convert bytes to MiB when value is known."""
    if value is None:
        return None
    return value / (1024.0 * 1024.0)


def format_number(value: float | None, suffix: str = "", precision: int = 1) -> str:
    """Format numeric values with fixed precision or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}{suffix}"


def format_gib(value: float | None) -> str:
    """Format GiB values with one decimal place or N/A."""
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def format_percent(value: float | None) -> str:
    """Format percent values with one decimal place plus % or N/A."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def get_cpu_telemetry() -> dict[str, Any]:
    """
    Collect CPU telemetry.

    Returns:
    - usage_per_core: list[float]
    - freq_mhz: float | None
    - per_core_freq_mhz: list[float] | None
    - temps: dict[str, float]
    - timestamp: str
    - state: str
    """
    usage_per_core = psutil.cpu_percent(interval=None, percpu=True)

    global_freq = psutil.cpu_freq(percpu=False)
    freq_mhz = global_freq.current if global_freq else None

    per_core_freq: list[float] | None = None
    try:
        per_core_freq_raw = psutil.cpu_freq(percpu=True)
        if per_core_freq_raw and isinstance(per_core_freq_raw, list):
            per_core_freq = [float(item.current) for item in per_core_freq_raw]
    except Exception:
        per_core_freq = None

    temps_out: dict[str, float] = {}
    try:
        all_temps = psutil.sensors_temperatures(fahrenheit=False) or {}
    except Exception:
        all_temps = {}

    allowed_key_fragments = ("k10temp", "zen", "cpu", "tctl", "ccd")
    for sensor_key, entries in all_temps.items():
        key_lower = sensor_key.lower()
        if not any(fragment in key_lower for fragment in allowed_key_fragments):
            continue

        for entry in entries:
            label = (entry.label or sensor_key).strip()
            current = getattr(entry, "current", None)
            if current is None:
                continue

            label_lower = label.lower()
            if any(fragment in label_lower for fragment in ("tctl", "ccd", "cpu", "package", "die")):
                temps_out[label] = float(current)
            elif sensor_key not in temps_out:
                temps_out[sensor_key] = float(current)

    return {
        "usage_per_core": usage_per_core,
        "freq_mhz": float(freq_mhz) if freq_mhz is not None else None,
        "per_core_freq_mhz": per_core_freq,
        "temps": dict(sorted(temps_out.items(), key=lambda item: item[0].lower())),
        "timestamp": now_ts(),
        "state": "OK",
    }


def get_ram_telemetry() -> dict[str, float]:
    """
    Collect RAM and swap telemetry.

    Returns:
    - ram_used_gib
    - ram_total_gib
    - ram_percent
    - swap_used_gib
    - swap_total_gib
    """
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()

    return {
        "ram_used_gib": float(vm.used) / (1024.0 ** 3),
        "ram_total_gib": float(vm.total) / (1024.0 ** 3),
        "ram_percent": float(vm.percent),
        "swap_used_gib": float(sm.used) / (1024.0 ** 3),
        "swap_total_gib": float(sm.total) / (1024.0 ** 3),
    }


def build_ram_swap_lines() -> tuple[str, str]:
    """Build RAM/SWAP display lines with fail-safe N/A fallback."""
    try:
        ram = get_ram_telemetry()
        ram_percent = float(ram["ram_percent"])
        ram_percent_text = format_percent(ram_percent)
        if ram_percent > RAM_WARN_THRESHOLD_PERCENT:
            ram_percent_text = f"{ram_percent_text}*"

        ram_line = (
            f"RAM : {format_gib(ram['ram_used_gib']):>6} / "
            f"{format_gib(ram['ram_total_gib']):>6} GiB "
            f"({ram_percent_text:>6})"
        )
        swap_line = (
            f"SWAP: {format_gib(ram['swap_used_gib']):>6} / "
            f"{format_gib(ram['swap_total_gib']):>6} GiB"
        )
        return ram_line, swap_line
    except Exception:
        return "RAM : N/A", "SWAP: N/A"


def _empty_gpu_record(index: int) -> dict[str, Any]:
    """Create empty metric shell for one GPU index."""
    return {
        "index": index,
        "temp_c": None,
        "power_w": None,
        "util_percent": None,
        "vram_used_mib": None,
        "vram_total_mib": None,
        "sclk_mhz": None,
        "mclk_mhz": None,
    }


def _ensure_gpu(gpus: dict[int, dict[str, Any]], gpu_index: int) -> dict[str, Any]:
    """Get or create mutable GPU record by index."""
    if gpu_index not in gpus:
        gpus[gpu_index] = _empty_gpu_record(gpu_index)
    return gpus[gpu_index]


def detect_gpu_backend(requested_backend: str) -> tuple[str, str | None]:
    """
    Resolve the GPU backend from CLI request and available system tools.

    Returns:
    - backend: one of nvidia, rocm, none
    - startup_note: optional warning/note for display
    """
    has_nvidia = shutil.which("nvidia-smi") is not None
    has_rocm = shutil.which("rocm-smi") is not None

    if requested_backend == "nvidia":
        if has_nvidia:
            return "nvidia", None
        return "none", "nvidia-smi not found"

    if requested_backend == "rocm":
        if has_rocm:
            return "rocm", None
        return "none", "rocm-smi not found"

    # auto mode
    if has_nvidia:
        return "nvidia", None
    if has_rocm:
        return "rocm", None
    return "none", "no supported GPU telemetry tool found"


def gpu_backend_tool_name(backend: str) -> str:
    """Return the CLI tool name used by a backend."""
    if backend == "nvidia":
        return "nvidia-smi"
    if backend == "rocm":
        return "rocm-smi"
    return "unknown"


def gpu_backend_display_name(backend: str) -> str:
    """Return human-readable backend display label for panel titles."""
    if backend == "nvidia":
        return "nvidia-smi"
    if backend == "rocm":
        return "rocm-smi"
    return "none"


def parse_rocm_smi_output(output: str) -> list[dict[str, Any]]:
    """
    Parse rocm-smi output into per-GPU telemetry records.

    Parsing is tolerant and regex-based. Missing metrics remain None.
    """
    gpus: dict[int, dict[str, Any]] = {}

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        index_match = GPU_INDEX_RE.search(line)
        if not index_match:
            continue

        gpu_index = int(index_match.group(1))
        record = _ensure_gpu(gpus, gpu_index)
        lower = line.lower()
        metric_segment = line.rsplit(":", 1)[-1]

        # Utilization
        if "gpu use" in lower and "(%)" in lower:
            record["util_percent"] = safe_float(metric_segment)
            continue

        # Temperature variants
        if "temp" in lower or "temperature" in lower:
            value = safe_float(metric_segment)
            if value is not None:
                record["temp_c"] = value
            continue

        # Power variants
        if ("power" in lower and "avg" in lower) or "average graphics package power" in lower:
            value = safe_float(metric_segment)
            if value is not None:
                record["power_w"] = value
            continue

        if "power" in lower and "w" in lower and record["power_w"] is None:
            value = safe_float(metric_segment)
            if value is not None:
                record["power_w"] = value
            continue

        # VRAM bytes metrics
        if "vram total used" in lower and "(b)" in lower:
            record["vram_used_mib"] = bytes_to_mib(safe_float(metric_segment))
            continue

        if "vram total" in lower and "(b)" in lower:
            record["vram_total_mib"] = bytes_to_mib(safe_float(metric_segment))
            continue

        # VRAM MiB direct metrics
        if "vram" in lower and "mib" in lower and "total" in lower:
            record["vram_total_mib"] = safe_float(metric_segment)
            continue

        if "vram" in lower and "mib" in lower and "used" in lower:
            record["vram_used_mib"] = safe_float(metric_segment)
            continue

        # Clocks
        if "sclk" in lower:
            mhz_match = MHZ_RE.search(metric_segment)
            if mhz_match:
                record["sclk_mhz"] = float(mhz_match.group(1))
            else:
                record["sclk_mhz"] = safe_float(metric_segment)
            continue

        if "mclk" in lower:
            mhz_match = MHZ_RE.search(metric_segment)
            if mhz_match:
                record["mclk_mhz"] = float(mhz_match.group(1))
            else:
                record["mclk_mhz"] = safe_float(metric_segment)
            continue

    return [gpus[index] for index in sorted(gpus.keys())]


def parse_nvidia_smi_output(output: str) -> list[dict[str, Any]]:
    """
    Parse nvidia-smi CSV output into per-GPU telemetry records.

    Expected columns:
    index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem
    """
    gpus: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 9:
            continue

        try:
            index = int(parts[0])
        except ValueError:
            continue

        name = parts[1]
        util = safe_float(parts[2])
        mem_used = safe_float(parts[3])
        mem_total = safe_float(parts[4])
        temp = safe_float(parts[5])
        power = safe_float(parts[6])
        sclk = safe_float(parts[7])
        mclk = safe_float(parts[8])

        gpus.append(
            {
                "index": index,
                "name": name,
                "temp_c": temp,
                "power_w": power,
                "util_percent": util,
                "vram_used_mib": mem_used,
                "vram_total_mib": mem_total,
                "sclk_mhz": sclk,
                "mclk_mhz": mclk,
            }
        )

    return sorted(gpus, key=lambda item: item["index"])


def parse_nvidia_gpu_uuid_map(output: str) -> dict[str, int]:
    """
    Parse nvidia-smi GPU UUID mapping output.

    Expected line format:
    0, GPU-xxxx
    """
    uuid_to_index: dict[str, int] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        uuid_to_index[parts[1]] = index
    return uuid_to_index


def parse_nvidia_compute_apps_output(output: str, uuid_to_index: dict[str, int]) -> list[dict[str, Any]]:
    """
    Parse nvidia-smi compute-app records and map GPU UUID to index.

    Expected line format:
    GPU-xxxx, PID, process_name, used_memory
    """
    processes: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) < 4:
            continue

        gpu_uuid = parts[0]
        gpu_index = uuid_to_index.get(gpu_uuid)
        if gpu_index is None:
            continue

        try:
            pid = int(parts[1])
        except ValueError:
            continue

        name = parts[2] or "unknown"
        used_memory_mib = safe_float(parts[3])
        if used_memory_mib is None:
            used_memory_mib = 0.0

        processes.append(
            {
                "gpu_index": gpu_index,
                "pid": pid,
                "name": name,
                "used_memory_mib": used_memory_mib,
            }
        )

    return sorted(processes, key=lambda item: (item["gpu_index"], item["pid"]))


def collect_nvidia_gpu_uuid_map(timeout: float) -> dict[str, int]:
    """Collect NVIDIA UUID-to-index map."""
    result = subprocess.run(
        NVIDIA_GPU_UUID_MAP_CMD,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output_text = result.stdout or ""
    parsed = parse_nvidia_gpu_uuid_map(output_text)
    if result.returncode != 0 and not parsed:
        raise RuntimeError((result.stderr or "nvidia-smi GPU UUID query failed").strip())
    if not parsed:
        raise RuntimeError("nvidia-smi returned no parseable GPU UUID mapping")
    return parsed


def collect_nvidia_process_telemetry(timeout: float) -> list[dict[str, Any]]:
    """Collect active NVIDIA compute process telemetry."""
    uuid_to_index = collect_nvidia_gpu_uuid_map(timeout)
    result = subprocess.run(
        NVIDIA_COMPUTE_APPS_CMD,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    output_text = result.stdout or ""
    parsed = parse_nvidia_compute_apps_output(output_text, uuid_to_index)
    if result.returncode != 0 and not parsed:
        raise RuntimeError((result.stderr or "nvidia-smi compute-app query failed").strip())
    return parsed


def collect_rocm_gpu_telemetry(gpu_timeout: float) -> list[dict[str, Any]]:
    """Run rocm-smi and parse telemetry rows."""
    result = subprocess.run(
        ROCM_SMI_CMD,
        capture_output=True,
        text=True,
        timeout=gpu_timeout,
        check=False,
    )

    output_text = result.stdout or ""
    parsed = parse_rocm_smi_output(output_text)
    if result.returncode != 0 and not parsed:
        raise RuntimeError((result.stderr or "rocm-smi failed").strip())
    if not parsed:
        raise RuntimeError("rocm-smi returned no parseable GPU metrics")
    return parsed


def collect_nvidia_gpu_telemetry(gpu_timeout: float) -> list[dict[str, Any]]:
    """Run nvidia-smi query and parse telemetry rows."""
    result = subprocess.run(
        NVIDIA_SMI_CMD,
        capture_output=True,
        text=True,
        timeout=gpu_timeout,
        check=False,
    )

    output_text = result.stdout or ""
    parsed = parse_nvidia_smi_output(output_text)
    if result.returncode != 0 and not parsed:
        raise RuntimeError((result.stderr or "nvidia-smi failed").strip())
    if not parsed:
        raise RuntimeError("nvidia-smi returned no parseable GPU metrics")
    return parsed


def get_gpu_telemetry(
    last_good: dict[str, Any] | None,
    backend: str,
    gpu_timeout: float,
    stale_seconds: float,
    max_fails_before_na: int,
    fail_count: int,
) -> tuple[dict[str, Any], int]:
    """
    Collect GPU summary telemetry with timeout and stale fallback.

    Returns:
    - gpu_data dict containing:
      - gpus: list[dict]
      - timestamp: str
      - state: OK | STALE | N/A
      - error: str | None
    - updated_fail_count
    """
    ts = now_ts()

    if backend == "none":
        return (
            {
                "gpus": [],
                "timestamp": ts,
                "state": "N/A",
                "error": "GPU backend unavailable",
                "backend": backend,
                "last_success_epoch": 0.0,
            },
            fail_count + 1,
        )

    try:
        if backend == "nvidia":
            parsed = collect_nvidia_gpu_telemetry(gpu_timeout)
        elif backend == "rocm":
            parsed = collect_rocm_gpu_telemetry(gpu_timeout)
        else:
            raise RuntimeError(f"unsupported backend: {backend}")

        data = {
            "gpus": parsed,
            "timestamp": ts,
            "state": "OK",
            "error": None,
            "backend": backend,
            "last_success_epoch": time.time(),
        }
        return data, 0

    except FileNotFoundError:
        tool = gpu_backend_tool_name(backend)
        error = f"{tool} not found"
    except subprocess.TimeoutExpired:
        tool = gpu_backend_tool_name(backend)
        error = f"{tool} timeout ({gpu_timeout:.2f}s)"
    except Exception as exc:
        error = str(exc)

    updated_fail_count = fail_count + 1
    if last_good:
        age = time.time() - float(last_good.get("last_success_epoch", 0.0))
        state = "STALE"
        if age > stale_seconds or updated_fail_count >= max_fails_before_na:
            state = "N/A"

        fallback = {
            "gpus": last_good.get("gpus", []),
            "timestamp": last_good.get("timestamp", ts),
            "state": state,
            "error": error,
            "backend": backend,
            "last_success_epoch": last_good.get("last_success_epoch", 0.0),
        }
        return fallback, updated_fail_count

    empty = {
        "gpus": [],
        "timestamp": ts,
        "state": "N/A",
        "error": error,
        "backend": backend,
        "last_success_epoch": 0.0,
    }
    return empty, updated_fail_count


def get_process_telemetry(
    last_good: dict[str, Any] | None,
    backend: str,
    process_timeout: float,
    stale_seconds: float,
    fail_count: int,
    max_fails_before_na: int,
) -> tuple[dict[str, Any], int]:
    """
    Collect process-focused GPU telemetry with stale fallback.

    State transitions mirror GPU summary telemetry but are tracked independently.
    """
    ts = now_ts()

    if backend == "none":
        return (
            {
                "processes": [],
                "timestamp": ts,
                "state": "N/A",
                "error": "GPU backend unavailable",
                "backend": backend,
                "last_success_epoch": 0.0,
            },
            fail_count,
        )

    if backend == "rocm":
        return (
            {
                "processes": [],
                "timestamp": ts,
                "state": "N/A",
                "error": "Process telemetry not implemented for ROCm backend",
                "backend": backend,
                "last_success_epoch": 0.0,
            },
            fail_count,
        )

    try:
        processes = collect_nvidia_process_telemetry(process_timeout)
        data = {
            "processes": processes,
            "timestamp": ts,
            "state": "OK",
            "error": None,
            "backend": backend,
            "last_success_epoch": time.time(),
        }
        return data, 0

    except FileNotFoundError:
        tool = gpu_backend_tool_name(backend)
        error = f"{tool} not found"
    except subprocess.TimeoutExpired:
        tool = gpu_backend_tool_name(backend)
        error = f"{tool} timeout ({process_timeout:.2f}s)"
    except Exception as exc:
        error = str(exc)

    updated_fail_count = fail_count + 1
    if last_good:
        age = time.time() - float(last_good.get("last_success_epoch", 0.0))
        state = "STALE"
        if age > stale_seconds or updated_fail_count >= max_fails_before_na:
            state = "N/A"

        fallback = {
            "processes": last_good.get("processes", []),
            "timestamp": last_good.get("timestamp", ts),
            "state": state,
            "error": error,
            "backend": backend,
            "last_success_epoch": last_good.get("last_success_epoch", 0.0),
        }
        return fallback, updated_fail_count

    empty = {
        "processes": [],
        "timestamp": ts,
        "state": "N/A",
        "error": error,
        "backend": backend,
        "last_success_epoch": 0.0,
    }
    return empty, updated_fail_count


def safe_addstr(win: curses.window, y: int, x: int, text: str) -> None:
    """Write text defensively without raising on boundary collisions."""
    try:
        max_y, max_x = win.getmaxyx()
        if y < 0 or y >= max_y or x >= max_x:
            return
        clipped = text[: max(0, max_x - x - 1)]
        win.addstr(y, x, clipped)
    except curses.error:
        return


def render_lines(win: curses.window, title: str, lines: list[str]) -> None:
    """Render fixed lines with terminal-height truncation notice."""
    win.erase()
    max_y, _ = win.getmaxyx()

    all_lines = [title, ""] + lines

    if max_y <= 0:
        return

    if len(all_lines) <= max_y:
        for y, line in enumerate(all_lines):
            safe_addstr(win, y, 0, line)
        return

    visible_capacity = max(1, max_y - 1)
    hidden = len(all_lines) - visible_capacity
    visible_lines = all_lines[:visible_capacity]
    if visible_lines:
        visible_lines[-1] = f"... ({hidden} hidden)"

    for y, line in enumerate(visible_lines):
        safe_addstr(win, y, 0, line)


def draw_cpu(win: curses.window, cpu_data: dict[str, Any]) -> None:
    """Draw CPU panel with fixed-width aligned rows."""
    state = cpu_data.get("state", "N/A")
    timestamp = cpu_data.get("timestamp", "--:--:--")
    title = f"CPU (Updated: {timestamp} | State: {state})"

    lines: list[str] = []
    freq_mhz = cpu_data.get("freq_mhz")
    lines.append(f"  Freq: {format_number(freq_mhz, ' MHz', precision=0):>10}")

    usage = cpu_data.get("usage_per_core", [])
    per_core_freq = cpu_data.get("per_core_freq_mhz")

    for idx, usage_value in enumerate(usage):
        usage_text = format_number(float(usage_value), "%", precision=1)
        if per_core_freq and idx < len(per_core_freq):
            core_freq = format_number(float(per_core_freq[idx]), " MHz", precision=0)
            lines.append(f"  Core {idx:02d}: {usage_text:>6} | {core_freq:>10}")
        else:
            lines.append(f"  Core {idx:02d}: {usage_text:>6}")

    temps = cpu_data.get("temps", {})
    if temps:
        lines.append("")
        for label, value in temps.items():
            lines.append(f"  {label:<12}: {format_number(float(value), 'C', precision=1):>8}")

    render_lines(win, title, lines)
    win.noutrefresh()


def draw_gpu(win: curses.window, gpu_data: dict[str, Any]) -> None:
    """Draw GPU panel with one block per GPU and fixed field alignment."""
    state = gpu_data.get("state", "N/A")
    timestamp = gpu_data.get("timestamp", "--:--:--")
    error = gpu_data.get("error")
    backend = gpu_data.get("backend", "none")

    title = f"GPU + Memory ({gpu_backend_display_name(backend)}) (Updated: {timestamp} | State: {state})"
    lines: list[str] = []

    ram_line, swap_line = build_ram_swap_lines()
    lines.append(ram_line)
    lines.append(swap_line)
    lines.append("")
    lines.append("GPU")
    lines.append("")

    if error and state != "OK":
        lines.append(f"  Note: {error}")
        lines.append("")

    gpus = gpu_data.get("gpus", [])
    if not gpus:
        lines.append("  No GPU telemetry available")
    else:
        for i, gpu in enumerate(gpus):
            idx = gpu.get("index", i)
            name = gpu.get("name")
            if name:
                lines.append(f"  GPU {idx} ({name})")
            else:
                lines.append(f"  GPU {idx}")
            lines.append(f"    Temp : {format_number(gpu.get('temp_c'), 'C', precision=1):>12}")
            lines.append(f"    Power: {format_number(gpu.get('power_w'), ' W', precision=2):>12}")
            lines.append(f"    Util : {format_number(gpu.get('util_percent'), '%', precision=1):>12}")

            used = format_number(gpu.get("vram_used_mib"), " MiB", precision=0)
            total = format_number(gpu.get("vram_total_mib"), " MiB", precision=0)
            lines.append(f"    VRAM : {used:>8} / {total:>8}")

            lines.append(f"    SCLK : {format_number(gpu.get('sclk_mhz'), ' MHz', precision=0):>12}")
            lines.append(f"    MCLK : {format_number(gpu.get('mclk_mhz'), ' MHz', precision=0):>12}")
            lines.append("")

    render_lines(win, title, lines)
    win.noutrefresh()


def draw_processes(win: curses.window, process_data: dict[str, Any], max_process_name: int) -> None:
    """Draw process telemetry panel with fixed-width aligned rows."""
    state = process_data.get("state", "N/A")
    timestamp = process_data.get("timestamp", "--:--:--")
    error = process_data.get("error")
    backend = process_data.get("backend", "none")

    title = (
        f"GPU Processes ({gpu_backend_display_name(backend)} | "
        f"Updated: {timestamp} | State: {state})"
    )
    lines: list[str] = []

    if error and state != "OK":
        lines.append(f"  Note: {error}")
        lines.append("")

    processes = process_data.get("processes", [])
    if not processes:
        lines.append("  No active compute processes")
    else:
        lines.append("  GPU PID      Mem MiB   Process")
        for item in processes:
            gpu_index = item.get("gpu_index", "?")
            pid = item.get("pid", "?")
            used_memory = item.get("used_memory_mib")
            name = str(item.get("name", "unknown"))
            if max_process_name > 0:
                name = name[:max_process_name]

            mem_text = format_number(used_memory, "", precision=0)
            lines.append(f"  {gpu_index:<3} {pid:<8} {mem_text:>8}   {name}")

    render_lines(win, title, lines)
    win.noutrefresh()


def render_gpu_with_processes(
    win: curses.window,
    gpu_data: dict[str, Any],
    process_data: dict[str, Any],
    max_process_name: int,
) -> None:
    """Render GPU summary followed by process telemetry in one panel for narrow terminals."""
    gpu_state = gpu_data.get("state", "N/A")
    gpu_ts = gpu_data.get("timestamp", "--:--:--")
    backend = gpu_data.get("backend", "none")
    gpu_error = gpu_data.get("error")

    title = f"GPU + Memory ({gpu_backend_display_name(backend)}) (Updated: {gpu_ts} | State: {gpu_state})"
    lines: list[str] = []

    ram_line, swap_line = build_ram_swap_lines()
    lines.append(ram_line)
    lines.append(swap_line)
    lines.append("")
    lines.append("GPU")
    lines.append("")

    if gpu_error and gpu_state != "OK":
        lines.append(f"  Note: {gpu_error}")
        lines.append("")

    gpus = gpu_data.get("gpus", [])
    if not gpus:
        lines.append("  No GPU telemetry available")
    else:
        for i, gpu in enumerate(gpus):
            idx = gpu.get("index", i)
            name = gpu.get("name")
            if name:
                lines.append(f"  GPU {idx} ({name})")
            else:
                lines.append(f"  GPU {idx}")
            lines.append(f"    Temp : {format_number(gpu.get('temp_c'), 'C', precision=1):>12}")
            lines.append(f"    Power: {format_number(gpu.get('power_w'), ' W', precision=2):>12}")
            lines.append(f"    Util : {format_number(gpu.get('util_percent'), '%', precision=1):>12}")
            used = format_number(gpu.get("vram_used_mib"), " MiB", precision=0)
            total = format_number(gpu.get("vram_total_mib"), " MiB", precision=0)
            lines.append(f"    VRAM : {used:>8} / {total:>8}")
            lines.append(f"    SCLK : {format_number(gpu.get('sclk_mhz'), ' MHz', precision=0):>12}")
            lines.append(f"    MCLK : {format_number(gpu.get('mclk_mhz'), ' MHz', precision=0):>12}")
            lines.append("")

    process_state = process_data.get("state", "N/A")
    process_ts = process_data.get("timestamp", "--:--:--")
    process_error = process_data.get("error")
    lines.append(
        f"GPU Processes ({gpu_backend_display_name(backend)} | Updated: {process_ts} | State: {process_state})"
    )
    if process_error and process_state != "OK":
        lines.append(f"  Note: {process_error}")
        lines.append("")

    processes = process_data.get("processes", [])
    if not processes:
        lines.append("  No active compute processes")
    else:
        lines.append("  GPU PID      Mem MiB   Process")
        for item in processes:
            gpu_index = item.get("gpu_index", "?")
            pid = item.get("pid", "?")
            used_memory = item.get("used_memory_mib")
            name = str(item.get("name", "unknown"))[:max_process_name]
            mem_text = format_number(used_memory, "", precision=0)
            lines.append(f"  {gpu_index:<3} {pid:<8} {mem_text:>8}   {name}")

    render_lines(win, title, lines)
    win.noutrefresh()


def run_tui(stdscr: curses.window, args: argparse.Namespace) -> None:
    """Main curses loop with decoupled CPU/GPU/process polling and resize-safe layout."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    # Prime CPU percent sampling to avoid first-sample artifacts.
    psutil.cpu_percent(interval=None, percpu=True)

    backend, backend_note = detect_gpu_backend(args.gpu_backend)

    last_good_gpu: dict[str, Any] | None = None
    gpu_data: dict[str, Any] = {
        "gpus": [],
        "timestamp": now_ts(),
        "state": "N/A",
        "error": backend_note or "waiting for first sample",
        "backend": backend,
        "last_success_epoch": 0.0,
    }
    fail_count = 0
    next_gpu_poll = 0.0
    process_fail_count = 0
    next_process_poll = 0.0

    last_good_process: dict[str, Any] | None = None
    process_data: dict[str, Any] = {
        "processes": [],
        "timestamp": now_ts(),
        "state": "N/A",
        "error": "waiting for first sample",
        "backend": backend,
        "last_success_epoch": 0.0,
    }

    show_processes = not args.hide_processes

    panel_height, panel_width = stdscr.getmaxyx()

    def build_windows(height: int, width: int) -> tuple[curses.window, curses.window, curses.window | None, bool]:
        if show_processes and width >= 120:
            cpu_w = max(30, width // 3)
            gpu_w = max(30, width // 3)
            proc_w = max(1, width - cpu_w - gpu_w)
            cpu_local = stdscr.derwin(height, cpu_w, 0, 0)
            gpu_local = stdscr.derwin(height, gpu_w, 0, cpu_w)
            proc_local = stdscr.derwin(height, proc_w, 0, cpu_w + gpu_w)
            return cpu_local, gpu_local, proc_local, True

        split = max(30, width // 2)
        cpu_local = stdscr.derwin(height, split, 0, 0)
        gpu_local = stdscr.derwin(height, max(1, width - split), 0, split)
        return cpu_local, gpu_local, None, False

    cpu_win, gpu_win, proc_win, use_three_panel = build_windows(panel_height, panel_width)

    while True:
        new_h, new_w = stdscr.getmaxyx()
        if (new_h, new_w) != (panel_height, panel_width):
            panel_height, panel_width = new_h, new_w
            cpu_win, gpu_win, proc_win, use_three_panel = build_windows(panel_height, panel_width)

        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            break

        cpu_data = get_cpu_telemetry()

        now_epoch = time.time()
        if now_epoch >= next_gpu_poll:
            gpu_data, fail_count = get_gpu_telemetry(
                last_good=last_good_gpu,
                backend=backend,
                gpu_timeout=args.gpu_timeout,
                stale_seconds=args.stale_seconds,
                max_fails_before_na=args.max_fails_before_na,
                fail_count=fail_count,
            )
            if gpu_data.get("state") == "OK":
                last_good_gpu = gpu_data
            next_gpu_poll = now_epoch + args.gpu_poll_interval

        if show_processes and now_epoch >= next_process_poll:
            process_data, process_fail_count = get_process_telemetry(
                last_good=last_good_process,
                backend=backend,
                process_timeout=args.process_timeout,
                stale_seconds=args.process_stale_seconds,
                fail_count=process_fail_count,
                max_fails_before_na=args.max_fails_before_na,
            )
            if process_data.get("state") == "OK":
                last_good_process = process_data
            next_process_poll = now_epoch + args.process_poll_interval

        draw_cpu(cpu_win, cpu_data)
        if show_processes and use_three_panel and proc_win is not None:
            draw_gpu(gpu_win, gpu_data)
            draw_processes(proc_win, process_data, args.max_process_name)
        elif show_processes:
            render_gpu_with_processes(gpu_win, gpu_data, process_data, args.max_process_name)
        else:
            draw_gpu(gpu_win, gpu_data)
        curses.doupdate()

        time.sleep(args.refresh_interval)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for polling and fail-safe tuning."""
    parser = argparse.ArgumentParser(
        description="Low-noise terminal telemetry dashboard for CPU and GPU telemetry"
    )
    parser.add_argument(
        "--gpu-backend",
        choices=("auto", "nvidia", "rocm"),
        default=DEFAULT_GPU_BACKEND,
        help=(
            "GPU telemetry backend selection "
            "(default: auto; prefer nvidia-smi when available, then rocm-smi)"
        ),
    )
    parser.add_argument(
        "--gpu-timeout",
        type=float,
        default=DEFAULT_GPU_TIMEOUT,
        help=f"GPU telemetry command timeout in seconds (default: {DEFAULT_GPU_TIMEOUT})",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=DEFAULT_REFRESH_INTERVAL,
        help=f"UI refresh interval in seconds (default: {DEFAULT_REFRESH_INTERVAL})",
    )
    parser.add_argument(
        "--gpu-poll-interval",
        type=float,
        default=DEFAULT_GPU_POLL_INTERVAL,
        help=f"GPU poll interval in seconds (default: {DEFAULT_GPU_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--process-poll-interval",
        type=float,
        default=DEFAULT_PROCESS_POLL_INTERVAL,
        help=f"Process telemetry poll interval in seconds (default: {DEFAULT_PROCESS_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--process-timeout",
        type=float,
        default=DEFAULT_PROCESS_TIMEOUT,
        help=f"Process telemetry command timeout in seconds (default: {DEFAULT_PROCESS_TIMEOUT})",
    )
    parser.add_argument(
        "--process-stale-seconds",
        type=float,
        default=DEFAULT_PROCESS_STALE_SECONDS,
        help=f"Seconds before process STALE escalates to N/A (default: {DEFAULT_PROCESS_STALE_SECONDS})",
    )
    parser.add_argument(
        "--max-process-name",
        type=int,
        default=DEFAULT_MAX_PROCESS_NAME,
        help=f"Maximum displayed process-name width (default: {DEFAULT_MAX_PROCESS_NAME})",
    )
    parser.add_argument(
        "--hide-processes",
        action="store_true",
        help="Hide process telemetry panel and keep two-panel CPU/GPU layout",
    )
    parser.add_argument(
        "--rocm-timeout",
        type=float,
        default=None,
        help="Deprecated alias for --gpu-timeout",
    )
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=DEFAULT_STALE_SECONDS,
        help=f"Seconds before STALE escalates to N/A (default: {DEFAULT_STALE_SECONDS})",
    )
    parser.add_argument(
        "--max-fails-before-na",
        type=int,
        default=DEFAULT_MAX_FAILS_BEFORE_NA,
        help=(
            "Consecutive telemetry failures before state escalates to N/A "
            f"(default: {DEFAULT_MAX_FAILS_BEFORE_NA})"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate runtime argument constraints for predictable behavior."""
    if args.rocm_timeout is not None:
        args.gpu_timeout = args.rocm_timeout

    if args.refresh_interval <= 0:
        raise ValueError("--refresh-interval must be > 0")
    if args.gpu_poll_interval <= 0:
        raise ValueError("--gpu-poll-interval must be > 0")
    if args.process_poll_interval <= 0:
        raise ValueError("--process-poll-interval must be > 0")
    if args.gpu_timeout <= 0:
        raise ValueError("--gpu-timeout must be > 0")
    if args.process_timeout <= 0:
        raise ValueError("--process-timeout must be > 0")
    if args.stale_seconds <= 0:
        raise ValueError("--stale-seconds must be > 0")
    if args.process_stale_seconds <= 0:
        raise ValueError("--process-stale-seconds must be > 0")
    if args.max_fails_before_na <= 0:
        raise ValueError("--max-fails-before-na must be > 0")
    if args.max_process_name <= 0:
        raise ValueError("--max-process-name must be > 0")


def main() -> int:
    """Program entrypoint."""
    try:
        args = parse_args()
        validate_args(args)
        curses.wrapper(run_tui, args)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"fatal: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
