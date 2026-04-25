#!/usr/bin/env python3
"""
Unit tests for scripts.telemetry_tui.

Purpose:
- Validate parser and helper behavior for the low-noise telemetry TUI.

Operational boundary:
- Tests focus on deterministic pure functions where practical.
- No curses rendering loop execution.
- No live subprocess telemetry calls.

Coverage targets:
- parse_nvidia_smi_output
- parse_rocm_smi_output
- parse_nvidia_gpu_uuid_map
- parse_nvidia_compute_apps_output
- detect_gpu_backend
- gpu_backend_tool_name
- gpu_backend_display_name
- format_number
- timeout alias handling in validate_args
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import telemetry_tui


class TestTelemetryTui(unittest.TestCase):
    """Focused tests for telemetry parsing and fail-safe helpers."""

    def test_parse_nvidia_smi_output(self) -> None:
        sample = """
0, NVIDIA GeForce RTX 5080, 99, 15552, 16303, 67, 211.93, 2850, 14801
1, NVIDIA GeForce RTX 5080, 13, 2048, 16303, 49, 55.10, 900, 405
""".strip()
        parsed = telemetry_tui.parse_nvidia_smi_output(sample)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["index"], 0)
        self.assertEqual(parsed[0]["name"], "NVIDIA GeForce RTX 5080")
        self.assertAlmostEqual(parsed[0]["util_percent"], 99.0)
        self.assertAlmostEqual(parsed[0]["vram_used_mib"], 15552.0)
        self.assertAlmostEqual(parsed[0]["vram_total_mib"], 16303.0)
        self.assertAlmostEqual(parsed[0]["power_w"], 211.93)
        self.assertAlmostEqual(parsed[1]["mclk_mhz"], 405.0)

    def test_parse_nvidia_gpu_uuid_map(self) -> None:
        sample = """
0, GPU-aaa
1, GPU-bbb
""".strip()
        parsed = telemetry_tui.parse_nvidia_gpu_uuid_map(sample)
        self.assertEqual(parsed, {"GPU-aaa": 0, "GPU-bbb": 1})

    def test_parse_nvidia_compute_apps_output(self) -> None:
        sample = """
GPU-aaa, 1958, VLLM::Worker_TP, 15762
GPU-bbb, 1959, python, 8120
""".strip()
        uuid_map = {"GPU-aaa": 0, "GPU-bbb": 1}
        parsed = telemetry_tui.parse_nvidia_compute_apps_output(sample, uuid_map)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["gpu_index"], 0)
        self.assertEqual(parsed[0]["pid"], 1958)
        self.assertEqual(parsed[0]["name"], "VLLM::Worker_TP")
        self.assertAlmostEqual(parsed[0]["used_memory_mib"], 15762.0)
        self.assertEqual(parsed[1]["gpu_index"], 1)

    def test_parse_nvidia_compute_apps_output_ignores_malformed_rows(self) -> None:
        sample = """
GPU-aaa, bad_pid, python, 123
GPU-missing, 1000, python, 456
totally malformed row
GPU-aaa, 2000, valid_proc, 1024
""".strip()
        uuid_map = {"GPU-aaa": 0}
        parsed = telemetry_tui.parse_nvidia_compute_apps_output(sample, uuid_map)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["gpu_index"], 0)
        self.assertEqual(parsed[0]["pid"], 2000)

    def test_parse_rocm_smi_output(self) -> None:
        sample = """
GPU[0]          : GPU use (%) 0
GPU[0]          : Temperature (Sensor edge) (C) 47.0
GPU[0]          : Average Graphics Package Power (W) 15.07
GPU[0]          : VRAM Total Memory (B) 17095028736
GPU[0]          : VRAM Total Used Memory (B) 2097152
GPU[0]          : sclk clock level: 500Mhz
GPU[0]          : mclk clock level: 96Mhz
GPU[1]          : GPU use (%) 12
GPU[1]          : Temperature (Sensor edge) (C) 45.0
GPU[1]          : Average Graphics Package Power (W) 14.55
""".strip()
        parsed = telemetry_tui.parse_rocm_smi_output(sample)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["index"], 0)
        self.assertAlmostEqual(parsed[0]["util_percent"], 0.0)
        self.assertAlmostEqual(parsed[0]["temp_c"], 47.0)
        self.assertAlmostEqual(parsed[0]["power_w"], 15.07)
        self.assertAlmostEqual(parsed[0]["vram_used_mib"], 2.0)
        self.assertGreater(parsed[0]["vram_total_mib"], 16000.0)
        self.assertAlmostEqual(parsed[0]["sclk_mhz"], 500.0)
        self.assertAlmostEqual(parsed[0]["mclk_mhz"], 96.0)

    def test_detect_gpu_backend_auto_prefers_nvidia(self) -> None:
        with patch("scripts.telemetry_tui.shutil.which") as which_mock:
            which_mock.side_effect = lambda tool: "/usr/bin/tool" if tool in ("nvidia-smi", "rocm-smi") else None
            backend, note = telemetry_tui.detect_gpu_backend("auto")

        self.assertEqual(backend, "nvidia")
        self.assertIsNone(note)

    def test_detect_gpu_backend_none_when_missing_tools(self) -> None:
        with patch("scripts.telemetry_tui.shutil.which", return_value=None):
            backend, note = telemetry_tui.detect_gpu_backend("auto")

        self.assertEqual(backend, "none")
        self.assertIn("no supported GPU telemetry tool found", note or "")

    def test_gpu_backend_tool_name(self) -> None:
        self.assertEqual(telemetry_tui.gpu_backend_tool_name("nvidia"), "nvidia-smi")
        self.assertEqual(telemetry_tui.gpu_backend_tool_name("rocm"), "rocm-smi")
        self.assertEqual(telemetry_tui.gpu_backend_tool_name("none"), "unknown")

    def test_gpu_backend_display_name(self) -> None:
        self.assertEqual(telemetry_tui.gpu_backend_display_name("nvidia"), "nvidia-smi")
        self.assertEqual(telemetry_tui.gpu_backend_display_name("rocm"), "rocm-smi")
        self.assertEqual(telemetry_tui.gpu_backend_display_name("none"), "none")

    def test_format_number_handles_none(self) -> None:
        self.assertEqual(telemetry_tui.format_number(None), "N/A")
        self.assertEqual(telemetry_tui.format_number(12.345, " W", precision=2), "12.35 W")

    def test_format_gib(self) -> None:
        self.assertEqual(telemetry_tui.format_gib(78.234), "78.2")

    def test_format_percent(self) -> None:
        self.assertEqual(telemetry_tui.format_percent(40.678), "40.7%")

    def test_get_ram_telemetry_keys_and_values(self) -> None:
        with patch("scripts.telemetry_tui.psutil.virtual_memory") as vm_mock, patch(
            "scripts.telemetry_tui.psutil.swap_memory"
        ) as sm_mock:
            vm_mock.return_value = SimpleNamespace(
                used=80 * (1024**3),
                total=192 * (1024**3),
                percent=41.7,
            )
            sm_mock.return_value = SimpleNamespace(
                used=1 * (1024**3),
                total=5 * (1024**3),
            )

            ram = telemetry_tui.get_ram_telemetry()

        required_keys = {
            "ram_used_gib",
            "ram_total_gib",
            "ram_percent",
            "swap_used_gib",
            "swap_total_gib",
        }
        self.assertTrue(required_keys.issubset(ram.keys()))
        for key in required_keys:
            self.assertIsInstance(ram[key], float)
            self.assertGreaterEqual(ram[key], 0.0)

    def test_build_ram_swap_lines_adds_pressure_marker_above_threshold(self) -> None:
        with patch("scripts.telemetry_tui.get_ram_telemetry", return_value={
            "ram_used_gib": 160.0,
            "ram_total_gib": 188.2,
            "ram_percent": 85.1,
            "swap_used_gib": 0.0,
            "swap_total_gib": 8.0,
        }):
            ram_line, _ = telemetry_tui.build_ram_swap_lines()

        self.assertIn("*", ram_line)

    def test_build_ram_swap_lines_no_pressure_marker_at_threshold(self) -> None:
        with patch("scripts.telemetry_tui.get_ram_telemetry", return_value={
            "ram_used_gib": 159.9,
            "ram_total_gib": 188.2,
            "ram_percent": 85.0,
            "swap_used_gib": 0.0,
            "swap_total_gib": 8.0,
        }):
            ram_line, _ = telemetry_tui.build_ram_swap_lines()

        self.assertNotIn("*", ram_line)

    def test_validate_args_deprecated_rocm_timeout_alias(self) -> None:
        args = argparse.Namespace(
            gpu_backend="auto",
            gpu_timeout=0.8,
            rocm_timeout=1.2,
            refresh_interval=1.0,
            gpu_poll_interval=2.0,
            process_poll_interval=2.0,
            process_timeout=0.8,
            stale_seconds=30.0,
            process_stale_seconds=30.0,
            max_fails_before_na=10,
            max_process_name=32,
            hide_processes=False,
        )

        telemetry_tui.validate_args(args)
        self.assertEqual(args.gpu_timeout, 1.2)

    def test_validate_args_rejects_invalid_process_timeout(self) -> None:
        args = argparse.Namespace(
            gpu_backend="auto",
            gpu_timeout=0.8,
            rocm_timeout=None,
            refresh_interval=1.0,
            gpu_poll_interval=2.0,
            process_poll_interval=2.0,
            process_timeout=0.0,
            stale_seconds=30.0,
            process_stale_seconds=30.0,
            max_fails_before_na=10,
            max_process_name=32,
            hide_processes=False,
        )

        with self.assertRaisesRegex(ValueError, "--process-timeout must be > 0"):
            telemetry_tui.validate_args(args)


if __name__ == "__main__":
    unittest.main()
