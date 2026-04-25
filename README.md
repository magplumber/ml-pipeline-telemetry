# ml-pipeline-telemetry

Deterministic, low-noise terminal telemetry for ML workloads.

## Included

- CPU telemetry (per-core usage, frequency, temperature when available)
- GPU telemetry via `nvidia-smi` and `rocm-smi` backends
- RAM and swap telemetry
- GPU process ownership panel (NVIDIA backend)

## Files

- `scripts/telemetry_tui.py` — main TUI app
- `scripts/run_telemetry_tui.sh` — launcher wrapper
- `TELEMETRY_TUI_USAGE.md` — full usage guide
- `tests/test_telemetry_tui.py` — unit tests
- `requirements.txt` — Python dependency list
- `LICENSE` — Apache-2.0

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_telemetry_tui.sh
```

## License

Apache-2.0
