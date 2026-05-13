# Arrest Analyzer — Phase 1: Live Dual-Channel Transcript

## Quick Start

```bash
# 1. Install system dependency first (Whisper needs this)
#    Windows: winget install ffmpeg
#    macOS:   brew install ffmpeg
#    Linux:   sudo apt install ffmpeg

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. List your audio device indices
python -c "from core.audio.capture import list_devices; list_devices()"

# 4. Run — device selection is done in the UI dropdowns
python main.py
```

---

## VB-Cable Setup (Windows)

1. Download & install VB-Cable: https://vb-audio.com/Cable/
2. In **Zoom → Settings → Audio → Speaker** → select `CABLE Input (VB-Audio)`
3. In the app's **REMOTE** dropdown → select `CABLE Output (VB-Audio Virtual Cable)`
4. Your mic stays as your normal input device

## VB-Cable Setup (macOS)

Use **BlackHole** instead: https://existential.audio/blackhole/
Same logic: set BlackHole as Zoom speaker → select it as REMOTE in the app.

---

## Bugs Fixed (vs original codebase)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `ui/` | Missing `__init__.py` — Python couldn't find the `ui` package | Added `ui/__init__.py` |
| 2 | `main_window.py` | `QLabel("MIC:").setStyleSheet(...)` — `setStyleSheet` returns `None`, so the `or` fallback always ran and the real label was discarded | Replaced with `self._small_label("MIC:")` directly |
| 3 | `main_window.py` | `_on_model_ready` connected to both `loader.finished` AND `signals.model_ready` — fired twice | Removed second connection; now only `loader.finished` triggers it |
| 4 | `main_window.py` | `status_update` signal was `pyqtSignal(str)` — color was lost when emitting from audio thread | Changed to `pyqtSignal(str, str)` — color now travels with the message |
| 5 | `config.py` | Hardcoded device indices `9` and `17` would crash on any other machine | Changed defaults to `None` (system default) |
| 6 | `core/` | Missing `core/__init__.py` | Added it |

---

## Phase Roadmap

- **Phase 1** ← You are here: Live dual-channel transcript
- **Phase 2**: CV emotion detection (webcam — OpenCV + DeepFace)
- **Phase 3**: Voice stress / fluctuation analysis
- **Phase 4**: Full UI fusion + real-time alerts
- **Phase 5**: Live Zoom test
