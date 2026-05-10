# Arrest Analyzer — Phase 1: Live Dual-Channel Transcript

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Find your audio device indices
python -c "from core.audio.capture import list_devices; list_devices()"

# 3. Set device indices in config.py (if not using UI dropdowns)
#    MIC_DEVICE_INDEX    = <your mic index>
#    REMOTE_DEVICE_INDEX = <VB-Cable index>

# 4. Run
python main.py
```

---

## VB-Cable Setup (Windows)

1. Download & install VB-Cable from https://vb-audio.com/Cable/
2. In **Zoom → Settings → Audio → Speaker** → select `CABLE Input (VB-Audio)`
3. In the app's **REMOTE** dropdown → select `CABLE Output (VB-Audio Virtual Cable)`
4. Your mic stays as your normal input device

> VB-Cable routes Zoom's audio output → becomes an input device → app reads it as "REMOTE"

---

## VB-Cable Setup (macOS)

Use **BlackHole** (free) instead of VB-Cable:
- https://existential.audio/blackhole/
- Same logic: set BlackHole as Zoom speaker → select it as REMOTE device in app

---

## UI Layout

```
┌─────────────────────────────────────────────────────┐
│  ◈ DIGITAL ARREST ANALYZER // PHASE-1 // TRANSCRIPT │
│                          [READY] [CLEAR] [▶ START]  │
├────────────────────────┬────────────────────────────┤
│  ◈ LOCAL // YOU        │  ◈ REMOTE // VB-CABLE      │
│                        │                            │
│  [10:24:01] Hello,     │  [10:24:05] We are calling │
│  my name is...         │  from the cybercrime...    │
│                        │                            │
├────────────────────────┴────────────────────────────┤
│  MIC: [Default ▾]   REMOTE: [CABLE Output ▾]  00:02:14 │
└─────────────────────────────────────────────────────┘
```

---

## How Utterance Capture Works

```
Audio frames (30ms each)
        │
        ▼
  WebRTC VAD ──► voiced? ──► buffer frames
                          │
                          ▼ (silence > 1.2s)
                    flush buffer → Whisper → transcript line
```

No 5-second chunks. Each complete sentence/utterance is sent to Whisper whole.

---

## config.py Tuning

| Key | Default | Notes |
|-----|---------|-------|
| `SILENCE_TIMEOUT_S` | 1.2 | Increase if sentences get cut off |
| `VAD_MODE` | 3 | Lower (0–2) if missing quiet speech |
| `WHISPER_MODEL` | `base` | Use `small` for better accuracy |
| `MIN_SPEECH_S` | 0.4 | Raise to filter short noise bursts |

---

## Phase Roadmap

- **Phase 1** ← You are here: Live dual-channel transcript
- **Phase 2**: CV emotion detection (webcam)
- **Phase 3**: Voice fluctuation / stress analysis
- **Phase 4**: Full UI fusion + alerts
- **Phase 5**: Live Zoom test
