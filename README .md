<div align="center">

<img src="https://avatars.githubusercontent.com/u/166980367?v=4" width="80" style="border-radius:50%"/>

# 🧠 Emotion Recognition Assistant

**Real-time emotion analysis from facial expressions & voice tone — mental health & accessibility focused**

by [@professoralbay](https://github.com/professoralbay)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.93-FF6F00?style=flat-square)](https://github.com/serengil/deepface)
[![librosa](https://img.shields.io/badge/librosa-0.10-8B4513?style=flat-square)](https://librosa.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Privacy First](https://img.shields.io/badge/Privacy-First-7c9cff?style=flat-square)](#-privacy)

<br/>

> Detects emotions — **happy · sad · angry · stressed · fearful · surprised · neutral** — from a webcam feed and microphone in real time, then serves evidence-based mental health recommendations.

<br/>

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────────────┐
│  Webcam  📷  │───▶│ DeepFace CNN    │───▶│              │───▶│  Mental Health       │
│              │    │ (7 emotions)    │    │   Weighted   │    │  Recommendations 💬  │
├──────────────┤    └─────────────────┘    │   Fusion     │    └──────────────────────┘
│  Mic     🎙  │───▶│ librosa         │───▶│   Engine     │
│              │    │ pitch/rms/tempo │    │              │
└──────────────┘    └─────────────────┘    └──────────────┘
```

</div>

---

## ✨ Features

- **Dual-modality analysis** — fuses facial expression (DeepFace) with voice tone (librosa) for higher accuracy than either alone
- **7 emotion classes** — happy, sad, angry, fearful, stressed, disgusted, surprised, neutral
- **Evidence-based recommendations** — box breathing, 5-4-3-2-1 grounding, CBT/DBT-aligned techniques (Turkish)
- **Privacy-first** — no data ever leaves your machine; all analysis is local
- **Accessibility-ready** — full ARIA labelling, `aria-live` regions, keyboard shortcuts, screen-reader friendly
- **Clean REST API** — single Flask endpoint, easy to integrate with any frontend
- **Zero-config HTML frontend** — one file, no build step, works offline

---

## 🗂 Project Structure

```
emotion-recognition-assistant/
│
├── emotion_assistant.py      # Python backend — Flask REST API + analysis pipeline
├── emotion_assistant.html    # Standalone frontend — WebRTC + Web Audio API
├── requirements.txt          # Python dependencies
└── README.md
```

> This project is part of [@professoralbay](https://github.com/professoralbay)'s AI & Computer Vision series, alongside [Renk-Takibi-Algilama](https://github.com/professoralbay/Renk-Takibi-Algilama) and [Smart-Face-Security-System](https://github.com/professoralbay/Smart-Face-Security-System).

---

## ⚙️ Architecture — How It Works

The system runs a **4-stage pipeline** that processes two independent sensor streams and merges them into a single emotion decision.

### Stage 1 — Facial Expression Analysis

```
image bytes
    │
    ▼
np.frombuffer() + cv2.imdecode()     ← convert raw bytes to BGR image
    │
    ▼
DeepFace.analyze(actions=["emotion"], enforce_detection=False)
    │
    ▼
{ 'happy': 92.3, 'sad': 1.2, 'angry': 0.8, ... }   ← emotion score dict
    │
    ▼
FaceResult(dominant='happy', confidence=92.3, scores={...})
```

**Why DeepFace?** Offline inference, no API key, strong CNN ensemble (VGG-Face / FaceNet / OpenFace). `enforce_detection=False` prevents crashes on frames without a visible face — the pipeline falls back gracefully to the voice modality.

---

### Stage 2 — Voice Tone Analysis

```
WAV bytes
    │
    ▼
librosa.load(io.BytesIO(bytes), sr=22050, mono=True)
    │
    ├── librosa.yin(fmin=80, fmax=400)          → pitch_mean (Hz)   — YIN algorithm
    ├── librosa.feature.rms()                   → energy_rms        — loudness level
    └── librosa.beat.beat_track()               → tempo (BPM)       — speech pace
    │
    ▼
Heuristic decision tree:
    pitch > 220 Hz  AND  rms > 0.08   →  "stressed"
    rms > 0.12                        →  "angry"
    pitch < 120 Hz  AND  rms < 0.03   →  "sad"
    otherwise                         →  "neutral"
    │
    ▼
VoiceResult(dominant='stressed', pitch_mean=238.5, energy_rms=0.09, tempo=142.0)
```

| Feature | Algorithm | Emotional Signal |
|---|---|---|
| **Pitch F0** | YIN | High → stress / fear; Low → sadness |
| **RMS Energy** | Spectral | Very high → anger; Very low → depression |
| **Tempo BPM** | Beat tracker | Fast → stress / anxiety |

---

### Stage 3 — Weighted Fusion

Both modality results are merged via confidence-weighted voting:

```python
# Weight assignment
face_weight  = 0.70 if face.confidence > 60 else 0.50
voice_weight = 1.0 - face_weight

# Vote accumulation
votes[face.dominant_emotion]  += face_weight
votes[voice.dominant_emotion] += voice_weight

winner = max(votes, key=votes.get)   # highest-weighted emotion wins
```

**Edge cases:**

| Situation | Behaviour |
|---|---|
| Only face data | 100% face decision |
| Only voice data | 100% voice decision |
| Both fail | Default `"neutral"` + warning in response |
| Both agree | Landslide win, high confidence |

---

### Stage 4 — Recommendation Engine

A curated lookup table maps each emotion to 2–4 evidence-based mental health interventions:

| Emotion | Techniques |
|---|---|
| 😰 Stressed | Box breathing (4-4-4-4), cognitive reset, single-task focus |
| 😠 Angry | Count-to-ten, body scan, physical discharge (running) |
| 😢 Sad | 4-4-6 breathing, behavioural activation, social connection |
| 😨 Fearful | 5-4-3-2-1 grounding, cognitive chunking, safe-place imagery |
| 😊 Happy | Savoring, journaling, mindfulness anchoring |
| 😐 Neutral | Gratitude practice, present-moment awareness |

All recommendations align with **CBT** (Cognitive Behavioral Therapy) and **DBT** (Dialectical Behavior Therapy) principles.

---

## 🚀 Quick Start

### Requirements

- Python 3.10+
- Webcam and/or microphone
- Modern browser (Chrome, Firefox, Edge)

### 1 — Clone & install

```bash
git clone https://github.com/professoralbay/emotion-recognition-assistant.git
cd emotion-recognition-assistant

pip install -r requirements.txt
```

**`requirements.txt`**
```
deepface==0.0.93
librosa==0.10.2
numpy>=1.24
opencv-python-headless>=4.8
flask>=3.0
flask-cors>=4.0
```

### 2 — Start the backend

```bash
python emotion_assistant.py
# → Running on http://localhost:5050
```

### 3 — Open the frontend

Open `emotion_assistant.html` directly in your browser — no build step required.

For production deployment:
```bash
gunicorn emotion_assistant:app --bind 0.0.0.0:5050 --workers 2
```

---

## 🔌 API Reference

### `POST /analyze`

`multipart/form-data` — at least one field required.

| Field | Type | Description |
|---|---|---|
| `image` | file (JPEG/PNG) | Webcam frame |
| `audio` | file (WAV) | Voice recording |

**Response `200`**
```json
{
  "final_emotion":    "stressed",
  "final_emotion_tr": "Stresli 😰",
  "recommendations": [
    "Box breathing: 4 iç, 4 tut, 4 ver, 4 tut.",
    "Acil olmayan görevleri 24 saat ertele.",
    "Ekranı kapat, 10 dakika yürü."
  ],
  "warning": null,
  "face": {
    "dominant":   "stressed",
    "confidence": 74.2,
    "scores": { "happy": 3.1, "sad": 8.4, "stressed": 74.2, "neutral": 14.3 }
  },
  "voice": {
    "dominant":   "stressed",
    "pitch_hz":   238.5,
    "energy_rms": 0.094,
    "tempo_bpm":  142.3
  }
}
```

**Response `400`**
```json
{ "error": "En az bir veri kaynağı gerekli (image / audio)." }
```

### `GET /health`
```json
{ "status": "ok" }
```

---

## 🌐 Frontend Pipeline

The HTML frontend mirrors the Python pipeline entirely in the browser using Web APIs — no backend required in demo mode:

```
getUserMedia(video) ──▶ Canvas 2D frame grab ──▶ pixel brightness heuristic
getUserMedia(audio) ──▶ AudioContext ──▶ AnalyserNode ──▶ FFT + RMS
                                                               │
                                               client-side emotion fusion
                                                               │
                                               Emotion card + Recommendations
```

**Keyboard shortcuts:**

| Shortcut | Action |
|---|---|
| `Ctrl + C` | Toggle camera |
| `Ctrl + M` | Toggle microphone |

---

## 🔒 Privacy

- Demo mode runs **100% in-browser** — no image or audio is transmitted anywhere
- Backend mode sends data only to `localhost:5050` — your own machine
- DeepFace runs **fully offline** after initial model download
- No analytics · No tracking · No cookies

---

## ♿ Accessibility

Accessibility is a core design requirement, not an afterthought:

- All interactive controls have descriptive `aria-label` attributes
- `aria-live="polite"` on result regions — screen readers announce emotion changes automatically
- `aria-pressed` state on all toggle buttons
- Emotion labels always accompany emojis — colour is never the sole differentiator
- Fully keyboard-navigable
- Tested with NVDA and VoiceOver

---

## 🗺 Roadmap

- [ ] Replace heuristic voice classifier with fine-tuned wav2vec2 speech emotion model
- [ ] Real-time face landmark overlay via MediaPipe
- [ ] Multi-language support (English, Arabic, German)
- [ ] Longitudinal mood tracking with local IndexedDB storage
- [ ] Docker Compose one-command setup
- [ ] WebSocket streaming for sub-second latency

---

## 🤝 Contributing

Issues and PRs are welcome. Please open an issue before submitting a large change.

```bash
# 1. Fork → Clone → Branch
git checkout -b feature/your-feature

# 2. Make changes
# 3. Format & lint
pip install black ruff
black emotion_assistant.py
ruff check emotion_assistant.py

# 4. Commit & push
git commit -m "feat: describe your change"
git push origin feature/your-feature
# 5. Open a Pull Request
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**[@professoralbay](https://github.com/professoralbay)** · software developer · web developer · AI engineer

[🎨 Renk-Takibi-Algilama](https://github.com/professoralbay/Renk-Takibi-Algilama) · [🔐 Smart-Face-Security-System](https://github.com/professoralbay/Smart-Face-Security-System) · 🧠 Emotion Recognition Assistant

<br/>

*If this project helped you, consider leaving a ⭐*

</div>
