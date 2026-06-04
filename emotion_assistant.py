"""
Emotion Recognition Assistant
================================
Kullanıcının ses tonu ve yüz ifadesinden duygu analizi yapan,
mental sağlık odaklı öneri sunan modül.

Mimari (adım adım):
  1. Yüz ifadesi  → DeepFace (offline, gizlilik dostu)
  2. Ses tonu      → librosa spektral analiz (pitch, energy, tempo)
  3. Füzyon        → Ağırlıklı oylama → nihai duygu etiketi
  4. Öneri motoru → Duyguya özel Türkçe rehberlik

Kurulum:
  pip install deepface librosa numpy opencv-python-headless flask flask-cors
"""

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import librosa
import numpy as np
from deepface import DeepFace
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Loglama ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("emotion_assistant")

# ── Sabitler ─────────────────────────────────────────────────────────────────
# Ses analizinde stres tespiti için eşik değerleri (heuristik kalibre)
STRESS_PITCH_THRESHOLD  = 220   # Hz — bu değerin üstü yüksek pitch (stres/korku)
STRESS_ENERGY_THRESHOLD = 0.08  # normalize RMS — yüksek ses enerjisi (öfke/stres)

# DeepFace → Türkçe duygu etiketleri
EMOTION_TR: dict[str, str] = {
    "happy":    "Mutlu 😊",
    "sad":      "Üzgün 😢",
    "angry":    "Kızgın 😠",
    "fear":     "Korkmuş 😨",
    "disgust":  "İğrenmiş 😖",
    "surprise": "Şaşırmış 😮",
    "neutral":  "Nötr 😐",
    "stressed": "Stresli 😰",  # ses analizinden türetilen ek etiket
}

# Duygu → mental sağlık odaklı Türkçe öneriler
RECOMMENDATIONS: dict[str, list[str]] = {
    "happy": [
        "Enerjini korumak için bugün sevdiğin biriyle zaman geçir.",
        "Bu mutluluğu bir günlüğe yaz; zor günlerde sana güç verir.",
        "Mindfulness pratiği bu anı pekiştirir — 5 dk nefese odaklan.",
    ],
    "sad": [
        "Kendine karşı nazik ol; üzüntü geçici bir misafirdir.",
        "4-4-6 nefes: 4 sayı iç, 4 tut, 6 sayı ver. 5 kez tekrarla.",
        "Güvendiğin biriyle konuşmayı dene ya da bir uzmanla görüş.",
        "Hafif bir yürüyüş serotonin seviyeni artırabilir.",
    ],
    "angry": [
        "Tepki vermeden önce 10'a kadar say.",
        "Omuzlarını bırak, çeneni gevşet — bedensel gerginliği fark et.",
        "Duyguyu kağıda yaz, sonra yırt — sembolik serbest bırakma.",
        "Yoğun egzersiz (koşu, yürüyüş) öfkeyi güvenli boşaltır.",
    ],
    "fear": [
        "5-4-3-2-1: 5 gördüğün, 4 dokunduğun, 3 duyduğun şeyi say.",
        "Korkuyu küçük adımlara böl; hepsine aynı anda bakmak zorunda değilsin.",
        "Güvende olduğun bir mekânı ya da kişiyi düşün.",
    ],
    "stressed": [
        "Box breathing: 4 iç, 4 tut, 4 ver, 4 tut — 4 tur yap.",
        "Acil olmayan görevleri 24 saat ertelemek kabul edilebilir.",
        "Ekranı kapat, 10 dakika yürü — beyin sıfırlanır.",
        "Bir öncelik listesi yap; sıradaki tek adımı seç.",
    ],
    "disgust": [
        "Bu duyguyu tetikleyen kaynaktan geçici uzaklaş.",
        "Soğuk suyla yüzünü yıka — fiziksel sıfırlama etkisi yapar.",
    ],
    "surprise": [
        "Nefes al, bilgiyi sindir, sonra tepki ver.",
        "Her sürpriz bir bilgi fırsatıdır — merakla yaklaş.",
    ],
    "neutral": [
        "Bu an için iyi hissediyorsun — varlığının tadını çıkar.",
        "Küçük bir şükran pratiği: bugün 3 güzel şeyi düşün.",
    ],
}


# ── Veri sınıfları ────────────────────────────────────────────────────────────
@dataclass
class FaceResult:
    """DeepFace analiz çıktısını taşıyan değer nesnesi."""
    dominant_emotion: str
    scores: dict[str, float]   # {'happy': 92.3, 'sad': 1.2, ...}
    confidence: float           # dominant duygunun yüzdelik skoru


@dataclass
class VoiceResult:
    """Ses tonu analiz çıktısını taşıyan değer nesnesi."""
    dominant_emotion: str
    pitch_mean: float   # ortalama temel frekans (Hz)
    energy_rms: float   # normalize RMS enerji (0-1 arası)
    tempo: float        # tahmini konuşma temposu (BPM)


@dataclass
class EmotionReport:
    """Tam analiz raporu — füzyon sonucu + öneriler."""
    final_emotion: str
    final_emotion_tr: str
    face: Optional[FaceResult] = None
    voice: Optional[VoiceResult] = None
    recommendations: list[str] = field(default_factory=list)
    warning: Optional[str] = None  # kısmi analiz başarısızsa mesaj


# ── 1. Yüz ifadesi analizi ───────────────────────────────────────────────────
def analyze_face(image_bytes: bytes) -> FaceResult:
    """
    Ham görüntü baytlarından yüz ifadesi analizi yapar.

    Adımlar:
      a) Bayt dizisini NumPy dizisine çevir
      b) OpenCV ile BGR görüntüyü çöz
      c) DeepFace ile duygu skor dağılımını al
      d) Dominant duyguyu ve güven skorunu döndür

    Args:
        image_bytes: JPEG veya PNG ham bayt verisi.

    Returns:
        FaceResult — dominant duygu ve tam skor sözlüğü.

    Raises:
        ValueError: Görüntü çözümlenemezse.
    """
    # a) Bayt → NumPy dizisi → BGR görüntü
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Görüntü çözümlenemedi — desteklenen format: JPEG, PNG.")

    # b) DeepFace analizi (enforce_detection=False → yüz bulunamazsa çökmez)
    result = DeepFace.analyze(
        img_path=img,
        actions=["emotion"],
        enforce_detection=False,
        silent=True,
    )

    # c) Sonuç bazen liste içinde gelir; ilk öğeyi güvenle al
    data = result[0] if isinstance(result, list) else result
    scores: dict[str, float] = data["emotion"]
    dominant: str = data["dominant_emotion"]
    confidence: float = scores.get(dominant, 0.0)

    log.info("Yüz analizi → %s (%.1f%%)", dominant, confidence)
    return FaceResult(dominant_emotion=dominant, scores=scores, confidence=confidence)


# ── 2. Ses tonu analizi ───────────────────────────────────────────────────────
def analyze_voice(audio_bytes: bytes, sr: int = 22050) -> VoiceResult:
    """
    Ham WAV baytlarından ses tonu özelliklerini çıkarır.

    Özellikler ve yorumu:
      - Pitch (YIN)  → yüksek pitch + yüksek enerji = stres/korku
      - RMS enerji   → çok yüksek = öfke; çok düşük + düşük pitch = üzgün
      - Tempo (BPM)  → hızlı konuşma temposu → stres göstergesi

    Args:
        audio_bytes: WAV formatında ham ses verisi.
        sr: Örnekleme hızı — 22050 Hz varsayılan.

    Returns:
        VoiceResult — heuristik duygu etiketi ve ham özellikler.
    """
    # a) Bayt akışını librosa'nın anlayacağı formata çevir
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)

    # b) Temel frekans (pitch) — YIN algoritması, konuşma aralığı
    f0 = librosa.yin(audio, fmin=80, fmax=400)
    valid_f0 = f0[f0 > 0]                               # sessizlik kareleri dışla
    pitch_mean = float(np.nanmean(valid_f0)) if len(valid_f0) else 0.0

    # c) Normalize RMS enerji (ses yüksekliği göstergesi)
    rms = librosa.feature.rms(y=audio)
    energy_rms = float(np.mean(rms))

    # d) Tempo — beat tracker; konuşma hızına yaklaşım
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = float(tempo)

    # e) Heuristik duygu etiketleme (eşik tabanlı karar ağacı)
    if pitch_mean > STRESS_PITCH_THRESHOLD and energy_rms > STRESS_ENERGY_THRESHOLD:
        emotion = "stressed"                             # yüksek pitch + yüksek enerji
    elif energy_rms > STRESS_ENERGY_THRESHOLD * 1.5:
        emotion = "angry"                                # aşırı yüksek enerji → öfke
    elif pitch_mean < 120 and energy_rms < 0.03:
        emotion = "sad"                                  # düşük pitch + sessiz → üzgün
    else:
        emotion = "neutral"                              # diğer durumlar

    log.info(
        "Ses analizi → %s | pitch=%.1f Hz | rms=%.4f | tempo=%.1f BPM",
        emotion, pitch_mean, energy_rms, tempo,
    )
    return VoiceResult(
        dominant_emotion=emotion,
        pitch_mean=pitch_mean,
        energy_rms=energy_rms,
        tempo=tempo,
    )


# ── 3. Füzyon: yüz + ses → nihai duygu ──────────────────────────────────────
def fuse_emotions(
    face: Optional[FaceResult],
    voice: Optional[VoiceResult],
) -> str:
    """
    İki kaynağı ağırlıklı oylama ile birleştirir.

    Ağırlık mantığı:
      - Yalnızca yüz varsa → %100 yüz kararı
      - Yalnızca ses varsa → %100 ses kararı
      - Her ikisi de varsa:
          * Yüz güveni > %60 → yüz:%70 / ses:%30
          * Yüz güveni ≤ %60 → yüz:%50 / ses:%50

    Args:
        face:  Yüz analiz sonucu (None olabilir).
        voice: Ses analiz sonucu (None olabilir).

    Returns:
        Kazanan duygu etiketi (str).
    """
    # Tek kaynak durumları
    if face is None and voice is None:
        return "neutral"
    if face is None:
        return voice.dominant_emotion       # type: ignore[union-attr]
    if voice is None:
        return face.dominant_emotion

    # Yüz güvenine göre ağırlık belirle
    face_w = 0.7 if face.confidence > 60 else 0.5
    voice_w = 1.0 - face_w

    # Her duyguya gelen toplam ağırlık
    votes: dict[str, float] = {}
    votes[face.dominant_emotion]  = votes.get(face.dominant_emotion, 0.0)  + face_w
    votes[voice.dominant_emotion] = votes.get(voice.dominant_emotion, 0.0) + voice_w

    winner = max(votes, key=votes.__getitem__)
    log.info("Füzyon → %s | oylar: %s", winner, votes)
    return winner


# ── 4. Öneri motoru ───────────────────────────────────────────────────────────
def get_recommendations(emotion: str) -> list[str]:
    """
    Duyguya özel mental sağlık önerilerini döndürür.

    Bilinmeyen duygu etiketi gelirse 'neutral' önerileri kullanılır.

    Args:
        emotion: Füzyon sonucu duygu etiketi.

    Returns:
        Türkçe öneri metinleri listesi.
    """
    return RECOMMENDATIONS.get(emotion, RECOMMENDATIONS["neutral"])


# ── Ana pipeline ──────────────────────────────────────────────────────────────
def run_analysis(
    image_bytes: Optional[bytes] = None,
    audio_bytes: Optional[bytes] = None,
) -> EmotionReport:
    """
    Tam analiz pipeline'ı — yüz ve/veya ses verisiyle çalışır.

    Hata yönetimi: Her kaynak bağımsız olarak hata verebilir;
    diğer kaynak varsa analiz devam eder, hata 'warning' alanına yazılır.

    Args:
        image_bytes: Ham görüntü baytları (opsiyonel).
        audio_bytes: Ham ses baytları (opsiyonel).

    Returns:
        EmotionReport — nihai duygu + öneriler + kaynak detayları.
    """
    face_result: Optional[FaceResult]  = None
    voice_result: Optional[VoiceResult] = None
    warnings: list[str] = []

    # Adım 1 — Yüz analizi
    if image_bytes:
        try:
            face_result = analyze_face(image_bytes)
        except Exception as exc:
            warnings.append(f"Yüz analizi atlandı: {exc}")
            log.warning(warnings[-1])

    # Adım 2 — Ses analizi
    if audio_bytes:
        try:
            voice_result = analyze_voice(audio_bytes)
        except Exception as exc:
            warnings.append(f"Ses analizi atlandı: {exc}")
            log.warning(warnings[-1])

    # Adım 3 — Füzyon
    final_emotion = fuse_emotions(face_result, voice_result)

    # Adım 4 — Öneri üret
    recs = get_recommendations(final_emotion)

    return EmotionReport(
        final_emotion=final_emotion,
        final_emotion_tr=EMOTION_TR.get(final_emotion, final_emotion),
        face=face_result,
        voice=voice_result,
        recommendations=recs,
        warning=" | ".join(warnings) or None,
    )


# ── Flask REST API ────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # HTML arayüzünden cross-origin isteklere izin ver


@app.post("/analyze")
def analyze_endpoint():
    """
    POST /analyze — multipart/form-data
      - image (opsiyonel): görüntü dosyası
      - audio (opsiyonel): ses dosyası

    Response 200: JSON EmotionReport
    Response 400: Veri eksikse hata mesajı
    """
    image_bytes = request.files["image"].read() if "image" in request.files else None
    audio_bytes = request.files["audio"].read() if "audio" in request.files else None

    # En az bir kaynak zorunlu
    if not image_bytes and not audio_bytes:
        return jsonify({"error": "En az bir veri kaynağı gerekli (image / audio)."}), 400

    report = run_analysis(image_bytes, audio_bytes)

    return jsonify({
        "final_emotion":    report.final_emotion,
        "final_emotion_tr": report.final_emotion_tr,
        "recommendations":  report.recommendations,
        "warning":          report.warning,
        "face": {
            "dominant":   report.face.dominant_emotion,
            "confidence": report.face.confidence,
            "scores":     report.face.scores,
        } if report.face else None,
        "voice": {
            "dominant":   report.voice.dominant_emotion,
            "pitch_hz":   report.voice.pitch_mean,
            "energy_rms": report.voice.energy_rms,
            "tempo_bpm":  report.voice.tempo,
        } if report.voice else None,
    })


@app.get("/health")
def health():
    """GET /health — sunucu canlılık kontrolü."""
    return jsonify({"status": "ok"})


# ── Giriş noktası ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Geliştirme modu — production'da: gunicorn emotion_assistant:app
    app.run(host="0.0.0.0", port=5050, debug=False)
