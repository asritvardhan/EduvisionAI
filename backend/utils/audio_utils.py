import librosa
import numpy as np
import tensorflow as tf

EMOTIONS = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

SR = 22050
DURATION = 3
N_MELS = 128
HOP_LENGTH = 512
MAX_LEN = 130


def extract_audio_features(audio_path):
    """
    MUST match OptimizedCREMADClassifier.extract_enhanced_features()
    Output shape: (time_steps, 176)
    """
    audio, sr = librosa.load(audio_path, sr=SR, duration=DURATION)
    target_len = SR * DURATION

    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='reflect')
    else:
        audio = audio[:target_len]

    # --- Mel Spectrogram (128) ---
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=2048,
        fmin=50,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).T  # (T, 128)

    # --- MFCC + Delta + Delta-Delta (39) ---
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_all = np.vstack([mfcc, mfcc_delta, mfcc_delta2]).T  # (T, 39)

    # --- Spectral features (1 + 1 + 7 = 9) ---
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH).T
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH).T
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=HOP_LENGTH).T

    # --- Align time steps ---
    T = min(
        mel_db.shape[0],
        mfcc_all.shape[0],
        spec_centroid.shape[0],
        spec_rolloff.shape[0],
        spec_contrast.shape[0]
    )

    features = np.concatenate([
        mel_db[:T],
        mfcc_all[:T],
        spec_centroid[:T],
        spec_rolloff[:T],
        spec_contrast[:T]
    ], axis=1)

    return features  # (T, 176)


def predict_audio_emotion(audio_path, model):
    features = extract_audio_features(audio_path)

    X = tf.keras.preprocessing.sequence.pad_sequences(
        [features],
        maxlen=MAX_LEN,
        dtype="float32",
        padding="post",
        truncating="post"
    )

    preds = model.predict(X, verbose=0)[0]
    emotion_idx = int(np.argmax(preds))

    probabilities = {
        EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))
    }

    return {
        "emotion": EMOTIONS[emotion_idx],
        "confidence": float(preds[emotion_idx]),
        "probabilities": probabilities
    }
