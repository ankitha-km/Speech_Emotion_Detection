import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, regularizers


# =========================================================
# PATHS
# =========================================================

MODEL_WEIGHTS = "models/ser_exp4_weights.weights.h5"
NORM_STATS = "models/norm_stats_exp4.npz"


# =========================================================
# MODEL SETTINGS
# =========================================================

SAMPLE_RATE = 22050

N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

INPUT_ROWS = 181
INPUT_FRAMES = 151

NUM_CLASSES = 5


# =========================================================
# ATTENTION LAYER
# EXACTLY LIKE YOUR ORIGINAL MODEL
# =========================================================

class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)

        self.units = units

        self.score_dense = tf.keras.layers.Dense(
            units,
            activation="tanh"
        )

        self.weight_dense = tf.keras.layers.Dense(1)

    def call(self, x):

        # x = (batch, time, features)

        score = self.score_dense(x)

        score = self.weight_dense(score)

        weights = tf.nn.softmax(score, axis=1)

        context = tf.reduce_sum(
            x * weights,
            axis=1
        )

        return context

    def get_config(self):

        config = super().get_config()

        config.update({
            "units": self.units
        })

        return config


# =========================================================
# BUILD MODEL
# =========================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(181, 151, 1)
    )

    # -------------------------
    # CNN 1
    # -------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 1)
    )(x)

    x = layers.Dropout(0.3)(x)

    # -------------------------
    # CNN 2
    # -------------------------

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 1)
    )(x)

    x = layers.Dropout(0.3)(x)

    # -------------------------
    # CNN 3
    # -------------------------

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 1)
    )(x)

    x = layers.Dropout(0.4)(x)

    # -------------------------
    # Convert CNN output
    # into sequence
    # -------------------------

    x = layers.Permute(
        (2, 1, 3)
    )(x)

    x = layers.Reshape(
        (151, -1)
    )(x)

    # -------------------------
    # TimeDistributed Dense
    # -------------------------

    x = layers.TimeDistributed(
        layers.Dense(
            96,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(x)

    # -------------------------
    # BiLSTM
    # -------------------------

    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(x)

    x = layers.Dropout(0.4)(x)

    # -------------------------
    # Attention
    # -------------------------

    x = AttentionLayer(64)(x)

    # -------------------------
    # Dense
    # -------------------------

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.Dropout(0.5)(x)

    # -------------------------
    # Output
    # -------------------------

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model


# =========================================================
# LOAD NORMALIZATION STATISTICS
# =========================================================
stats = np.load(NORM_STATS, allow_pickle=True)

MFCC_MEAN = float(stats["mfcc_mean"])
MFCC_STD = float(stats["mfcc_std"])

MEL_MEAN = float(stats["mel_mean"])
MEL_STD = float(stats["mel_std"])

CHROMA_MEAN = float(stats["chroma_mean"])
CHROMA_STD = float(stats["chroma_std"])

ZCR_MEAN = float(stats["zcr_mean"])
ZCR_STD = float(stats["zcr_std"])

CLASSES = stats["classes"]


# =========================================================
# BUILD + LOAD MODEL
# =========================================================

print("Building model...")

model = build_model()

# Build variables before loading weights
dummy_input = np.zeros(
    (1, 181, 151, 1),
    dtype=np.float32
)

model(dummy_input)

print("Loading weights...")

model.load_weights(
    MODEL_WEIGHTS
)

print("MODEL LOADED ✅")

print("Classes:", CLASSES)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # -------------------------
    # MFCC
    # -------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # -------------------------
    # MEL
    # -------------------------

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # -------------------------
    # CHROMA
    # -------------------------

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # -------------------------
    # ZERO CROSSING RATE
    # -------------------------

    zcr = librosa.feature.zero_crossing_rate(
        y,
        hop_length=HOP_LENGTH
    )

    # =====================================================
    # NORMALIZATION
    # =====================================================

    mfcc = (
        mfcc - MFCC_MEAN
    ) / (MFCC_STD + 1e-8)

    mel = (
        mel - MEL_MEAN
    ) / (MEL_STD + 1e-8)

    chroma = (
        chroma - CHROMA_MEAN
    ) / (CHROMA_STD + 1e-8)

    zcr = (
        zcr - ZCR_MEAN
    ) / (ZCR_STD + 1e-8)

    # =====================================================
    # MAKE ALL FEATURES 151 FRAMES
    # =====================================================

    def fix_frames(feature, target=151):

        current = feature.shape[1]

        if current < target:

            feature = np.pad(
                feature,
                ((0, 0), (0, target - current)),
                mode="constant"
            )

        elif current > target:

            feature = feature[:, :target]

        return feature

    mfcc = fix_frames(mfcc)
    mel = fix_frames(mel)
    chroma = fix_frames(chroma)
    zcr = fix_frames(zcr)

    # =====================================================
    # CONCATENATE
    # =====================================================

    combined = np.concatenate(
        [
            mfcc,
            mel,
            chroma,
            zcr
        ],
        axis=0
    )

    # Expected:
    # 40 + 128 + 12 + 1 = 181

    assert combined.shape == (181, 151), \
        f"Unexpected feature shape: {combined.shape}"

    # Add channel dimension

    combined = combined.astype(
        np.float32
    )

    combined = np.expand_dims(
        combined,
        axis=-1
    )

    # Add batch dimension

    combined = np.expand_dims(
        combined,
        axis=0
    )

    return combined


# =========================================================
# PREDICTION
# =========================================================

def predict_emotion(audio_path):

    features = extract_features(
        audio_path
    )

    probabilities = model.predict(
        features,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        probabilities
    )

    predicted_emotion = CLASSES[
        predicted_index
    ]

    return predicted_emotion, probabilities