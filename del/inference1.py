import numpy as np
import librosa
import torch
import torch.nn as nn

SR, N_MFCC, N_MELS, N_CHROMA = 22050, 40, 128, 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/leakage_experiment_model.pt"


def extract_features(path):
    y, sr = librosa.load(path, sr=SR)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS), ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    zcr    = librosa.feature.zero_crossing_rate(y=y)
    t = min(mfcc.shape[1], mel_db.shape[1], chroma.shape[1], zcr.shape[1])
    feat = np.concatenate([mfcc[:, :t], mel_db[:, :t], chroma[:, :t], zcr[:, :t]], axis=0)
    return feat.T.astype(np.float32)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        w = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1), w


class CNNLSTMAttention(nn.Module):
    def __init__(self, input_dim, num_classes, cnn_channels=64, lstm_hidden=128, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, 5, padding=2), nn.BatchNorm1d(cnn_channels), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, 5, padding=2), nn.BatchNorm1d(cnn_channels * 2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(cnn_channels * 2, lstm_hidden, lstm_layers, batch_first=True, bidirectional=True,
                             dropout=dropout if lstm_layers > 1 else 0.0)
        self.attention = Attention(lstm_hidden * 2)
        self.classifier = nn.Sequential(nn.Linear(lstm_hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.classifier(context)


def load_model(path=MODEL_PATH):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = CNNLSTMAttention(input_dim=ckpt["input_dim"], num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model, ckpt["classes"], ckpt["feat_mean"], ckpt["feat_std"]


def predict(wav_path, model, classes, feat_mean, feat_std):
    feat = extract_features(wav_path)
    feat = (feat - feat_mean) / (feat_std + 1e-8)
    x = torch.tensor(feat).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], {classes[i]: float(probs[i]) for i in range(len(classes))}