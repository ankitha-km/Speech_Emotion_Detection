import numpy as np, torch, librosa, joblib
from transformers import Wav2Vec2Model

SR = 16000
_w2v, _bundle = None, None

def _init():
    global _w2v, _bundle
    if _w2v is None:
        _w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()
        _bundle = joblib.load("models/english_ser.joblib")

def _embed(path):
    y, _ = librosa.load(path, sr=SR, duration=6)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = np.pad(y, (0, max(0, SR - len(y))))
    y = ((y - y.mean()) / (y.std() + 1e-7)).astype(np.float32)
    with torch.no_grad():
        hs = _w2v(torch.tensor(y).unsqueeze(0), output_hidden_states=True).hidden_states[3:10]
    return torch.cat([h.mean(1)[0] for h in hs] + [h.std(1)[0] for h in hs]).numpy()

def predict_english(audio_path):
    _init()
    clf, sc = _bundle["clf"], _bundle["scaler"]
    probs = clf.predict_proba(sc.transform(_embed(audio_path)[None]))[0]
    return clf.classes_[probs.argmax()], dict(zip(clf.classes_, probs.round(3)))