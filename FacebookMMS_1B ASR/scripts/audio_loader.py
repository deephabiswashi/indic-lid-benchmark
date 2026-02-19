import os
import random
import torchaudio
import torch
try:
    import soundfile as sf
except Exception:
    sf = None

TARGET_SR = 16000
AUDIO_EXTS = (".wav", ".flac", ".mp3")


def _load_audio(file_path):
    """
    Load mono waveform and auto-resample to TARGET_SR.
    Returns numpy audio + sample rate.
    """
    try:
        waveform, sr = torchaudio.load(file_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
    except Exception:
        if sf is None:
            raise
        audio, sr = sf.read(file_path)
        waveform = torch.tensor(audio, dtype=torch.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=TARGET_SR
        )
        sr = TARGET_SR

    return waveform.numpy(), sr


def _collect_audio_files(path):
    files = []
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(AUDIO_EXTS):
                files.append(os.path.join(dirpath, filename))
    return files


def load_audio_dataset(path, limit=None, seed=42):
    """
    Recursive dataset loader with optional sample limit.
    """
    files = _collect_audio_files(path)
    files.sort()

    if limit is not None:
        random.seed(seed)
        files = random.sample(files, min(limit, len(files)))

    return [(f, *_load_audio(f)) for f in files]


def load_samples(root, lang, n, seed=42):
    """
    Backward-compatible loader for CommonVoice & VoxLingua.
    """
    path = os.path.join(root, lang, "clips")
    return load_audio_dataset(path, limit=n, seed=seed)


def load_indicvoices(path):
    """
    Backward-compatible loader for IndicVoices.
    """
    return load_audio_dataset(path)
