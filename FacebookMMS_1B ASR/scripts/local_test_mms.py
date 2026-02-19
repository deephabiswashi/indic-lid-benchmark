import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import time
import os

from mms_asr import MMSASR

TARGET_SR = 16000
AUDIO_EXTS = (".wav", ".mp3", ".flac")


# -----------------------------
# Audio Utilities
# -----------------------------
def record_audio(duration_sec):
    print(f"🎙 Recording for {duration_sec} seconds...")
    audio = sd.rec(
        int(duration_sec * TARGET_SR),
        samplerate=TARGET_SR,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("✅ Recording finished.")
    return audio.squeeze(), TARGET_SR


def load_audio_file(path):
    if not path.lower().endswith(AUDIO_EXTS):
        raise ValueError("Unsupported audio format")

    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        audio = torch.tensor(audio, dtype=torch.float32)
        audio = torchaudio.functional.resample(
            audio, orig_freq=sr, new_freq=TARGET_SR
        ).numpy()
        sr = TARGET_SR

    return audio, sr


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Local testing for facebook/mms-1b-all (no server)"
    )

    parser.add_argument(
        "--mode",
        choices=["record", "file"],
        required=True,
        help="record = live mic | file = pre-recorded audio"
    )

    parser.add_argument(
        "--lang",
        choices=["as", "bn", "ory"],
        required=True,
        help="Language code"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Recording duration (seconds) [record mode]"
    )

    parser.add_argument(
        "--audio_path",
        type=str,
        help="Path to audio file [file mode]"
    )

    args = parser.parse_args()

    asr = MMSASR()

    # ---------- Live Recording ----------
    if args.mode == "record":
        audio, sr = record_audio(args.duration)

    # ---------- File Upload ----------
    elif args.mode == "file":
        if not args.audio_path:
            raise ValueError("--audio_path is required for file mode")
        audio, sr = load_audio_file(args.audio_path)

    # ---------- Transcription ----------
    print("🧠 Running MMS ASR...")
    start = time.time()
    text = asr.transcribe(audio, sr, args.lang)
    elapsed = time.time() - start

    print("\n==============================")
    print(f"Language        : {args.lang}")
    print(f"Inference Time  : {elapsed:.2f} sec")
    print("Transcription  :")
    print(text)
    print("==============================\n")


if __name__ == "__main__":
    main()
