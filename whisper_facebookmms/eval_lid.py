import os
import json
import argparse
import librosa
import torch
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# -----------------------------
# Language mappings
# -----------------------------
LANG_MAP = {
    "hindi": "hi",
    "bengali": "bn",
    "assamese": "as",
    "odia": "or",
    "tamil": "ta",
    "telegu": "te"
}

WHISPER_LANGS = {"hi", "bn", "as", "ta", "te"}   # NOT odia
ODIA_LANG = "or"

DATA_DIR = "data"
OUT_JSON = "lid_results_whisper_mms.json"

# -----------------------------
# Load models
# -----------------------------
parser = argparse.ArgumentParser(description="Evaluate LID with Whisper and MMS-LID")
parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Device to run models on (overrides auto-detection)")
args = parser.parse_args()

# Determine device
detected_cuda = torch.cuda.is_available()
if args.device:
    device = args.device
else:
    device = "cuda" if detected_cuda else "cpu"

print("🔊 Loading Whisper-small...")
whisper = WhisperModel(
    "small",
    device=device,
    compute_type="float16" if device == "cuda" else "float32"
)

print("🌍 Loading MMS-LID (Facebook)...")
mms_processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-1024")
mms_model = AutoModelForAudioClassification.from_pretrained("facebook/mms-lid-1024").to(torch.device(device))

id2label = mms_model.config.id2label

# -----------------------------
# Utility: load audio
# -----------------------------
def load_audio(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return audio

# -----------------------------
# Whisper LID
# -----------------------------
def whisper_lid(audio):
    segments, info = whisper.transcribe(
        audio,
        language=None,
        task="transcribe"
    )

    lang = info.language
    conf = info.language_probability

    return {
        "pred": lang,
        "confidence": float(conf),
        "topk": [lang]  # Whisper only gives top-1
    }

# -----------------------------
# MMS LID (for Odia)
# -----------------------------
def mms_lid(audio):
    # Feature-extractor returns tensors when return_tensors="pt"
    inputs = mms_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    # move tensors to model device
    inputs = {k: v.to(mms_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mms_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    topk = torch.topk(probs, k=3)

    preds = [
        id2label[idx.item()].replace("lang_", "")
        for idx in topk.indices
    ]

    return {
        "pred": preds[0],
        "confidence": float(topk.values[0]),
        "topk": preds
    }

# -----------------------------
# Main evaluation loop
# -----------------------------
results = []

print("🚀 Starting LID Evaluation...\n")

# Print device info
try:
    print(f"Whisper device: {whisper_device}")
except NameError:
    print("Whisper device: unknown")
try:
    print(f"MMS device: {mms_device}")
except NameError:
    print("MMS device: unknown")

# Resolve data directory: check cwd/data, repo root (parent of scripts)/data, then scripts/data
cwd_candidate = Path(DATA_DIR)
script_dir = Path(__file__).resolve().parent
repo_root_candidate = script_dir.parent / DATA_DIR
script_local_candidate = script_dir / DATA_DIR
if cwd_candidate.exists():
    data_root = cwd_candidate.resolve()
elif repo_root_candidate.exists():
    data_root = repo_root_candidate.resolve()
elif script_local_candidate.exists():
    data_root = script_local_candidate.resolve()
else:
    raise SystemExit(f"Data directory not found. Checked: {cwd_candidate.resolve()}, {repo_root_candidate.resolve()}, {script_local_candidate.resolve()}")

allowed_ext = {".wav", ".mp3", ".flac"}
tasks = []
for root, _, files in os.walk(data_root):
    for file in files:
        if not file.lower().endswith(tuple(allowed_ext)):
            continue
        file_path = Path(root) / file
        try:
            rel = file_path.relative_to(data_root)
        except Exception:
            continue
        if len(rel.parts) == 0:
            continue
        lang_dir = rel.parts[0].lower()
        true_lang = LANG_MAP.get(lang_dir)
        if not true_lang:
            continue
        posix_path = file_path.resolve().as_posix()
        tasks.append((posix_path, true_lang))

total = len(tasks)
if total == 0:
    print("No audio files found under:", data_root)
else:
    print(f"Found {total} audio files. Processing with progress bar...")
    for posix_path, true_lang in tqdm(tasks, desc="Processing files", unit="file"):
        try:
            audio = load_audio(posix_path)
        except Exception as e:
            print(f"⚠️ Failed to load {posix_path}: {e}")
            continue

        try:
            if true_lang == ODIA_LANG:
                out = mms_lid(audio)
                model_used = "MMS-LID"
            else:
                out = whisper_lid(audio)
                model_used = "Whisper-small"

            results.append({
                "file": posix_path,
                "true_lang": true_lang,
                "pred_lang": out["pred"],
                "confidence": out["confidence"],
                "topk": out["topk"],
                "model": model_used
            })

        except Exception as e:
            print(f"⚠️ Failed on {posix_path}: {e}")

# -----------------------------
# Save results
# -----------------------------
os.makedirs("evaluation", exist_ok=True)

with open(f"evaluation/{OUT_JSON}", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ LID Evaluation Completed")
print(f"📄 Results saved to evaluation/{OUT_JSON}")
