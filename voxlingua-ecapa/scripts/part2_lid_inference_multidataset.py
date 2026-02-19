import os
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from tqdm import tqdm

# =========================================================
# WINDOWS SAFE SETTINGS
# =========================================================
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["SPEECHBRAIN_CACHE"] = r"C:\speechbrain_cache"

# =========================================================
# DEVICE
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================================================
# LOAD MODEL
# =========================================================
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir=r"C:\speechbrain_cache\voxlingua107",
    run_opts={"device": device},
)

# =========================================================
# DATASET DEFINITIONS
# =========================================================
DATASETS = [
    # CommonVoice
    {
        "dataset": "commonvoice",
        "true_language": "as",
        "source_dataset_language": "as",
        "path": r"commonvoice\as\clips",
    },
    {
        "dataset": "commonvoice",
        "true_language": "bn",
        "source_dataset_language": "bn",
        "path": r"commonvoice\bn\clips",
    },
    # VoxLingua107
    {
        "dataset": "voxlingua107",
        "true_language": "as",
        "source_dataset_language": "as",
        "path": r"voxlingua107\as",
    },
    {
        "dataset": "voxlingua107",
        "true_language": "bn",
        "source_dataset_language": "bn",
        "path": r"voxlingua107\bn",
    },
    # IndicVoices
    {
        "dataset": "indicvoices",
        "true_language": "or",
        "source_dataset_language": "ory",
        "path": r"indicvoices\ory",
    },
    # AikoshDataset
    {
        "dataset": "aikoshdataset",
        "true_language": "as",
        "source_dataset_language": "assamese",
        "path": r"aikoshdataset\assamese",
    },
    {
        "dataset": "aikoshdataset",
        "true_language": "bn",
        "source_dataset_language": "bengali",
        "path": r"aikoshdataset\bengali",
    },
    {
        "dataset": "aikoshdataset",
        "true_language": "hi",
        "source_dataset_language": "hindi",
        "path": r"aikoshdataset\hindi",
    },
    {
        "dataset": "aikoshdataset",
        "true_language": "or",
        "source_dataset_language": "odia",
        "path": r"aikoshdataset\odia",
    },
    {
        "dataset": "aikoshdataset",
        "true_language": "ta",
        "source_dataset_language": "tamil",
        "path": r"aikoshdataset\tamil",
    },
    {
        "dataset": "aikoshdataset",
        "true_language": "te",
        "source_dataset_language": "telegu",
        "path": r"aikoshdataset\telegu",
    },
    # Google FLEURS
    {
        "dataset": "google fleurs",
        "true_language": "or",
        "source_dataset_language": "ory",
        "path": r"google fleurs\ory\test",
    },
]

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
BASE_DIR = Path(".")
results = []
seen_keys = set()


def get_audio_files(audio_dir):
    if not audio_dir.exists():
        return None
    files = [
        file_path
        for file_path in audio_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


# =========================================================
# DATASET FILE COUNTS
# =========================================================
total_selected_files = 0
for ds in DATASETS:
    dataset = ds["dataset"]
    true_lang = ds["true_language"]
    source_dataset_language = ds["source_dataset_language"]
    audio_dir = BASE_DIR / ds["path"]

    audio_files = get_audio_files(audio_dir)
    if audio_files is None:
        print(f"[WARN] Directory not found: {audio_dir}")
        continue

    print(
        f"[COUNT] {dataset} | source={source_dataset_language} | "
        f"true={true_lang} | files={len(audio_files)}"
    )
    total_selected_files += len(audio_files)

print(f"\nTotal selected files across configured datasets: {total_selected_files}")

# =========================================================
# INFERENCE LOOP (ROBUST + MULTI-FORMAT)
# =========================================================
for ds in DATASETS:
    dataset = ds["dataset"]
    true_lang = ds["true_language"]
    source_dataset_language = ds["source_dataset_language"]
    audio_dir = BASE_DIR / ds["path"]

    audio_files = get_audio_files(audio_dir)
    if audio_files is None:
        continue

    print(
        f"\nProcessing {dataset} | source={source_dataset_language} | "
        f"{true_lang} | files: {len(audio_files)}"
    )

    for wav in tqdm(audio_files, desc=f"{dataset}-{true_lang}"):
        row_key = (dataset, str(wav))
        if row_key in seen_keys:
            continue

        try:
            signal, sr = torchaudio.load(wav)

            # Ensure mono
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != 16000:
                signal = torchaudio.functional.resample(signal, sr, 16000)

            prediction = classifier.classify_batch(signal)

            results.append(
                {
                    "dataset": dataset,
                    "source_dataset_language": source_dataset_language,
                    "true_language": true_lang,
                    "predicted_language": prediction[3][0],  # "as: Assamese"
                    "confidence": float(prediction[1].max()),
                    "audio_path": str(wav),
                }
            )
            seen_keys.add(row_key)

        except Exception as e:
            print(f"[SKIP] {wav.name} | {e}")
            continue

# =========================================================
# SAVE RESULTS
# =========================================================
out_df = pd.DataFrame(results)
out_df.to_csv("results/predictions_multidataset.csv", index=False)

print("\nInference completed successfully.")
print("Saved to: results/predictions_multidataset.csv")
