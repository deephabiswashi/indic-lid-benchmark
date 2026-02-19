import os
import pandas as pd
import json
import time
from tqdm import tqdm

from audio_loader import load_audio_dataset
from mms_asr import MMSASR


DATASETS = {
    "commonvoice_as": {
        "path": os.path.join("commonvoice", "as", "clips"),
        "lang": "as",
        "limit": 1000,
    },
    "commonvoice_bn": {
        "path": os.path.join("commonvoice", "bn", "clips"),
        "lang": "bn",
        "limit": 1000,
    },
    "voxlingua_as": {
        "path": os.path.join("voxlingua", "as", "clips"),
        "lang": "as",
        "limit": 2000,
    },
    "voxlingua_bn": {
        "path": os.path.join("voxlingua", "bn", "clips"),
        "lang": "bn",
        "limit": 2000,
    },
    "indicvoices_ory": {
        "path": [
            os.path.join("indicvoices", "ory"),
            os.path.join("indicvoices", "ory_wav_16k"),
        ],
        "lang": "ory",
        "limit": None,
    },
    "google_fleurs_ory": {
        "path": os.path.join("google fleurs", "ory"),
        "lang": "ory",
        "limit": None,
    },
    "aikosh_assamese": {
        "path": os.path.join("aikoshdataset", "assamese"),
        "lang": "assamese",
        "limit": None,
    },
    "aikosh_bengali": {
        "path": os.path.join("aikoshdataset", "bengali"),
        "lang": "bengali",
        "limit": None,
    },
    "aikosh_hindi": {
        "path": os.path.join("aikoshdataset", "hindi"),
        "lang": "hindi",
        "limit": None,
    },
    "aikosh_odia": {
        "path": os.path.join("aikoshdataset", "odia"),
        "lang": "odia",
        "limit": None,
    },
    "aikosh_tamil": {
        "path": os.path.join("aikoshdataset", "tamil"),
        "lang": "tamil",
        "limit": None,
    },
    "aikosh_telegu": {
        "path": os.path.join("aikoshdataset", "telegu"),
        "lang": "telegu",
        "limit": None,
    },
}


def _resolve_dataset_path(path_or_paths):
    if isinstance(path_or_paths, (list, tuple)):
        for candidate in path_or_paths:
            if os.path.isdir(candidate):
                return candidate
        return path_or_paths[0]
    return path_or_paths


def run_lang(asr, dataset, lang, samples):
    rows = []
    for path, audio, sr in tqdm(samples, desc=f"{dataset}-{lang}"):
        start = time.time()
        text = asr.transcribe(audio, sr, lang)
        t = time.time() - start

        rows.append({
            "dataset": dataset,
            "lang": lang,
            "file": path,
            "transcription": text,
            "inference_time_sec": round(t, 4)
        })
    return rows


if __name__ == "__main__":
    asr = MMSASR()
    rows = []

    for dataset_name, cfg in DATASETS.items():
        dataset_path = _resolve_dataset_path(cfg["path"])
        samples = load_audio_dataset(
            dataset_path,
            limit=cfg["limit"],
            seed=42,
        )
        rows += run_lang(asr, dataset_name, cfg["lang"], samples)

    os.makedirs("results", exist_ok=True)

    # ---------- Save Predictions ----------
    df = pd.DataFrame(rows)
    df.to_csv("results/all_predictions.csv", index=False)

    # ---------- JSON-SAFE SUMMARY ----------
    summary = {}
    for (dataset, lang), count in df.groupby(["dataset", "lang"]).size().items():
        summary.setdefault(dataset, {})[lang] = int(count)

    with open("results/stats.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(json.dumps(summary, indent=4))
