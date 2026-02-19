import pandas as pd
from pathlib import Path
import shutil
import random
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

AS_BASE = Path(r"D:\CommonVoice Dataset\Assamese\cv-corpus-24.0-2025-12-05\as")
BN_BASE = Path(r"D:\CommonVoice Dataset\Bengali\cv-corpus-24.0-2025-12-05\bn")

LOCAL_DATA = Path(r"C:\Users\admin\Desktop\Major Project\INDIC-LID\data")

N_SAMPLES = 1000
OUTPUT_CSV = Path("results/sample_2000_clips.csv")

# =========================================================
# FUNCTION
# =========================================================

def sample_and_copy(tsv_path, clips_dir, out_dir, lang, n):
    print(f"\n[{lang}] Reading metadata:", tsv_path)
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["path"].notna()]

    print(f"[{lang}] Scanning audio files on disk...")
    valid = []

    for path in tqdm(df["path"], desc=f"{lang} scanning"):
        src = clips_dir / path
        if src.exists():
            valid.append(src)

    print(f"[{lang}] Existing validated audio files:", len(valid))

    if len(valid) == 0:
        raise RuntimeError(f"No audio files found for language {lang}")

    sampled = random.sample(valid, min(n, len(valid)))

    out_lang_dir = out_dir / lang / "clips"
    out_lang_dir.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"[{lang}] Copying {len(sampled)} files locally...")

    for src in tqdm(sampled, desc=f"{lang} copying"):
        dst = out_lang_dir / src.name
        shutil.copy2(src, dst)
        records.append({
            "true_language": lang,
            "audio_path": str(dst)
        })

    print(f"[{lang}] Copy completed")
    return pd.DataFrame(records)

# =========================================================
# RUN
# =========================================================

LOCAL_DATA.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(exist_ok=True)

as_df = sample_and_copy(
    AS_BASE / "validated.tsv",
    AS_BASE / "clips",
    LOCAL_DATA,
    "as",
    N_SAMPLES
)

bn_df = sample_and_copy(
    BN_BASE / "validated.tsv",
    BN_BASE / "clips",
    LOCAL_DATA,
    "bn",
    N_SAMPLES
)

final_df = pd.concat([as_df, bn_df], ignore_index=True)
final_df.to_csv(OUTPUT_CSV, index=False)

print("\nFINAL SAMPLE COUNTS:")
print(final_df["true_language"].value_counts())
print("Saved to:", OUTPUT_CSV.resolve())
