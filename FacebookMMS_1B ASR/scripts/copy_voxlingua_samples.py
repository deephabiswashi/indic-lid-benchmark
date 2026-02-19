import os
import random
import shutil

SRC_ROOT = r"D:\VoxLingua Dataset"
DST_ROOT = r"C:\Users\admin\Desktop\Major Project\FacebookMMS_LID\voxlingua"

REQUESTED_SAMPLES = 2000
SEED = 42
AUDIO_EXTS = (".wav", ".flac", ".mp3")

def collect_audio_files(root):
    audio_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(AUDIO_EXTS):
                audio_files.append(os.path.join(dirpath, f))
    return audio_files

def copy_samples(lang):
    src_lang = os.path.join(SRC_ROOT, lang)
    dst_lang = os.path.join(DST_ROOT, lang, "clips")
    os.makedirs(dst_lang, exist_ok=True)

    files = collect_audio_files(src_lang)

    if len(files) == 0:
        raise RuntimeError(f"No audio files found for {lang}")

    random.seed(SEED)
    n = min(len(files), REQUESTED_SAMPLES)
    selected = random.sample(files, n)

    for f in selected:
        shutil.copy2(f, os.path.join(dst_lang, os.path.basename(f)))

    print(
        f"{lang.upper()} | "
        f"Available: {len(files)} | "
        f"Copied: {n}"
    )

if __name__ == "__main__":
    copy_samples("as")
    copy_samples("bn")
