import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ CONFIG ------------------

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")

AUDIO_EXTS = (".wav", ".flac", ".mp3")
RANDOM_SEED = 42
MAX_POINTS_PER_DATASET = 1200

# ------------------ AUDIO METRICS ------------------

def audio_stats(file_path):
    audio, sr = sf.read(file_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio ** 2))
    return duration, rms, sr


# ------------------ CORE ANALYSIS ------------------

def shorten_dataset_name(name):
    return (
        name.replace("commonvoice_", "cv_")
        .replace("voxlingua_", "vox_")
        .replace("indicvoices_", "indic_")
        .replace("google_fleurs_", "fleurs_")
        .replace("aikosh_", "ak_")
    )


def dataset_order(df):
    ordered = (
        df.groupby("dataset")["inference_time_sec"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    return [shorten_dataset_name(x) for x in ordered]


def downsample_for_plot(df):
    sampled = []
    for _, group in df.groupby("dataset", sort=False):
        if len(group) > MAX_POINTS_PER_DATASET:
            sampled.append(group.sample(MAX_POINTS_PER_DATASET, random_state=RANDOM_SEED))
        else:
            sampled.append(group)
    return pd.concat(sampled, ignore_index=True)


def enrich_dataframe(df):
    durations, rms_vals, lengths, empty_flags = [], [], [], []

    for row in df.itertuples(index=False):
        dur, rms, _ = audio_stats(row.file)
        durations.append(dur)
        rms_vals.append(rms)

        text = str(row.transcription)
        lengths.append(len(text))
        empty_flags.append(len(text.strip()) < 3)

    df["duration_sec"] = durations
    df["rms_energy"] = rms_vals
    df["output_length"] = lengths
    df["empty_output"] = empty_flags

    return df


def compute_stats(df):
    return {
        "samples": int(len(df)),
        "mean_duration_sec": float(df["duration_sec"].mean()),
        "median_duration_sec": float(df["duration_sec"].median()),
        "mean_output_length": float(df["output_length"].mean()),
        "empty_output_rate_%": round(100 * df["empty_output"].mean(), 2),
        "mean_rms_energy": float(df["rms_energy"].mean()),
        "mean_inference_time_sec": float(df["inference_time_sec"].mean())
    }


# ------------------ PLOTTING ------------------

def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_duration(df):
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    order = dataset_order(df)
    sns.boxplot(
        data=df,
        y="dataset_short",
        x="duration_sec",
        order=order,
        showfliers=False,
        linewidth=0.8,
        color="#72B6A1",
        ax=ax,
    )
    ax.set_title("Audio Duration Distribution Across Datasets")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Dataset")
    save_plot(fig, "duration_distribution.png")


def plot_inference_time(df):
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    order = dataset_order(df)
    sns.boxplot(
        data=df,
        y="dataset_short",
        x="inference_time_sec",
        hue="lang",
        order=order,
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )
    ax.set_title("Inference Time Distribution (MMS-1B)")
    ax.set_xlabel("Inference Time (seconds)")
    ax.set_ylabel("Dataset")
    ax.legend(title="lang", ncol=3, fontsize=8, title_fontsize=9, loc="upper right")
    save_plot(fig, "inference_time_boxplot.png")


def plot_output_length(df):
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    order = dataset_order(df)
    sns.boxplot(
        data=df,
        y="dataset_short",
        x="output_length",
        hue="lang",
        order=order,
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )
    ax.set_title("Output Length Distribution")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Dataset")
    ax.legend(title="lang", ncol=3, fontsize=8, title_fontsize=9, loc="upper right")
    save_plot(fig, "output_length_violin.png")


def plot_empty_rate(df):
    rate = (
        df.groupby(["dataset_short", "lang"])["empty_output"]
        .mean()
        .reset_index()
    )
    rate["empty_output"] *= 100

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    order = dataset_order(df)
    sns.barplot(
        data=rate,
        y="dataset_short",
        x="empty_output",
        hue="lang",
        order=order,
        ax=ax,
    )
    ax.set_title("Empty / Near-Empty Transcription Rate")
    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel("Dataset")
    ax.legend(title="lang", ncol=3, fontsize=8, title_fontsize=9, loc="upper right")
    save_plot(fig, "empty_output_rate.png")


def plot_rms_energy(df):
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    order = dataset_order(df)
    sns.boxplot(
        data=df,
        y="dataset_short",
        x="rms_energy",
        hue="lang",
        order=order,
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )
    ax.set_title("RMS Energy Distribution (Audio Quality Proxy)")
    ax.set_xlabel("RMS Energy")
    ax.set_ylabel("Dataset")
    ax.legend(title="lang", ncol=3, fontsize=8, title_fontsize=9, loc="upper right")
    save_plot(fig, "rms_energy_boxplot.png")


# ------------------ MAIN ------------------

def main():
    df = pd.read_csv("results/all_predictions.csv")

    df = enrich_dataframe(df)
    df["dataset_short"] = df["dataset"].apply(shorten_dataset_name)

    # Downsample only for heavy distribution plots to make rendering faster.
    plot_df = downsample_for_plot(df)

    # ---------- Global Stats ----------
    global_stats = compute_stats(df)

    # ---------- Dataset-wise Stats ----------
    dataset_stats = {
        dataset: compute_stats(subdf)
        for dataset, subdf in df.groupby("dataset")
    }

    # ---------- Language-wise Stats ----------
    language_stats = {
        lang: compute_stats(subdf)
        for lang, subdf in df.groupby("lang")
    }

    # ---------- Save Stats ----------
    with open(os.path.join(RESULTS_DIR, "global_analysis.json"), "w") as f:
        json.dump(global_stats, f, indent=4)

    with open(os.path.join(RESULTS_DIR, "dataset_analysis.json"), "w") as f:
        json.dump(dataset_stats, f, indent=4)

    with open(os.path.join(RESULTS_DIR, "language_analysis.json"), "w") as f:
        json.dump(language_stats, f, indent=4)

    # ---------- Plots ----------
    plot_duration(plot_df)
    plot_inference_time(plot_df)
    plot_output_length(plot_df)
    plot_empty_rate(df)
    plot_rms_energy(plot_df)

    print("Combined multi-dataset analysis completed successfully.")


if __name__ == "__main__":
    main()
