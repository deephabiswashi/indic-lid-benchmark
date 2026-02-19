import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


RESULTS_CSV = "results/all_predictions.csv"
OUT = "results/plots"
MAX_POINTS_PER_GROUP = 1200
RANDOM_SEED = 42


def shorten_dataset_name(name):
    return (
        str(name)
        .replace("commonvoice_", "cv_")
        .replace("voxlingua_", "vox_")
        .replace("indicvoices_", "indic_")
        .replace("google_fleurs_", "fleurs_")
        .replace("aikosh_", "ak_")
    )


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def sample_by_group(df, group_cols, max_points=MAX_POINTS_PER_GROUP):
    sampled = []
    for _, group in df.groupby(group_cols, dropna=False):
        if len(group) > max_points:
            sampled.append(group.sample(max_points, random_state=RANDOM_SEED))
        else:
            sampled.append(group)
    return pd.concat(sampled, ignore_index=True)


def dataset_order(df):
    return (
        df.groupby("dataset_short").size()
        .sort_values(ascending=False)
        .index.tolist()
    )


def plot_sample_distribution(df):
    pivot = (
        df.groupby(["dataset_short", "lang"]).size()
        .unstack(fill_value=0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig_h = max(4.5, 0.55 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    pivot.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Sample Distribution per Dataset and Language")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Dataset")
    ax.legend(title="Language", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, "dataset_language_distribution.png")


def plot_inference_time(df):
    sampled = sample_by_group(df, ["dataset_short", "lang"])
    order = dataset_order(df)

    fig_h = max(4.5, 0.48 * len(order))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    sns.boxplot(
        data=sampled,
        y="dataset_short",
        x="inference_time_sec",
        hue="lang",
        order=order,
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )

    p99 = float(df["inference_time_sec"].quantile(0.99))
    if p99 > 0:
        ax.set_xlim(0, p99 * 1.05)
    ax.set_title("Inference Time Distribution (99th Percentile View)")
    ax.set_xlabel("Inference Time (seconds)")
    ax.set_ylabel("Dataset")
    ax.legend(title="Language", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, "inference_time_boxplot.png")


def plot_output_length(df):
    sampled = sample_by_group(df, ["dataset_short", "lang"])
    order = dataset_order(df)

    fig_h = max(4.5, 0.48 * len(order))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    sns.boxplot(
        data=sampled,
        y="dataset_short",
        x="output_len",
        hue="lang",
        order=order,
        showfliers=False,
        linewidth=0.8,
        ax=ax,
    )

    p99 = float(df["output_len"].quantile(0.99))
    if p99 > 0:
        ax.set_xlim(0, p99 * 1.05)
    ax.set_title("Output Length Distribution (99th Percentile View)")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Dataset")
    ax.legend(title="Language", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, "output_length_violin.png")


def plot_empty_output_heatmap(df):
    rate = (
        df.groupby(["dataset_short", "lang"])["is_empty"]
        .mean()
        .mul(100)
        .unstack(fill_value=0.0)
    )
    rate = rate.loc[rate.mean(axis=1).sort_values(ascending=False).index]

    fig_h = max(4.5, 0.50 * len(rate.index))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    sns.heatmap(rate, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.4, ax=ax)
    ax.set_title("Empty / Near-Empty Output Rate (%)")
    ax.set_xlabel("Language")
    ax.set_ylabel("Dataset")
    save_figure(fig, "empty_output_rate_heatmap.png")


def plot_inference_ecdf(df):
    sampled = sample_by_group(df, ["lang"], max_points=6000)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.ecdfplot(data=sampled, x="inference_time_sec", hue="lang", ax=ax)
    p99 = float(df["inference_time_sec"].quantile(0.99))
    if p99 > 0:
        ax.set_xlim(0, p99 * 1.05)
    ax.set_title("Inference Time ECDF by Language")
    ax.set_xlabel("Inference Time (seconds)")
    ax.set_ylabel("Cumulative Probability")
    save_figure(fig, "inference_time_ecdf_by_lang.png")


def plot_latency_vs_output(df):
    sampled = sample_by_group(df, ["dataset_short"], max_points=1500)
    fig, ax = plt.subplots(figsize=(10, 5))
    hb = ax.hexbin(
        sampled["output_len"],
        sampled["inference_time_sec"],
        gridsize=55,
        mincnt=1,
        cmap="viridis",
        bins="log",
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")
    ax.set_title("Output Length vs Inference Time Density")
    ax.set_xlabel("Output Length (characters)")
    ax.set_ylabel("Inference Time (seconds)")
    save_figure(fig, "latency_vs_output_length_hexbin.png")


def plot_throughput(df):
    summary = (
        df.groupby("dataset_short")
        .agg(samples=("file", "count"), total_time=("inference_time_sec", "sum"))
        .reset_index()
    )
    summary["samples_per_sec"] = summary["samples"] / summary["total_time"].clip(lower=1e-9)
    summary = summary.sort_values("samples_per_sec", ascending=True)

    fig_h = max(4.5, 0.42 * len(summary))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    sns.barplot(data=summary, y="dataset_short", x="samples_per_sec", color="#4C78A8", ax=ax)
    ax.set_title("Throughput by Dataset")
    ax.set_xlabel("Samples per second (higher is better)")
    ax.set_ylabel("Dataset")
    save_figure(fig, "throughput_samples_per_sec.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")

    df = pd.read_csv(RESULTS_CSV)
    required = {"dataset", "lang", "file", "transcription", "inference_time_sec"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {RESULTS_CSV}: {sorted(missing)}")

    df["transcription"] = df["transcription"].fillna("").astype(str)
    df["output_len"] = df["transcription"].str.len()
    df["is_empty"] = df["transcription"].str.strip().str.len() < 3
    df["dataset_short"] = df["dataset"].astype(str).apply(shorten_dataset_name)

    # Backward-compatible output filenames (improved layout + performance).
    plot_sample_distribution(df)
    plot_inference_time(df)
    plot_output_length(df)

    # Additional evaluation visuals.
    plot_empty_output_heatmap(df)
    plot_inference_ecdf(df)
    plot_latency_vs_output(df)
    plot_throughput(df)

    print("Plot generation completed.")


if __name__ == "__main__":
    main()
