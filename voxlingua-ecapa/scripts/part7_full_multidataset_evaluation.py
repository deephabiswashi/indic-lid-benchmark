from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = Path("results/predictions_multidataset.csv")
OUT_DIR = Path("results/plots_multidataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_DATASETS = [
    ("commonvoice", "confusion_commonvoice.png", "CommonVoice"),
    ("voxlingua107", "confusion_voxlingua107.png", "VoxLingua107"),
    ("aikoshdataset", "confusion_aikoshdataset.png", "AikoshDataset"),
    ("indicvoices", "confusion_indicvoices.png", "IndicVoices"),
    ("google_fleurs", "confusion_googlefleurs.png", "Google FLEURS"),
]


def save_no_data_plot(title, filename):
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


def prepare_dataframe(df):
    if "dataset" not in df.columns:
        print("[WARN] Missing 'dataset' column. Filling with 'unknown'.")
        df["dataset"] = "unknown"

    if "source_dataset_language" not in df.columns:
        print(
            "[WARN] Missing 'source_dataset_language' column. "
            "Falling back to true_language."
        )
        df["source_dataset_language"] = df.get("true_language", "unknown")

    if "predicted_language" not in df.columns:
        print("[WARN] Missing 'predicted_language' column. Using empty values.")
        df["predicted_language"] = ""

    if "true_language" not in df.columns:
        print("[WARN] Missing 'true_language' column. Using empty values.")
        df["true_language"] = ""

    if "confidence" not in df.columns:
        print("[WARN] Missing 'confidence' column. Filling with NaN.")
        df["confidence"] = np.nan

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["true_language"] = df["true_language"].fillna("").astype(str).str.strip()
    df["predicted_code"] = (
        df["predicted_language"].fillna("").astype(str).str.split(":").str[0].str.strip()
    )
    df["correct"] = df["true_language"] == df["predicted_code"]
    return df


def plot_overall_accuracy_by_dataset(df):
    acc_table = df.groupby("dataset", dropna=False)["correct"].mean().reset_index()
    if acc_table.empty:
        save_no_data_plot("Overall Accuracy by Dataset", "overall_accuracy_by_dataset.png")
        return

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=acc_table,
        x="dataset",
        y="correct",
        hue="dataset",
        palette="Set2",
        dodge=False,
        legend=False,
    )
    plt.ylim(0, 1)
    plt.title("Overall Accuracy by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "overall_accuracy_by_dataset.png")
    plt.close()


def plot_per_language_accuracy_by_dataset(df):
    lang_acc = (
        df.groupby(["dataset", "true_language"], dropna=False)["correct"]
        .mean()
        .reset_index()
    )
    if lang_acc.empty:
        save_no_data_plot(
            "Per-Language Accuracy by Dataset", "per_language_accuracy_by_dataset.png"
        )
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=lang_acc,
        x="dataset",
        y="correct",
        hue="true_language",
        palette="Set1",
    )
    plt.ylim(0, 1)
    plt.title("Per-Language Accuracy by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.legend(title="True Language")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_language_accuracy_by_dataset.png")
    plt.close()


def plot_global_confusion_matrix(df, true_labels):
    df_cm = df[
        df["true_language"].astype(str).str.len().gt(0)
        & df["predicted_code"].astype(str).str.len().gt(0)
    ]
    if df_cm.empty or len(true_labels) == 0:
        save_no_data_plot("Global Confusion Matrix", "global_confusion_matrix.png")
        return

    cm = confusion_matrix(
        df_cm["true_language"],
        df_cm["predicted_code"],
        labels=true_labels,
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=true_labels,
        yticklabels=true_labels,
    )
    plt.title("Global Confusion Matrix")
    plt.xlabel("Predicted Language")
    plt.ylabel("True Language")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "global_confusion_matrix.png")
    plt.close()


def plot_dataset_confusion_matrices(df):
    for dataset_key, filename, dataset_title in REQUIRED_DATASETS:
        subset = df[
            (df["dataset"] == dataset_key)
            & df["true_language"].astype(str).str.len().gt(0)
            & df["predicted_code"].astype(str).str.len().gt(0)
        ]
        labels = sorted(subset["true_language"].dropna().astype(str).unique())

        if subset.empty or len(labels) == 0:
            save_no_data_plot(
                f"Confusion Matrix - {dataset_title}",
                filename,
            )
            continue

        cm = confusion_matrix(
            subset["true_language"],
            subset["predicted_code"],
            labels=labels,
        )

        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(f"Confusion Matrix - {dataset_title}")
        plt.xlabel("Predicted Language")
        plt.ylabel("True Language")
        plt.tight_layout()
        plt.savefig(OUT_DIR / filename)
        plt.close()


def plot_confidence_vs_correctness(df):
    df_plot = df.dropna(subset=["confidence"]).copy()
    if df_plot.empty:
        save_no_data_plot("Confidence vs Correctness", "confidence_vs_correctness.png")
        return

    df_plot["correct_label"] = np.where(df_plot["correct"], "Correct", "Incorrect")
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df_plot,
        x="correct_label",
        y="confidence",
        hue="correct_label",
        palette={"Incorrect": "red", "Correct": "green"},
        legend=False,
    )
    plt.title("Confidence vs Correctness")
    plt.xlabel("Prediction Correctness")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confidence_vs_correctness.png")
    plt.close()


def plot_routing_analysis(df):
    thresholds = np.arange(0.1, 1.01, 0.1)
    stats = []
    total = len(df)

    for threshold in thresholds:
        routed = df[df["confidence"] >= threshold]
        if len(routed) == 0 or total == 0:
            stats.append((threshold, np.nan, 0.0))
            continue
        stats.append((threshold, routed["correct"].mean(), len(routed) / total))

    routing_df = pd.DataFrame(stats, columns=["threshold", "accuracy", "coverage"])

    plt.figure(figsize=(8, 5))
    plt.plot(
        routing_df["threshold"],
        routing_df["accuracy"],
        marker="o",
        color="blue",
    )
    plt.title("Routing Accuracy vs Confidence Threshold")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "routing_accuracy_vs_threshold.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        routing_df["threshold"],
        routing_df["coverage"],
        marker="o",
        color="orange",
    )
    plt.title("Routing Coverage vs Confidence Threshold")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Coverage")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "routing_coverage_vs_threshold.png")
    plt.close()


def plot_top_out_of_set_predictions(df, true_labels):
    out_of_set = df[
        (df["correct"] == False)
        & (~df["predicted_code"].isin(true_labels))
        & df["predicted_code"].astype(str).str.len().gt(0)
    ]
    counts = out_of_set["predicted_code"].value_counts().head(10)

    if counts.empty:
        save_no_data_plot(
            "Top Out-of-Set Prediction Distribution",
            "top_out_of_set_predictions.png",
        )
        return

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar", color="orange")
    plt.title("Top Out-of-Set Prediction Distribution")
    plt.xlabel("Predicted Language Code")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_out_of_set_predictions.png")
    plt.close()


def plot_confidence_distribution_per_language(df):
    df_plot = df.dropna(subset=["confidence"]).copy()
    df_plot = df_plot[df_plot["true_language"].astype(str).str.len().gt(0)]
    if df_plot.empty:
        save_no_data_plot(
            "Confidence Distribution per Language",
            "confidence_distribution_per_language.png",
        )
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_plot,
        x="true_language",
        y="confidence",
        hue="true_language",
        palette="Set1",
        dodge=False,
        legend=False,
    )
    plt.title("Confidence Distribution per Language")
    plt.xlabel("True Language")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confidence_distribution_per_language.png")
    plt.close()


def main():
    sns.set_style("whitegrid")

    if not INPUT_CSV.exists():
        print(f"[WARN] Missing input CSV: {INPUT_CSV}")
        print("No evaluation plots generated.")
        return

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        print(f"[WARN] Input CSV is empty: {INPUT_CSV}")
        print("Generating placeholder plots.")
        df = prepare_dataframe(df)
    else:
        df = prepare_dataframe(df)

    true_labels = sorted(df["true_language"].dropna().astype(str).unique())

    plot_overall_accuracy_by_dataset(df)
    plot_per_language_accuracy_by_dataset(df)
    plot_global_confusion_matrix(df, true_labels)
    plot_dataset_confusion_matrices(df)
    plot_confidence_vs_correctness(df)
    plot_routing_analysis(df)
    plot_top_out_of_set_predictions(df, true_labels)
    plot_confidence_distribution_per_language(df)

    print("Multidataset evaluation plots generated successfully.")
    print(f"Saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
