import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def ensure_out(dirpath):
    os.makedirs(dirpath, exist_ok=True)
    return dirpath


def plot_confusion(df, out_dir, model_name=None, lang_order=None):
    if model_name is not None:
        df = df[df["model"] == model_name]
    labels = lang_order or sorted(df["true_lang"].unique())
    y_true = df["true_lang"]
    y_pred = df["pred_lang"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(max(6, len(labels)), max(5, len(labels) * 0.6)))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    title = f"Confusion Matrix"
    if model_name:
        title += f" - {model_name}"
    plt.title(title)
    plt.tight_layout()
    fname = f"confusion_{model_name or 'combined'}.png"
    plt.savefig(Path(out_dir) / fname)
    plt.close()


def plot_confidence_violin(df, out_dir, model_name=None, lang_order=None):
    if model_name is not None:
        df = df[df["model"] == model_name]
    df = df.copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["correct"] = df["pred_lang"] == df["true_lang"]
    labels = lang_order or sorted(df["true_lang"].unique())
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="true_lang", y="confidence", hue="correct", data=df, order=labels, split=True, inner="quartile")
    title = "Accuracy vs Confidence"
    if model_name:
        title += f" - {model_name}"
    plt.title(title)
    plt.xlabel("Language")
    plt.ylabel("Confidence (Top-1)")
    plt.tight_layout()
    fname = f"confidence_violin_{model_name or 'combined'}.png"
    plt.savefig(Path(out_dir) / fname)
    plt.close()


def plot_topk_accuracy(df, out_dir, model_name=None, lang_order=None):
    if model_name is not None:
        df = df[df["model"] == model_name]
    # Ensure topk column exists and compute top3 correctness
    df = df.copy()
    if "topk" in df.columns:
        df["correct_top3"] = df.apply(lambda r: r["true_lang"] in (r["topk"] or []), axis=1)
    else:
        df["correct_top3"] = df["pred_lang"] == df["true_lang"]
    df["correct_top1"] = df["pred_lang"] == df["true_lang"]
    labels = lang_order or sorted(df["true_lang"].unique())
    acc = (
        df.groupby("true_lang")
        .agg(top1_accuracy=("correct_top1", "mean"), top3_accuracy=("correct_top3", "mean"))
        .reindex(labels)
        .reset_index()
    )
    acc_m = acc.melt(id_vars=["true_lang"], value_vars=["top1_accuracy", "top3_accuracy"], var_name="metric", value_name="accuracy")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="true_lang", y="accuracy", hue="metric", data=acc_m, order=labels)
    plt.ylim(0, 1)
    title = "Top-1 vs Top-3 Accuracy per language"
    if model_name:
        title += f" - {model_name}"
    plt.title(title)
    plt.xlabel("Language")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    fname = f"topk_accuracy_{model_name or 'combined'}.png"
    plt.savefig(Path(out_dir) / fname)
    plt.close()


def summary_csv(df, out_dir):
    # per-model, per-language summary
    df = df.copy()
    df["confidence"] = pd.to_numeric(df.get("confidence", 0), errors="coerce").fillna(0.0)
    if "topk" in df.columns:
        df["correct_top3"] = df.apply(lambda r: r["true_lang"] in (r["topk"] or []), axis=1)
    else:
        df["correct_top3"] = df["pred_lang"] == df["true_lang"]
    df["correct_top1"] = df["pred_lang"] == df["true_lang"]
    agg = (
        df.groupby(["model", "true_lang"]).agg(
            n=("file", "count"),
            top1_accuracy=("correct_top1", "mean"),
            top3_accuracy=("correct_top3", "mean"),
            mean_confidence=("confidence", "mean")
        ).reset_index()
    )
    agg.to_csv(Path(out_dir) / "lid_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Plot LID evaluation results")
    parser.add_argument("--input", "-i", default="evaluation/lid_results_whisper_mms.json", help="Input JSON results file")
    parser.add_argument("--out", "-o", default="outputs/plots", help="Output directory for plots")
    parser.add_argument("--langs", "-l", default=None, help="Comma-separated language order (e.g. hi,bn,or,as,ta,te)")
    parser.add_argument("--models", "-m", default=None, help="Comma-separated models to plot (default: all)")
    args = parser.parse_args()

    df = load_results(args.input)
    out_dir = ensure_out(args.out)

    lang_order = None
    if args.langs:
        lang_order = [s.strip() for s in args.langs.split(",") if s.strip()]

    models = sorted(df["model"].unique())
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    # global/combined plots for best model and combined
    # combined overall plots
    plot_confusion(df, out_dir, model_name=None, lang_order=lang_order)
    plot_confidence_violin(df, out_dir, model_name=None, lang_order=lang_order)
    plot_topk_accuracy(df, out_dir, model_name=None, lang_order=lang_order)
    summary_csv(df, out_dir)

    # per-model plots
    for m in models:
        plot_confusion(df, out_dir, model_name=m, lang_order=lang_order)
        plot_confidence_violin(df, out_dir, model_name=m, lang_order=lang_order)
        plot_topk_accuracy(df, out_dir, model_name=m, lang_order=lang_order)

    print("Plots saved to:", out_dir)


if __name__ == "__main__":
    main()
