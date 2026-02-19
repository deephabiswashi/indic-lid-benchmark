#!/usr/bin/env python3
"""
Compare Indian-language prediction quality across model outputs with compact charts.

Inputs (defaults point to your current files):
- Facebook MMS LID 1024/lid_results_whisper_mms.json
- FacebookMMS_1BASR/results/all_predictions.csv
- INDIC-LID/results/predictions_multidataset.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


LANG_MAP = {
    "as": "as",
    "assamese": "as",
    "bn": "bn",
    "bengali": "bn",
    "hi": "hi",
    "hindi": "hi",
    "or": "or",
    "oriya": "or",
    "odia": "or",
    "ory": "or",
    "ta": "ta",
    "tamil": "ta",
    "te": "te",
    "telugu": "te",
    "telegu": "te",
}
DEFAULT_FOCUS_LANGS = ["as", "bn", "hi", "or", "ta", "te"]
MODEL_ID_ALIASES = {
    "mms-lid": "facebook/mms-lid-1024",
    "facebook-mms-lid": "facebook/mms-lid-1024",
    "facebook/mms-lid-1024": "facebook/mms-lid-1024",
    "indic-lid": "speechbrain/lang-id-voxlingua107-ecapa",
    "speechbrain/lang-id-voxlingua107-ecapa": "speechbrain/lang-id-voxlingua107-ecapa",
    "whisper-small": "openai/whisper-small",
    "openai/whisper-small": "openai/whisper-small",
    "facebookmms-1basr": "facebook/mms-1b-all",
    "facebook/mms-1b-all": "facebook/mms-1b-all",
}
MODEL_LABELS = {
    "facebook/mms-lid-1024": "Facebook MMS-LID-1024",
    "speechbrain/lang-id-voxlingua107-ecapa": "SpeechBrain ECAPA LID",
    "openai/whisper-small": "OpenAI Whisper-small",
    "facebook/mms-1b-all": "Facebook MMS-1B-All",
}
DEFAULT_MODEL_ORDER = [
    "facebook/mms-lid-1024",
    "speechbrain/lang-id-voxlingua107-ecapa",
    "openai/whisper-small",
    "facebook/mms-1b-all",
]


def normalize_lang(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text or text == "nan":
        return ""
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    text = re.sub(r"\s+", " ", text)
    return LANG_MAP.get(text, text)


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def canonical_model_id(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "unknown-model"
    key = re.sub(r"\s+", "-", text.lower())
    return MODEL_ID_ALIASES.get(key, text)


def model_label(model_id: object) -> str:
    text = "" if model_id is None else str(model_id)
    return MODEL_LABELS.get(text, text)


def model_order(model_ids: list[str]) -> list[str]:
    rank = {name: idx for idx, name in enumerate(DEFAULT_MODEL_ORDER)}
    return sorted(model_ids, key=lambda x: (rank.get(x, 10_000), x))


def resolve_input_path(path: Path, candidates: list[Path], label: str) -> Path:
    if path.exists():
        return path
    for candidate in candidates:
        if candidate.exists():
            print(f"[INFO] Using detected {label} path: {candidate}")
            return candidate
    return path


def sample_id(path_value: object, true_lang: str) -> str:
    path_str = "" if path_value is None else str(path_value)
    filename = path_str.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
    return f"{true_lang}::{filename}"


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "true_lang",
            "pred_lang",
            "confidence",
            "sample_id",
            "label_quality",
            "source",
        ]
    )


def load_whisper_mms_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing JSON file: {path}")
        return empty_frame()

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        true_lang = normalize_lang(row.get("true_lang"))
        pred_lang = normalize_lang(row.get("pred_lang"))
        model = canonical_model_id(row.get("model") or "Whisper-MMS")
        rows.append(
            {
                "model": model,
                "true_lang": true_lang,
                "pred_lang": pred_lang,
                "confidence": to_float(row.get("confidence")),
                "sample_id": sample_id(row.get("file"), true_lang),
                "label_quality": "direct",
                "source": str(path),
            }
        )

    return pd.DataFrame(rows)


def load_indic_lid_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing INDIC-LID CSV: {path}")
        return empty_frame()

    df = pd.read_csv(path)
    if df.empty:
        return empty_frame()

    true_series = (
        df["true_language"]
        if "true_language" in df.columns
        else df.get("source_dataset_language", "")
    )
    pred_series = (
        df["predicted_language"]
        if "predicted_language" in df.columns
        else df.get("pred_lang", "")
    )

    true_norm = pd.Series(true_series).map(normalize_lang)
    pred_norm = pd.Series(pred_series).map(normalize_lang)

    out = pd.DataFrame(
        {
            "model": "speechbrain/lang-id-voxlingua107-ecapa",
            "true_lang": true_norm,
            "pred_lang": pred_norm,
            "confidence": pd.to_numeric(df.get("confidence"), errors="coerce"),
            "sample_id": [
                sample_id(p, t)
                for p, t in zip(df.get("audio_path", ""), true_norm, strict=False)
            ],
            "label_quality": "direct",
            "source": str(path),
        }
    )
    return out


def load_mms_asr_csv_as_proxy(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing FacebookMMS_1BASR CSV: {path}")
        return empty_frame()

    df = pd.read_csv(path)
    if df.empty:
        return empty_frame()

    if "lang" not in df.columns:
        print("[WARN] all_predictions.csv has no 'lang' column. Skipping proxy model.")
        return empty_frame()

    pred_norm = df["lang"].map(normalize_lang)
    true_norm = pred_norm.copy()

    out = pd.DataFrame(
        {
            "model": "facebook/mms-1b-all",
            "true_lang": true_norm,
            "pred_lang": pred_norm,
            "confidence": np.nan,
            "sample_id": [
                sample_id(p, t) for p, t in zip(df.get("file", ""), true_norm, strict=False)
            ],
            "label_quality": "proxy",
            "source": str(path),
        }
    )
    print(
        "[INFO] Loaded facebook/mms-1b-all as proxy labels from 'lang' column "
        "(no explicit predicted_language field in CSV)."
    )
    return out


def clean_eval_frame(df: pd.DataFrame, focus_langs: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    data = df.copy()
    data["model"] = data["model"].map(canonical_model_id)
    data["true_lang"] = data["true_lang"].map(normalize_lang)
    data["pred_lang"] = data["pred_lang"].map(normalize_lang)

    data = data[
        data["true_lang"].astype(str).str.len().gt(0)
        & data["pred_lang"].astype(str).str.len().gt(0)
    ].copy()
    data = data[data["true_lang"].isin(focus_langs)].copy()

    dup_mask = data.duplicated(subset=["model", "sample_id"], keep="first")
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        print(f"[WARN] Dropping duplicate rows by (model, sample_id): {dup_count}")
        data = data[~dup_mask].copy()

    data["correct"] = data["true_lang"] == data["pred_lang"]
    data["model_label"] = data["model"].map(model_label)
    return data


def compute_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "model_label",
                "samples",
                "overall_accuracy",
                "macro_accuracy",
                "mean_confidence",
                "label_quality",
            ]
        )

    per_lang = (
        df.groupby(["model", "true_lang"], dropna=False)["correct"].mean().reset_index()
    )
    macro = per_lang.groupby("model", dropna=False)["correct"].mean().rename("macro_accuracy")

    summary = (
        df.groupby("model", dropna=False)
        .agg(
            samples=("correct", "size"),
            overall_accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            label_quality=("label_quality", lambda s: ",".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
    )
    summary = summary.merge(macro.reset_index(), on="model", how="left")
    summary["model_label"] = summary["model"].map(model_label)
    ordered = model_order(summary["model"].astype(str).tolist())
    summary["__order"] = pd.Categorical(summary["model"], categories=ordered, ordered=True)
    summary = summary.sort_values(["__order", "overall_accuracy"], ascending=[True, False]).drop(
        columns=["__order"]
    )
    return summary


def compute_per_language_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["model", "model_label", "true_lang", "accuracy", "samples"])

    out = (
        df.groupby(["model", "true_lang"], dropna=False)
        .agg(accuracy=("correct", "mean"), samples=("correct", "size"))
        .reset_index()
    )
    out["model_label"] = out["model"].map(model_label)
    return out


def compute_pairwise_agreement(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = model_order(sorted(df["model"].unique().tolist()))
    if not models:
        return pd.DataFrame(), pd.DataFrame()

    pivot = df.pivot_table(index="sample_id", columns="model", values="pred_lang", aggfunc="first")

    long_rows = []
    matrix = pd.DataFrame(np.nan, index=models, columns=models)
    for model in models:
        matrix.loc[model, model] = 1.0

    for left, right in itertools.combinations(models, 2):
        pair = pivot[[left, right]].dropna()
        n = len(pair)
        agreement = float("nan") if n == 0 else float((pair[left] == pair[right]).mean())
        long_rows.append(
            {
                "model_a": left,
                "model_a_label": model_label(left),
                "model_b": right,
                "model_b_label": model_label(right),
                "shared_samples": n,
                "agreement": agreement,
            }
        )
        matrix.loc[left, right] = agreement
        matrix.loc[right, left] = agreement

    return pd.DataFrame(long_rows), matrix


def save_placeholder(path: Path, title: str, text: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_overview(summary: pd.DataFrame, out_dir: Path) -> None:
    path = out_dir / "comparison_overview.png"
    if summary.empty:
        save_placeholder(path, "Model Comparison Overview", "No data")
        return

    order = summary["model"].tolist()
    label_order = [model_label(x) for x in order]
    plot_df = summary.copy()
    plot_df["model_label"] = plot_df["model"].map(model_label)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))

    sns.barplot(
        data=plot_df,
        x="model_label",
        y="overall_accuracy",
        hue="model_label",
        order=label_order,
        ax=axes[0, 0],
        palette="Set2",
        legend=False,
    )
    axes[0, 0].set_title("Overall Accuracy")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis="x", rotation=25)

    sns.barplot(
        data=plot_df,
        x="model_label",
        y="macro_accuracy",
        hue="model_label",
        order=label_order,
        ax=axes[0, 1],
        palette="Set2",
        legend=False,
    )
    axes[0, 1].set_title("Macro Accuracy (Mean Across Languages)")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis="x", rotation=25)

    sns.barplot(
        data=plot_df,
        x="model_label",
        y="samples",
        hue="model_label",
        order=label_order,
        ax=axes[1, 0],
        palette="Set2",
        legend=False,
    )
    axes[1, 0].set_title("Samples Evaluated")
    axes[1, 0].tick_params(axis="x", rotation=25)

    sns.barplot(
        data=plot_df,
        x="model_label",
        y="mean_confidence",
        hue="model_label",
        order=label_order,
        ax=axes[1, 1],
        palette="Set2",
        legend=False,
    )
    axes[1, 1].set_title("Mean Confidence")
    axes[1, 1].tick_params(axis="x", rotation=25)

    for ax in axes.ravel():
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=16, labelsize=9)

    fig.suptitle("Indian Language Model Comparison (Compact Overview)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_per_language_heatmap(
    per_lang: pd.DataFrame,
    summary: pd.DataFrame,
    focus_langs: list[str],
    out_dir: Path,
) -> None:
    path = out_dir / "per_language_accuracy_heatmap.png"
    if per_lang.empty:
        save_placeholder(path, "Per-Language Accuracy", "No data")
        return

    order = summary["model"].tolist() if not summary.empty else model_order(sorted(per_lang["model"].unique()))
    table = (
        per_lang.pivot(index="model", columns="true_lang", values="accuracy")
        .reindex(index=order, columns=focus_langs)
    )
    table.index = [model_label(x) for x in table.index]

    fig, ax = plt.subplots(figsize=(8.5, 3.5 + 0.45 * max(1, len(order))))
    sns.heatmap(
        table,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Per-Language Accuracy by Model")
    ax.set_xlabel("Indian Language")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_pairwise_agreement_heatmap(matrix: pd.DataFrame, out_dir: Path) -> None:
    path = out_dir / "pairwise_agreement_heatmap.png"
    if matrix.empty:
        save_placeholder(path, "Pairwise Agreement", "No comparable shared samples")
        return

    plot_matrix = matrix.copy()
    plot_matrix.index = [model_label(x) for x in plot_matrix.index]
    plot_matrix.columns = [model_label(x) for x in plot_matrix.columns]

    fig, ax = plt.subplots(figsize=(5.8 + 0.55 * len(plot_matrix), 4.8 + 0.35 * len(plot_matrix)))
    sns.heatmap(
        plot_matrix,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Pairwise Prediction Agreement")
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_confusion_panel(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    focus_langs: list[str],
    out_dir: Path,
) -> None:
    path = out_dir / "confusion_matrices_by_model.png"
    if df.empty:
        save_placeholder(path, "Confusion Matrices", "No data")
        return

    models = summary["model"].tolist() if not summary.empty else model_order(sorted(df["model"].unique()))
    labels = focus_langs + ["other"]

    n_models = len(models)
    ncols = 3 if n_models >= 3 else max(1, n_models)
    nrows = int(math.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.9 * nrows))
    axes = np.array(axes).reshape(-1)

    for idx, model in enumerate(models):
        ax = axes[idx]
        subset = df[df["model"] == model].copy()
        subset["pred_cm"] = subset["pred_lang"].where(subset["pred_lang"].isin(focus_langs), "other")

        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        cm = confusion_matrix(
            subset["true_lang"],
            subset["pred_cm"],
            labels=labels,
            normalize="true",
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            linewidths=0.2,
            linecolor="white",
            ax=ax,
        )
        model_acc = float(subset["correct"].mean()) if len(subset) > 0 else float("nan")
        ax.set_title(f"{model_label(model)} (n={len(subset)}, acc={model_acc:.2f})", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for idx in range(n_models, len(axes)):
        axes[idx].set_axis_off()

    fig.suptitle("Normalized Confusion Matrices by Model", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare Indian-language prediction outputs across models."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("Facebook MMS LID 1024/lid_results_whisper_mms.json"),
        help="Path to JSON with whisper/mms LID predictions.",
    )
    parser.add_argument(
        "--asr-csv",
        type=Path,
        default=Path("FacebookMMS_1B ASR/results/all_predictions.csv"),
        help="Path to facebook/mms-1b-all result CSV (optional proxy input).",
    )
    parser.add_argument(
        "--indic-csv",
        type=Path,
        default=Path("voxlingua-ecapa/results/predictions_multidataset.csv"),
        help="Path to speechbrain/indic-lid predictions_multidataset.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("model_comparison_results"),
        help="Output directory for charts and CSV summaries.",
    )
    parser.add_argument(
        "--focus-langs",
        type=str,
        default=",".join(DEFAULT_FOCUS_LANGS),
        help="Comma-separated language codes to include as true labels.",
    )
    parser.add_argument(
        "--include-proxy",
        action="store_true",
        help="Include facebook/mms-1b-all as proxy labels from all_predictions.csv.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Comma-separated model IDs/aliases to keep, for example: "
            "facebook/mms-lid-1024,speechbrain/lang-id-voxlingua107-ecapa,facebook/mms-1b-all"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        },
    )

    focus_langs = [normalize_lang(x) for x in args.focus_langs.split(",") if x.strip()]
    if not focus_langs:
        focus_langs = DEFAULT_FOCUS_LANGS

    args.json_path = resolve_input_path(
        args.json_path,
        [Path("Facebook MMS LID 1024/lid_results_whisper_mms.json")],
        "JSON",
    )
    args.indic_csv = resolve_input_path(
        args.indic_csv,
        [
            Path("voxlingua-ecapa/results/predictions_multidataset.csv"),
            Path("INDIC-LID/results/predictions_multidataset.csv"),
        ],
        "INDIC-LID CSV",
    )
    args.asr_csv = resolve_input_path(
        args.asr_csv,
        [
            Path("FacebookMMS_1B ASR/results/all_predictions.csv"),
            Path("FacebookMMS_1BASR/results/all_predictions.csv"),
        ],
        "MMS-1B-ASR CSV",
    )

    frames = [
        load_whisper_mms_json(args.json_path),
        load_indic_lid_csv(args.indic_csv),
    ]
    if args.include_proxy:
        frames.append(load_mms_asr_csv_as_proxy(args.asr_csv))
    else:
        print("[INFO] Skipping proxy model. Pass --include-proxy to add facebook/mms-1b-all.")

    valid_frames = [frame for frame in frames if not frame.empty]
    raw = pd.concat(valid_frames, ignore_index=True) if valid_frames else empty_frame()
    evaluated = clean_eval_frame(raw, focus_langs)

    if args.models.strip():
        selected_models = [
            canonical_model_id(item) for item in args.models.split(",") if item.strip()
        ]
        evaluated = evaluated[evaluated["model"].isin(selected_models)].copy()
        print(
            "[INFO] Keeping models: "
            + ", ".join(model_label(model_id) for model_id in model_order(selected_models))
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if evaluated.empty:
        print("[WARN] No valid rows after normalization/filtering.")
        save_placeholder(args.out_dir / "comparison_overview.png", "Overview", "No valid rows")
        return

    summary = compute_model_summary(evaluated)
    per_lang = compute_per_language_summary(evaluated)
    pairwise_long, pairwise_matrix = compute_pairwise_agreement(evaluated)
    pairwise_matrix_labeled = pairwise_matrix.copy()
    pairwise_matrix_labeled.index = [model_label(x) for x in pairwise_matrix_labeled.index]
    pairwise_matrix_labeled.columns = [model_label(x) for x in pairwise_matrix_labeled.columns]

    summary.to_csv(args.out_dir / "model_summary.csv", index=False)
    per_lang.to_csv(args.out_dir / "per_language_summary.csv", index=False)
    pairwise_long.to_csv(args.out_dir / "pairwise_agreement_long.csv", index=False)
    pairwise_matrix.to_csv(args.out_dir / "pairwise_agreement_matrix.csv")
    pairwise_matrix_labeled.to_csv(args.out_dir / "pairwise_agreement_matrix_labeled.csv")
    evaluated.to_csv(args.out_dir / "normalized_predictions.csv", index=False)

    plot_overview(summary, args.out_dir)
    plot_per_language_heatmap(per_lang, summary, focus_langs, args.out_dir)
    plot_pairwise_agreement_heatmap(pairwise_matrix, args.out_dir)
    plot_confusion_panel(evaluated, summary, focus_langs, args.out_dir)

    print("Model comparison finished.")
    print(f"Output directory: {args.out_dir.resolve()}")
    print(
        "Saved CSVs: model_summary, per_language_summary, pairwise_agreement, "
        "pairwise_agreement_matrix_labeled, normalized_predictions"
    )
    print("Saved plots: comparison_overview, per_language_accuracy_heatmap, pairwise_agreement_heatmap, confusion_matrices_by_model")


if __name__ == "__main__":
    main()

