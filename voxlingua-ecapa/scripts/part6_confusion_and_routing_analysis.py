import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = "results/predictions_multidataset.csv"
OUT_DIR = Path("results/plots_multidataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LANGS = ["as", "bn"]

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_CSV)

# Normalize predicted labels: "as: Assamese" → "as"
df["predicted_code"] = df["predicted_language"].str.split(":").str[0].str.strip()

# Correctness
df["correct"] = df["true_language"] == df["predicted_code"]

# =========================================================
# ===================== TEST A ============================
# PER-DATASET CONFUSION MATRICES
# =========================================================

def plot_confusion_matrix(sub_df, title, filename):
    cm = confusion_matrix(
        sub_df["true_language"],
        sub_df["predicted_code"],
        labels=TARGET_LANGS
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_LANGS,
        yticklabels=TARGET_LANGS
    )
    plt.xlabel("Predicted Language")
    plt.ylabel("True Language")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()

# -------- CommonVoice --------
cv_df = df[
    (df["dataset"] == "commonvoice") &
    (df["true_language"].isin(TARGET_LANGS)) &
    (df["predicted_code"].isin(TARGET_LANGS))
]

plot_confusion_matrix(
    cv_df,
    "Confusion Matrix – CommonVoice (as vs bn)",
    "confusion_commonvoice.png"
)

# -------- VoxLingua107 --------
vx_df = df[
    (df["dataset"] == "voxlingua107") &
    (df["true_language"].isin(TARGET_LANGS)) &
    (df["predicted_code"].isin(TARGET_LANGS))
]

plot_confusion_matrix(
    vx_df,
    "Confusion Matrix – VoxLingua107 (as vs bn)",
    "confusion_voxlingua107.png"
)

# -------- IndicVoices (Oriya correctness only) --------
iv_df = df[df["dataset"] == "indicvoices"]

plt.figure(figsize=(4, 4))
sns.countplot(
    data=iv_df,
    x="correct",
    hue="correct",
    palette=["#d62728", "#2ca02c"],
    legend=False
)
plt.xticks([0, 1], ["Incorrect", "Correct"])
plt.title("IndicVoices (Oriya) – Correct vs Incorrect")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "indicvoices_correctness.png")
plt.close()

# =========================================================
# ===================== TEST B ============================
# CONFIDENCE-THRESHOLD ROUTING ANALYSIS
# =========================================================

thresholds = np.arange(0.1, 1.01, 0.1)
routing_stats = []

for t in thresholds:
    routed = df[df["confidence"] >= t]

    if len(routed) == 0:
        routing_stats.append((t, np.nan, 0))
        continue

    acc = routed["correct"].mean()
    coverage = len(routed) / len(df)

    routing_stats.append((t, acc, coverage))

routing_df = pd.DataFrame(
    routing_stats,
    columns=["threshold", "accuracy", "coverage"]
)

# -------- Accuracy vs Threshold --------
plt.figure(figsize=(6, 4))
plt.plot(
    routing_df["threshold"],
    routing_df["accuracy"],
    marker="o",
    color="#1f77b4"
)
plt.xlabel("Confidence Threshold")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Confidence Threshold (Routing)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "routing_accuracy_vs_threshold.png")
plt.close()

# -------- Coverage vs Threshold --------
plt.figure(figsize=(6, 4))
plt.plot(
    routing_df["threshold"],
    routing_df["coverage"],
    marker="o",
    color="#ff7f0e"
)
plt.xlabel("Confidence Threshold")
plt.ylabel("Fraction of Samples Routed")
plt.title("Coverage vs Confidence Threshold")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "routing_coverage_vs_threshold.png")
plt.close()

print("Per-dataset confusion matrices and routing analysis plots generated successfully.")
