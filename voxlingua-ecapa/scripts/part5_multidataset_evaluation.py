import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("results/predictions_multidataset.csv")

# Normalize predicted labels: "as: Assamese" → "as"
df["predicted_code"] = df["predicted_language"].str.split(":").str[0].str.strip()

# Correctness
df["correct"] = df["true_language"] == df["predicted_code"]

OUT_DIR = Path("results/plots_multidataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. OVERALL ACCURACY BY DATASET
# =========================================================
acc_table = (
    df.groupby("dataset")["correct"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(6,4))
sns.barplot(data=acc_table, x="dataset", y="correct", palette="Set2")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.title("Overall Accuracy by Dataset")
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_by_dataset.png")
plt.close()

# =========================================================
# 2. PER-LANGUAGE ACCURACY BY DATASET
# =========================================================
lang_acc = (
    df.groupby(["dataset", "true_language"])["correct"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(7,4))
sns.barplot(
    data=lang_acc,
    x="dataset",
    y="correct",
    hue="true_language",
    palette="Set1"
)
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.title("Per-Language Accuracy by Dataset")
plt.legend(title="Language")
plt.tight_layout()
plt.savefig(OUT_DIR / "per_language_accuracy_by_dataset.png")
plt.close()

# =========================================================
# 3. CONFIDENCE vs CORRECTNESS (GLOBAL)
# =========================================================
plt.figure(figsize=(6,4))
sns.boxplot(
    data=df,
    x="correct",
    y="confidence",
    palette=["#d62728", "#2ca02c"]
)
plt.xticks([0,1], ["Incorrect", "Correct"])
plt.title("Confidence vs Correctness (All Datasets)")
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_vs_correctness.png")
plt.close()

# =========================================================
# 4. TOP CONFUSIONS (OUT-OF-SET)
# =========================================================
top_preds = (
    df[df["correct"] == False]["predicted_code"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(8,4))
top_preds.plot(kind="bar", color="#ff7f0e")
plt.ylabel("Count")
plt.title("Top Out-of-Set Predictions (All Datasets)")
plt.tight_layout()
plt.savefig(OUT_DIR / "top_out_of_set_predictions.png")
plt.close()

# =========================================================
# 5. CONFUSION MATRIX (CommonVoice only, as/bn)
# =========================================================
cv_df = df[
    (df["dataset"] == "commonvoice") &
    (df["true_language"].isin(["as","bn"])) &
    (df["predicted_code"].isin(["as","bn"]))
]

cm = confusion_matrix(
    cv_df["true_language"],
    cv_df["predicted_code"],
    labels=["as","bn"]
)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["as","bn"],
    yticklabels=["as","bn"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – CommonVoice (as vs bn)")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_commonvoice.png")
plt.close()

print("All multidataset evaluation plots generated successfully.")
