import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path
import json

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = "results/predictions.csv"
OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LANGS = ["as", "bn"]

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_CSV)

# Normalize predicted labels: "as: Assamese" → "as"
df["predicted_code"] = df["predicted_language"].str.split(":").str[0].str.strip()

# Correctness flag
df["correct"] = df["true_language"] == df["predicted_code"]

# In-set predictions only
df_in = df[df["predicted_code"].isin(TARGET_LANGS)]

print("Total samples:", len(df))
print("In-set samples:", len(df_in))

# =========================================================
# 1. OVERALL ACCURACY (IN-SET)
# =========================================================
overall_acc = accuracy_score(
    df_in["true_language"],
    df_in["predicted_code"]
)

with open("results/metrics.json", "w") as f:
    json.dump(
        {
            "overall_accuracy": overall_acc,
            "total_samples": int(len(df)),
            "in_set_samples": int(len(df_in)),
        },
        f,
        indent=2
    )

print("Overall in-set accuracy:", overall_acc)

# =========================================================
# 2. CONFUSION MATRIX (as vs bn)
# =========================================================
cm = confusion_matrix(
    df_in["true_language"],
    df_in["predicted_code"],
    labels=TARGET_LANGS
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=TARGET_LANGS,
    yticklabels=TARGET_LANGS
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – VoxLingua107 ECAPA (2000 samples)")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png")
plt.close()

# =========================================================
# 3. PER-LANGUAGE ACCURACY
# =========================================================
lang_acc = df_in.groupby("true_language")["correct"].mean()

plt.figure(figsize=(5, 4))
lang_acc.plot(kind="bar")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Per-Language Accuracy")
plt.tight_layout()
plt.savefig(OUT_DIR / "per_language_accuracy.png")
plt.close()

# =========================================================
# 4. TOP PREDICTED LANGUAGES (OUT-OF-SET ANALYSIS)
# =========================================================
top_preds = df["predicted_code"].value_counts().head(10)

plt.figure(figsize=(8, 5))
top_preds.plot(kind="bar")
plt.ylabel("Count")
plt.xlabel("Language Code")
plt.title("Top Predicted Languages (VoxLingua107)")
plt.tight_layout()
plt.savefig(OUT_DIR / "top_predicted_languages.png")
plt.close()

# =========================================================
# 5. CONFIDENCE vs CORRECTNESS
# =========================================================
plt.figure(figsize=(6, 5))
sns.boxplot(x=df["correct"], y=df["confidence"])
plt.xticks([0, 1], ["Incorrect", "Correct"])
plt.ylabel("Confidence")
plt.title("Confidence vs Prediction Correctness")
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_vs_correctness.png")
plt.close()

# =========================================================
# 6. ACCURACY vs CONFIDENCE THRESHOLD
# =========================================================
thresholds = np.arange(0.1, 1.01, 0.1)
accs = []

for t in thresholds:
    subset = df_in[df_in["confidence"] >= t]
    if len(subset) > 0:
        accs.append(subset["correct"].mean())
    else:
        accs.append(np.nan)

plt.figure(figsize=(7, 5))
plt.plot(thresholds, accs, marker="o")
plt.xlabel("Confidence Threshold")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Confidence Threshold")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_vs_confidence.png")
plt.close()

# =========================================================
# 7. CONFUSION DESTINATION ANALYSIS
# =========================================================
confusion_dest = (
    df[df["true_language"] != df["predicted_code"]]
    .groupby(["true_language", "predicted_code"])
    .size()
    .unstack(fill_value=0)
)

confusion_dest.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5)
)
plt.ylabel("Count")
plt.title("Confusion Destination by True Language")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_distribution.png")
plt.close()

# =========================================================
# 8. CONFIDENCE DISTRIBUTION (as vs bn)
# =========================================================
plt.figure(figsize=(8, 5))
sns.boxplot(
    x="true_language",
    y="confidence",
    data=df_in
)
plt.title("Confidence Distribution by Language")
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_distribution.png")
plt.close()

print("All evaluation plots generated successfully.")
