import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path
import json

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("results/predictions.csv")

# =========================================================
# NORMALIZE PREDICTED LABELS
# predicted_language format: "as: Assamese"
# =========================================================
df["predicted_code"] = df["predicted_language"].str.split(":").str[0].str.strip()

# =========================================================
# MAP CODES TO DATASET LABELS
# =========================================================
LANG_MAP = {
    "as": "as",   # Assamese
    "bn": "bn",   # Bengali
}

df["predicted_mapped"] = df["predicted_code"].map(LANG_MAP)

# Keep only samples relevant to our evaluation
df_eval = df.dropna(subset=["predicted_mapped"])

print("Evaluation samples:", len(df_eval))

# =========================================================
# METRICS
# =========================================================
accuracy = accuracy_score(
    df_eval["true_language"],
    df_eval["predicted_mapped"]
)

Path("results/plots").mkdir(parents=True, exist_ok=True)

with open("results/metrics.json", "w") as f:
    json.dump(
        {
            "accuracy": accuracy,
            "total_samples": int(len(df_eval))
        },
        f,
        indent=2
    )

print("Accuracy:", accuracy)

# =========================================================
# CONFUSION MATRIX
# =========================================================
labels = ["as", "bn"]
cm = confusion_matrix(
    df_eval["true_language"],
    df_eval["predicted_mapped"],
    labels=labels
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels
)
plt.title("Confusion Matrix – VoxLingua107 ECAPA (2000 samples)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/plots/confusion_matrix.png")
plt.close()

# =========================================================
# CONFIDENCE DISTRIBUTION
# =========================================================
plt.figure(figsize=(8, 5))
sns.boxplot(
    x="true_language",
    y="confidence",
    data=df_eval
)
plt.title("Confidence Distribution – VoxLingua107 ECAPA")
plt.tight_layout()
plt.savefig("results/plots/confidence_distribution.png")
plt.close()

print("Plots regenerated successfully")
