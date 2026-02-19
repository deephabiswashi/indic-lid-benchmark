# PROJECT REQUIREMENTS DOCUMENT (PRD)

## 1. Project Overview
**Project Name:** Indic LID (Language Identification for Low-Resource Indian Languages)

**Aim:** To develop and benchmark a robust Language Identification (LID) system capable of recognizing and distinguishing between low-resource Indian languages, specifically focusing on **Assamese (as)**, **Bengali (bn)**, and **Odia (or)**.

**Context:**
LID is a critical front-end component for multilingual speech systems (ASR, TTS). Low-resource languages often suffer from poor detection rates due to data scarcity and acoustic similarities. This project evaluates current state-of-the-art models to identify gaps and best performers for these specific regional languages.

## 2. Objectives
1.  **Compare Models:** Benchmark 3 diverse speech models.
2.  **Evaluate on Real Data:** Use 5 major datasets (Indic Voices, Voxlingua 107, etc.) to ensure real-world applicability.
3.  **Analyze Performance:** Generate quantitative metrics (Accuracy, Confusion Matrices) and visual insights (Heatmaps) to understand model behavior.
4.  **Identify Weaknesses:** Specifically pinpoint which languages and models require further tuning (e.g., failure modes in Odia detection).

## 3. Data Strategy
The project leverages a varying mix of open-source datasets to cover the target languages.

| Dataset | URL | Description |
| :--- | :--- | :--- |
| **Indic Voices** | [Link](https://huggingface.co/datasets/ai4bharat/IndicVoices) | Large-scale, diverse Indian language dataset. |
| **VoxLingua107** | [Link](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/) | 107 languages, web-scraped data. |
| **Google FLEURS** | [Link](https://huggingface.co/datasets/google/fleurs) | Few-shot evaluation dataset. |
| **Mozilla Common Voice** | [Link](https://datacollective.mozillafoundation.org/datasets) | Crowdsourced voice data. |
| **AIKOSH** | [Link](https://aikosh.indiaai.gov.in/home/datasets/all) | Government of India initiative dataset. |

## 4. Technical Architecture & Models

### A. Facebook MMS-LID (1024)
*   **Type:** End-to-end LID model.
*   **Source:** [HuggingFace](https://huggingface.co/facebook/mms-lid-1024)
*   **Role:** Primary candidate for massive multilingual support.
*   **Performance:** Shown to be highly accurate (1.0 on test subset) and robust.

### B. Facebook MMS-1B-All
*   **Type:** Massive ASR (Automatic Speech Recognition) model.
*   **Source:** [HuggingFace](https://huggingface.co/facebook/mms-1b-all)
*   **Role:** Used as a high-quality "Proxy" or baseline. Since it is a larger, more powerful model, its language detection (implicit or explicit) serves as a ground truth or "upper bound" for performance comparison.

### C. SpeechBrain ECAPA-TDNN (VoxLingua107)
*   **Type:** TDNN (Time Delay Neural Network) based embedding model.
*   **Source:** [HuggingFace](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
*   **Role:** A lightweight, popular embedding model often used for speaker and language verification.
*   **Performance:** Good on most Indic languages but failed on Odia in this specific evaluation.

## 5. Implementation Details
The core logic is encapsulated in `evaluate_indian_language_model_comparison.py`.

*   **Workflow:**
    1.  **Input Parsing:** Reads predictions from JSON (MMS) and CSV (SpeechBrain/ASR) files.
    2.  **Normalization:** standardized language codes (e.g., "assamese" -> "as", "oriya" -> "or") using a `LANG_MAP` dictionary.
    3.  **Filtering:** Focuses analysis on target languages: `['as', 'bn', 'hi', 'or', 'ta', 'te']`.
    4.  **Metrics Calculation:** Computes Overall Accuracy, Macro Accuracy, and Per-Language Accuracy.
    5.  **Visualization:** Uses `matplotlib` and `seaborn` to generate:
        *   Bar charts for overall comparison.
        *   Heatmaps for pairwise agreement and per-language accuracy.
        *   Confusion matrices for error analysis.

## 6. Findings & Results

Based on the generated artifacts in `model_comparison_results/`:

1.  **ECAPA-TDNN Failure on Odia:**
    *   The SpeechBrain model achieved **0.0% accuracy on Odia**. This is a critical critical finding, suggesting either a mismatch in language codes/labels (e.g., model outputting 'or' vs 'ory') or a fundamental lack of training data for Odia in the specific version of the model used.
    *   *Action Item:* Investigate label mapping for SpeechBrain output or retrain/finetune for Odia.

2.  **High Performance of MMS:**
    *   The Facebook MMS models (both LID and 1B-ASR) demonstrated near-perfect accuracy on the tested samples.
    *   This suggests MMS is currently the superior choice for deploying a production LID system for these languages.

3.  **Language Confusion:**
    *   Confusion matrices indicate distinct acoustic signatures for Assamese and Bengali are generally well-separated by the models, except for the specific failure mode of ECAPA on Odia.

## 7. Operational Recommendations
*   **Deploy MMS-LID:** For immediate production needs, `facebook/mms-lid-1024` is the recommended model.
*   **Debug ECAPA:** If a lightweight model is required, the Odia failure in ECAPA-TDNN must be root-caused (likely a mapping issue or specific domain mismatch) before deployment.
*   **Data Augmentation:** Continue sourcing diverse data from AIKOSH and Indic Voices to stress-test the MMS model further.
