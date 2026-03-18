# ethix-ai-evaluation-platform
=======
# ⚖️ ETHIX — AI Evaluation and Monitoring Platform

ETHIX is a production-ready Streamlit application that evaluates machine learning
models not only on predictive performance, but also on **fairness**, **bias
detection**, and **ethical responsibility**.

---

## 🚀 Quick Start

```bash
# 1. Clone or download the project
cd ethix-ai-fairness-project

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/streamlit_app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
ethix-ai-fairness-project/
│
├── app/
│   └── streamlit_app.py          # Main Streamlit entry point
│
├── src/
│   ├── data_loader.py            # CSV loading, column type detection, validation
│   ├── preprocessing.py          # Cleaning, encoding, scaling, train/test split
│   ├── eda.py                    # All EDA charts (returns Matplotlib Figures)
│   ├── model.py                  # Training, evaluation, confusion matrix, ROC
│   ├── fairness.py               # SPD, DIR, EOD, individual fairness metrics + plots
│   └── ethical_score.py          # Composite ethical score calculator + gauge chart
│
├── data/
│   └── sample_dataset.csv        # Adult Income sample (50 rows, for quick demo)
│
├── notebooks/
│   └── ethix_analysis.ipynb      # Jupyter notebook for offline exploration
│
├── requirements.txt
└── README.md
```

---

## 📊 Application Workflow

| Step | Description |
|------|-------------|
| 1 | Upload a CSV dataset (or use the built-in sample) |
| 2 | Auto-detect column types and identifier columns |
| 3 | Select target column and sensitive attribute |
| 4 | Automatic EDA: distributions, correlations, missing values |
| 5 | Preprocessing: imputation, encoding, scaling, splitting |
| 6 | Train Logistic Regression baseline model |
| 7 | Evaluate: accuracy, precision, recall, F1, ROC-AUC |
| 8 | Fairness: SPD, DIR, EOD, individual fairness |
| 9 | Bias mitigation: re-train without sensitive attribute |
| 10 | Ethical Score: weighted combination of performance + fairness |

---

## ⚖️ Fairness Metrics Explained

### Statistical Parity Difference (SPD)
Difference in positive prediction rates between groups.
- **Ideal value**: 0
- **Fair range**: −0.1 to +0.1

### Disparate Impact Ratio (DIR)
Ratio of positive prediction rates (unprivileged ÷ privileged).
- **Ideal value**: 1.0
- **Fair range**: ≥ 0.8 (the "80% rule")

### Equal Opportunity Difference (EOD)
Difference in True Positive Rates between groups.
- **Ideal value**: 0
- **Fair range**: −0.1 to +0.1

### Individual Fairness Score
Fraction of similar individual pairs that receive the same prediction.
- **Ideal value**: 1.0
- **Fair range**: ≥ 0.8

---

## 🏆 Ethical Score

```
Ethical Score = 0.35 × Accuracy
              + 0.25 × SPD score
              + 0.25 × DIR score
              + 0.15 × EOD score
```

| Score | Grade | Meaning |
|-------|-------|---------|
| ≥ 0.85 | Excellent | Accurate and fair |
| ≥ 0.70 | Good | Minor fairness concerns |
| ≥ 0.55 | Fair | Investigate fairness issues |
| ≥ 0.40 | Poor | Significant bias present |
| < 0.40 | Critical | Do not deploy without mitigation |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — model training and preprocessing
- **Matplotlib / Seaborn** — all visualisations

---

## 💡 Using Your Own Dataset

1. Upload any CSV with a binary target column (0/1 or two class names).
2. Select the column you want to predict as the **Target**.
3. Select the column representing a protected attribute (e.g. gender, race) as the **Sensitive attribute**.
4. Click **Run Analysis**.

The app handles missing values, categorical encoding, and scaling automatically.

---

## 📝 Notes

- The sample dataset is a 50-row excerpt of the classic **Adult Income** dataset.
- For best fairness results, the sensitive attribute should have exactly **2 unique values**.
- The bias mitigation strategy used here is **pre-processing** (removing the sensitive
  attribute). More advanced techniques (reweighing, adversarial debiasing) can be
  added as future enhancements.