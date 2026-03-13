`Kmohamedalie/eda-mL` is a **Python + Streamlit** project (default branch: `master`) that provides a **no-code “Universal EDA & Machine Learning” web app** for quickly exploring datasets and training baseline models.

### What it does
- **Upload data**: accepts **CSV or Excel** uploads (`st.file_uploader(..., type=['csv','xlsx'])`).
- **General EDA tab**:
  - dataset overview (rows, columns, duplicates)
  - descriptive statistics (`df.describe(include='all')`)
  - interactive Plotly visuals (scatter/bar/histogram/boxplot)
- **Text & Sentiment tab**:
  - detects text columns and runs **TextBlob** sentiment (polarity/subjectivity)
  - shows word-frequency counts (top 20 common words)
- **ML “bake-off” tab**:
  - supports **Regression** and **Classification**
  - compares multiple scikit-learn models (e.g., Linear/Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN)
  - basic **hyperparameter tuning UI** for RF / SVM / Logistic Regression / KNN
  - shows metrics (R²/MSE/MAE for regression; Accuracy/Precision/Recall/F1 for classification)
  - diagnostics: **feature importance** (when supported) and **ROC-AUC curves** (binary classification only)
  - lets you **export the best model** as a downloadable `.pkl` (pickle)

### Repo structure (top-level)
- `.devcontainer/` (dev container setup)
- `app.py` (the Streamlit application)
- `requirements.txt` (Python dependencies)
- `mtcars.csv` (sample dataset)

### Tech stack / dependencies
From `requirements.txt`: `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `textblob`, `scikit-learn`, plus `altair<5`.

### How to run locally
1. Install deps:
   - `pip install -r requirements.txt`
2. Launch Streamlit:
   - `streamlit run app.py`

If you tell me the tone you want (e.g., “portfolio-ready”, “README intro”, or “short repo description”), I can rewrite this into a polished paragraph/blurb.
