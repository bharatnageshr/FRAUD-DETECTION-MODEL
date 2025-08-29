# FRAUD-DETECTION-MODEL
# Fraud Classification with SMOTE/ADASYN + Grid Search (ROC AUC ~0.97 with LR)

This repository contains a clean, reproducible pipeline to train **Logistic Regression, XGBoost, Decision Trees, and Random Forest** on highly imbalanced data. It uses **SMOTE or ADASYN** to address class imbalance and **GridSearchCV** for hyperparameter tuning. On a typical credit-card style dataset, **Logistic Regression** can achieve **ROC AUC ~0.97** with thoughtful preprocessing and resampling—demonstrating how a simple model can perform strongly.

> Replace the dataset with your own CSV. By default, the target column is named `target` (you can change it via a CLI argument).

## Features

- Handles mixed numeric & categorical columns
- Choice of **SMOTE** or **ADASYN** oversampling
- Trains and tunes **LogReg / DT / RF / XGBoost**
- Robust preprocessing: imputation, one-hot encoding, scaling (optional)
- **Stratified** CV with `roc_auc` scoring
- Saves best model(s) + JSON reports
- Minimal, production-ready, single-command training

## Project Structure

```
.
├── models/            # Saved best estimators (.joblib)
├── reports/           # Metrics & best params (.json/.txt)
├── src/
│   └── train.py       # Main training entrypoint
├── README.md
└── requirements.txt
```

## Quickstart

1. **Install dependencies** (recommend a fresh virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data** as a CSV with a binary `target` column (0/1). Example:
   ```text
   amount,age,merchant,city,target
   123.45,33,StoreA,London,0
   987.65,58,StoreB,Paris,1
   ```

3. **Train all models** with SMOTE (default) and scale numeric features:
   ```bash
   python src/train.py --data path/to/data.csv --target target --imbalance smote --model all --scale-numeric --outdir .
   ```

   Switch to **ADASYN**:
   ```bash
   python src/train.py --data path/to/data.csv --imbalance adasyn --model all --scale-numeric
   ```

   Train a **single model**, e.g., Logistic Regression:
   ```bash
   python src/train.py --data path/to/data.csv --model logreg --scale-numeric
   ```

## Outputs

- `models/<model>_best_model.joblib` – best pipeline (preprocessing + sampler + classifier)
- `reports/<model>_report.json` – metrics (ROC AUC, PR AUC, confusion matrix, classification report, best params)
- `reports/summary.json` & `reports/summary.txt` – quick comparison across models

## Repro Tips

- Use `--random-state` for deterministic splits and SMOTE/ADASYN.
- For Logistic Regression, scaling numeric features (`--scale-numeric`) is recommended.
- If XGBoost is not installed, the script will skip it with a warning.

## Hyperparameter Grids (default)

- **LogisticRegression**
  - `C`: `[0.1, 1.0, 3.0, 10.0]`
  - `solver`: `["lbfgs", "liblinear"]`
- **DecisionTree**
  - `max_depth`: `[None, 5, 10, 20]`
  - `min_samples_split`: `[2, 5, 10]`
  - `min_samples_leaf`: `[1, 2, 5]`
- **RandomForest**
  - `n_estimators`: `[200, 400]`
  - `max_depth`: `[None, 10, 20]`
  - `max_features`: `["sqrt", "log2", None]`
  - `min_samples_leaf`: `[1, 2, 5]`
- **XGBoost** (if installed)
  - `n_estimators`: `[300, 400, 600]`
  - `max_depth`: `[3, 5, 8]`
  - `learning_rate`: `[0.03, 0.1]`
  - `subsample`: `[0.8, 1.0]`
  - `colsample_bytree`: `[0.8, 1.0]`

## Requirements

```
pandas
numpy
scikit-learn>=1.3
imblearn
xgboost  # optional; script will skip if unavailable
joblib
```

## Reproducing Your Claimed Result (ROC AUC 0.97 with LR)

Once trained, check `reports/logreg_report.json` and `reports/summary.txt`. Your **test ROC AUC** will be printed in the console and written to the report files. If you achieved **0.97** previously, the same data & settings should result in a similar score.

## Notes

- The pipeline applies oversampling **after** preprocessing to ensure the sampler sees the fully numeric feature matrix.
- Classification threshold is `0.5` by default for the printed confusion matrix & report; for deployment, you may tune thresholds to optimize precision/recall or cost-weighted outcomes.

---

**Happy modeling!** If you want me to tailor the grids or add cost-sensitive metrics (e.g., expected financial loss), open an issue or ask for a tweak.
