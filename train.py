#!/usr/bin/env python3
"""
Train classification models with SMOTE/ADASYN and Grid Search CV.

Supports: Logistic Regression, Decision Tree, Random Forest, XGBoost.
Handles mixed numeric/categorical features with robust preprocessing.
Evaluates on a held-out test set and saves the best model + metrics.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN


def infer_feature_types(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def make_preprocessor(num_cols, cat_cols, scale_numeric: bool):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    num_tr = Pipeline(steps=num_steps)

    cat_tr = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tr, num_cols),
            ("cat", cat_tr, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def get_models_and_grids():
    models = {}

    # Logistic Regression
    models["logreg"] = (
        LogisticRegression(max_iter=1000, n_jobs=None if "n_jobs" not in LogisticRegression().get_params() else -1),
        {
            "clf__C": [0.1, 1.0, 3.0, 10.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "liblinear"],
        },
    )

    # Decision Tree
    models["dt"] = (
        DecisionTreeClassifier(random_state=42),
        {
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 5],
        },
    )

    # Random Forest
    models["rf"] = (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 10, 20],
            "clf__max_features": ["sqrt", "log2", None],
            "clf__min_samples_leaf": [1, 2, 5],
        },
    )

    # XGBoost
    if HAS_XGB:
        models["xgb"] = (
            XGBClassifier(
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=400,
                tree_method="hist",
                n_jobs=-1,
            ),
            {
                "clf__n_estimators": [300, 400, 600],
                "clf__max_depth": [3, 5, 8],
                "clf__learning_rate": [0.03, 0.1],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            },
        )
    return models


def make_sampler(kind: str, random_state: int):
    if kind.lower() == "smote":
        return SMOTE(random_state=random_state, n_jobs=-1)
    elif kind.lower() == "adasyn":
        return ADASYN(random_state=random_state, n_neighbors=5, n_jobs=-1)
    else:
        raise ValueError("Sampler must be 'smote' or 'adasyn'")


def build_pipeline(preprocessor, sampler, clf):
    # Order: preprocess -> sample -> classifier
    return ImbPipeline(steps=[
        ("prep", preprocessor),
        ("sampler", sampler),
        ("clf", clf),
    ])


def main():
    parser = argparse.ArgumentParser(description="Fraud Classification with SMOTE/ADASYN + Grid Search")
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--target", type=str, default="target", help="Target column name (default: target)")
    parser.add_argument("--imbalance", type=str, default="smote", choices=["smote", "adasyn"], help="Over-sampler")
    parser.add_argument("--model", type=str, default="all", choices=["logreg", "dt", "rf", "xgb", "all"], help="Model to train")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--scale-numeric", action="store_true", help="Apply StandardScaler to numeric features")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs for CV")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for models/reports")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    (outdir / "reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data. Found: {list(df.columns)[:10]} ...")

    y = df[args.target]
    X = df.drop(columns=[args.target])

    num_cols, cat_cols = infer_feature_types(df, args.target)
    preprocessor = make_preprocessor(num_cols, cat_cols, scale_numeric=args.scale_numeric)
    sampler = make_sampler(args.imbalance, args.random_state)

    models_grids = get_models_and_grids()

    model_keys = list(models_grids.keys()) if args.model == "all" else [args.model]
    if "xgb" in model_keys and not HAS_XGB:
        print("WARNING: xgboost not available; skipping XGBClassifier. Install 'xgboost' to enable.")
        model_keys = [k for k in model_keys if k != "xgb"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    results = {}

    for key in model_keys:
        base_clf, grid = models_grids[key]
        pipe = build_pipeline(preprocessor, sampler, base_clf)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=args.jobs,
            verbose=1,
            refit=True,
        )

        gs.fit(X_train, y_train)

        y_proba = gs.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        roc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        model_path = outdir / "models" / f"{key}_best_model.joblib"
        joblib.dump(gs.best_estimator_, model_path)

        results[key] = {
            "best_params": gs.best_params_,
            "best_cv_score_roc_auc": float(gs.best_score_),
            "test_roc_auc": float(roc),
            "test_pr_auc": float(pr_auc),
            "confusion_matrix": cm,
            "classification_report": report,
            "model_path": str(model_path),
        }

        # Write per-model report
        with open(outdir / "reports" / f"{key}_report.json", "w") as f:
            json.dump(results[key], f, indent=2)

        print(f"[{key}] ROC AUC (test): {roc:.4f} | PR AUC: {pr_auc:.4f}")
        print(f"[{key}] Best params: {gs.best_params_}")
        print(f"[{key}] Saved model to: {model_path}")

    # Write combined results
    with open(outdir / "reports" / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also create a quick human-readable txt
    lines = []
    for k, v in results.items():
        lines.append(f"Model: {k}")
        lines.append(f"  Test ROC AUC: {v['test_roc_auc']:.4f}")
        lines.append(f"  Test PR AUC:  {v['test_pr_auc']:.4f}")
        lines.append(f"  Best CV ROC AUC: {v['best_cv_score_roc_auc']:.4f}")
        lines.append(f"  Best Params: {v['best_params']}")
        lines.append("")
    with open(outdir / "reports" / "summary.txt", "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
