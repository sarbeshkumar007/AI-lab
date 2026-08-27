"""
Train the cardiovascular-disease Random Forest model and save it for the
Streamlit app.

Run:
    python train_model.py

Produces:
    model.pkl        — trained RandomForestClassifier
    feature_cols.pkl — exact column order the model expects (app.py needs this)
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from data_prep import prepare_training_frame

DATA_PATH = "cardiovascular_diseases_dv3.csv"
MODEL_PATH = "model.pkl"
FEATURE_COLS_PATH = "feature_cols.pkl"


def main():
    df = pd.read_csv(DATA_PATH, sep=";")
    X, y = prepare_training_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURE_COLS_PATH)
    print(f"\nSaved {MODEL_PATH} and {FEATURE_COLS_PATH}")


if __name__ == "__main__":
    main()
