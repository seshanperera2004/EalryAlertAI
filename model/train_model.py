import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
import shap

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "students_dummy.csv")
METRICS_OUT = os.path.join(os.path.dirname(__file__), "..", "dashboard", "model_metrics.json")
RISK_OUT = os.path.join(os.path.dirname(__file__), "..", "dashboard", "risk_scores.json")

FEATURE_COLS = [
    "attendance_decline_pct",
    "lms_inactivity_gap_days",
    "assignment_delay_index",
    "grade_volatility_score",
    "current_avg_mark",
]

FEATURE_LABELS = {
    "attendance_decline_pct": "Attendance decline",
    "lms_inactivity_gap_days": "LMS inactivity gap",
    "assignment_delay_index": "Assignment delay index",
    "grade_volatility_score": "Grade volatility",
    "current_avg_mark": "Current average mark",
}

RISK_TIERS = ["low", "moderate", "high"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def train_and_evaluate(df):
    X = df[FEATURE_COLS].copy()
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df["true_risk_tier"])  # low=?, moderate=?, high=?

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test) if False else scaler.transform(X_test)

    # Ensemble: Random Forest + XGBoost (soft-voting), with Logistic Regression
    # included as the interpretable baseline named in the proposal's timeline.
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        eval_metric="mlogloss",
        random_state=42,
    )
    logreg = LogisticRegression(max_iter=1000)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("logreg", logreg)],
        voting="soft",
    )
    ensemble.fit(X_train_scaled, y_train)

    # VotingClassifier clones its estimators internally, so the `rf` object
    # itself is never fitted. Fit a standalone copy for SHAP explanations.
    rf_for_shap = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    rf_for_shap.fit(X_train_scaled, y_train)

    y_pred = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro"), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro"), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "roc_auc_macro_ovr": round(
            roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"), 4
        ),
        "cv_accuracy_5fold_mean": round(
            float(np.mean(cross_val_score(rf, X_train_scaled, y_train, cv=5))), 4
        ),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "note": (
            "Metrics are computed on a SYNTHETIC dataset for pipeline "
            "validation only. They will be recalculated once real, "
            "anonymized LMS/attendance/CA data is available."
        ),
    }

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model metrics:")
    print(json.dumps(metrics, indent=2))

    return ensemble, scaler, y_encoder, rf_for_shap, X_train_scaled, X_train


def score_all_students(df, ensemble, scaler, y_encoder, rf, X_train_scaled_ref, X_train_ref):
    X_all = df[FEATURE_COLS].copy()
    X_all_scaled = scaler.transform(X_all)

    proba = ensemble.predict_proba(X_all_scaled)
    class_order = list(y_encoder.classes_)  # e.g. ['high', 'low', 'moderate']
    high_idx = class_order.index("high")
    mod_idx = class_order.index("moderate")

    # Composite risk score 0-100: weighted toward "high" probability
    risk_score = (proba[:, high_idx] * 100) + (proba[:, mod_idx] * 40)
    risk_score = np.clip(risk_score, 0, 100)

    predicted_tier_idx = ensemble.predict(X_all_scaled)
    predicted_tier = y_encoder.inverse_transform(predicted_tier_idx)

    # SHAP explanations using the Random Forest branch of the ensemble
    # (fast TreeExplainer; used to surface the top contributing factors
    # per student, echoing the proposal's Explainable AI requirement)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_all_scaled)

    # shap_values shape for multiclass RF: (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_for_high = shap_values[high_idx]
    else:
        shap_for_high = shap_values[:, :, high_idx]

    records = []
    for i, row in df.iterrows():
        contribs = list(zip(FEATURE_COLS, shap_for_high[i]))
        contribs.sort(key=lambda t: abs(t[1]), reverse=True)
        top_reasons = [
            {
                "feature": FEATURE_LABELS[feat],
                "value": float(row[feat]),
                "impact": round(float(val), 4),
            }
            for feat, val in contribs[:3]
        ]

        tier = predicted_tier[i]
        records.append(
            {
                "student_id": row["student_id"],
                "faculty": row["faculty"],
                "year_of_study": int(row["year_of_study"]),
                "risk_score": round(float(risk_score[i]), 1),
                "risk_tier": tier,  # low / moderate / high
                "current_avg_mark": float(row["current_avg_mark"]),
                "attendance_decline_pct": float(row["attendance_decline_pct"]),
                "lms_inactivity_gap_days": float(row["lms_inactivity_gap_days"]),
                "assignment_delay_index": float(row["assignment_delay_index"]),
                "grade_volatility_score": float(row["grade_volatility_score"]),
                "top_reasons": top_reasons,
            }
        )

    records.sort(key=lambda r: r["risk_score"], reverse=True)

    summary = {
        "generated_from": "synthetic_dummy_dataset",
        "total_students": len(records),
        "counts": {
            tier: sum(1 for r in records if r["risk_tier"] == tier) for tier in RISK_TIERS
        },
        "students": records,
    }

    with open(RISK_OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote risk scores for {len(records)} students -> {RISK_OUT}")
    print("Risk tier counts:", summary["counts"])


def main():
    df = load_data()
    ensemble, scaler, y_encoder, rf, X_train_scaled, X_train = train_and_evaluate(df)
    score_all_students(df, ensemble, scaler, y_encoder, rf, X_train_scaled, X_train)


if __name__ == "__main__":
    main()
