"""Prediction using the trained ANN. All probabilities come from predict_proba."""
import math

import joblib
import pandas as pd

import config
from ml.preprocess import prepare_frame

_BUNDLE = None


def load_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(config.MODEL_PATH)
    return _BUNDLE


def risk_level(dropout_prob):
    """dropout_prob in 0-1 -> LOW / MEDIUM / HIGH (thresholds in config.py)."""
    if dropout_prob >= config.RISK_HIGH:
        return "HIGH"
    if dropout_prob >= config.RISK_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _to_frame(features):
    b = load_bundle()
    row = {}
    for col in b["feature_cols"]:
        v = features.get(col)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            row[col] = b["defaults"][col]
        else:
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"'{col}' must be a number, got {v!r}")
            if math.isnan(f) or math.isinf(f):
                raise ValueError(f"'{col}' must be a finite number")
            row[col] = f
    return prepare_frame(pd.DataFrame([row], columns=b["feature_cols"]))


def _observations(f, risk):
    obs = []

    def num(k):
        try:
            return float(f.get(k))
        except (TypeError, ValueError):
            return None

    for sem, label in (("1st", "first"), ("2nd", "second")):
        enr = num(f"Curricular units {sem} sem (enrolled)")
        app = num(f"Curricular units {sem} sem (approved)")
        gr = num(f"Curricular units {sem} sem (grade)")
        if enr and enr > 0 and app is not None:
            rate = app / enr
            if rate < 0.5:
                obs.append(f"Low pass rate in {label} semester "
                           f"({int(app)} of {int(enr)} units approved).")
            elif rate >= 0.9:
                obs.append(f"Strong pass rate in {label} semester "
                           f"({int(app)} of {int(enr)} units approved).")
        elif enr == 0:
            obs.append(f"No curricular units enrolled in {label} semester.")
        if gr is not None and enr and gr < 10:
            obs.append(f"Average grade in {label} semester is below the pass mark ({gr:.1f}/20).")
    if num("Tuition fees up to date") == 0:
        obs.append("Tuition fees are not up to date.")
    if num("Debtor") == 1:
        obs.append("Student is flagged as a debtor.")
    if num("Scholarship holder") == 1:
        obs.append("Scholarship holder (protective factor).")
    age = num("Age at enrollment")
    if age is not None and age >= 25:
        obs.append(f"Older enrolment age ({int(age)}) - may need flexible support.")
    head = {"HIGH": "Overall academic and background profile indicates elevated dropout risk.",
            "MEDIUM": "Profile shows a moderate dropout risk; early monitoring is advisable.",
            "LOW": "Profile indicates a low dropout risk."}[risk]
    return [head] + obs


def _interventions(f, risk):
    base = {
        "HIGH": ["Academic counselling", "Regular performance monitoring",
                 "Faculty follow-up"],
        "MEDIUM": ["Periodic mentor check-ins", "Academic progress review each semester"],
        "LOW": ["Routine monitoring", "Encourage continued participation"],
    }[risk]
    extra = []
    try:
        if float(f.get("Tuition fees up to date", 1)) == 0 or float(f.get("Debtor", 0)) == 1:
            extra.append("Financial aid / fee-payment counselling")
    except (TypeError, ValueError):
        pass
    return base + extra


def predict_student(features):
    """features: dict keyed by dataset column names. Returns a result dict."""
    b = load_bundle()
    X = _to_frame(features)
    proba = b["pipeline"].predict_proba(X)[0]
    classes = [config.CLASSES[int(c)] for c in b["pipeline"].named_steps["ann"].classes_]
    probs = {c: float(p) for c, p in zip(classes, proba)}
    predicted = max(probs, key=probs.get)
    dropout_p = probs["Dropout"]
    risk = risk_level(dropout_p)
    used = {c: X.iloc[0][c].item() if hasattr(X.iloc[0][c], "item") else X.iloc[0][c]
            for c in b["feature_cols"]}
    return {
        "predicted_status": predicted,
        "probabilities": {c: round(probs[c] * 100, 1) for c in config.CLASSES},
        "dropout_probability": round(dropout_p * 100, 1),
        "risk_level": risk,
        "observations": _observations(used, risk),
        "interventions": _interventions(used, risk),
        "features": used,
    }


def format_report(student_id, r):
    line = "-" * 38
    out = ["STUDENT DROPOUT PREDICTION", line,
           f"Student ID          : {student_id}",
           f"Predicted Status    : {r['predicted_status']}",
           f"Dropout Probability : {r['dropout_probability']:.1f}%",
           f"Risk Level          : {r['risk_level']}", "",
           "Class Probabilities:"]
    for c in config.CLASSES:
        out.append(f"  {c:<9} : {r['probabilities'][c]:.1f}%")
    out += ["", "Key Observations:"] + [f"  * {o}" for o in r["observations"]]
    out += ["", "Suggested Intervention:"] + [f"  * {i}" for i in r["interventions"]]
    return "\n".join(out)