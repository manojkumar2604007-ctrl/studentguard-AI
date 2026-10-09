"""End-to-end check on a REAL held-out student from the dataset.

Run:  python test_prediction.py            (first held-out Dropout student)
      python test_prediction.py --index 12 (any dataset row)
"""
import argparse
import json

import config
from ml.predict import format_report, predict_student, risk_level
from ml.preprocess import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int)
    ap.add_argument("--label", default="Dropout")
    ap.add_argument("--id", default="STU001")
    a = ap.parse_args()

    df = load_dataset()
    with open(config.METRICS_PATH) as f:
        m = json.load(f)
    idx = a.index
    if idx is None:
        idx = next(int(i) for i, l in m["test_labels"].items() if l == a.label)
    row = df.loc[idx]
    feats = {c: row[c].item() if hasattr(row[c], "item") else row[c]
             for c in df.columns if c != config.TARGET_COL}

    r = predict_student(feats)
    print(format_report(a.id, r))
    print(f"\nDataset row index   : {idx}  (held-out test set: "
          f"{str(idx) in m['test_labels']})")
    print(f"Actual label        : {row[config.TARGET_COL]}")

    # --- integrity checks ---
    total = sum(r["probabilities"].values())
    top = max(r["probabilities"], key=r["probabilities"].get)
    exp_risk = risk_level(r["dropout_probability"] / 100)
    checks = {
        f"probabilities sum to ~100 (got {total:.1f})": abs(total - 100) <= 0.3,
        "predicted class == highest probability": r["predicted_status"] == top,
        "risk level matches dropout probability": r["risk_level"] == exp_risk,
    }
    print("\nIntegrity checks:")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nModel test accuracy : {m['test_accuracy']*100:.2f}% "
          f"({m['n_test']} held-out students)")
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()