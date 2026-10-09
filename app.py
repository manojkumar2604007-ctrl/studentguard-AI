"""StudentGuard AI - Flask application."""
import json
import os

from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request, url_for)

import config
from database import db
from ml import features as F
from ml.predict import load_bundle, predict_student
from ml.preprocess import load_dataset

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
db.init_db()

_DF = None


def dataset():
    global _DF
    if _DF is None:
        _DF = load_dataset()
    return _DF


def metrics():
    if not os.path.exists(config.METRICS_PATH):
        return None
    with open(config.METRICS_PATH) as f:
        return json.load(f)


def model_ready():
    return os.path.exists(config.MODEL_PATH) and os.path.exists(config.METRICS_PATH)


@app.context_processor
def inject():
    m = metrics()
    return {"model_acc": round(m["test_accuracy"] * 100, 1) if m else None}


@app.route("/")
def index():
    return render_template("index.html", m=metrics(), stats=db.stats(), ready=model_ready())


@app.route("/assess")
def assess():
    if not model_ready():
        flash("Model not trained yet. Run: python -m ml.train", "error")
    b = load_bundle() if model_ready() else {"defaults": {}, "feature_cols": []}
    adv = [c for c in b["feature_cols"] if c not in F.KEY_NAMES]
    return render_template("assess.html", key_fields=F.KEY_FIELDS, adv=adv,
                           defaults=b["defaults"])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    sid = (data.pop("student_id", "") or "").strip()
    name = (data.pop("student_name", "") or "").strip()
    try:
        res = predict_student(data)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("assess"))
    pid = db.save_prediction(sid, name, res)
    return redirect(url_for("result", pid=pid))


@app.route("/result/<int:pid>")
def result(pid):
    r = db.get_prediction(pid)
    if not r:
        abort(404)
    return render_template("result.html", r=r, classes=config.CLASSES)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=db.stats(), recent=db.recent(10), m=metrics())


@app.route("/analytics")
def analytics():
    m = metrics()
    pts = ""
    if m:
        lc = m["loss_curve"]
        hi, lo = max(lc), min(lc)
        span = (hi - lo) or 1
        n = max(len(lc) - 1, 1)
        pts = " ".join(f"{10 + 480 * i / n:.1f},{10 + 140 * (1 - (v - lo) / span):.1f}"
                       for i, v in enumerate(lc))
    return render_template("analytics.html", m=m, pts=pts)


# ---------------- JSON API ----------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    sid = str(data.pop("student_id", "") or "").strip()
    try:
        res = predict_student(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    pid = db.save_prediction(sid, str(data.pop("student_name", "") or ""), res)
    res["id"] = pid
    res["student_id"] = db.get_prediction(pid)["student_id"]
    return jsonify(res)


@app.route("/api/sample")
def api_sample():
    """Return a REAL held-out student row from the dataset (for form filling/testing)."""
    m = metrics()
    if not m:
        return jsonify({"error": "model not trained"}), 400
    df = dataset()
    label = request.args.get("label")
    n = int(request.args.get("n", 0))
    idx = request.args.get("index", type=int)
    if idx is None:
        pool = [int(i) for i, l in m["test_labels"].items() if label in (None, "", l)]
        if not pool:
            return jsonify({"error": "no such label"}), 404
        idx = pool[n % len(pool)]
    if idx not in df.index:
        return jsonify({"error": "index out of range"}), 404
    row = df.loc[idx]
    feats = {c: (row[c].item() if hasattr(row[c], "item") else row[c])
             for c in df.columns if c != config.TARGET_COL}
    return jsonify({"index": idx, "actual": row[config.TARGET_COL], "features": feats})


@app.route("/api/metrics")
def api_metrics():
    return jsonify(metrics() or {})


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "model_ready": model_ready()})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)