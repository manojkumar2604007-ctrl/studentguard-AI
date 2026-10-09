"""Train the ANN and store model + REAL evaluation metrics.

Run:  python -m ml.train
"""
import json
import time

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import config
from ml.preprocess import (NOMINAL_COLS, build_preprocessor, load_dataset,
                           prepare_frame, split_xy)


def main():
    t0 = time.time()
    df = load_dataset()
    X, y = split_xy(df)
    X = prepare_frame(X)
    feature_cols = list(X.columns)
    print(f"Dataset      : {df.shape[0]} rows, {len(feature_cols)} features")
    print("Class counts :", y.value_counts().to_dict())

    # Integer-encode labels (0=Dropout, 1=Enrolled, 2=Graduate); the ANN's
    # early-stopping needs numeric labels in this scikit-learn version.
    label_to_id = {c: i for i, c in enumerate(config.CLASSES)}
    y_names = y
    y = y.map(label_to_id).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE)

    ann = MLPClassifier(
        hidden_layer_sizes=config.HIDDEN_LAYERS, activation="relu",
        solver="adam", alpha=1e-3, batch_size=64, learning_rate_init=1e-3,
        max_iter=300, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15, random_state=config.RANDOM_STATE)
    pipe = Pipeline([("prep", build_preprocessor(feature_cols)), ("ann", ann)])
    pipe.fit(X_tr, y_tr)

    classes = [config.CLASSES[int(i)] for i in pipe.named_steps["ann"].classes_]
    pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, pred)
    ids = list(range(len(classes)))
    report = classification_report(y_te, pred, labels=ids, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=ids)
    train_acc = accuracy_score(y_tr, pipe.predict(X_tr))

    imp = permutation_importance(pipe, X_te, y_te, n_repeats=3,
                                 random_state=config.RANDOM_STATE,
                                 scoring="accuracy")
    order = np.argsort(imp.importances_mean)[::-1][:12]
    importance = [{"feature": feature_cols[i],
                   "importance": float(imp.importances_mean[i])} for i in order]

    defaults = {}
    for c in feature_cols:
        defaults[c] = (int(X[c].mode().iloc[0]) if c in NOMINAL_COLS
                       else float(X[c].median()))

    bundle = {"pipeline": pipe, "feature_cols": feature_cols,
              "classes": classes, "defaults": defaults}
    joblib.dump(bundle, config.MODEL_PATH)

    ann_step = pipe.named_steps["ann"]
    metrics = {
        "model": "ANN (MLPClassifier)",
        "hidden_layers": list(config.HIDDEN_LAYERS),
        "n_rows": int(df.shape[0]), "n_features": len(feature_cols),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "epochs_run": int(ann_step.n_iter_),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(acc),
        "classes": classes,
        "class_distribution": {k: int(v) for k, v in y_names.value_counts().items()},
        "report": {k: report[k] for k in classes},
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "feature_importance": importance,
        "loss_curve": [float(v) for v in ann_step.loss_curve_],
        "test_indices": [int(i) for i in X_te.index],
        "test_labels": {str(i): config.CLASSES[int(l)] for i, l in zip(X_te.index, y_te)},
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Epochs run   : {metrics['epochs_run']}")
    print(f"Train acc    : {train_acc:.4f}")
    print(f"Test  acc    : {acc:.4f}   (held-out {len(X_te)} students)")
    print(classification_report(y_te, pred, labels=ids, target_names=classes, zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted):", classes)
    print(cm)
    print(f"Saved model  : {config.MODEL_PATH}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()