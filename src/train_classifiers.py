"""
STEP 2 - train_classifiers.py   (the "signature" / known-attack path)

Trains three supervised models on the SMOTE-balanced training data:
  - Random Forest  (many decision trees voting)
  - XGBoost        (trees built one after another, each fixing earlier mistakes)
  - Deep Neural Network (PyTorch)
Then scores each on the official NSL-KDD test set and picks the one the
hybrid engine will use.

Run:  python src/train_classifiers.py
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from config import (BATCH_SIZE, CLASSES, CLASSIFIER_CHOICE, DATA_DIR, DNN_EPOCHS, MODEL_DIR,
                    RANDOM_STATE, RESULT_DIR)
from deep_models import DNNClassifier
from evaluation import multiclass_metrics, plot_history, plot_metric_bars


def latency_ms_per_row(model, X, n=200):
    """Average time to classify ONE connection (rows sent one at a time)."""
    rows = X[:n]
    if hasattr(model, "n_jobs"):      # one row at a time: extra threads only add overhead
        model.n_jobs = 1
    start = time.perf_counter()
    for i in range(len(rows)):
        model.predict_proba(rows[i:i + 1])
    return (time.perf_counter() - start) / len(rows) * 1000


def main():
    d = np.load(DATA_DIR / "processed.npz")
    sel = joblib.load(MODEL_DIR / "preprocessor.joblib")["selected"]
    X_tr, y_tr = d["X_train_smote"][:, sel], d["y_train_smote"]
    X_val, y_val = d["X_val"][:, sel], d["y_val"]
    X_te, y_te = d["X_test"][:, sel], d["y_test"]
    print(f"Training on {len(X_tr):,} rows x {X_tr.shape[1]} features")

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss",
            n_jobs=-1, random_state=RANDOM_STATE),
        "dnn": DNNClassifier(n_features=X_tr.shape[1], n_classes=len(CLASSES),
                             epochs=DNN_EPOCHS, batch_size=BATCH_SIZE, seed=RANDOM_STATE),
    }

    rows, val_f1 = {}, {}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        if name == "dnn":
            hist = model.fit(X_tr, y_tr, X_val, y_val)
            plot_history(hist, "DNN classifier - training curve", RESULT_DIR / "dnn_training_curve.png",
                         "Cross-entropy loss")
            model.save(MODEL_DIR / "dnn.pt")
        else:
            model.fit(X_tr, y_tr)
            joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        train_s = time.perf_counter() - t0

        val_f1[name] = f1_score(y_val, model.predict(X_val), average="macro")
        m = multiclass_metrics(y_te, model.predict(X_te), name)
        m["train_time_s"] = train_s
        m["latency_ms_per_row"] = latency_ms_per_row(model, X_te)
        rows[name] = m
        print(f"  trained in {train_s:.1f}s | val macro-F1 {val_f1[name]:.4f} | "
              f"TEST accuracy {m['accuracy']:.4f}  macro-F1 {m['f1_macro']:.4f}  "
              f"attack recall {m['bin_recall']:.4f}  FPR {m['bin_fpr']:.4f}")

    table = pd.DataFrame(rows).T
    table.index.name = "model"
    table.to_csv(RESULT_DIR / "classifier_comparison.csv", float_format="%.4f")
    plot_metric_bars(table, ["accuracy", "precision_macro", "recall_macro", "f1_macro", "bin_recall", "bin_fpr"],
                     "Supervised classifiers on NSL-KDD test set", RESULT_DIR / "classifier_comparison.png")

    best = max(val_f1, key=val_f1.get) if CLASSIFIER_CHOICE == "auto" else CLASSIFIER_CHOICE
    json.dump({"name": best, "val_macro_f1": val_f1},
              open(MODEL_DIR / "best_classifier.json", "w"), indent=2)
    print("\n", table[["accuracy", "f1_macro", "bin_recall", "bin_fpr", "latency_ms_per_row"]].round(4))
    print(f"\nHybrid engine will use: {best}")


if __name__ == "__main__":
    main()
