"""
STEP 4 - evaluate_hybrid.py
The key results table of the project: does combining both detectors beat each one alone?

Compares, on the official NSL-KDD test set:
  - Signature only   (best supervised classifier)  ~ how a traditional, signature-style IDS behaves
  - Anomaly only     (autoencoder)
  - Hybrid "or"      (alert if either fires)
  - Hybrid weighted  (blended score)
...and reports how many ZERO-DAY attacks (types never seen in training) each catches.

Run:  python src/evaluate_hybrid.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLASSES, DATA_DIR, MODEL_DIR, RESULT_DIR, UNKNOWN_LABEL
from evaluation import binary_metrics, plot_metric_bars
from hybrid_engine import SentinelEngine


def score(attack, novel, flagged):
    m = binary_metrics(attack, flagged)
    m["zero_day_detection"] = float(flagged[novel].mean())
    m["known_attack_detection"] = float(flagged[attack & ~novel].mean())
    return m


def confusion_with_unknown(y_true, labels, path):
    cols = CLASSES + [UNKNOWN_LABEL]
    cm = pd.crosstab(pd.Categorical([CLASSES[i] for i in y_true], CLASSES),
                     pd.Categorical(labels, cols), dropna=False)
    pct = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(cols)):
            v = pct.values[i, j]
            ax.text(j, i, f"{cm.values[i, j]:,}\n{v:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 else "#0b0b0b")
    ax.set_xlabel("Hybrid engine verdict")
    ax.set_ylabel("Actual class")
    ax.set_title("Hybrid engine - what each actual class was labelled as")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of actual class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    cm.to_csv(path.with_suffix(".csv"))


def main():
    d = np.load(DATA_DIR / "processed.npz")
    X_te, y_te, novel = d["X_test"], d["y_test"], d["test_is_novel"]
    attack = y_te != 0
    print(f"Test set: {len(y_te):,} rows, {attack.sum():,} attacks, {novel.sum():,} zero-day attacks\n")

    engine = SentinelEngine()
    proba, err = engine.scores(X_te, X_te[:, engine.selected])
    or_ = engine.decide(proba, err, mode="or")
    wtd = engine.decide(proba, err, mode="weighted")

    rows = {
        f"Signature only ({engine.classifier_name})": score(attack, novel, or_["clf_flag"]),
        "Anomaly only (autoencoder)": score(attack, novel, or_["ae_flag"]),
        "Hybrid (OR rule)": score(attack, novel, or_["is_attack"]),
        "Hybrid (weighted)": score(attack, novel, wtd["is_attack"]),
    }
    table = pd.DataFrame(rows).T
    table.index.name = "detector"

    # ---- end-to-end latency: raw row -> verdict, one row at a time
    sample = pd.read_csv(DATA_DIR / "demo_stream.csv").head(200)
    t0 = time.perf_counter()
    for i in range(len(sample)):
        engine.analyze(sample.iloc[[i]])
    single_ms = (time.perf_counter() - t0) / len(sample) * 1000
    batch = pd.read_csv(DATA_DIR / "demo_stream.csv")
    t0 = time.perf_counter()
    engine.analyze(batch)
    batch_ms = (time.perf_counter() - t0) / len(batch) * 1000

    table.to_csv(RESULT_DIR / "hybrid_comparison.csv", float_format="%.4f")
    plot_metric_bars(table, ["accuracy", "precision", "recall", "f1", "fpr", "zero_day_detection"],
                     "Signature vs anomaly vs hybrid (NSL-KDD test set)",
                     RESULT_DIR / "hybrid_comparison.png")
    confusion_with_unknown(y_te, or_["label"], RESULT_DIR / "cm_hybrid.png")

    # ---- does the hybrid help no matter which classifier is used?
    per_clf = {}
    for name in ["random_forest", "xgboost", "dnn"]:
        e = SentinelEngine(classifier=name)
        p = e.classifier.predict_proba(X_te[:, e.selected])
        h = e.decide(p, err, mode="or")
        sig, hyb = score(attack, novel, h["clf_flag"]), score(attack, novel, h["is_attack"])
        per_clf[name] = {"signature_recall": sig["recall"], "hybrid_recall": hyb["recall"],
                         "signature_zero_day": sig["zero_day_detection"],
                         "hybrid_zero_day": hyb["zero_day_detection"],
                         "signature_fpr": sig["fpr"], "hybrid_fpr": hyb["fpr"]}
    pd.DataFrame(per_clf).T.to_csv(RESULT_DIR / "hybrid_gain_per_classifier.csv", float_format="%.4f")

    # ---- sensitivity analysis for "weighted" mode: how do weight & cutoff trade recall vs false alarms?
    sweep = []
    for cutoff in (0.3, 0.4, 0.5):
        for w in np.round(np.arange(0.1, 0.91, 0.1), 1):
            m = score(attack, novel, engine.decide(proba, err, "weighted", w, cutoff)["is_attack"])
            sweep.append({"cutoff": cutoff, "weight": w, **m})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(RESULT_DIR / "hybrid_weight_sweep.csv", index=False, float_format="%.4f")
    fig, ax = plt.subplots(figsize=(7, 5))
    for color, (cutoff, grp) in zip(["#2a78d6", "#eb6834", "#1baf7a"], sweep.groupby("cutoff")):
        ax.plot(grp["fpr"], grp["recall"], marker="o", ms=5, lw=2, color=color,
                label=f"Weighted, cutoff {cutoff} (dots = weight 0.1 ... 0.9)")
    for name, marker in [("Hybrid (OR rule)", "*"), (f"Signature only ({engine.classifier_name})", "s"),
                         ("Anomaly only (autoencoder)", "^")]:
        ax.scatter(table.loc[name, "fpr"], table.loc[name, "recall"], s=110, marker=marker,
                   color="#0b0b0b", zorder=3, label=name)
    ax.set_xlabel("False positive rate (false alarms)")
    ax.set_ylabel("Recall (attacks caught)")
    ax.set_title("Hybrid trade-off: catching attacks vs false alarms")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "hybrid_tradeoff.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 160)
    print(table.round(4).to_string())
    print("\nHybrid gain for every classifier:")
    print(pd.DataFrame(per_clf).T.round(4).to_string())
    print(f"\nDetection latency (raw row -> verdict): {single_ms:.2f} ms/row one-at-a-time, "
          f"{batch_ms:.4f} ms/row in batches")
    (RESULT_DIR / "latency.txt").write_text(
        f"single_row_ms={single_ms:.3f}\nbatched_ms_per_row={batch_ms:.4f}\n")


if __name__ == "__main__":
    main()
