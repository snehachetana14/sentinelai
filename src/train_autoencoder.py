"""
STEP 3 - train_autoencoder.py   (the "anomaly" / unknown-attack path)

  1. Train the autoencoder on NORMAL traffic only (it learns what 'normal' looks like).
  2. Measure reconstruction error on the validation set (normal + attacks).
  3. Pick the error threshold from the ROC curve (best balance of
     catching attacks vs. false alarms = Youden's J = TPR - FPR).
  4. Score on the test set, including how many NEVER-SEEN attack types it catches.

Run:  python src/train_autoencoder.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from config import AE_EPOCHS, BATCH_SIZE, DATA_DIR, MODEL_DIR, RANDOM_STATE, RESULT_DIR
from deep_models import Autoencoder
from evaluation import binary_metrics, plot_history


def main():
    d = np.load(DATA_DIR / "processed.npz")
    X_tr, y_tr = d["X_train"], d["y_train"]          # NOT the SMOTE version: we only want real normal rows
    X_val, y_val = d["X_val"], d["y_val"]
    X_te, y_te, novel = d["X_test"], d["y_test"], d["test_is_novel"]

    X_normal = X_tr[y_tr == 0]
    print(f"Training autoencoder on {len(X_normal):,} NORMAL rows x {X_normal.shape[1]} features")
    ae = Autoencoder(n_features=X_tr.shape[1], epochs=AE_EPOCHS, batch_size=BATCH_SIZE, seed=RANDOM_STATE)
    hist = ae.fit(X_normal, X_val[y_val == 0])
    ae.save(MODEL_DIR / "autoencoder.pt")
    plot_history(hist, "Autoencoder - training curve (normal traffic)",
                 RESULT_DIR / "autoencoder_training_curve.png", "Reconstruction MSE")

    # ---- choose threshold on VALIDATION data (test set stays untouched)
    err_val = ae.reconstruction_error(X_val)
    fpr, tpr, thr = roc_curve(y_val != 0, err_val)
    best = np.argmax(tpr - fpr)
    threshold = float(thr[best])
    print(f"\nThreshold (Youden's J on validation): {threshold:.6f}  "
          f"-> val TPR {tpr[best]:.4f}, val FPR {fpr[best]:.4f}")

    # ---- evaluate on TEST
    err_te = ae.reconstruction_error(X_te)
    attack = y_te != 0
    flagged = err_te > threshold
    m = binary_metrics(attack, flagged)
    m["roc_auc"] = roc_auc_score(attack, err_te)
    m["novel_attack_detection_rate"] = float(flagged[novel].mean())
    m["known_attack_detection_rate"] = float(flagged[attack & ~novel].mean())
    print("TEST: " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))
    json.dump({"threshold": threshold, "val_tpr": float(tpr[best]), "val_fpr": float(fpr[best]),
               "test_metrics": m}, open(MODEL_DIR / "ae_threshold.json", "w"), indent=2)
    np.save(RESULT_DIR / "ae_test_errors.npy", err_te)

    # ---- chart 1: error distribution, normal vs attack
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.logspace(np.log10(max(err_te.min(), 1e-7)), np.log10(err_te.max()), 80)
    ax.hist(err_te[~attack], bins=bins, alpha=0.65, color="#2a78d6", label="Normal")
    ax.hist(err_te[attack & ~novel], bins=bins, alpha=0.65, color="#eb6834", label="Known attacks")
    ax.hist(err_te[novel], bins=bins, alpha=0.75, color="#4a3aa7", label="Zero-day attacks")
    ax.axvline(threshold, color="#0b0b0b", ls="--", lw=1.5, label=f"Threshold = {threshold:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction error (log scale)")
    ax.set_ylabel("Number of connections")
    ax.set_title("Autoencoder reconstruction error on test set")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "ae_error_distribution.png", dpi=150)
    plt.close(fig)

    # ---- chart 2: ROC curve on test, with the chosen operating point
    fpr_t, tpr_t, _ = roc_curve(attack, err_te)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr_t, tpr_t, color="#2a78d6", lw=2, label=f"Autoencoder (AUC = {m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#898781", ls=":", lw=1, label="Random guess")
    ax.scatter([m["fpr"]], [m["recall"]], s=60, color="#eb6834", zorder=3, label="Chosen threshold")
    ax.set_xlabel("False positive rate (false alarms)")
    ax.set_ylabel("True positive rate (attacks caught)")
    ax.set_title("ROC curve - anomaly detector (test set)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "ae_roc_curve.png", dpi=150)
    plt.close(fig)
    print("Saved models/autoencoder.pt, models/ae_threshold.json and charts in results/")


if __name__ == "__main__":
    main()
