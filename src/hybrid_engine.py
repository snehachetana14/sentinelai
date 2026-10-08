"""
hybrid_engine.py - the "boss" that combines both detectors.

    raw connection(s) -> preprocessor -> classifier   -> P(attack), attack type
                                      -> autoencoder  -> reconstruction error
                                      -> decision rule -> verdict

Decision rule, mode "or" (default):
    classifier=attack, AE=anomaly -> Known attack (confirmed)   severity High
    classifier=attack, AE=normal  -> Known attack               severity Medium
    classifier=normal, AE=anomaly -> Unknown / zero-day attack  severity Medium
    classifier=normal, AE=normal  -> Normal

Mode "weighted" (defaults: w = 0.3, cutoff = 0.4, see config.py):
    score = w * P(attack) + (1 - w) * AE_score     (AE_score = error / (error + threshold),
    attack if score >= cutoff                        which is exactly 0.5 at the threshold)

Used by:  evaluate_hybrid.py, api.py, dashboard.py
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from config import (CLASSES, CLF_ATTACK_THRESHOLD, FEATURES, HYBRID_CUTOFF, HYBRID_MODE, HYBRID_WEIGHT,
                    MODEL_DIR, UNKNOWN_LABEL)
from deep_models import Autoencoder, DNNClassifier


class SentinelEngine:
    def __init__(self, classifier=None):
        art = joblib.load(MODEL_DIR / "preprocessor.joblib")
        self.preprocessor, self.selected = art["preprocessor"], art["selected"]

        self.classifier_name = classifier or json.load(open(MODEL_DIR / "best_classifier.json"))["name"]
        if self.classifier_name == "dnn":
            self.classifier = DNNClassifier.load(MODEL_DIR / "dnn.pt")
        else:
            self.classifier = joblib.load(MODEL_DIR / f"{self.classifier_name}.joblib")
            if hasattr(self.classifier, "n_jobs"):
                self.classifier.n_jobs = 1          # faster for small live batches

        self.autoencoder = Autoencoder.load(MODEL_DIR / "autoencoder.pt")
        self.threshold = json.load(open(MODEL_DIR / "ae_threshold.json"))["threshold"]

    # ------------------------------------------------------------------ helpers
    def encode(self, df):
        """Raw rows (DataFrame or list of dicts) -> (all features, selected features)."""
        df = pd.DataFrame(df)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        X = self.preprocessor.transform(df[FEATURES]).astype(np.float32)
        return X, X[:, self.selected]

    def scores(self, X_full, X_sel):
        """The raw outputs of both detectors (used by analyze and by evaluation)."""
        proba = self.classifier.predict_proba(X_sel)
        err = self.autoencoder.reconstruction_error(X_full)
        return proba, err

    def decide(self, proba, err, mode=None, weight=None, cutoff=None):
        """Apply the hybrid rule to detector outputs. Returns a dict of numpy arrays."""
        mode = mode or HYBRID_MODE
        w = HYBRID_WEIGHT if weight is None else weight
        cutoff = HYBRID_CUTOFF if cutoff is None else cutoff

        p_attack = 1.0 - proba[:, 0]
        attack_type = proba[:, 1:].argmax(axis=1) + 1            # most likely attack family
        clf_flag = p_attack >= CLF_ATTACK_THRESHOLD
        ae_score = err / (err + self.threshold)
        ae_flag = err > self.threshold

        if mode == "or":
            is_attack = clf_flag | ae_flag
            risk = np.maximum(p_attack, ae_score)
        elif mode == "weighted":
            risk = w * p_attack + (1 - w) * ae_score
            is_attack = risk >= cutoff
        else:
            raise ValueError("mode must be 'or' or 'weighted'")

        label = np.where(~is_attack, "Normal",
                         np.where(clf_flag, np.array(CLASSES, dtype=object)[attack_type], UNKNOWN_LABEL))
        verdict = np.where(~is_attack, "Normal",
                           np.where(clf_flag, "Known attack", "Zero-day suspect"))
        severity = np.where(~is_attack, "None",
                            np.where(clf_flag & ae_flag, "High",
                                     np.where(clf_flag | ae_flag, "Medium", "Low")))
        return dict(is_attack=is_attack, label=label, verdict=verdict, severity=severity,
                    risk=risk, p_attack=p_attack, clf_flag=clf_flag,
                    ae_error=err, ae_score=ae_score, ae_flag=ae_flag)

    # ------------------------------------------------------------------ main entry point
    def analyze(self, rows, mode=None, weight=None, cutoff=None):
        """Raw connections in -> one verdict per row out (DataFrame)."""
        t0 = time.perf_counter()
        X_full, X_sel = self.encode(rows)
        proba, err = self.scores(X_full, X_sel)
        out = self.decide(proba, err, mode, weight, cutoff)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        res = pd.DataFrame({
            "verdict": out["verdict"],
            "label": out["label"],
            "severity": out["severity"],
            "risk": out["risk"].round(4),
            "clf_p_attack": out["p_attack"].round(4),
            "clf_top_class": [CLASSES[i] for i in proba.argmax(axis=1)],
            "ae_error": out["ae_error"].round(6),
            "ae_flag": out["ae_flag"],
        })
        res.attrs["latency_ms_per_row"] = elapsed_ms / max(len(res), 1)
        return res


if __name__ == "__main__":
    # quick smoke test:  python src/hybrid_engine.py
    from config import DATA_DIR
    engine = SentinelEngine()
    sample = pd.read_csv(DATA_DIR / "demo_stream.csv").head(10)
    result = engine.analyze(sample)
    print(pd.concat([sample[["attack_name", "category"]], result], axis=1).to_string())
    print(f"classifier = {engine.classifier_name}, "
          f"latency {result.attrs['latency_ms_per_row']:.3f} ms/row")
