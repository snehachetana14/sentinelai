"""
STEP 1 - preprocess.py
Turns the raw NSL-KDD text files into clean numeric arrays the models can learn from.

What happens here (in order):
  1. Load the files and give the columns names.
  2. Map ~40 attack names into 5 classes (Normal, DoS, Probe, R2L, U2R).
  3. Mark test rows whose attack type NEVER appears in training  -> "novel" (zero-day).
  4. Encode text columns (one-hot) and scale numbers to 0..1.
  5. Feature selection (mutual information) for the classifiers.
  6. Split off a validation set (used later to tune thresholds).
  7. SMOTE: create synthetic rows for the rare classes R2L and U2R.
  8. Save everything to data/ and models/.

Run:  python src/preprocess.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

from config import (ATTACK_MAP, CATEGORICAL, CLASS_COLORS, CLASSES, COLUMNS, DATA_DIR,
                    FEATURES, MODEL_DIR, N_SELECTED_FEATURES, NUMERIC, RANDOM_STATE,
                    RESULT_DIR, SMOTE_TARGETS, TEST_FILE, TRAIN_FILE, VAL_SIZE)


def load(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["attack_name"] = df["label"].str.strip().str.lower()
    unknown = set(df["attack_name"]) - set(ATTACK_MAP)
    if unknown:
        raise ValueError(f"Attack names missing from ATTACK_MAP in config.py: {unknown}")
    df["category"] = df["attack_name"].map(ATTACK_MAP)
    return df


def build_preprocessor():
    """Text -> one-hot columns.  Numbers -> log(1+x) -> scaled to 0..1.

    log(1+x) first because columns like src_bytes range from 0 to ~1 billion;
    without it, almost every value would be squashed to ~0 after scaling.
    """
    numeric = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", MinMaxScaler(clip=True)),   # clip: unseen huge values stay at 1
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def plot_class_counts(before, after, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, counts, title in [(axes[0], before, "Before SMOTE"), (axes[1], after, "After SMOTE")]:
        vals = [counts.get(c, 0) for c in CLASSES]
        bars = ax.bar(CLASSES, vals, color=[CLASS_COLORS[c] for c in CLASSES])
        ax.set_yscale("log")
        ax.set_title(title)
        ax.bar_label(bars, labels=[f"{v:,}" for v in vals], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Rows (log scale)")
    fig.suptitle("Training class balance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    print("Loading NSL-KDD ...")
    train_df, test_df = load(TRAIN_FILE), load(TEST_FILE)
    print(f"  train: {len(train_df):,} rows   test: {len(test_df):,} rows")

    # ---- novel (zero-day) attacks: names present in test but never in train
    train_names = set(train_df["attack_name"])
    test_df["is_novel"] = ~test_df["attack_name"].isin(train_names)
    novel_names = sorted(set(test_df.loc[test_df["is_novel"], "attack_name"]))
    print(f"  {test_df['is_novel'].sum():,} test rows use {len(novel_names)} attack types "
          f"never seen in training (our 'zero-day' set): {', '.join(novel_names)}")

    # ---- train / validation split (stratified = same class mix in both)
    tr_df, val_df = train_test_split(train_df, test_size=VAL_SIZE,
                                     stratify=train_df["category"], random_state=RANDOM_STATE)

    # ---- fit encoder + scaler on TRAIN ONLY (never peek at test data)
    pre = build_preprocessor()
    X_tr = pre.fit_transform(tr_df[FEATURES])
    X_val = pre.transform(val_df[FEATURES])
    X_te = pre.transform(test_df[FEATURES])
    feature_names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    to_y = lambda d: d["category"].map(CLASSES.index).to_numpy()
    y_tr, y_val, y_te = to_y(tr_df), to_y(val_df), to_y(test_df)
    print(f"  after encoding: {X_tr.shape[1]} columns")

    # ---- feature selection: rank columns by mutual information with the label
    sample = np.random.default_rng(RANDOM_STATE).choice(len(X_tr), min(20000, len(X_tr)), replace=False)
    mi = mutual_info_classif(X_tr[sample], y_tr[sample], random_state=RANDOM_STATE)
    selected = np.sort(np.argsort(mi)[::-1][:N_SELECTED_FEATURES])
    print(f"  feature selection: keeping top {len(selected)} of {X_tr.shape[1]} columns for the classifiers")

    order = np.argsort(mi)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in order][::-1], mi[order][::-1], color="#2a78d6")
    ax.set_xlabel("Mutual information with attack class")
    ax.set_title("Top 20 most informative features")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "feature_importance_mi.png", dpi=150)
    plt.close(fig)

    # ---- SMOTE on the classifier training set only
    counts = {c: int((y_tr == i).sum()) for i, c in enumerate(CLASSES)}
    strategy = {CLASSES.index(c): max(n, counts[c]) for c, n in SMOTE_TARGETS.items()}
    smote = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE, k_neighbors=5)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
    counts_sm = {c: int((y_tr_sm == i).sum()) for i, c in enumerate(CLASSES)}
    print(f"  class counts before SMOTE: {counts}")
    print(f"  class counts after  SMOTE: {counts_sm}")
    plot_class_counts(counts, counts_sm, RESULT_DIR / "class_balance_smote.png")

    # ---- save
    f32 = lambda a: a.astype(np.float32)
    np.savez_compressed(
        DATA_DIR / "processed.npz",
        X_train=f32(X_tr), y_train=y_tr,
        X_train_smote=f32(X_tr_sm), y_train_smote=y_tr_sm,
        X_val=f32(X_val), y_val=y_val,
        X_test=f32(X_te), y_test=y_te,
        test_is_novel=test_df["is_novel"].to_numpy(),
    )
    joblib.dump({"preprocessor": pre, "selected": selected, "feature_names": feature_names},
                MODEL_DIR / "preprocessor.joblib")
    test_df[["attack_name", "category", "is_novel"]].to_csv(DATA_DIR / "test_meta.csv", index=False)

    # demo files for the dashboard: raw rows replayed as "live traffic"
    keep = FEATURES + ["attack_name", "category", "is_novel"]
    test_df[keep].sample(n=5000, random_state=RANDOM_STATE).to_csv(DATA_DIR / "demo_stream.csv", index=False)
    test_df.loc[test_df["is_novel"], keep].to_csv(DATA_DIR / "demo_zero_day.csv", index=False)

    json.dump({"novel_attack_types": novel_names,
               "n_features_encoded": int(X_tr.shape[1]),
               "n_features_selected": int(len(selected)),
               "selected_features": [feature_names[i] for i in selected],
               "class_counts_before_smote": counts,
               "class_counts_after_smote": counts_sm},
              open(RESULT_DIR / "preprocessing_summary.json", "w"), indent=2)
    print("Done. Saved data/processed.npz and models/preprocessor.joblib")


if __name__ == "__main__":
    main()
