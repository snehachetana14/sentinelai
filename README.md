# 🛡️ SentinelAI
**A Hybrid Signature and Anomaly-Based Intrusion Detection System using Machine Learning and Deep Learning**

SentinelAI combines two detectors:
- a **supervised classifier** (Random Forest / XGBoost / Deep Neural Network) that recognises *known* attacks and names their type
- an **autoencoder**, trained only on normal traffic, that flags anything that doesn't look normal, including *never-seen-before (zero-day)* attacks

A **hybrid decision engine** merges both, served through a **Flask API** and a live **Streamlit dashboard**.

---

## 1. Setup (one time)

You need **Python 3.10, 3.11 or 3.12** (3.13 also works if PyTorch installs for you).

Open a terminal **inside the `SentinelAI` folder** and run:

**Windows**
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> The install takes a few minutes (PyTorch is the big one). The NSL-KDD dataset is already in `data/`.

## 2. Train everything (about 3-8 minutes)
```bash
python run_all.py
```
This runs 5 steps in order and prints the results as it goes:

| Step | Script | What it does |
|---|---|---|
| 1 | `src/preprocess.py` | cleans, encodes and scales the data, selects features, applies SMOTE |
| 2 | `src/train_classifiers.py` | trains Random Forest, XGBoost and a DNN, then picks the best |
| 3 | `src/train_autoencoder.py` | trains the autoencoder on normal traffic and picks the threshold from the ROC curve |
| 4 | `src/evaluate_hybrid.py` | compares signature vs anomaly vs hybrid on the test set |
| 5 | `src/architecture_diagram.py` | draws the architecture figure |

All charts and tables land in `results/` (ready for your report). Trained models land in `models/`.

## 3. Run the live dashboard
```bash
streamlit run dashboard.py
```
Your browser opens at `http://localhost:8501`.
- Press **▶ Start** to begin replaying traffic.
- Press **🚨 Inject zero-day attack burst** to send attack types the models never saw in training, then watch the purple "Zero-day suspect" alerts.
- Switch the decision rule (OR / weighted) in the sidebar and watch recall and false alarms change.
- The **Model performance** tab shows all the result charts.

## 4. (Optional) Run the Flask API
In a **second** terminal (with the venv activated):
```bash
python src/api.py
```
- `http://localhost:8000/health` checks the server is up
- `http://localhost:8000/sample` returns a random raw connection record
- `POST http://localhost:8000/predict` takes one record (JSON) and returns a verdict

In the dashboard sidebar, set **Run models via → Flask API** to route traffic through it.

Quick test from Python:
```python
import requests
rec = requests.get("http://localhost:8000/sample").json()["record"]
print(requests.post("http://localhost:8000/predict", json=rec).json())
```

---

## Project structure
```
SentinelAI/
├── data/                     NSL-KDD files (+ processed files after step 1)
├── models/                   trained models (after training)
├── results/                  charts + metric tables for the report
├── src/
│   ├── config.py             all settings in one place
│   ├── preprocess.py         step 1
│   ├── deep_models.py        PyTorch DNN classifier + autoencoder
│   ├── evaluation.py         metrics + chart helpers
│   ├── train_classifiers.py  step 2
│   ├── train_autoencoder.py  step 3
│   ├── hybrid_engine.py      the decision engine (used by API + dashboard)
│   ├── evaluate_hybrid.py    step 4
│   ├── architecture_diagram.py
│   └── api.py                Flask backend
├── dashboard.py              Streamlit dashboard
├── run_all.py                runs steps 1-5
└── requirements.txt
```

## Key results files (for the report)
| File | Use it for |
|---|---|
| `results/hybrid_comparison.csv` / `.png` | **main results table**: signature vs anomaly vs hybrid |
| `results/hybrid_tradeoff.png` | recall vs false-alarm trade-off of the weighted mode |
| `results/hybrid_gain_per_classifier.csv` | shows the hybrid helps with *every* classifier |
| `results/classifier_comparison.csv` / `.png` | RF vs XGBoost vs DNN |
| `results/cm_*.png` | confusion matrices |
| `results/ae_roc_curve.png`, `ae_error_distribution.png` | autoencoder ROC-AUC + threshold |
| `results/class_balance_smote.png` | effect of SMOTE |
| `results/feature_importance_mi.png` | top features |
| `results/latency.txt` | detection latency |
| `results/architecture.png` | system architecture diagram |

## Things to change (all in `src/config.py`)
- `DNN_EPOCHS`, `AE_EPOCHS`: train longer or shorter
- `CLASSIFIER_CHOICE`: force `"random_forest"`, `"xgboost"` or `"dnn"` instead of `"auto"`
- `HYBRID_MODE`, `HYBRID_WEIGHT`, `HYBRID_CUTOFF`: decision-engine behaviour
- `N_SELECTED_FEATURES`, `SMOTE_TARGETS`: preprocessing

## Notes and limitations (good material for the report)
- **Why is test accuracy ~75-80% for the classifiers?** The official NSL-KDD test set is deliberately harder than the training set and contains 17 attack types that never appear in training. Well-known papers report similar numbers. That gap is exactly what the anomaly path fixes.
- "Live" traffic is a **replay** of the test set, not packets captured from a real network card. Capturing live packets (e.g. with CICFlowMeter + Scapy) is future work.
- The anomaly threshold is chosen on validation data (Youden's J on the ROC curve). The weighted-mode defaults (w = 0.3, cutoff = 0.4) are a design choice; `hybrid_weight_sweep.csv` shows the full trade-off.
- Future work: adversarial attacks on the IDS itself, concept drift (retraining as traffic changes), a newer dataset (CICIDS2017), LSTM models on time-ordered flows, and scaling with Kafka/Spark.

## Troubleshooting
- `ModuleNotFoundError` → activate the venv, then `pip install -r requirements.txt`
- Dashboard says "Models not found" → run `python run_all.py` first
- PyTorch won't install → use Python 3.11 or 3.12, or see https://pytorch.org/get-started/locally/
- Port already in use → change `API_PORT` in `src/config.py`, or run `streamlit run dashboard.py --server.port 8502`
