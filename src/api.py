"""
api.py - Flask backend that serves the hybrid engine over HTTP.
Any other system (a firewall, a SIEM, our dashboard) can send connection records
here and get verdicts back - that's how this would plug into a real network.

Run:   python src/api.py            -> http://localhost:8000

Endpoints
  GET  /health    is the server up + which models are loaded
  GET  /sample    one random raw record from the test set (handy for trying /predict)
  POST /predict   body: one record {...} or {"records": [{...}, ...]}
                  optional: "mode": "or" | "weighted", "weight": 0..1, "cutoff": 0..1
"""
import pandas as pd
from flask import Flask, jsonify, request

from config import API_PORT, DATA_DIR, FEATURES
from hybrid_engine import SentinelEngine

app = Flask(__name__)
engine = SentinelEngine()
_samples = pd.read_csv(DATA_DIR / "demo_stream.csv")


@app.get("/health")
def health():
    return jsonify(status="ok", classifier=engine.classifier_name,
                   anomaly_threshold=engine.threshold, n_features=len(FEATURES))


@app.get("/sample")
def sample():
    row = _samples.sample(1).iloc[0]
    return jsonify(record={k: row[k].item() if hasattr(row[k], "item") else row[k] for k in FEATURES},
                   true_label=row["category"], attack_name=row["attack_name"])


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not body:
        return jsonify(error="Send JSON: one record or {'records': [...]}"), 400
    mode = body.get("mode")
    records = body["records"] if "records" in body else [body]
    weight, cutoff = body.get("weight"), body.get("cutoff")
    records = [{k: v for k, v in r.items() if k not in ("mode", "weight", "cutoff")} for r in records]
    try:
        result = engine.analyze(records, mode=mode, weight=weight, cutoff=cutoff)
    except (ValueError, KeyError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(results=result.to_dict(orient="records"),
                   latency_ms_per_row=round(result.attrs["latency_ms_per_row"], 4))


if __name__ == "__main__":
    print(f"SentinelAI API running on http://localhost:{API_PORT}  (classifier: {engine.classifier_name})")
    app.run(host="0.0.0.0", port=API_PORT, debug=False)
