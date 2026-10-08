"""
dashboard.py - SentinelAI live monitoring dashboard (Streamlit).

Replays connections from the NSL-KDD TEST set as if they were arriving live,
runs each one through the hybrid engine, and shows verdicts, alerts and charts.

Run:  streamlit run dashboard.py
"""
import inspect
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import (API_PORT, CLASS_COLORS, CLASSES, DATA_DIR, FEATURES, HYBRID_CUTOFF,  # noqa: E402
                    HYBRID_WEIGHT, MODEL_DIR, RESULT_DIR, UNKNOWN_LABEL)

st.set_page_config(page_title="SentinelAI - Hybrid IDS", page_icon="🛡️", layout="wide")

VERDICT_ICON = {"Normal": "✅ Normal", "Known attack": "🚨 Known attack", "Zero-day suspect": "⚠️ Zero-day suspect"}
VERDICT_BG = {"Normal": "rgba(12,163,12,0.15)", "Known attack": "rgba(208,59,59,0.20)",
              "Zero-day suspect": "rgba(74,58,167,0.22)"}
LABEL_ORDER = CLASSES + [UNKNOWN_LABEL]
API_URL = f"http://localhost:{API_PORT}/predict"


def full_width(element):
    """Streamlit renamed 'use_container_width=True' to 'width="stretch"' in newer versions.
    This picks whichever one your installed version understands."""
    params = inspect.signature(element).parameters
    return {"use_container_width": True} if "use_container_width" in params else {"width": "stretch"}


BTN, TABLE, CHART = full_width(st.button), full_width(st.dataframe), full_width(st.plotly_chart)


# --------------------------------------------------------------------------- loading
needed = ["preprocessor.joblib", "best_classifier.json", "autoencoder.pt", "ae_threshold.json"]
if not all((MODEL_DIR / f).exists() for f in needed):
    st.error("Models not found. Train them first:  `python run_all.py`")
    st.stop()


@st.cache_resource
def load_engine():
    from hybrid_engine import SentinelEngine
    return SentinelEngine()


@st.cache_data
def load_streams():
    return pd.read_csv(DATA_DIR / "demo_stream.csv"), pd.read_csv(DATA_DIR / "demo_zero_day.csv")


engine = load_engine()
stream, zero_day_pool = load_streams()

ss = st.session_state


def reset():
    ss.running = False
    ss.ptr = 0
    ss.inject = 0
    ss.next_id = 1
    ss.history = pd.DataFrame()
    ss.latencies = []


if "running" not in ss:
    reset()

# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ Controls")
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Start", **BTN):
    ss.running = True
if c2.button("⏸ Pause", **BTN):
    ss.running = False
if st.sidebar.button("🚨 Inject zero-day attack burst", **BTN,
                     help="Next 25 connections come from attack types the models NEVER saw in training"):
    ss.inject += 25
if st.sidebar.button("↺ Reset", **BTN):
    reset()

st.sidebar.divider()
rows_per_tick = st.sidebar.slider("Connections per tick", 1, 50, 8)
delay = st.sidebar.slider("Seconds between ticks", 0.2, 3.0, 1.0, 0.1)
mode = st.sidebar.radio("Hybrid decision rule", ["or", "weighted"],
                        format_func=lambda m: "OR - alert if either model fires" if m == "or"
                        else "Weighted - blended score")
weight, cutoff = HYBRID_WEIGHT, HYBRID_CUTOFF
if mode == "weighted":
    weight = st.sidebar.slider("Classifier weight (autoencoder gets the rest)", 0.0, 1.0, HYBRID_WEIGHT, 0.05)
    cutoff = st.sidebar.slider("Alert cutoff", 0.1, 0.9, HYBRID_CUTOFF, 0.05)
backend = st.sidebar.radio("Run models via", ["Local engine", "Flask API"],
                           help="Flask API needs `python src/api.py` running in another terminal")
st.sidebar.caption(f"Classifier: **{engine.classifier_name}** · Anomaly threshold: **{engine.threshold:.5f}**")


# --------------------------------------------------------------------------- one tick of "live" traffic
def next_batch(n):
    parts = []
    k = min(ss.inject, n)
    if k:
        parts.append(zero_day_pool.sample(k))
        ss.inject -= k
    rest = n - k
    if rest:
        idx = [(ss.ptr + i) % len(stream) for i in range(rest)]
        parts.append(stream.iloc[idx])
        ss.ptr = (ss.ptr + rest) % len(stream)
    return pd.concat(parts, ignore_index=True)


def classify(batch):
    t0 = time.perf_counter()
    if backend == "Flask API":
        try:
            payload = {"records": batch[FEATURES].to_dict(orient="records"),
                       "mode": mode, "weight": weight, "cutoff": cutoff}
            r = requests.post(API_URL, json=payload, timeout=5)
            r.raise_for_status()
            res = pd.DataFrame(r.json()["results"])
        except requests.RequestException:
            st.toast("Flask API not reachable - using the local engine. Start it with `python src/api.py`.")
            res = engine.analyze(batch[FEATURES], mode=mode, weight=weight, cutoff=cutoff)
    else:
        res = engine.analyze(batch[FEATURES], mode=mode, weight=weight, cutoff=cutoff)
    ss.latencies.append((time.perf_counter() - t0) * 1000 / len(batch))
    return res


if ss.running:
    batch = next_batch(rows_per_tick)
    res = classify(batch)
    now = datetime.now().strftime("%H:%M:%S")
    rec = pd.DataFrame({
        "id": range(ss.next_id, ss.next_id + len(batch)),
        "time": now,
        "protocol": batch["protocol_type"],
        "service": batch["service"],
        "src_bytes": batch["src_bytes"],
        "verdict": res["verdict"],
        "detected_as": res["label"],
        "severity": res["severity"],
        "risk": res["risk"],
        "true_class": batch["category"],
        "true_attack": batch["attack_name"],
        "zero_day_truth": batch["is_novel"].astype(bool),
    })
    ss.next_id += len(batch)
    ss.history = pd.concat([ss.history, rec], ignore_index=True).tail(5000)

# --------------------------------------------------------------------------- main page
st.title("🛡️ SentinelAI")
st.caption("Hybrid signature + anomaly intrusion detection · live replay of the NSL-KDD test set")
status = "🟢 Monitoring" if ss.running else "⚪ Paused - press ▶ Start in the sidebar"
if ss.inject:
    status += f" · {ss.inject} zero-day connections queued"
st.markdown(f"**Status:** {status}")

tab_live, tab_perf, tab_how = st.tabs(["📡 Live monitor", "📊 Model performance", "🧠 How it works"])
h = ss.history

with tab_live:
    total = len(h)
    attacks = int((h["verdict"] != "Normal").sum()) if total else 0
    zero_day = int((h["verdict"] == "Zero-day suspect").sum()) if total else 0
    real_attacks = h[h["true_class"] != "Normal"] if total else h
    caught = (real_attacks["verdict"] != "Normal").mean() if len(real_attacks) else None
    normals = h[h["true_class"] == "Normal"] if total else h
    false_alarm = (normals["verdict"] != "Normal").mean() if len(normals) else None
    lat = sum(ss.latencies[-50:]) / len(ss.latencies[-50:]) if ss.latencies else None

    k = st.columns(5)
    k[0].metric("Connections analysed", f"{total:,}")
    k[1].metric("Alerts raised", f"{attacks:,}", f"{attacks / total:.0%} of traffic" if total else None,
                delta_color="off")
    k[2].metric("Zero-day suspects", f"{zero_day:,}")
    k[3].metric("Attacks caught (recall)", f"{caught:.1%}" if caught is not None else "-",
                f"false alarms {false_alarm:.1%}" if false_alarm is not None else None, delta_color="off")
    k[4].metric("Detection latency", f"{lat:.2f} ms/conn" if lat is not None else "-")

    if not total:
        st.info("No traffic yet. Press **▶ Start** in the sidebar. "
                "Try **🚨 Inject zero-day attack burst** to see the anomaly detector catch unseen attacks.")
    else:
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Live traffic feed")
            feed = h.tail(15).iloc[::-1].copy()
            feed["true_class"] = feed["true_class"] + feed["zero_day_truth"].map({True: " (zero-day)", False: ""})
            feed = feed[["id", "time", "protocol", "service", "src_bytes", "verdict", "detected_as",
                         "severity", "risk", "true_class", "true_attack"]]
            styled = (feed.style
                      .apply(lambda r: [f"background-color: {VERDICT_BG[r['verdict']]}"] * len(r), axis=1)
                      .format({"verdict": VERDICT_ICON.get, "risk": "{:.2f}"}))
            st.dataframe(styled, hide_index=True, **TABLE, height=565)
        with right:
            st.subheader("Detected attack types")
            alerts = h[h["verdict"] != "Normal"]
            if len(alerts):
                counts = (alerts["detected_as"].value_counts()
                          .reindex(LABEL_ORDER).dropna().astype(int).reset_index())
                counts.columns = ["type", "count"]
                fig = px.bar(counts, x="count", y="type", orientation="h", color="type",
                             color_discrete_map=CLASS_COLORS, text="count")
                fig.update_layout(showlegend=False, height=260, margin=dict(l=0, r=10, t=10, b=0),
                                  yaxis=dict(title=None, categoryorder="array",
                                             categoryarray=list(reversed(LABEL_ORDER))),
                                  xaxis_title="Alerts")
                fig.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(fig, **CHART, key="types")
            else:
                st.caption("No alerts yet.")

            st.subheader("Verdict mix")
            mix = h["verdict"].value_counts().reindex(list(VERDICT_ICON)).fillna(0).astype(int)
            for v, n in mix.items():
                st.markdown(f"{VERDICT_ICON[v]}: **{n:,}**")

        st.subheader("Risk score - last 300 connections")
        tail = h.tail(300)
        cut = cutoff if mode == "weighted" else 0.5
        fig = px.scatter(tail, x="id", y="risk", color="detected_as", color_discrete_map=CLASS_COLORS,
                         category_orders={"detected_as": LABEL_ORDER},
                         hover_data=["time", "service", "verdict", "true_attack"])
        fig.add_hline(y=cut, line_dash="dash", line_color="#898781",
                      annotation_text=f"alert line ({cut:.2f})", annotation_position="top left")
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="Detected as",
                          xaxis_title="Connection #", yaxis_title="Risk", yaxis_range=[0, 1.05])
        st.plotly_chart(fig, **CHART, key="risk")

        st.subheader("🚨 Alert log")
        log = h[h["verdict"] != "Normal"].iloc[::-1]
        st.dataframe(log[["id", "time", "verdict", "detected_as", "severity", "risk", "protocol",
                          "service", "true_class", "true_attack"]].head(200)
                     .style.format({"verdict": VERDICT_ICON.get, "risk": "{:.2f}"}),
                     hide_index=True, **TABLE, height=300)
        st.download_button("Download alert log (CSV)", log.to_csv(index=False), "sentinelai_alerts.csv",
                           "text/csv")

with tab_perf:
    st.subheader("Signature vs anomaly vs hybrid (full NSL-KDD test set, 22,544 connections)")
    table = RESULT_DIR / "hybrid_comparison.csv"
    if table.exists():
        st.dataframe(pd.read_csv(table).style.format(precision=3), hide_index=True, **TABLE)
    else:
        st.info("Run `python src/evaluate_hybrid.py` to generate results.")
    charts = [("hybrid_comparison.png", "Detector comparison"), ("hybrid_tradeoff.png", "Recall vs false alarms"),
              ("cm_hybrid.png", "Hybrid confusion matrix"), ("classifier_comparison.png", "Classifier comparison"),
              ("ae_error_distribution.png", "Reconstruction error"), ("ae_roc_curve.png", "Autoencoder ROC")]
    available = [(f, t) for f, t in charts if (RESULT_DIR / f).exists()]
    for i in range(0, len(available), 2):
        cols = st.columns(2)
        for col, (f, t) in zip(cols, available[i:i + 2]):
            col.markdown(f"**{t}**")
            col.image(str(RESULT_DIR / f))

with tab_how:
    if (RESULT_DIR / "architecture.png").exists():
        st.image(str(RESULT_DIR / "architecture.png"))
    st.markdown("""
**1 · Preprocessing** - text columns are one-hot encoded, numbers are log-scaled to 0..1.

**2 · Signature path (supervised classifier)** - trained on labelled attacks, it names the
attack family: DoS, Probe, R2L or U2R. Strong on attacks it has seen, blind to new ones.

**3 · Anomaly path (autoencoder)** - trained only on normal traffic, it tries to rebuild each
connection. A large rebuild error means *"this doesn't look like anything normal"*.

**4 · Hybrid decision engine**

| Classifier | Autoencoder | Verdict |
|---|---|---|
| attack | anomaly | 🚨 Known attack (severity High) |
| attack | normal | 🚨 Known attack (severity Medium) |
| normal | anomaly | ⚠️ Zero-day suspect |
| normal | normal | ✅ Normal |

**Zero-day burst** replays attack types (e.g. *mscan, apache2, snmpguess*) that appear only in
the test set, so neither model saw them during training.
""")

# --------------------------------------------------------------------------- keep the loop going
if ss.running:
    time.sleep(delay)
    st.rerun()
