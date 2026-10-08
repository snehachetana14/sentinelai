"""
architecture_diagram.py - draws the system architecture figure for the report
(results/architecture.png).

Run:  python src/architecture_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import RESULT_DIR

INK, MUTED = "#0b0b0b", "#52514e"


def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color + "22", ec=color, lw=2))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.3)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=MUTED))


def main():
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    box(ax, 0.2, 2.0, 2.0, 1.3, "Network traffic", "NSL-KDD connections\n(replayed live)", "#898781")
    box(ax, 2.8, 2.0, 2.2, 1.3, "Preprocessing", "one-hot encoding\nlog + min-max scaling", "#2a78d6")
    box(ax, 5.7, 3.3, 2.6, 1.4, "Signature path", "feature selection (MI)\nRF / XGBoost / DNN\n+ SMOTE balancing", "#eb6834")
    box(ax, 5.7, 0.5, 2.6, 1.4, "Anomaly path", "Autoencoder trained on\nnormal traffic only\nerror > threshold?", "#4a3aa7")
    box(ax, 9.0, 2.0, 2.3, 1.3, "Hybrid decision\nengine", "\nOR rule / weighted score", "#1baf7a")
    box(ax, 11.9, 3.3, 1.9, 1.3, "Flask API", "POST /predict", "#eda100")
    box(ax, 11.9, 0.7, 1.9, 1.3, "Streamlit\ndashboard", "\nlive feed, alerts", "#e87ba4")

    arrow(ax, (2.2, 2.65), (2.8, 2.65))
    arrow(ax, (5.0, 2.9), (5.7, 3.9))
    arrow(ax, (5.0, 2.4), (5.7, 1.3))
    arrow(ax, (8.3, 3.9), (9.0, 2.95))
    arrow(ax, (8.3, 1.3), (9.0, 2.35))
    arrow(ax, (11.3, 2.9), (11.9, 3.8))
    arrow(ax, (11.3, 2.4), (11.9, 1.5))
    ax.text(8.95, 3.75, "attack type\n+ P(attack)", fontsize=7.5, color=MUTED, ha="center")
    ax.text(8.95, 1.15, "reconstruction\nerror", fontsize=7.5, color=MUTED, ha="center")
    ax.text(7.0, 4.95, "SentinelAI - hybrid intrusion detection architecture", ha="center",
            fontsize=13, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "architecture.png", dpi=160)
    plt.close(fig)
    print("Saved results/architecture.png")


if __name__ == "__main__":
    main()
