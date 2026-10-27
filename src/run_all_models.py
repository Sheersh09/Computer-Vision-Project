"""
Step 5 of the build order: once train.py works for Xception, generalize to
the remaining five backbones, produce the Table 8-style comparison, and plot
the Figure 2-style weighted-accuracy bar chart.

Usage:
    python run_all_models.py            # skips backbones that already have a report
    python run_all_models.py --force    # retrain everything
"""
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

from config import BACKBONES, RESULTS_TABLES_DIR, RESULTS_PLOTS_DIR
from train import train_and_evaluate


def _plot_comparison(df: pd.DataFrame):
    os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)
    plot_df = df.sort_values("weighted_accuracy_avg")
    plt.figure(figsize=(8, 5))
    bars = plt.barh(plot_df["model"], plot_df["weighted_accuracy_avg"], color="#4C72B0")
    for bar, value in zip(bars, plot_df["weighted_accuracy_avg"]):
        plt.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.4f}", va="center")
    plt.xlabel("Weighted Accuracy Avg")
    plt.title("Model Accuracy")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_PLOTS_DIR, "model_comparison_bar.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    rows = []
    for name in BACKBONES:
        print(f"\n{'=' * 25} {name} {'=' * 25}")
        report_path = os.path.join(RESULTS_TABLES_DIR, f"{name}_report.json")
        if os.path.exists(report_path) and "--force" not in sys.argv:
            print(f"[skip] {name}: found {report_path} (add --force to retrain)")
            with open(report_path) as f:
                summary = json.load(f)["summary"]
        else:
            summary, _ = train_and_evaluate(name)
        rows.append({"model": name, **summary})

    df = pd.DataFrame(rows).sort_values("weighted_accuracy_avg", ascending=False).reset_index(drop=True)
    os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_TABLES_DIR, "model_comparison.csv")
    df.to_csv(out_path, index=False)

    print("\n=== Final comparison (Table 8 equivalent) ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_path}")

    _plot_comparison(df)


if __name__ == "__main__":
    main()
