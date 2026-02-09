#!/usr/bin/env python3
"""
Plot model comparisons between pass@k and repair-loop summaries.

Expected inputs:
- pass_at_k summary CSV with columns like:
  model,total_tasks,avg_att_to_pass*,p1_rate*,p_any_rate*,avg_code_bleu,avg_code_bert_f1
- repair summary CSV with columns like:
  model,total_tasks,avg_att_to_pass*,p1_rate*,p_any_rate*,avg_code_bleu,avg_codebert_f1

Outputs several PNG charts in the output directory.
"""

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting. Install with: pip install matplotlib"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_summary(csv_path: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["source"] = source

    # Normalize column naming between reports.
    if "avg_codebert_f1" in df.columns and "avg_code_bert_f1" not in df.columns:
        df = df.rename(columns={"avg_codebert_f1": "avg_code_bert_f1"})

    return df


def _plot_metric(df: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    # Handle duplicate model/source rows by averaging the metric first.
    grouped = df.groupby(["model", "source"], as_index=False)[metric].mean()
    pivot = grouped.pivot(index="model", columns="source", values=metric)
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=30)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.legend(title="Source")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pass@k vs repair-loop model comparisons")
    parser.add_argument(
        "--pass_at_k_csv",
        type=str,
        default=str(PROJECT_ROOT / "results/reports/pass_at_k/summary_pass_at_k.csv"),
        help="Path to pass@k summary CSV",
    )
    parser.add_argument(
        "--repair_csv",
        type=str,
        default=str(PROJECT_ROOT / "results/reports/repairs/summary_repairs.csv"),
        help="Path to repair-loop summary CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "results/reports/plots"),
        help="Directory for output charts",
    )

    args = parser.parse_args()

    pass_df = _load_summary(args.pass_at_k_csv, "pass_at_k")
    repair_df = _load_summary(args.repair_csv, "repair_loop")

    df = pd.concat([pass_df, repair_df], ignore_index=True)
    df = df.sort_values(["model", "source"]).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_titles = {
        "p1_rate*": "Pass@1 Rate (Attempt)",
        "p_any_rate*": "Pass Any Rate",
        "avg_att_to_pass*": "Average Attempts to Pass",
        "avg_code_bleu": "Average Code BLEU",
        "avg_code_bert_f1": "Average CodeBERT F1",
    }

    for metric, title in metric_titles.items():
        if metric not in df.columns:
            print(f"Skipping {metric}: column not found in input summaries")
            continue
        output_path = output_dir / f"compare_{metric.replace('*', '')}.png"
        _plot_metric(df, metric, title, output_path)

    print(f"Charts saved to: {output_dir}")


if __name__ == "__main__":
    main()