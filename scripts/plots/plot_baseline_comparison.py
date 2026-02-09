#!/usr/bin/env python3
"""
Plot comparison charts between baseline pilot and tests summary CSVs.

Each chart shows per-model values with split by pilot vs tests.
Total task counts are shown in the legend labels (if consistent).
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


def _load_summary(csv_path: str, split_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["split"] = split_label
    return df


def _legend_label(df: pd.DataFrame, split_label: str) -> str:
    totals = df.loc[df["split"] == split_label, "total_tasks"].dropna().unique()
    if len(totals) == 1:
        return f"{split_label} (n={int(totals[0])})"
    return f"{split_label} (n=varies)"


def _plot_metric(df: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    grouped = df.groupby(["model", "split"], as_index=False)[metric].mean()
    pivot = grouped.pivot(index="model", columns="split", values=metric)

    split_labels = {split: _legend_label(df, split) for split in pivot.columns}
    pivot = pivot.rename(columns=split_labels)

    ax = pivot.plot(kind="bar", figsize=(11, 5), rot=30)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.legend(title="Split")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline pilot vs tests comparison")
    parser.add_argument(
        "--pilot_csv",
        type=str,
        default=str(PROJECT_ROOT / "results/reports/baseline/summary_pilot.csv"),
        help="Path to pilot summary CSV",
    )
    parser.add_argument(
        "--tests_csv",
        type=str,
        default=str(PROJECT_ROOT / "results/reports/baseline/summary_tests.csv"),
        help="Path to tests summary CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "results/reports/baseline/plots"),
        help="Directory for output charts",
    )

    args = parser.parse_args()

    pilot_df = _load_summary(args.pilot_csv, "pilot")
    tests_df = _load_summary(args.tests_csv, "tests")

    df = pd.concat([pilot_df, tests_df], ignore_index=True)
    df = df.sort_values(["model", "split"]).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_titles = {
        "compilable_count": "Compilable Count",
        "compile_rate_%": "Compile Rate (%)",
        "pass_count": "Pass Count",
        "pass_rate_%": "Pass Rate (%)",
        "avg_duration_s": "Average Duration (s)",
        "avg_runtime_ms": "Average Runtime (ms)",
        "avg_code_bleu": "Average Code BLEU",
        "avg_codebert_f1": "Average CodeBERT F1",
    }

    for metric, title in metric_titles.items():
        if metric not in df.columns:
            print(f"Skipping {metric}: column not found in input summaries")
            continue
        output_path = output_dir / f"compare_{metric}.png"
        _plot_metric(df, metric, title, output_path)

    print(f"Charts saved to: {output_dir}")


if __name__ == "__main__":
    main()