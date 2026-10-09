#!/usr/bin/env python3
"""
analyze.py
Loads all experiment result JSON files and optional pcap traces, performs
statistical analysis, and produces publication-quality plots for the research paper.

Outputs:
    - flow_size_cdf.png            CDF of flow sizes by phase and class
    - ttfb_by_concurrency.png      TTFB distributions across concurrency levels
    - inter_token_intervals.png    Decode phase inter-token timing
    - burst_analysis.png           Burst characterization
    - arrival_process.png          Request inter-arrival time analysis
    - summary_stats.csv            Full numerical summary table

Usage:
    python3 scripts/analyze.py
    python3 scripts/analyze.py --results-dir ./results/ --output-dir ./results/plots/
    python3 scripts/analyze.py --no-pcap   # skip pcap analysis if traces not available
"""

import json
import argparse
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

warnings.filterwarnings("ignore", category=FutureWarning)

console = Console()

RESULTS_DIR = Path(__file__).parent.parent / "results"
TRACES_DIR = RESULTS_DIR / "traces"
PLOTS_DIR = RESULTS_DIR / "plots"

CONCURRENCY_LEVELS = [1, 8, 32, 64]
PROMPT_CLASSES = ["short", "medium", "long"]

# Plot styling
PALETTE = {
    "short":  "#2E75B6",
    "medium": "#ED7D31",
    "long":   "#70AD47",
    1:        "#1F3864",
    8:        "#2E75B6",
    32:       "#ED7D31",
    64:       "#C00000",
}
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    records = []
    files = list(results_dir.glob("exp_c*.json"))
    if not files:
        console.print(f"[red]No experiment result files found in {results_dir}.[/red]")
        console.print("Run: python3 scripts/run_experiments.py --full-matrix")
        raise SystemExit(1)

    for f in sorted(files):
        with open(f) as fh:
            data = json.load(fh)
        records.extend(data)
        console.print(f"  Loaded {len(data):>4} records from {f.name}")

    df = pd.DataFrame(records)
    df = df[df["success"]].copy()
    df = df[df["ttfb"] > 0].copy()
    console.print(f"\n[green]Total successful requests: {len(df):,}[/green]")
    return df


# ---------------------------------------------------------------------------
# Plot 1: Flow Size CDF (prefill proxy = request bytes, decode proxy = response bytes)
# ---------------------------------------------------------------------------

def plot_flow_size_cdf(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Flow Size Distributions: Prefill vs. Decode Phases", fontweight="bold")

    for cls in PROMPT_CLASSES:
        sub = df[df["prompt_class"] == cls]
        color = PALETTE[cls]

        # Prefill phase: bytes sent (request payload size)
        req_bytes = np.sort(sub["request_bytes_sent"].values)
        cdf = np.arange(1, len(req_bytes) + 1) / len(req_bytes)
        axes[0].plot(req_bytes, cdf, label=cls, color=color, linewidth=2)

        # Decode phase: bytes received (response payload size)
        resp_bytes = np.sort(sub["response_bytes_received"].values)
        cdf = np.arange(1, len(resp_bytes) + 1) / len(resp_bytes)
        axes[1].plot(resp_bytes, cdf, label=cls, color=color, linewidth=2)

    for ax, title in zip(axes, ["Prefill Phase (Request Bytes)", "Decode Phase (Response Bytes)"]):
        ax.set_xlabel("Bytes")
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.legend(title="Prompt class")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "flow_size_cdf.png", bbox_inches="tight")
    plt.close()
    console.print(f"  [green]Saved:[/green] flow_size_cdf.png")


# ---------------------------------------------------------------------------
# Plot 2: TTFB by Concurrency
# ---------------------------------------------------------------------------

def plot_ttfb_by_concurrency(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Time-to-First-Byte (Prefill Duration Proxy) by Concurrency Level",
                 fontweight="bold")

    for ax, cls in zip(axes, PROMPT_CLASSES):
        sub = df[df["prompt_class"] == cls]
        data_by_conc = {c: sub[sub["concurrency"] == c]["ttfb"].values
                        for c in CONCURRENCY_LEVELS}
        data_by_conc = {k: v for k, v in data_by_conc.items() if len(v) > 0}

        positions = list(range(len(data_by_conc)))
        bp = ax.boxplot(
            list(data_by_conc.values()),
            positions=positions,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, conc in zip(bp["boxes"], data_by_conc.keys()):
            patch.set_facecolor(PALETTE[conc])
            patch.set_alpha(0.75)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"c={c}" for c in data_by_conc.keys()])
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("TTFB (seconds)")
        ax.set_title(f"Prompt class: {cls}")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "ttfb_by_concurrency.png", bbox_inches="tight")
    plt.close()
    console.print(f"  [green]Saved:[/green] ttfb_by_concurrency.png")


# ---------------------------------------------------------------------------
# Plot 3: Inter-Token Intervals (Decode Phase)
# ---------------------------------------------------------------------------

def plot_inter_token_intervals(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Decode Phase: Inter-Token Interval Distributions", fontweight="bold")

    # Left: CDF by prompt class (at lowest concurrency)
    ax = axes[0]
    for cls in PROMPT_CLASSES:
        sub = df[(df["prompt_class"] == cls) & (df["concurrency"] == 1)]
        all_iti = np.concatenate(sub["inter_token_intervals"].apply(
            lambda x: x if isinstance(x, list) else []
        ).values)
        if len(all_iti) == 0:
            continue
        sorted_iti = np.sort(all_iti)
        cdf = np.arange(1, len(sorted_iti) + 1) / len(sorted_iti)
        ax.plot(sorted_iti * 1000, cdf, label=cls, color=PALETTE[cls], linewidth=2)

    ax.set_xlabel("Inter-token interval (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("By Prompt Class (concurrency=1)")
    ax.legend(title="Prompt class")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(left=0)

    # Right: median ITI by concurrency level (short prompts)
    ax = axes[1]
    for cls in PROMPT_CLASSES:
        medians = []
        concs = []
        for c in CONCURRENCY_LEVELS:
            sub = df[(df["prompt_class"] == cls) & (df["concurrency"] == c)]
            all_iti = np.concatenate(sub["inter_token_intervals"].apply(
                lambda x: x if isinstance(x, list) else []
            ).values)
            if len(all_iti) > 0:
                medians.append(np.median(all_iti) * 1000)
                concs.append(c)
        if medians:
            ax.plot(concs, medians, marker="o", label=cls, color=PALETTE[cls], linewidth=2)

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Median ITI (ms)")
    ax.set_title("Median Inter-Token Interval vs. Concurrency")
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.legend(title="Prompt class")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "inter_token_intervals.png", bbox_inches="tight")
    plt.close()
    console.print(f"  [green]Saved:[/green] inter_token_intervals.png")


# ---------------------------------------------------------------------------
# Plot 4: Burst Analysis
# ---------------------------------------------------------------------------

def detect_bursts(send_times: np.ndarray, window_s: float = 0.5) -> list[dict]:
    """Identify bursts: clusters of requests within a sliding window."""
    if len(send_times) == 0:
        return []
    send_times = np.sort(send_times)
    bursts = []
    i = 0
    while i < len(send_times):
        j = i
        while j < len(send_times) and send_times[j] - send_times[i] <= window_s:
            j += 1
        count = j - i
        duration = send_times[j - 1] - send_times[i] if count > 1 else 0.0
        bursts.append({"count": count, "duration": duration, "start": send_times[i]})
        i = j
    return bursts


def plot_burst_analysis(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Request Burst Characterization", fontweight="bold")

    ax = axes[0]
    for c in CONCURRENCY_LEVELS:
        sub = df[df["concurrency"] == c]
        if len(sub) == 0:
            continue
        send_times = sub["send_time"].values
        bursts = detect_bursts(send_times)
        counts = [b["count"] for b in bursts]
        sorted_counts = np.sort(counts)
        cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        ax.plot(sorted_counts, cdf, label=f"c={c}", color=PALETTE[c], linewidth=2)

    ax.set_xlabel("Requests per burst window (0.5s)")
    ax.set_ylabel("CDF")
    ax.set_title("Burst Magnitude CDF by Concurrency")
    ax.legend(title="Concurrency")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for c in CONCURRENCY_LEVELS:
        sub = df[df["concurrency"] == c]
        if len(sub) == 0:
            continue
        send_times = sub["send_time"].values
        iat = np.diff(np.sort(send_times))
        if len(iat) == 0:
            continue
        sorted_iat = np.sort(iat)
        cdf = np.arange(1, len(sorted_iat) + 1) / len(sorted_iat)
        ax.plot(sorted_iat * 1000, cdf, label=f"c={c}", color=PALETTE[c], linewidth=2)

    ax.set_xlabel("Inter-request arrival time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Inter-Request Arrival Time CDF by Concurrency")
    ax.legend(title="Concurrency")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(out / "burst_analysis.png", bbox_inches="tight")
    plt.close()
    console.print(f"  [green]Saved:[/green] burst_analysis.png")


# ---------------------------------------------------------------------------
# Plot 5: Arrival Process Characterization
# ---------------------------------------------------------------------------

def plot_arrival_process(df: pd.DataFrame, out: Path):
    """
    Test whether the inter-arrival process is Poisson (exponential IAT)
    by comparing empirical CDF against a fitted exponential.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Arrival Process Analysis: Empirical vs. Poisson Model", fontweight="bold")

    for ax, c in zip(axes.flat, CONCURRENCY_LEVELS):
        sub = df[df["concurrency"] == c]
        if len(sub) < 10:
            ax.set_visible(False)
            continue

        send_times = np.sort(sub["send_time"].values)
        iat = np.diff(send_times)
        if len(iat) == 0:
            ax.set_visible(False)
            continue

        rate = 1.0 / np.mean(iat)
        x = np.linspace(0, np.percentile(iat, 99), 300)
        fitted_cdf = 1 - np.exp(-rate * x)

        sorted_iat = np.sort(iat)
        emp_cdf = np.arange(1, len(sorted_iat) + 1) / len(sorted_iat)

        ax.plot(sorted_iat * 1000, emp_cdf, color=PALETTE[c], linewidth=2, label="Empirical")
        ax.plot(x * 1000, fitted_cdf, color="black", linewidth=1.5, linestyle="--",
                label=f"Poisson fit (λ={rate:.1f}/s)")

        ks_stat, ks_p = stats.kstest(iat, "expon", args=(0, 1 / rate))
        ax.set_title(f"Concurrency = {c}  (KS p={ks_p:.3f})")
        ax.set_xlabel("Inter-arrival time (ms)")
        ax.set_ylabel("CDF")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(left=0)

        note = "Consistent with Poisson" if ks_p > 0.05 else "Departs from Poisson"
        ax.text(0.97, 0.05, note, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                color="green" if ks_p > 0.05 else "red")

    plt.tight_layout()
    plt.savefig(out / "arrival_process.png", bbox_inches="tight")
    plt.close()
    console.print(f"  [green]Saved:[/green] arrival_process.png")


# ---------------------------------------------------------------------------
# Summary Statistics CSV
# ---------------------------------------------------------------------------

def build_summary_csv(df: pd.DataFrame, out: Path):
    rows = []
    for c, cls in product(CONCURRENCY_LEVELS, PROMPT_CLASSES):
        sub = df[(df["concurrency"] == c) & (df["prompt_class"] == cls)]
        if len(sub) == 0:
            continue

        ttfbs = sub["ttfb"].values
        durations = sub["total_duration"].values
        req_bytes = sub["request_bytes_sent"].values
        resp_bytes = sub["response_bytes_received"].values
        token_counts = sub["tokens_received"].values
        all_iti = np.concatenate(sub["inter_token_intervals"].apply(
            lambda x: x if isinstance(x, list) else []
        ).values)

        def q(arr, p): return float(np.percentile(arr, p)) if len(arr) else float("nan")

        rows.append({
            "concurrency": c,
            "prompt_class": cls,
            "n": len(sub),
            "ttfb_median_s": q(ttfbs, 50),
            "ttfb_p95_s": q(ttfbs, 95),
            "ttfb_p99_s": q(ttfbs, 99),
            "duration_median_s": q(durations, 50),
            "duration_p95_s": q(durations, 95),
            "req_bytes_median": q(req_bytes, 50),
            "resp_bytes_median": q(resp_bytes, 50),
            "tokens_median": q(token_counts, 50),
            "iti_median_ms": q(all_iti, 50) * 1000 if len(all_iti) else float("nan"),
            "iti_p95_ms": q(all_iti, 95) * 1000 if len(all_iti) else float("nan"),
        })

    summary_df = pd.DataFrame(rows)
    summary_path = out / "summary_stats.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")
    console.print(f"  [green]Saved:[/green] summary_stats.csv")

    table = Table(title="Summary Statistics (selected)", border_style="blue")
    for col in ["concurrency", "prompt_class", "n", "ttfb_median_s", "ttfb_p95_s",
                "resp_bytes_median", "iti_median_ms"]:
        table.add_column(col, style="bold" if col in ("concurrency", "prompt_class") else "")
    for _, row in summary_df.iterrows():
        table.add_row(*[str(round(row[c], 3)) if isinstance(row[c], float) else str(row[c])
                        for c in ["concurrency", "prompt_class", "n", "ttfb_median_s",
                                  "ttfb_p95_s", "resp_bytes_median", "iti_median_ms"]])
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results and produce plots.")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Directory with experiment JSON files.")
    parser.add_argument("--traces-dir", default=str(TRACES_DIR), help="Directory with pcap trace files.")
    parser.add_argument("--output-dir", default=str(PLOTS_DIR), help="Output directory for plots and CSV.")
    parser.add_argument("--no-pcap", action="store_true", help="Skip pcap analysis.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        "[bold blue]LLM Inference Traffic Research[/bold blue]\n"
        "Analysis and Plotting",
        border_style="blue"
    ))

    console.print("\n[blue]Loading experiment results...[/blue]")
    df = load_results(results_dir)

    # Ensure inter_token_intervals column is list type
    df["inter_token_intervals"] = df["inter_token_intervals"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    console.print("\n[blue]Generating plots...[/blue]")
    plot_flow_size_cdf(df, out_dir)
    plot_ttfb_by_concurrency(df, out_dir)
    plot_inter_token_intervals(df, out_dir)
    plot_burst_analysis(df, out_dir)
    plot_arrival_process(df, out_dir)
    build_summary_csv(df, out_dir)

    if not args.no_pcap:
        traces = list(Path(args.traces_dir).glob("*.pcap"))
        if traces:
            console.print(
                f"\n[blue]Found {len(traces)} pcap file(s). "
                "For packet-level analysis, open them in Wireshark or run:[/blue]\n"
                "  tshark -r ./results/traces/<file>.pcap -T fields "
                "-e frame.time_relative -e tcp.len -e ip.src > frame_data.txt"
            )
        else:
            console.print("\n[yellow]No pcap files found in traces directory. Skipping packet-level analysis.[/yellow]")

    console.print(f"\n[bold green]Analysis complete![/bold green]")
    console.print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
