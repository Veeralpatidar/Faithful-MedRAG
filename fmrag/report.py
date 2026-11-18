"""
Faithful-MedRAG : results persistence and plotting.

Every experiment writes through here so that:
  - each results file carries the provenance block (hardware, versions, seed,
    model ids, sampling), satisfying Part E #9;
  - every number in a presentation is traceable to a CSV on disk;
  - graphs are saved with titles, axis labels and legends (Part D).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger("fmrag.report")

# Matplotlib: headless, consistent styling.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = ["#2E5EAA", "#D1495B", "#00798C", "#EDAE49", "#7B6D8D", "#3B7A57"]


def save_table(df: pd.DataFrame, phase_dir: Path, name: str, provenance: bool = True) -> Path:
    """Write a results table as CSV, with a sidecar provenance JSON."""
    out_dir = phase_dir / "Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{name}.csv"
    df.to_csv(csv, index=False)
    if provenance:
        (out_dir / f"{name}_provenance.json").write_text(
            json.dumps(C.provenance(), indent=2, default=str)
        )
    log.info("saved %s (%d rows)", csv.name, len(df))
    return csv


def save_json(obj, phase_dir: Path, name: str) -> Path:
    out_dir = phase_dir / "Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=str))
    log.info("saved %s", p.name)
    return p


def save_excel(tables: dict[str, pd.DataFrame], phase_dir: Path, name: str) -> Path:
    """Bundle every table of a phase into one workbook, provenance included."""
    out_dir = phase_dir / "Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx = out_dir / f"{name}.xlsx"
    prov = pd.DataFrame(
        sorted((k, str(v)) for k, v in C.provenance().items()),
        columns=["key", "value"],
    )
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        prov.to_excel(xw, sheet_name="provenance", index=False)
        for sheet, df in tables.items():
            df.to_excel(xw, sheet_name=sheet[:31], index=False)
    log.info("saved workbook %s (%d sheets)", xlsx.name, len(tables) + 1)
    return xlsx


def _finish(fig, phase_dir: Path, name: str) -> Path:
    out_dir = phase_dir / "Graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    log.info("saved graph %s", p.name)
    return p


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_grouped_bars(
    df: pd.DataFrame, x: str, series: list[str], phase_dir: Path, name: str,
    title: str = "", ylabel: str = "", caption: str = "", err: dict | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(df)), 4.2))
    idx = np.arange(len(df))
    w = 0.8 / max(len(series), 1)
    for i, s in enumerate(series):
        yerr = df[err[s]].values if err and s in err and err[s] in df else None
        ax.bar(idx + i * w - 0.4 + w / 2, df[s].values, width=w,
               label=s, color=PALETTE[i % len(PALETTE)],
               yerr=yerr, capsize=3, error_kw={"elinewidth": 0.8})
    ax.set_xticks(idx)
    ax.set_xticklabels(df[x].astype(str).values, rotation=20, ha="right")
    ax.set_title(title or name)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x)
    if len(series) > 1:
        ax.legend(frameon=False)
    if caption:
        fig.text(0.01, -0.07, caption, fontsize=8, style="italic", wrap=True)
    return _finish(fig, phase_dir, name)


def plot_accuracy_vs_faithfulness(
    df: pd.DataFrame, phase_dir: Path, name: str, label_col: str = "config",
    acc_col: str = "overall_accuracy", faith_col: str = "faithfulness_micro",
) -> Path:
    """The central claim of the thesis: these two axes are not the same axis."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    d = df.dropna(subset=[acc_col, faith_col])
    for i, (_, row) in enumerate(d.iterrows()):
        ax.scatter(row[acc_col], row[faith_col], s=130,
                   color=PALETTE[i % len(PALETTE)], zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(str(row[label_col]), (row[acc_col], row[faith_col]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    lo = min(d[acc_col].min(), d[faith_col].min()) - 0.05
    hi = max(d[acc_col].max(), d[faith_col].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "--", color="#999", linewidth=1,
            label="accuracy = faithfulness")
    ax.set_xlabel("Answer accuracy")
    ax.set_ylabel("Claim-level faithfulness (supported / total claims)")
    ax.set_title("Accuracy does not imply faithfulness")
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.01, -0.04,
             "Points below the diagonal answer correctly while leaving claims unsupported.",
             fontsize=8, style="italic")
    return _finish(fig, phase_dir, name)


def plot_risk_coverage(rows: list[dict], phase_dir: Path, name: str,
                       title: str = "Risk-coverage curve") -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    cov = [r["coverage"] for r in rows]
    risk = [r["risk"] for r in rows]
    ax.plot(cov, risk, "-o", color=PALETTE[0], markersize=4)
    ax.set_xlabel("Coverage (fraction of questions answered)")
    ax.set_ylabel("Risk (1 - selective accuracy)")
    ax.set_title(title)
    ax.invert_xaxis()
    fig.text(0.01, -0.05,
             "Lower-left is better: high coverage at low risk. Sweeping the gate "
             "threshold trades helpfulness for safety.",
             fontsize=8, style="italic")
    return _finish(fig, phase_dir, name)


def plot_calibration(bins, phase_dir: Path, name: str, ece: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    xs = [b[0] for b in bins]
    ys = [b[1] for b in bins]
    ns = [b[2] for b in bins]
    ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1, label="perfect calibration")
    ok = [(x, y) for x, y, n in zip(xs, ys, ns) if n > 0 and not np.isnan(y)]
    if ok:
        ax.plot([p[0] for p in ok], [p[1] for p in ok], "-o",
                color=PALETTE[1], markersize=5, label="gate (isotonic-calibrated)")
    ax.set_xlabel("Predicted probability of answerable")
    ax.set_ylabel("Observed fraction answerable")
    t = "Answerability-gate calibration"
    if ece is not None:
        t += f"  (ECE = {ece:.3f})"
    ax.set_title(t)
    ax.legend(frameon=False, loc="upper left")
    return _finish(fig, phase_dir, name)


def plot_threshold_sweep(df: pd.DataFrame, phase_dir: Path, name: str) -> Path:
    """Coverage / selective accuracy / faithfulness against the gate threshold."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(df["threshold"], df["coverage"], "-o", color=PALETTE[0],
            markersize=4, label="coverage")
    if "selective_accuracy" in df:
        ax.plot(df["threshold"], df["selective_accuracy"], "-s", color=PALETTE[1],
                markersize=4, label="selective accuracy")
    if "faithfulness_micro" in df:
        ax.plot(df["threshold"], df["faithfulness_micro"], "-^", color=PALETTE[2],
                markersize=4, label="faithfulness")
    if "hallucination_rate" in df:
        ax.plot(df["threshold"], df["hallucination_rate"], "-v", color=PALETTE[3],
                markersize=4, label="hallucination rate")
    ax.set_xlabel("Answerability-gate threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Faithfulness-helpfulness trade-off across gate thresholds")
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, -0.05,
             "Raising the threshold abstains more often: coverage falls while "
             "faithfulness and selective accuracy rise.",
             fontsize=8, style="italic")
    return _finish(fig, phase_dir, name)


def plot_claim_breakdown(df: pd.DataFrame, phase_dir: Path, name: str) -> Path:
    """Stacked claim-verdict composition per configuration."""
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(df)), 4.4))
    labels = ["n_supported", "n_contradicted", "n_neutral", "abstained_claims"]
    pretty = ["supported", "contradicted", "neutral (unverifiable)", "abstained"]
    colors = ["#3B7A57", "#D1495B", "#EDAE49", "#7B6D8D"]
    present = [l for l in labels if l in df.columns]
    totals = df[present].sum(axis=1).replace(0, np.nan)
    bottom = np.zeros(len(df))
    for l, p, c in zip(labels, pretty, colors):
        if l not in df.columns:
            continue
        vals = (df[l] / totals).fillna(0).values
        ax.bar(np.arange(len(df)), vals, bottom=bottom, label=p, color=c, width=0.65)
        bottom += vals
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["config"].astype(str).values, rotation=20, ha="right")
    ax.set_ylabel("Fraction of generated claims")
    ax.set_title("Claim-level verdict composition by configuration")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _finish(fig, phase_dir, name)


def plot_feature_importance(imp: dict, phase_dir: Path, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    items = sorted(imp.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items], color=PALETTE[0])
    ax.set_xlabel("Normalised |coefficient| / importance")
    ax.set_title("Answerability-gate feature importance")
    fig.text(0.01, -0.05,
             "Which retrieval signals carry the answerability decision.",
             fontsize=8, style="italic")
    return _finish(fig, phase_dir, name)
