
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmrag import config as C
from fmrag import data as D
from fmrag import report as R
from fmrag.gate import AnswerabilityGate
from fmrag.pipeline import ABLATION_CONFIGS, FaithfulMedRAG, aggregate
from fmrag.retrieval import HybridRetriever

log = logging.getLogger("exp7")

COMPARE = ["A_vanilla_medrag", "F_plus_claim_repair", "H_full_faithful_medrag"]
PRETTY = {
    "A_vanilla_medrag": "Vanilla baseline",
    "F_plus_claim_repair": "Faithfulness-aware, no gate",
    "H_full_faithful_medrag": "Complete Faithful-MedRAG",
}


def main(n_eval: int | None = None) -> pd.DataFrame:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    C.set_all_seeds()
    C.ensure_dirs()
    n_eval = n_eval or C.SCALE.n_ood_eval

    print("=" * 78)
    print("EXPERIMENT 7 : out-of-distribution generalization (MedQA / USMLE)")
    print("=" * 78)
    print(C.describe())

    mirage = D.load_mirage()
    ood = sorted(mirage["medqa"], key=lambda q: q.qid)[:n_eval]
    log.info("OOD set: MedQA-US n=%d (of %d in MIRAGE)", len(ood), len(mirage["medqa"]))

    corpus, grounded = D.build_corpus()
    ind = sorted(grounded, key=lambda q: q.qid)[-n_eval:]
    log.info("in-distribution set: PubMedQA grounded n=%d", len(ind))

    retriever = HybridRetriever(corpus, cache_tag="main")
    ckpt = C.PHASE2 / "Checkpoints" / "answerability_gate.pkl"
    gate = AnswerabilityGate.load(ckpt) if ckpt.exists() else None
    if gate is None:
        log.warning("no gate checkpoint -- the complete system cannot be evaluated. "
                    "Run exp5_gate.py first.")
    system = FaithfulMedRAG(retriever=retriever, gate=gate)

    rows, items = [], []
    for name in COMPARE:
        cfg = next(c for c in ABLATION_CONFIGS if c.name == name)
        if cfg.use_gate and gate is None:
            log.warning("skipping %s (needs the gate)", name)
            continue
        for dist, qs in [("in_distribution", ind), ("out_of_distribution", ood)]:
            log.info("--- %s on %s (n=%d) ---", name, dist, len(qs))
            res = system.run_batch(qs, cfg)
            row = aggregate(res, name)
            row["distribution"] = dist
            row["dataset"] = "PubMedQA-grounded" if dist == "in_distribution" else "MedQA-US"
            row["system"] = PRETTY.get(name, name)
            rows.append(row)
            for r in res:
                items.append({
                    "system": name, "distribution": dist, "qid": r.qid,
                    "gold": r.gold_answer, "predicted": r.predicted,
                    "correct": r.correct, "answered": r.answered,
                    "abstain_reason": r.abstain_reason, "gate_prob": r.gate_prob,
                    "faithfulness": r.faith.get("faithfulness", float("nan")),
                    "n_claims": r.faith.get("n_claims", 0),
                })

    df = pd.DataFrame(rows)

    # ---- generalization gap per system ----
    gap_rows = []
    for name in df["config"].unique():
        d = df[df["config"] == name]
        try:
            i = d[d.distribution == "in_distribution"].iloc[0]
            o = d[d.distribution == "out_of_distribution"].iloc[0]
        except IndexError:
            continue
        g = {"config": name, "system": PRETTY.get(name, name)}
        for m in ["overall_accuracy", "selective_accuracy", "coverage",
                  "faithfulness_micro", "hallucination_rate", "abstain_rate"]:
            iv, ov = i.get(m, np.nan), o.get(m, np.nan)
            g[f"{m}_in"] = iv
            g[f"{m}_ood"] = ov
            g[f"{m}_gap"] = (iv - ov) if (pd.notna(iv) and pd.notna(ov)) else np.nan
        gap_rows.append(g)
    gaps = pd.DataFrame(gap_rows)

    R.save_table(df, C.PHASE2, "exp7_ood_results")
    R.save_table(gaps, C.PHASE2, "exp7_generalization_gap")
    R.save_table(pd.DataFrame(items), C.PHASE2, "exp7_ood_per_item")

    for metric, label in [("overall_accuracy", "Accuracy"),
                          ("faithfulness_micro", "Faithfulness"),
                          ("hallucination_rate", "Hallucination rate")]:
        if metric not in df.columns:
            continue
        piv = df.pivot_table(index="system", columns="distribution",
                             values=metric, aggfunc="first").reset_index()
        cols = [c for c in ["in_distribution", "out_of_distribution"] if c in piv.columns]
        if not cols:
            continue
        R.plot_grouped_bars(
            piv, "system", cols, C.PHASE2, f"exp7_ood_{metric}",
            title=f"{label}: in-distribution vs out-of-distribution",
            ylabel=label,
            caption="In-distribution: PubMedQA grounded. OOD: MedQA-US (USMLE) from "
                    "MIRAGE. The gate is transferred unchanged, never retrained on MedQA.",
        )

    print("\n" + "=" * 78)
    print("EXPERIMENT 7 RESULTS  (actually executed)")
    print("=" * 78)
    show = ["system", "distribution", "dataset", "n", "overall_accuracy",
            "selective_accuracy", "coverage", "faithfulness_micro",
            "hallucination_rate", "abstain_rate"]
    print(df[[c for c in show if c in df.columns]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    if not gaps.empty:
        print("\nGENERALIZATION GAP (in-distribution minus OOD; positive = degrades OOD)")
        gshow = ["system", "overall_accuracy_gap", "faithfulness_micro_gap",
                 "hallucination_rate_gap", "coverage_gap"]
        print(gaps[[c for c in gshow if c in gaps.columns]].to_string(
            index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nNOTE: MedQA items carry no gold evidence documents, so retrieval metrics")
    print("(Recall@k / nDCG / MRR) are not computable on the OOD split.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    main(a.n)
