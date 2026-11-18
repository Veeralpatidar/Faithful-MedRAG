
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmrag import config as C
from fmrag import data as D
from fmrag import metrics as M
from fmrag import report as R
from fmrag.gate import AnswerabilityGate
from fmrag.pipeline import ABLATION_CONFIGS, FaithfulMedRAG, SystemConfig
from fmrag.retrieval import HybridRetriever

log = logging.getLogger("exp9")


def sweep(items: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Post-hoc threshold sweep over recorded gate probabilities."""
    rows = []
    for t in thresholds:
        ans = items["gate_prob"].values >= t
        n_ans = int(ans.sum())
        correct = items["correct"].values.astype(float)

        # Claim-level faithfulness counted over ANSWERED items only: an
        # abstention emits no claims, so it can neither help nor hurt.
        sup = items.loc[ans, "n_supported"].sum()
        con = items.loc[ans, "n_contradicted"].sum()
        neu = items.loc[ans, "n_neutral"].sum()
        tot = sup + con + neu

        rows.append({
            "threshold": float(t),
            "coverage": float(ans.mean()),
            "n_answered": n_ans,
            "selective_accuracy": float(correct[ans].mean()) if n_ans else float("nan"),
            "risk": 1.0 - float(correct[ans].mean()) if n_ans else float("nan"),
            # accuracy over ALL questions, abstentions scored as wrong
            "overall_accuracy_abstain_as_wrong": float((correct * ans).mean()),
            "faithfulness_micro": float(sup / tot) if tot else float("nan"),
            "hallucination_rate": float((con + neu) / tot) if tot else float("nan"),
            "n_claims_emitted": int(tot),
        })
    return pd.DataFrame(rows)


def main(n_eval: int | None = None) -> pd.DataFrame:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    C.set_all_seeds()
    C.ensure_dirs()
    n_eval = n_eval or C.SCALE.n_generation_eval

    print("=" * 78)
    print("EXPERIMENT 9 : faithfulness-helpfulness trade-off")
    print("=" * 78)

    corpus, grounded = D.build_corpus()
    pool = sorted(grounded, key=lambda q: q.qid)[-n_eval:]

    retriever = HybridRetriever(corpus, cache_tag="main")
    ckpt = C.PHASE2 / "Checkpoints" / "answerability_gate.pkl"
    if not ckpt.exists():
        raise SystemExit("No gate checkpoint. Run experiments/exp5_gate.py first.")
    gate = AnswerabilityGate.load(ckpt)
    system = FaithfulMedRAG(retriever=retriever, gate=gate)

    # Complete pipeline, gating disabled but gate probability recorded.
    base = next(c for c in ABLATION_CONFIGS if c.name == "H_full_faithful_medrag")
    cfg = SystemConfig(
        name="H_ungated_for_sweep", retrieval_mode=base.retrieval_mode,
        grounded=True, verify=True, use_repair=True, use_gate=False,
        description="Complete pipeline, gate scored but not enforced.",
    )

    log.info("running complete pipeline (ungated) on n=%d", len(pool))
    results = system.run_batch(pool, cfg)

    # Score the gate on every item so the sweep is exact.
    log.info("scoring the gate on every item for the post-hoc sweep")
    rows = []
    for i, (q, r) in enumerate(zip(pool, results), start=1):
        ev, _ = retriever.retrieve(q.text, cfg.retrieval_mode, k=cfg.k_context)
        feats = system.gate_features(q.text, ev)
        _, prob = gate.should_answer(feats)
        rows.append({
            "qid": r.qid, "gate_prob": prob, "correct": r.correct,
            "n_claims": r.faith.get("n_claims", 0),
            "n_supported": r.faith.get("n_supported", 0),
            "n_contradicted": r.faith.get("n_contradicted", 0),
            "n_neutral": r.faith.get("n_neutral", 0),
            "n_repaired": r.n_repaired,
        })
        if i % 50 == 0:
            log.info("  gate-scored %d/%d", i, len(pool))
    items = pd.DataFrame(rows)

    # ---- validation / test split for honest threshold selection ----
    qids = sorted(items["qid"].unique())
    rng = random.Random(C.SEED)
    rng.shuffle(qids)
    half = len(qids) // 2
    val_ids = set(qids[:half])
    items["split"] = items["qid"].map(lambda q: "val" if q in val_ids else "test")
    val, test = items[items.split == "val"], items[items.split == "test"]

    thresholds = np.linspace(0.0, 1.0, 21)
    sweep_val = sweep(val, thresholds);   sweep_val["split"] = "validation"
    sweep_test = sweep(test, thresholds); sweep_test["split"] = "test"
    sweep_all = sweep(items, thresholds); sweep_all["split"] = "all"

    # ---- choose the operating point on VALIDATION ----
    cand = sweep_val[(sweep_val["coverage"] >= 0.5) & sweep_val["faithfulness_micro"].notna()]
    if len(cand):
        pick = cand.loc[cand["faithfulness_micro"].idxmax()]
        chosen = float(pick["threshold"])
        reason = ("highest validation faithfulness subject to coverage >= 0.50")
    else:
        chosen = 0.5
        reason = "fallback: no validation threshold met the coverage floor"
    log.info("operating threshold chosen on validation: %.3f (%s)", chosen, reason)

    at_test = sweep_test.iloc[(sweep_test["threshold"] - chosen).abs().argmin()]

    df = pd.concat([sweep_val, sweep_test, sweep_all], ignore_index=True)
    R.save_table(df, C.PHASE2, "exp9_threshold_sweep")
    R.save_table(items, C.PHASE2, "exp9_per_item_gate_scores")
    R.save_json(
        {"chosen_threshold": chosen, "selection_rule": reason,
         "selected_on": "validation split", "test_metrics_at_threshold": at_test.to_dict()},
        C.PHASE2, "exp9_operating_point",
    )

    R.plot_threshold_sweep(sweep_test, C.PHASE2, "exp9_threshold_sweep_test")
    rc = [{"coverage": r["coverage"], "risk": r["risk"]}
          for _, r in sweep_test.iterrows() if pd.notna(r["risk"])]
    if rc:
        R.plot_risk_coverage(rc, C.PHASE2, "exp9_risk_coverage_test",
                             title="Risk-coverage curve, complete Faithful-MedRAG (test)")

    fig_df = sweep_test.dropna(subset=["faithfulness_micro"])
    if len(fig_df):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.plot(fig_df["coverage"], fig_df["faithfulness_micro"], "-o",
                color=R.PALETTE[2], markersize=4)
        ax.set_xlabel("Coverage (fraction of questions answered)")
        ax.set_ylabel("Claim-level faithfulness")
        ax.set_title("Coverage versus faithfulness")
        ax.invert_xaxis()
        fig.text(0.01, -0.05,
                 "Abstaining on low-confidence retrievals raises the faithfulness of "
                 "what remains. Moving left trades helpfulness for safety.",
                 fontsize=8, style="italic")
        R._finish(fig, C.PHASE2, "exp9_coverage_vs_faithfulness")

    print("\n" + "=" * 78)
    print("EXPERIMENT 9 RESULTS  (actually executed)")
    print("=" * 78)
    show = ["threshold", "coverage", "n_answered", "selective_accuracy",
            "faithfulness_micro", "hallucination_rate", "n_claims_emitted"]
    print("\nTEST-split sweep:")
    print(sweep_test[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\noperating threshold selected on validation: {chosen:.3f}")
    print(f"  rule: {reason}")
    print("  test metrics at that threshold:")
    for k in ["coverage", "selective_accuracy", "faithfulness_micro", "hallucination_rate"]:
        print(f"    {k:24s} {at_test[k]:.4f}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    main(a.n)
