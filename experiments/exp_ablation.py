 
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmrag import config as C
from fmrag import data as D
from fmrag import report as R
from fmrag.gate import AnswerabilityGate
from fmrag.pipeline import ABLATION_CONFIGS, FaithfulMedRAG, aggregate
from fmrag.retrieval import HybridRetriever

log = logging.getLogger("exp_ablation")


def main(n_eval: int | None = None, only: list[str] | None = None,
         phase: str = "phase2") -> pd.DataFrame:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    C.set_all_seeds()
    C.ensure_dirs()
    n_eval = n_eval or C.SCALE.n_generation_eval
    out_dir = C.PHASE1 if phase == "phase1" else C.PHASE2

    print("=" * 78)
    print("ABLATION : configurations A-H  (Experiments 1, 2, 4, 6)")
    print("=" * 78)
    print(C.describe())

    corpus, grounded = D.build_corpus()
    # Hold the generation eval set to the TAIL of the grounded questions so it
    # does not overlap the head slice used for gate training.
    pool = sorted(grounded, key=lambda q: q.qid)
    questions = pool[-n_eval:]
    log.info("eval questions=%d (tail slice, disjoint from gate-training head)",
             len(questions))

    retriever = HybridRetriever(corpus, cache_tag="main")

    # Load the trained gate for configurations G and H.
    gate = None
    ckpt = C.PHASE2 / "Checkpoints" / "answerability_gate.pkl"
    if ckpt.exists():
        gate = AnswerabilityGate.load(ckpt)
        log.info("loaded answerability gate (threshold=%.3f)", gate.threshold)
    else:
        log.warning("no gate checkpoint -- configs G/H will be skipped. "
                    "Run exp5_gate.py first.")

    system = FaithfulMedRAG(retriever=retriever, gate=gate)

    configs = ABLATION_CONFIGS
    if only:
        configs = [c for c in configs if c.name in only or c.name[0] in only]
    if gate is None:
        configs = [c for c in configs if not c.use_gate]

    rows, all_items = [], []
    for cfg in configs:
        log.info("=" * 60)
        log.info("CONFIG %s : %s", cfg.name, cfg.description)
        results = system.run_batch(questions, cfg)
        row = aggregate(results, cfg.name)
        row["description"] = cfg.description
        row["retrieval_mode"] = cfg.retrieval_mode
        row["grounded"] = cfg.grounded
        row["verify"] = cfg.verify
        row["use_repair"] = cfg.use_repair
        row["use_gate"] = cfg.use_gate
        rows.append(row)

        for r in results:
            all_items.append({
                "config": cfg.name, "qid": r.qid, "subset": r.subset,
                "gold_answer": r.gold_answer, "predicted": r.predicted,
                "correct": r.correct, "answered": r.answered,
                "abstain_reason": r.abstain_reason, "gate_prob": r.gate_prob,
                "n_claims": r.faith.get("n_claims", 0),
                "n_supported": r.faith.get("n_supported", 0),
                "n_contradicted": r.faith.get("n_contradicted", 0),
                "n_neutral": r.faith.get("n_neutral", 0),
                "faithfulness": r.faith.get("faithfulness", float("nan")),
                "citation_precision": r.faith.get("citation_precision", float("nan")),
                "n_repaired": r.n_repaired,
                "n_claim_abstained": r.n_claim_abstained,
                "repair_rounds": r.repair_rounds,
                "t_total_s": r.t_total_s,
                "rationale": r.rationale[:1500],
                "final_answer_text": r.final_answer_text[:1500],
            })

        log.info(
            "%s -> acc=%.4f sel_acc=%.4f cov=%.4f faith=%.4f halluc=%.4f "
            "repaired=%d abstained_claims=%d %.1fs/item",
            cfg.name, row.get("overall_accuracy", float("nan")),
            row.get("selective_accuracy", float("nan")), row.get("coverage", float("nan")),
            row.get("faithfulness_micro", float("nan")),
            row.get("hallucination_rate", float("nan")),
            row.get("n_repaired_claims", 0), row.get("n_abstained_claims", 0),
            row.get("mean_t_total_s", float("nan")),
        )

    df = pd.DataFrame(rows)
    items = pd.DataFrame(all_items)

    front = ["config", "n", "overall_accuracy", "selective_accuracy", "coverage",
             "faithfulness_micro", "hallucination_rate", "citation_precision",
             "n_repaired_claims", "n_abstained_claims", "abstain_rate",
             "mean_t_total_s"]
    df = df[[c for c in front if c in df.columns]
            + [c for c in df.columns if c not in front]]

    R.save_table(df, out_dir, "ablation_configurations")
    R.save_table(items, out_dir, "ablation_per_item")

    # ---- plots ----
    if "faithfulness_micro" in df.columns:
        R.plot_grouped_bars(
            df, "config", ["overall_accuracy", "faithfulness_micro"], out_dir,
            "ablation_accuracy_vs_faithfulness_bars",
            title="Accuracy and claim-level faithfulness by configuration",
            ylabel="Metric value",
            caption=f"PubMedQA grounded questions (n={len(questions)}). Accuracy is "
                    "answer-level; faithfulness is the fraction of generated claims "
                    "entailed by retrieved evidence.",
        )
        R.plot_accuracy_vs_faithfulness(df, out_dir, "ablation_accuracy_faithfulness_scatter")
        R.plot_claim_breakdown(df, out_dir, "ablation_claim_verdict_breakdown")
    if "hallucination_rate" in df.columns:
        R.plot_grouped_bars(
            df, "config", ["hallucination_rate"], out_dir, "ablation_hallucination_rate",
            title="Hallucination rate by configuration",
            ylabel="Unsupported claims / total claims",
            caption="A claim counts as a hallucination when the verifier assigns "
                    "contradiction, or neutral with no entailing evidence.",
        )
    R.plot_grouped_bars(
        df, "config", ["mean_t_total_s"], out_dir, "ablation_latency",
        title="Mean end-to-end latency per question",
        ylabel="Seconds per question",
        caption=f"Measured on {C.DEVICE.upper()}; generator "
                f"{C.GEN_PROFILES[C.GEN_PROFILE]['model']}.",
    )

    print("\n" + "=" * 78)
    print("ABLATION RESULTS  (actually executed)")
    print("=" * 78)
    show = ["config", "n", "overall_accuracy", "selective_accuracy", "coverage",
            "faithfulness_micro", "hallucination_rate", "n_repaired_claims",
            "mean_t_total_s"]
    print(df[[c for c in show if c in df.columns]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated config names or leading letters")
    ap.add_argument("--phase", type=str, default="phase2", choices=["phase1", "phase2"])
    a = ap.parse_args()
    main(a.n, a.only.split(",") if a.only else None, a.phase)
