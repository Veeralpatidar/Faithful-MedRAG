 
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
from fmrag.pipeline import ABLATION_CONFIGS, FaithfulMedRAG, aggregate
from fmrag.retrieval import HybridRetriever

log = logging.getLogger("exp1")

# MIRAGE subset name -> column name in the published tables
SUBSET_TO_PUBLISHED = {
    "mmlu": "MMLU-Med",
    "medqa": "MedQA-US",
    "medmcqa": "MedMCQA",
    "pubmedqa": "PubMedQA*",
    "bioasq": "BioASQ-Y/N",
}


def main(n_eval: int | None = None) -> pd.DataFrame:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    C.set_all_seeds()
    C.ensure_dirs()
    n_eval = n_eval or C.SCALE.n_generation_eval

    print("=" * 78)
    print("EXPERIMENT 1 : vanilla MedRAG baseline on MIRAGE")
    print("=" * 78)
    print(C.describe())

    mirage = D.load_mirage()
    sample = D.stratified_sample(mirage, n_eval, seed=C.SEED)
    log.info("stratified MIRAGE sample: %d questions", len(sample))
    by_subset = pd.Series([q.subset for q in sample]).value_counts().to_dict()
    log.info("sample composition: %s", by_subset)

    corpus, _ = D.build_corpus()
    retriever = HybridRetriever(corpus, cache_tag="main")
    system = FaithfulMedRAG(retriever=retriever)

    cfg = next(c for c in ABLATION_CONFIGS if c.name == "A_vanilla_medrag")
    results = system.run_batch(sample, cfg)

    # ---- per-subset accuracy, with binomial standard deviation ----
    rows = []
    for subset in sorted({r.subset for r in results}):
        sub = [r for r in results if r.subset == subset]
        acc = float(np.mean([r.correct for r in sub]))
        n = len(sub)
        sd = float(np.sqrt(max(acc * (1 - acc), 0.0) / n)) if n else float("nan")
        rows.append({
            "subset": subset,
            "published_column": SUBSET_TO_PUBLISHED.get(subset, ""),
            "n_evaluated": n,
            "n_in_full_mirage": len(mirage.get(subset, [])),
            "accuracy_executed_pct": 100 * acc,
            "std_pct": 100 * sd,
            "evidence": "EXECUTED",
        })
    overall = float(np.mean([r.correct for r in results]))
    rows.append({
        "subset": "AVERAGE(macro over subsets)",
        "published_column": "Average",
        "n_evaluated": len(results),
        "n_in_full_mirage": sum(len(v) for v in mirage.values()),
        "accuracy_executed_pct": 100 * float(np.mean([r["accuracy_executed_pct"] / 100
                                                      for r in rows])),
        "std_pct": float("nan"),
        "evidence": "EXECUTED",
    })
    df = pd.DataFrame(rows)

    # ---- attach the published reference frame ----
    ref_path = C.SHARED / "References" / "published_medrag_table6_mirage.csv"
    published = None
    if ref_path.exists():
        pub = pd.read_csv(ref_path)
        published = pub[pub["method"] == "MedRAG"].copy()
        log.info("loaded published reference (%d MedRAG rows)", len(published))

    R.save_table(df, C.PHASE1, "exp1_mirage_baseline_accuracy")
    R.save_table(
        pd.DataFrame([{
            "qid": r.qid, "subset": r.subset, "gold": r.gold_answer,
            "predicted": r.predicted, "correct": r.correct,
            "t_total_s": r.t_total_s, "rationale": r.rationale[:800],
        } for r in results]),
        C.PHASE1, "exp1_mirage_per_item",
    )

    agg = aggregate(results, cfg.name)
    R.save_json(agg, C.PHASE1, "exp1_mirage_aggregate")

    R.plot_grouped_bars(
        df[df.subset != "AVERAGE(macro over subsets)"], "subset",
        ["accuracy_executed_pct"], C.PHASE1, "exp1_mirage_accuracy_by_subset",
        title="Vanilla MedRAG baseline accuracy by MIRAGE subset (executed)",
        ylabel="Accuracy (%)",
        err={"accuracy_executed_pct": "std_pct"},
        caption=f"Seeded stratified sample of n={len(sample)} from the real 7,663-question "
                f"MIRAGE benchmark. Generator: {C.GEN_PROFILES[C.GEN_PROFILE]['model']}; "
                f"corpus: {len(corpus):,} passages. Error bars are binomial SD.",
    )

    # Side-by-side with published numbers, clearly labelled as different scale.
    if published is not None:
        comp_rows = []
        for _, r in df.iterrows():
            col = r["published_column"]
            if not col or col == "Average":
                continue
            for _, p in published.iterrows():
                comp_rows.append({
                    "subset": r["subset"],
                    "system": f"{p['llm']} MedRAG (PUBLISHED)",
                    "accuracy_pct": p[col],
                    "evidence": "PUBLISHED",
                })
            comp_rows.append({
                "subset": r["subset"],
                "system": f"{C.GEN_PROFILES[C.GEN_PROFILE]['model']} (EXECUTED, subset corpus)",
                "accuracy_pct": r["accuracy_executed_pct"],
                "evidence": "EXECUTED",
            })
        comp = pd.DataFrame(comp_rows)
        R.save_table(comp, C.PHASE1, "exp1_published_vs_executed")

    print("\n" + "=" * 78)
    print("EXPERIMENT 1 RESULTS  (actually executed)")
    print("=" * 78)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nmicro accuracy over the whole sample: {100*overall:.2f}%  (n={len(results)})")
    if published is not None:
        print("\nPUBLISHED reference frame (Xiong et al. 2024, Table 6, MedRAG rows):")
        print(published[["llm", "MMLU-Med", "MedQA-US", "MedMCQA",
                         "PubMedQA*", "BioASQ-Y/N", "Average"]].to_string(index=False))
        gen = C.GEN_PROFILES[C.GEN_PROFILE]["model"]
        print(f"\nNOTE: published runs use GPT-4/GPT-3.5/70B-class backbones over the "
              f"full\n23.9M-doc PubMed corpus. This run used {gen} over a "
              f"{len(corpus):,}-passage\nsubset (n={len(sample)}). The comparison is a "
              f"reference frame, NOT a like-for-like claim.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    main(a.n)
