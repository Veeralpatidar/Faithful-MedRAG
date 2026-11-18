
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmrag import config as C
from fmrag import data as D
from fmrag import report as R
from fmrag.metrics import retrieval_report
from fmrag.retrieval import RETRIEVAL_MODES, HybridRetriever

log = logging.getLogger("exp3")


def main(n_eval: int | None = None, modes: list[str] | None = None,
         tag: str = "") -> pd.DataFrame:
    """`tag` suffixes the output filenames so separate runs never overwrite
    each other (the main bake-off and the reranker diagnostic are distinct
    experiments and must not collide)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    C.set_all_seeds()
    C.ensure_dirs()
    modes = modes or RETRIEVAL_MODES
    n_eval = n_eval or C.SCALE.n_retrieval_eval

    print("=" * 78)
    print("EXPERIMENT 3 : hybrid retrieval + reranking bake-off  (E1-E2)")
    print("=" * 78)
    print(C.describe())

    corpus, grounded = D.build_corpus()
    questions = sorted(grounded, key=lambda q: q.qid)[:n_eval]
    log.info("corpus=%d passages | eval questions=%d", len(corpus), len(questions))

    retriever = HybridRetriever(corpus, cache_tag="main")

    rows: list[dict] = []
    per_query: list[dict] = []

    for mode in modes:
        log.info("--- arm: %s ---", mode)
        runs: list[tuple[list[str], set[str]]] = []
        timings: list[dict] = []
        t0 = time.time()

        for i, q in enumerate(questions, start=1):
            res, timing = retriever.retrieve(q.text, mode, k=20)
            rids = [s.passage.doc_id for s in res]
            gold = set(q.gold_doc_ids)
            runs.append((rids, gold))
            timings.append(timing)
            per_query.append({
                "mode": mode, "qid": q.qid,
                "n_gold": len(gold),
                "recall_at_5": len(set(rids[:5]) & gold) / max(len(gold), 1),
                "recall_at_10": len(set(rids[:10]) & gold) / max(len(gold), 1),
                "latency_s": timing.get("total_s", float("nan")),
            })
            if i % 100 == 0:
                el = time.time() - t0
                log.info("  %s: %d/%d (%.2fs/query, eta %.1f min)",
                         mode, i, len(questions), el / i, (len(questions) - i) * el / i / 60)

        rep = retrieval_report(runs, ks=(5, 10, 20))
        rep["mode"] = mode
        rep["wall_clock_s"] = time.time() - t0
        for key in ["total_s", "lexical_s", "dense_s", "fuse_s", "rerank_s"]:
            vals = [t[key] for t in timings if key in t]
            if vals:
                rep[f"mean_{key}"] = float(np.mean(vals))
        rows.append(rep)
        log.info(
            "%-9s R@5=%.4f R@10=%.4f R@20=%.4f MRR=%.4f nDCG@10=%.4f  %.3fs/query",
            mode, rep.get("Recall@5", float("nan")), rep.get("Recall@10", float("nan")),
            rep.get("Recall@20", float("nan")), rep.get("MRR", float("nan")),
            rep.get("nDCG@10", float("nan")), rep.get("mean_total_s", float("nan")),
        )

    df = pd.DataFrame(rows)
    front = ["mode", "n_queries", "Recall@5", "Recall@10", "Recall@20",
             "MRR", "nDCG@5", "nDCG@10", "nDCG@20", "mean_total_s"]
    df = df[[c for c in front if c in df.columns]
            + [c for c in df.columns if c not in front]]

    sfx = f"_{tag}" if tag else ""
    R.save_table(df, C.PHASE1, f"exp3_retrieval_bakeoff{sfx}")
    R.save_table(pd.DataFrame(per_query), C.PHASE1, f"exp3_retrieval_per_query{sfx}")

    R.plot_grouped_bars(
        df, "mode", ["Recall@5", "Recall@10", "Recall@20"], C.PHASE1,
        f"exp3_recall_by_retriever{sfx}",
        title="Retrieval recall by retriever configuration",
        ylabel="Recall@k (gold passages retrieved)",
        err={"Recall@5": "Recall@5_ci95", "Recall@10": "Recall@10_ci95",
             "Recall@20": "Recall@20_ci95"},
        caption=f"PubMedQA grounded questions (n={df['n_queries'].iloc[0]}) over a "
                f"{len(corpus):,}-passage corpus. Error bars are 95% CIs.",
    )
    R.plot_grouped_bars(
        df, "mode", ["MRR", "nDCG@10"], C.PHASE1, f"exp3_mrr_ndcg_by_retriever{sfx}",
        title="Ranking quality by retriever configuration",
        ylabel="Metric value",
        caption="MRR and nDCG@10 reward placing gold evidence at the very top.",
    )
    R.plot_grouped_bars(
        df, "mode", ["mean_total_s"], C.PHASE1, f"exp3_latency_by_retriever{sfx}",
        title="Mean retrieval latency per query",
        ylabel="Seconds per query",
        caption="Reranking adds cross-encoder cost; measured on "
                f"{C.DEVICE.upper()} ({C.provenance()['platform']}).",
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT 3 RESULTS  (actually executed)")
    print("=" * 78)
    show = ["mode", "n_queries", "Recall@5", "Recall@10", "Recall@20", "MRR",
            "nDCG@10", "mean_total_s"]
    print(df[[c for c in show if c in df.columns]].to_string(index=False,
                                                             float_format=lambda v: f"{v:.4f}"))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="number of eval questions")
    ap.add_argument("--modes", type=str, default=None, help="comma-separated arms")
    ap.add_argument("--tag", type=str, default="", help="output filename suffix")
    a = ap.parse_args()
    main(a.n, a.modes.split(",") if a.modes else None, a.tag)
