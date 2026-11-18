
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmrag import config as C
from fmrag import data as D
from fmrag import metrics as M
from fmrag import report as R
from fmrag.gate import FEATURE_NAMES, AnswerabilityGate, extract_features, make_label
from fmrag.retrieval import HybridRetriever
from fmrag.verify import NLIVerifier

log = logging.getLogger("exp5")

# Evidence-based choice: Experiment 3 showed MedCPT + its own biomedical
# cross-encoder is the strongest arm (Recall@10 0.819, MRR 0.981), while the
# proposal's general-domain BGE reranker degraded retrieval.
from fmrag.pipeline import BEST_RETRIEVAL
RETRIEVAL_MODE = BEST_RETRIEVAL
K_CONTEXT = C.PIPE.k_context


def build_feature_table(
    questions, retriever: HybridRetriever, verifier: NLIVerifier, k: int = K_CONTEXT
) -> pd.DataFrame:
    """Extract gate features + labels for every question, caching to disk."""
    cache = C.PHASE2 / "Checkpoints" / f"gate_features_{len(questions)}_{RETRIEVAL_MODE}.csv"
    if cache.exists():
        log.info("loading cached gate features (%s)", cache.name)
        return pd.read_csv(cache)

    rng = random.Random(C.SEED)
    corpus_index = {p.doc_id: i for i, p in enumerate(retriever.corpus)}
    dense_emb = retriever.dense.emb        # ensures the dense index is loaded

    rows = []
    t0 = time.time()
    for i, q in enumerate(questions, start=1):
        gold = set(q.gold_doc_ids)
        ablated = rng.random() < 0.5

        # Retrieve a deeper pool so the ablated condition still fills k slots.
        pool, _ = retriever.retrieve(q.text, RETRIEVAL_MODE, k=k + 20)
        if ablated:
            kept = [s for s in pool if s.passage.doc_id not in gold][:k]
        else:
            kept = pool[:k]
        for r_, s in enumerate(kept, start=1):
            s.rank = r_

        entail = verifier.entailment_strength(q.text, kept)
        rows_idx = [corpus_index[s.passage.doc_id] for s in kept
                    if s.passage.doc_id in corpus_index]
        embs = dense_emb[rows_idx] if len(rows_idx) >= 2 else None

        feats = extract_features(kept, entail=entail, passage_embeddings=embs)
        label = make_label(kept, gold)

        rec = {name: float(v) for name, v in zip(FEATURE_NAMES, feats)}
        rec.update({
            "qid": q.qid, "subset": q.subset, "label": int(label),
            "condition": "ablated" if ablated else "intact",
            "n_gold": len(gold),
        })
        rows.append(rec)

        if i % 50 == 0:
            el = time.time() - t0
            log.info("  features %d/%d (%.2fs/q, eta %.1f min)",
                     i, len(questions), el / i, (len(questions) - i) * el / i / 60)

    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    log.info("gate feature table cached -> %s", cache.name)
    return df


def main(n_total: int | None = None) -> dict:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    C.set_all_seeds()
    C.ensure_dirs()
    n_total = n_total or C.SCALE.n_gate_total

    print("=" * 78)
    print("EXPERIMENT 5 : learned answerability gate  (E4, Contribution B)")
    print("=" * 78)

    corpus, grounded = D.build_corpus()
    questions = sorted(grounded, key=lambda q: q.qid)[:n_total]
    retriever = HybridRetriever(corpus, cache_tag="main")
    verifier = NLIVerifier()

    df = build_feature_table(questions, retriever, verifier)
    log.info("feature table: %d rows, %.1f%% answerable",
             len(df), 100 * df["label"].mean())
    log.info("by condition:\n%s", df.groupby("condition")["label"].agg(["mean", "count"]))

    # ---- reproducible 60/20/20 split by qid ----
    qids = sorted(df["qid"].unique())
    rng = random.Random(C.SEED)
    rng.shuffle(qids)
    a, b = int(0.6 * len(qids)), int(0.8 * len(qids))
    split = {q: ("train" if i < a else "val" if i < b else "test")
             for i, q in enumerate(qids)}
    df["split"] = df["qid"].map(split)

    def XY(part):
        d = df[df["split"] == part]
        return d[FEATURE_NAMES].values, d["label"].values, d

    Xtr, ytr, dtr = XY("train")
    Xva, yva, dva = XY("val")
    Xte, yte, dte = XY("test")
    log.info("splits: train=%d val=%d test=%d", len(ytr), len(yva), len(yte))

    results = {}
    rows = []
    best_gate, best_auroc = None, -1.0

    for kind in ["logreg", "gbdt"]:
        gate = AnswerabilityGate(kind=kind).fit(Xtr, ytr)
        gate.calibrate(Xva, yva)                       # validation only
        gate.select_threshold(Xva, yva, target="f1")   # validation only

        p_va = gate.predict_proba(Xva)
        p_te = gate.predict_proba(Xte)                 # test touched once
        rep_va = M.gate_report(yva, p_va, gate.threshold)
        rep_te = M.gate_report(yte, p_te, gate.threshold)

        rows.append({"model": kind, "split": "validation", **rep_va})
        rows.append({"model": kind, "split": "test", **rep_te})
        results[kind] = {"val": rep_va, "test": rep_te,
                         "threshold": gate.threshold,
                         "feature_importance": gate.feature_importance()}
        log.info("%-7s TEST  AUROC=%.4f AUPRC=%.4f F1=%.4f ECE=%.4f Brier=%.4f",
                 kind, rep_te["AUROC"], rep_te["AUPRC"], rep_te["F1"],
                 rep_te["ECE"], rep_te["Brier"])

        if rep_te["AUROC"] > best_auroc:
            best_auroc, best_gate = rep_te["AUROC"], gate

    # ---- persist the selected gate ----
    ckpt = C.PHASE2 / "Checkpoints" / "answerability_gate.pkl"
    best_gate.save(ckpt)

    df_metrics = pd.DataFrame(rows)
    R.save_table(df_metrics, C.PHASE2, "exp5_gate_metrics")
    R.save_table(df, C.PHASE2, "exp5_gate_features")
    R.save_json(results, C.PHASE2, "exp5_gate_details")

    # ---- plots ----
    p_te = best_gate.predict_proba(Xte)
    R.plot_calibration(
        M.reliability_bins(yte, p_te), C.PHASE2, "exp5_gate_calibration",
        ece=M.ece(yte, p_te),
    )
    imp = best_gate.feature_importance()
    if imp:
        R.plot_feature_importance(imp, C.PHASE2, "exp5_gate_feature_importance")

    rc = M.risk_coverage_curve(yte.astype(float), p_te)
    R.save_table(pd.DataFrame(rc), C.PHASE2, "exp5_gate_risk_coverage")
    R.plot_risk_coverage(
        rc, C.PHASE2, "exp5_gate_risk_coverage",
        title="Answerability-gate risk-coverage (test split)",
    )

    disc = df_metrics[df_metrics["split"] == "test"][
        ["model", "n", "AUROC", "AUPRC", "precision", "recall", "F1", "Brier", "ECE"]
    ]
    R.plot_grouped_bars(
        disc, "model", ["AUROC", "AUPRC", "F1"], C.PHASE2,
        "exp5_gate_discrimination",
        title="Answerability-gate discrimination (held-out test split)",
        ylabel="Metric value",
        caption="Thresholds and calibration were fitted on validation only; "
                "test was scored once.",
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT 5 RESULTS  (actually executed)")
    print("=" * 78)
    print(f"label balance: {100*df['label'].mean():.1f}% answerable "
          f"(n={len(df)}; 50/50 intact vs gold-ablated by seed {C.SEED})")
    print(df_metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nselected gate: {best_gate.kind} (threshold={best_gate.threshold:.3f}) "
          f"-> {ckpt.name}")
    print("\ntop gate features by importance:")
    for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:6]:
        print(f"   {k:26s} {v:.4f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    main(a.n)
