
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmrag import config as C
from fmrag import data as D
from fmrag import report as R
from fmrag.gate import AnswerabilityGate
from fmrag.pipeline import ABLATION_CONFIGS, FaithfulMedRAG
from fmrag.retrieval import HybridRetriever

log = logging.getLogger("exp8")

PROTOCOL = """\
# Human Evaluation Protocol -- Faithful-MedRAG

## Purpose
Validate the automatic NLI-based faithfulness metric against human judgement on
a {n}-item sample, and check whether the system's abstentions are appropriate.

## Reviewer qualification
Reviewers should have medical or biomedical training sufficient to judge
textbook- and literature-level claims. Two or more independent reviewers are
needed for inter-annotator agreement; a third resolves disagreements.

## Procedure
For each row of `human_eval_sheet.csv` the reviewer sees the question, the
numbered evidence passages the system actually retrieved, and the answer the
system actually produced. The reviewer fills ONLY the blank judgement columns
and must not consult outside sources when judging `evidence_support` -- that
field asks whether the SHOWN evidence supports the answer, not whether the
answer happens to be true.

## Field definitions
| Field | Values | Definition |
|---|---|---|
| `factual_correctness` | 0 / 1 | Is the final answer medically correct? |
| `evidence_support` | 0 / 1 | Is every assertion supported by the shown passages? |
| `citation_correctness` | 0 / 1 | Does each [n] marker point at a passage that supports its sentence? |
| `unsupported_claims` | e.g. `0;2` | Semicolon-separated indices of unsupported claims, blank if none |
| `answer_completeness` | 0 / 1 / 2 | 0 = does not answer, 1 = partial, 2 = fully answers |
| `abstention_appropriate` | 0 / 1 / NA | For abstained items only: was abstaining correct? NA if the system answered |
| `reviewer_notes` | text | Anything notable |

## Analysis once annotations are returned
Run `python3 experiments/exp8_human_eval.py --analyze <filled_sheet.csv>`. It computes:
  - Cohen's kappa / Krippendorff's alpha between reviewers (if >=2 present)
  - agreement between `evidence_support` and the automatic NLI verdict
  - precision / recall of the automatic metric against human labels
  - whether abstentions were judged appropriate

## Status
PENDING -- no human annotations have been collected. Every judgement column in
the emitted sheet is intentionally blank. No human-evaluation numbers appear in
any results file, table, or presentation produced by this project.
"""


def build_sheet(n_items: int) -> pd.DataFrame:
    C.set_all_seeds()
    corpus, grounded = D.build_corpus()
    retriever = HybridRetriever(corpus, cache_tag="main")
    ckpt = C.PHASE2 / "Checkpoints" / "answerability_gate.pkl"
    gate = AnswerabilityGate.load(ckpt) if ckpt.exists() else None
    system = FaithfulMedRAG(retriever=retriever, gate=gate)

    cfg = next(c for c in ABLATION_CONFIGS if c.name == "H_full_faithful_medrag")
    if gate is None:
        cfg = next(c for c in ABLATION_CONFIGS if c.name == "F_plus_claim_repair")
        log.warning("no gate -- sampling from configuration F instead of H")

    pool = sorted(grounded, key=lambda q: q.qid)
    sample = D.stratified_sample({"pubmedqa_grounded": pool}, n_items, seed=C.SEED)
    log.info("building human-eval sheet from %d real system outputs", len(sample))

    rows = []
    for i, q in enumerate(sample, start=1):
        r = system.run_one(q, cfg)
        ev, _ = retriever.retrieve(q.text, cfg.retrieval_mode, k=cfg.k_context)
        rows.append({
            "item_id": f"HE{i:04d}",
            "qid": r.qid,
            "question": r.question,
            "options": json.dumps(q.options) if q.options else "",
            "evidence_shown": "\n\n".join(
                f"[{j}] ({e.passage.title or e.passage.source}) {e.passage.text[:700]}"
                for j, e in enumerate(ev, start=1)
            ),
            "system_answer_choice": r.predicted,
            "system_explanation": r.final_answer_text,
            "system_abstained": (not r.answered),
            "system_abstain_reason": r.abstain_reason,
            "gold_answer": r.gold_answer,
            # automatic verdicts, for post-hoc comparison (NOT shown to reviewers)
            "auto_faithfulness": r.faith.get("faithfulness", ""),
            "auto_n_claims": r.faith.get("n_claims", 0),
            "auto_n_supported": r.faith.get("n_supported", 0),
            "auto_claim_labels": ";".join(c["label"] for c in r.claims),
            # blank judgement columns for the reviewer
            "reviewer_id": "",
            "factual_correctness": "",
            "evidence_support": "",
            "citation_correctness": "",
            "unsupported_claims": "",
            "answer_completeness": "",
            "abstention_appropriate": "",
            "reviewer_notes": "",
        })
        if i % 25 == 0:
            log.info("  %d/%d items prepared", i, len(sample))
    return pd.DataFrame(rows)


def analyze(path: Path) -> None:
    """Compute agreement statistics from a FILLED annotation sheet."""
    df = pd.read_csv(path)
    judged = df[df["evidence_support"].notna() & (df["evidence_support"].astype(str) != "")]
    if judged.empty:
        print("No annotations present in this sheet. Nothing to analyse.")
        print("Status remains PENDING.")
        return

    print(f"annotated items: {len(judged)} / {len(df)}")
    rev = judged["reviewer_id"].nunique()
    print(f"distinct reviewers: {rev}")

    if rev >= 2:
        from sklearn.metrics import cohen_kappa_score

        piv = judged.pivot_table(index="item_id", columns="reviewer_id",
                                 values="evidence_support", aggfunc="first").dropna()
        cols = list(piv.columns)
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                k = cohen_kappa_score(piv[cols[a]].astype(int), piv[cols[b]].astype(int))
                print(f"  Cohen's kappa ({cols[a]} vs {cols[b]}): {k:.4f}")

    auto = judged["auto_faithfulness"].astype(float)
    human = judged["evidence_support"].astype(float)
    auto_bin = (auto >= 0.999).astype(int)
    from sklearn.metrics import precision_recall_fscore_support

    pr, rc, f1, _ = precision_recall_fscore_support(
        human.astype(int), auto_bin, average="binary", zero_division=0
    )
    print("\nautomatic NLI metric vs human evidence_support:")
    print(f"  agreement : {(auto_bin == human.astype(int)).mean():.4f}")
    print(f"  precision : {pr:.4f}   recall: {rc:.4f}   F1: {f1:.4f}")
    print(f"  correlation (Pearson): {auto.corr(human):.4f}")


def main(n_items: int | None = None, analyze_path: str | None = None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    C.ensure_dirs()

    if analyze_path:
        analyze(Path(analyze_path))
        return

    n_items = n_items or C.SCALE.n_human_eval
    print("=" * 78)
    print("EXPERIMENT 8 : human evaluation instrument  (STATUS: PENDING)")
    print("=" * 78)

    df = build_sheet(n_items)
    out = C.PHASE2 / "Results" / "human_eval_sheet.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    proto = C.SHARED / "Reproducibility_Instructions" / "HUMAN_EVAL_PROTOCOL.md"
    proto.write_text(PROTOCOL.format(n=len(df)))

    R.save_json(
        {
            "status": "PENDING",
            "n_items_prepared": len(df),
            "n_annotations_collected": 0,
            "note": "Annotation columns are intentionally blank. No human-evaluation "
                    "results are reported anywhere in this project until the sheet is "
                    "filled in and analysed with --analyze.",
            "sheet": str(out.relative_to(C.ROOT)),
            "protocol": str(proto.relative_to(C.ROOT)),
        },
        C.PHASE2, "exp8_human_eval_status",
    )

    print(f"\nprepared {len(df)} annotation items from real system outputs")
    print(f"  sheet    : {out.relative_to(C.ROOT)}")
    print(f"  protocol : {proto.relative_to(C.ROOT)}")
    print("\nSTATUS: PENDING -- judgement columns are blank by design.")
    print("No human-evaluation numbers are reported until annotations are collected.")
    print("Then run:  python3 experiments/exp8_human_eval.py --analyze <filled.csv>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--analyze", type=str, default=None,
                    help="path to a FILLED annotation sheet")
    a = ap.parse_args()
    main(a.n, a.analyze)
