
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# 1. Retrieval metrics
# --------------------------------------------------------------------------
def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    hit = len(set(retrieved_ids[:k]) & gold_ids)
    return hit / len(gold_ids)


def hit_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    return 1.0 if set(retrieved_ids[:k]) & gold_ids else 0.0


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return float("nan")
    for i, d in enumerate(retrieved_ids, start=1):
        if d in gold_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not gold_ids:
        return float("nan")
    dcg = sum(
        (1.0 / math.log2(i + 1))
        for i, d in enumerate(retrieved_ids[:k], start=1)
        if d in gold_ids
    )
    ideal_n = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else float("nan")


def retrieval_report(runs: list[tuple[list[str], set[str]]], ks=(5, 10, 20)) -> dict:
    """Aggregate retrieval metrics over many queries, with 95% CIs."""
    acc: dict[str, list[float]] = {}
    for rids, gids in runs:
        if not gids:
            continue
        for k in ks:
            acc.setdefault(f"Recall@{k}", []).append(recall_at_k(rids, gids, k))
            acc.setdefault(f"Hit@{k}", []).append(hit_at_k(rids, gids, k))
            acc.setdefault(f"nDCG@{k}", []).append(ndcg_at_k(rids, gids, k))
        acc.setdefault("MRR", []).append(mrr(rids, gids))

    out: dict[str, float] = {"n_queries": len(acc.get("MRR", []))}
    for key, vals in acc.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        out[key] = float(arr.mean())
        out[f"{key}_ci95"] = float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return out


# --------------------------------------------------------------------------
# 2. Answer accuracy
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def choice_accuracy(pred: str, gold: str, options: dict | None) -> float:
    
    p_raw = (pred or "").strip()
    g_raw = (gold or "").strip()
    # `parse_answer` normally strips this, but tolerate it reaching us directly.
    p_raw = re.sub(r"^(?:the\s+)?answer\s*(?:is)?\s*[:\-]?\s*", "", p_raw, flags=re.I).strip()
    p_raw = p_raw.strip("*").strip()
    if not p_raw:
        return 0.0

    opts = {str(k).strip().upper(): str(v) for k, v in (options or {}).items()}

    # ---- resolve the gold side to (letter, text) ----
    gold_letter, gold_text = None, g_raw
    if g_raw.upper() in opts:
        gold_letter, gold_text = g_raw.upper(), opts[g_raw.upper()]
    else:
        for k, v in opts.items():
            if _norm(v) == _norm(g_raw):
                gold_letter, gold_text = k, v
                break

    # ---- resolve the prediction side to (letter, text) ----
    pred_letter, pred_text = None, p_raw
    m = re.match(r"^\(?([A-Za-z])\)?(?:[\)\.\:\-,]|\s|$)", p_raw)
    if m and m.group(1).upper() in opts:
        pred_letter = m.group(1).upper()
        rest = p_raw[m.end():].strip()
        pred_text = rest or opts[pred_letter]
    else:
        for k, v in opts.items():
            if _norm(v) == _norm(p_raw):
                pred_letter, pred_text = k, v
                break

    # ---- compare ----
    if gold_letter and pred_letter:
        return 1.0 if pred_letter == gold_letter else 0.0
    gt, pt = _norm(gold_text), _norm(pred_text)
    if gt and pt and (pt == gt or gt == pt):
        return 1.0
    # last resort: the gold text appears verbatim in a short prediction
    if gt and pt and len(pt) <= len(gt) + 24 and gt in pt:
        return 1.0
    return 0.0


# --------------------------------------------------------------------------
# 3. Faithfulness / hallucination
# --------------------------------------------------------------------------
@dataclass
class FaithfulnessResult:
    n_claims: int
    n_supported: int
    n_contradicted: int
    n_neutral: int
    n_abstained: int
    faithfulness: float          # supported / total
    hallucination_rate: float    # (contradicted + unsupported-neutral) / total
    unsupported_rate: float
    citation_precision: float


def faithfulness_report(verdicts: list) -> FaithfulnessResult:
    """
    Aggregate claim-level verdicts for one answer.

    `verdicts` is a list of objects exposing `.label` in
    {'supported','contradicted','neutral','abstained'} and `.citation_ok`.
    """
    n = len(verdicts)
    if n == 0:
        return FaithfulnessResult(0, 0, 0, 0, 0, float("nan"), float("nan"),
                                  float("nan"), float("nan"))
    sup = sum(1 for v in verdicts if v.label == "supported")
    con = sum(1 for v in verdicts if v.label == "contradicted")
    neu = sum(1 for v in verdicts if v.label == "neutral")
    abst = sum(1 for v in verdicts if v.label == "abstained")

    # Explicit hallucination rule: contradiction, or neutral left in the answer.
    halluc = con + neu
    cited = [v for v in verdicts if getattr(v, "has_citation", False)]
    cit_prec = (
        sum(1 for v in cited if getattr(v, "citation_ok", False)) / len(cited)
        if cited else float("nan")
    )
    return FaithfulnessResult(
        n_claims=n, n_supported=sup, n_contradicted=con, n_neutral=neu,
        n_abstained=abst,
        faithfulness=sup / n,
        hallucination_rate=halluc / n,
        unsupported_rate=(con + neu) / n,
        citation_precision=cit_prec,
    )


def aggregate_faithfulness(results: list[FaithfulnessResult]) -> dict:
    """Micro-averaged (claim-weighted) faithfulness across many answers."""
    tot = sum(r.n_claims for r in results)
    if tot == 0:
        return {"n_answers": len(results), "n_claims": 0}
    sup = sum(r.n_supported for r in results)
    con = sum(r.n_contradicted for r in results)
    neu = sum(r.n_neutral for r in results)
    abst = sum(r.n_abstained for r in results)

    per_answer = np.array(
        [r.faithfulness for r in results if not math.isnan(r.faithfulness)], dtype=float
    )
    ci = (
        float(1.96 * per_answer.std(ddof=1) / math.sqrt(per_answer.size))
        if per_answer.size > 1 else 0.0
    )
    cits = [r.citation_precision for r in results if not math.isnan(r.citation_precision)]
    return {
        "n_answers": len(results),
        "n_claims": tot,
        "faithfulness_micro": sup / tot,
        "faithfulness_macro": float(per_answer.mean()) if per_answer.size else float("nan"),
        "faithfulness_macro_ci95": ci,
        "hallucination_rate": (con + neu) / tot,
        "contradicted_rate": con / tot,
        "neutral_rate": neu / tot,
        "abstained_claims": abst,
        "claims_per_answer": tot / max(len(results), 1),
        "citation_precision": float(np.mean(cits)) if cits else float("nan"),
    }


# --------------------------------------------------------------------------
# 4. Gate quality : discrimination, calibration, selective risk
# --------------------------------------------------------------------------
def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    # rank-based (Mann-Whitney U)
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks over ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        total += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """(bin_centre, empirical_rate, count) for the calibration plot."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            rows.append(((lo + hi) / 2, float("nan"), 0))
        else:
            rows.append(((lo + hi) / 2, float(y[m].mean()), int(m.sum())))
    return rows


def selective_metrics(correct: np.ndarray, answered: np.ndarray) -> dict:
    """
    Coverage and selective accuracy for an abstaining system.

    coverage          = fraction of questions answered
    selective_acc     = accuracy among answered questions
    risk              = 1 - selective_acc
    """
    correct = np.asarray(correct, dtype=float)
    answered = np.asarray(answered, dtype=bool)
    cov = float(answered.mean()) if answered.size else float("nan")
    sel = float(correct[answered].mean()) if answered.any() else float("nan")
    return {
        "coverage": cov,
        "selective_accuracy": sel,
        "risk": 1.0 - sel if not math.isnan(sel) else float("nan"),
        "overall_accuracy": float(correct.mean()) if correct.size else float("nan"),
        "n_answered": int(answered.sum()),
        "n_total": int(answered.size),
    }


def risk_coverage_curve(correct: np.ndarray, confidence: np.ndarray, n_points: int = 21):
    """
    Risk-coverage curve: sweep the abstention threshold over confidence and
    report (coverage, risk, selective_accuracy, threshold) at each point.
    """
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    rows = []
    for thr in np.linspace(0.0, 1.0, n_points):
        ans = confidence >= thr
        if ans.sum() == 0:
            rows.append({"threshold": float(thr), "coverage": 0.0,
                         "selective_accuracy": float("nan"), "risk": float("nan")})
            continue
        sel = float(correct[ans].mean())
        rows.append({
            "threshold": float(thr),
            "coverage": float(ans.mean()),
            "selective_accuracy": sel,
            "risk": 1.0 - sel,
        })
    return rows


def gate_report(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Full gate evaluation: discrimination + calibration + operating point."""
    from sklearn.metrics import precision_recall_fscore_support

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "AUROC": auroc(y, p),
        "AUPRC": auprc(y, p),
        "precision": float(pr),
        "recall": float(rc),
        "F1": float(f1),
        "accuracy": float((pred == y).mean()),
        "Brier": brier(y, p),
        "ECE": ece(y, p),
    }
