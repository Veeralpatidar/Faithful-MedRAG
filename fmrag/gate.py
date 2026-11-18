
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C
from .retrieval import Scored

log = logging.getLogger("fmrag.gate")

FEATURE_NAMES = [
    "score_max", "score_mean", "score_min", "score_std",
    "gap_1_2", "gap_1_rest", "margin_norm",
    "score_entropy",
    "interpassage_agreement",
    "q_entail_max", "q_entail_mean", "q_contra_max",
    "n_evidence",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def extract_features(
    evidence: list[Scored],
    entail: dict | None = None,
    passage_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build the gate feature vector for one query's retrieval result.

    `entail` comes from NLIVerifier.entailment_strength (query-evidence
    entailment). `passage_embeddings` (k x d) powers inter-passage agreement;
    when absent, agreement falls back to 0.0 and is reported as unavailable.
    """
    scores = np.array([e.score for e in evidence], dtype=float)
    f: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
    f["n_evidence"] = float(len(evidence))

    if scores.size:
        f["score_max"] = float(scores.max())
        f["score_mean"] = float(scores.mean())
        f["score_min"] = float(scores.min())
        f["score_std"] = float(scores.std())
        if scores.size >= 2:
            s = np.sort(scores)[::-1]
            f["gap_1_2"] = float(s[0] - s[1])
            f["gap_1_rest"] = float(s[0] - s[1:].mean())
            denom = abs(s[0]) + abs(s[1]) + 1e-9
            f["margin_norm"] = float((s[0] - s[1]) / denom)
        p = _softmax(scores)
        nz = p[p > 0]
        f["score_entropy"] = float(-(nz * np.log(nz)).sum())

    if passage_embeddings is not None and len(passage_embeddings) >= 2:
        X = np.asarray(passage_embeddings, dtype=float)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        S = Xn @ Xn.T
        iu = np.triu_indices(len(Xn), k=1)
        f["interpassage_agreement"] = float(S[iu].mean())

    if entail:
        f["q_entail_max"] = float(entail.get("q_entail_max", 0.0))
        f["q_entail_mean"] = float(entail.get("q_entail_mean", 0.0))
        f["q_contra_max"] = float(entail.get("q_contra_max", 0.0))

    return np.array([f[n] for n in FEATURE_NAMES], dtype=float)


def make_label(evidence: list[Scored], gold_doc_ids: set[str]) -> int:
    """
    ANSWERABLE (1) iff the retrieved evidence contains a gold passage.

    Gold ids are used only here, to form the target. No feature touches them.
    """
    if not gold_doc_ids:
        return 0
    return int(any(e.passage.doc_id in gold_doc_ids for e in evidence))


@dataclass
class GateArtifacts:
    """What gets checkpointed: model, calibrator, scaler, threshold, metadata."""
    model: object
    calibrator: object | None
    scaler: object
    threshold: float
    feature_names: list[str]
    meta: dict


class AnswerabilityGate:
    """
    Logistic-regression / gradient-boosted gate with isotonic calibration.

    Calibration matters here: the abstention threshold is only meaningful if
    the predicted probability means what it says, so the gate is calibrated on
    the validation split and ECE/Brier are reported alongside AUROC.
    """

    def __init__(self, kind: str = "logreg"):
        self.kind = kind
        self.model = None
        self.calibrator = None
        self.scaler = None
        self.threshold = C.PIPE.gate_threshold
        self.meta: dict = {}

    # -------------------------------------------------- training
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AnswerabilityGate":
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        if self.kind == "logreg":
            self.model = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=C.SEED
            )
        elif self.kind == "gbdt":
            self.model = GradientBoostingClassifier(random_state=C.SEED)
        else:
            raise ValueError(f"unknown gate kind: {self.kind!r}")

        self.model.fit(Xs, y)
        self.meta = {
            "kind": self.kind,
            "n_train": int(len(y)),
            "positive_rate_train": float(y.mean()),
            "feature_names": FEATURE_NAMES,
        }
        log.info("gate fitted (%s) on %d examples, %.1f%% answerable",
                 self.kind, len(y), 100 * y.mean())
        return self

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> "AnswerabilityGate":
        """Isotonic calibration on the VALIDATION split only."""
        from sklearn.isotonic import IsotonicRegression

        raw = self._raw_proba(X_val)
        y_val = np.asarray(y_val).astype(int)
        if len(np.unique(y_val)) < 2:
            log.warning("validation split is single-class -- skipping calibration")
            return self
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y_val)
        log.info("gate calibrated on %d validation examples", len(y_val))
        return self

    # -------------------------------------------------- inference
    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        return self.model.predict_proba(Xs)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._raw_proba(X)
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.0, 1.0)

    def should_answer(self, x: np.ndarray) -> tuple[bool, float]:
        p = float(self.predict_proba(np.asarray(x, dtype=float).reshape(1, -1))[0])
        return p >= self.threshold, p

    # -------------------------------------------------- threshold selection
    def select_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        min_coverage: float = 0.5,
        target: str = "f1",
    ) -> float:
        """
        Choose the abstention threshold on VALIDATION data (Part E #3).

        `target='f1'` maximises F1; `target='precision_at_coverage'` picks the
        highest-precision threshold that still answers `min_coverage` of
        questions -- the clinically sensible objective.
        """
        from sklearn.metrics import f1_score, precision_score

        p = self.predict_proba(X_val)
        y_val = np.asarray(y_val).astype(int)
        best_thr, best_val = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 91):
            pred = (p >= thr).astype(int)
            cov = pred.mean()
            if target == "f1":
                val = f1_score(y_val, pred, zero_division=0)
            else:
                if cov < min_coverage:
                    continue
                val = precision_score(y_val, pred, zero_division=0)
            if val > best_val:
                best_val, best_thr = val, float(thr)
        self.threshold = best_thr
        self.meta["threshold_target"] = target
        self.meta["threshold_value"] = best_thr
        self.meta["threshold_score"] = float(best_val)
        log.info("threshold selected on validation: %.3f (%s=%.4f)",
                 best_thr, target, best_val)
        return best_thr

    def feature_importance(self) -> dict:
        if self.model is None:
            return {}
        if hasattr(self.model, "coef_"):
            vals = np.abs(self.model.coef_[0])
        elif hasattr(self.model, "feature_importances_"):
            vals = self.model.feature_importances_
        else:
            return {}
        tot = vals.sum() or 1.0
        return {n: float(v / tot) for n, v in zip(FEATURE_NAMES, vals)}

    # -------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                GateArtifacts(
                    model=self.model, calibrator=self.calibrator, scaler=self.scaler,
                    threshold=self.threshold, feature_names=FEATURE_NAMES, meta=self.meta,
                ),
                f,
            )
        log.info("gate checkpoint saved -> %s", path)

    @classmethod
    def load(cls, path: Path) -> "AnswerabilityGate":
        with Path(path).open("rb") as f:
            art: GateArtifacts = pickle.load(f)
        g = cls(kind=art.meta.get("kind", "logreg"))
        g.model, g.calibrator, g.scaler = art.model, art.calibrator, art.scaler
        g.threshold, g.meta = art.threshold, art.meta
        return g
