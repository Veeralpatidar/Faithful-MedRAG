
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .retrieval import Scored

log = logging.getLogger("fmrag.verify")

# --------------------------------------------------------------------------
# Claim decomposition
# --------------------------------------------------------------------------
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_ABSTAIN_MARK = re.compile(r"\b(insufficient evidence|cannot determine|not enough evidence)\b", re.I)

# Abbreviations that must not end a sentence during splitting.
_ABBREV = r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)(?<!\bDr)(?<!\bFig)(?<!\bno)(?<!\bca)"
_SENT_SPLIT = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Claim:
    """One atomic claim extracted from a generated answer."""
    index: int
    text: str
    cited: list[int] = field(default_factory=list)   # 1-based citation markers

    # filled in by the verifier
    label: str = "unverified"        # supported|contradicted|neutral|abstained
    p_entail: float = 0.0
    p_contradict: float = 0.0
    best_evidence_idx: int = -1      # 0-based index into the evidence list
    citation_ok: bool = False
    has_citation: bool = False
    repaired: bool = False
    repair_round: int = 0


def decompose_claims(answer: str) -> list[Claim]:
    """
    Split an answer into atomic claims (sentence granularity) and parse its
    citation markers. Sentence-level decomposition is what the proposal
    specifies and what the NLI verifier is trained for.
    """
    text = (answer or "").strip()
    if not text:
        return []
    claims: list[Claim] = []
    for i, raw in enumerate(s for s in _SENT_SPLIT.split(text) if s.strip()):
        sent = raw.strip()
        cited: list[int] = []
        for m in _CITE.finditer(sent):
            cited.extend(int(x) for x in re.findall(r"\d+", m.group(1)))
        bare = _CITE.sub("", sent).strip()
        bare = re.sub(r"\s{2,}", " ", bare)
        bare = re.sub(r"\s+([.,;:!?])", r"\1", bare).strip()
        if len(bare) < 8:        # drop fragments that carry no proposition
            continue
        claims.append(
            Claim(index=len(claims), text=bare, cited=sorted(set(cited)))
        )
    for c in claims:
        c.has_citation = bool(c.cited)
    return claims


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN_MARK.search(answer or ""))


# --------------------------------------------------------------------------
# NLI verification
# --------------------------------------------------------------------------
class NLIVerifier:
    """
    DeBERTa-NLI claim verifier. premise = retrieved passage, hypothesis = claim.

    Scores all (claim, evidence) pairs in batches and keeps, per claim, the
    maximum entailment and maximum contradiction probability across evidence.
    """

    def __init__(self, model_name: str = C.NLI_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = C.DEVICE
        log.info("loading NLI verifier (%s)", model_name)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        # Resolve label positions from the model config rather than assuming.
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.i_entail = next((i for i, l in id2label.items() if "entail" in l), 0)
        self.i_contra = next((i for i, l in id2label.items() if "contra" in l), 2)
        self.i_neutral = next((i for i, l in id2label.items() if "neutral" in l), 1)
        log.info("NLI label map: %s (entail=%d contra=%d)",
                 id2label, self.i_entail, self.i_contra)

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Return softmax probabilities, shape (n_pairs, 3)."""
        torch = self._torch
        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)
        out = []
        bs = C.PIPE.nli_batch_size
        with torch.no_grad():
            for s in range(0, len(pairs), bs):
                chunk = pairs[s : s + bs]
                enc = self.tok(
                    [p for p, _ in chunk],
                    [h for _, h in chunk],
                    truncation=True, padding=True, max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)
                out.append(probs.cpu().numpy())
        return np.vstack(out)

    def verify(
        self,
        claims: list[Claim],
        evidence: list[Scored],
        support_threshold: float = C.PIPE.nli_support_threshold,
        contradict_threshold: float = C.PIPE.nli_contradict_threshold,
    ) -> list[Claim]:
        """Assign a verdict to every claim. Mutates and returns `claims`."""
        if not claims:
            return claims
        if not evidence:
            for c in claims:
                c.label, c.p_entail, c.p_contradict = "neutral", 0.0, 0.0
            return claims

        premises = [e.passage.for_index()[:1800] for e in evidence]
        pairs = [(prem, c.text) for c in claims for prem in premises]
        probs = self._score_pairs(pairs)

        n_ev = len(premises)
        for ci, c in enumerate(claims):
            block = probs[ci * n_ev : (ci + 1) * n_ev]
            ent = block[:, self.i_entail]
            con = block[:, self.i_contra]
            best = int(np.argmax(ent))
            c.p_entail = float(ent[best])
            c.p_contradict = float(con.max())
            c.best_evidence_idx = best

            if c.p_entail >= support_threshold:
                c.label = "supported"
            elif c.p_contradict >= contradict_threshold:
                c.label = "contradicted"
            else:
                c.label = "neutral"

            # Citation correctness: does the passage the claim cites entail it?
            if c.cited:
                ok = False
                for marker in c.cited:
                    j = marker - 1                      # markers are 1-based
                    if 0 <= j < n_ev and ent[j] >= support_threshold:
                        ok = True
                        break
                c.citation_ok = ok
        return claims

    def entailment_strength(self, query: str, evidence: list[Scored]) -> dict:
        """
        Query-evidence entailment strength -- a feature the proposal lists for
        the answerability gate. Treats the question as the hypothesis and each
        passage as the premise.
        """
        if not evidence:
            return {"q_entail_max": 0.0, "q_entail_mean": 0.0, "q_contra_max": 0.0}
        pairs = [(e.passage.for_index()[:1800], query) for e in evidence]
        probs = self._score_pairs(pairs)
        ent = probs[:, self.i_entail]
        return {
            "q_entail_max": float(ent.max()),
            "q_entail_mean": float(ent.mean()),
            "q_contra_max": float(probs[:, self.i_contra].max()),
        }
