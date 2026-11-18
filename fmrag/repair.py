"""
Faithful-MedRAG : claim-level surgical re-retrieval and repair loop.

This is Contribution A -- the headline novelty (proposal Objective 3).

What makes it 'surgical': prior work corrects at the *query* level (CRAG) or
judges support at the *answer* level (Self-RAG). Here the unit of correction is
the individual claim. A supported sentence is never touched; only the specific
sentences the verifier rejected are re-retrieved for and rewritten.

One round:
  1. verify every claim against the current evidence (DeBERTa-NLI)
  2. for each unsupported claim, build a targeted query *from that claim*
  3. retrieve fresh evidence for that claim alone
  4. rewrite only that sentence against the new evidence
  5. re-verify the rewrite; keep it only if support strictly improved

After `max_repair_rounds`, any claim still unsupported is abstained on
individually -- dropped from the answer and recorded, rather than left in as an
unsupported assertion. That per-claim abstention is what separates this from
all-or-nothing answer-level abstention.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from . import config as C
from .retrieval import HybridRetriever, Scored
from .verify import Claim, NLIVerifier

log = logging.getLogger("fmrag.repair")

_STOP = {
    "the", "a", "an", "of", "in", "is", "are", "was", "were", "to", "and", "or",
    "for", "with", "that", "this", "these", "those", "it", "its", "as", "by",
    "be", "been", "has", "have", "had", "which", "can", "may", "also", "from",
    "on", "at", "but", "not", "no", "than", "then", "there", "their",
}


def targeted_query(claim_text: str, question: str, max_terms: int = 14) -> str:
    """
    Build a retrieval query from the *claim*, not the original question.

    This is the key difference from query-level correction: the query targets
    exactly the proposition that failed verification. The question is appended
    as light context so the query stays on-topic.
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", claim_text.lower())
    content = [w for w in words if w not in _STOP and len(w) > 2]
    seen, terms = set(), []
    for w in content:
        if w not in seen:
            seen.add(w)
            terms.append(w)
        if len(terms) >= max_terms:
            break
    q_words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", question.lower())
               if w not in _STOP and len(w) > 3][:6]
    return " ".join(terms + [w for w in q_words if w not in seen])


@dataclass
class RepairTrace:
    """Audit trail for one repair attempt -- used in the results tables."""
    claim_index: int
    round: int
    original: str
    query: str
    rewritten: str | None
    p_entail_before: float
    p_entail_after: float
    accepted: bool
    abstained: bool
    retrieve_s: float
    generate_s: float


@dataclass
class RepairOutcome:
    claims: list[Claim]
    traces: list[RepairTrace] = field(default_factory=list)
    rounds_used: int = 0
    n_repair_retrievals: int = 0
    repair_latency_s: float = 0.0
    evidence: list[Scored] = field(default_factory=list)

    @property
    def n_repaired(self) -> int:
        return sum(1 for c in self.claims if c.repaired)

    @property
    def n_abstained(self) -> int:
        return sum(1 for c in self.claims if c.label == "abstained")

    def final_answer_text(self) -> str:
        """The answer after repair, with abstained claims removed."""
        kept = [c for c in self.claims if c.label != "abstained"]
        if not kept:
            return "Insufficient evidence to answer from the retrieved documents."
        return " ".join(
            f"{c.text} [{c.cited[0]}]" if c.cited else c.text for c in kept
        )


def repair_answer(
    question: str,
    claims: list[Claim],
    evidence: list[Scored],
    retriever: HybridRetriever,
    verifier: NLIVerifier,
    generator,                      # GroundedGenerator
    retrieval_mode: str = "rrf_bge",
    max_rounds: int = None,
) -> RepairOutcome:
    """
    Run the surgical repair loop. `claims` must already be verified once.

    Returns a RepairOutcome carrying the repaired claims plus a full audit
    trail of every attempt (accepted or not).
    """
    max_rounds = C.PIPE.max_repair_rounds if max_rounds is None else max_rounds
    out = RepairOutcome(claims=claims, evidence=list(evidence))
    t_start = time.time()

    for rnd in range(1, max_rounds + 1):
        targets = [c for c in out.claims if c.label in ("contradicted", "neutral")]
        if not targets:
            break
        out.rounds_used = rnd
        log.debug("repair round %d: %d unsupported claim(s)", rnd, len(targets))

        for c in targets:
            # ---- 2/3. targeted re-retrieval for this claim alone ----
            q = targeted_query(c.text, question)
            t0 = time.time()
            fresh, _ = retriever.retrieve(q, retrieval_mode, k=C.PIPE.k_repair_retrieve)
            t_ret = time.time() - t0
            out.n_repair_retrievals += 1

            # Merge new evidence with the existing context, de-duplicated, so
            # citation markers stay stable for claims already supported.
            known = {e.passage.doc_id for e in out.evidence}
            added = [e for e in fresh if e.passage.doc_id not in known]
            merged = out.evidence + added
            for i, e in enumerate(merged, start=1):
                e.rank = i

            # ---- 4. rewrite only this sentence ----
            rewritten, t_gen = generator.repair_claim(question, c.text, merged)

            p_before = c.p_entail
            accepted = False
            abstained = False

            if not rewritten or re.search(r"insufficient evidence", rewritten, re.I):
                p_after = p_before
            else:
                # ---- 5. re-verify the rewrite; keep only a strict improvement ----
                probe = Claim(index=c.index, text=re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", rewritten).strip())
                for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", rewritten):
                    probe.cited.extend(int(x) for x in re.findall(r"\d+", m.group(1)))
                probe.cited = sorted(set(probe.cited))
                probe.has_citation = bool(probe.cited)
                verifier.verify([probe], merged)
                p_after = probe.p_entail

                if probe.label == "supported" and p_after > p_before:
                    c.text = probe.text
                    c.cited = probe.cited
                    c.has_citation = probe.has_citation
                    c.citation_ok = probe.citation_ok
                    c.label = probe.label
                    c.p_entail = probe.p_entail
                    c.p_contradict = probe.p_contradict
                    c.best_evidence_idx = probe.best_evidence_idx
                    c.repaired = True
                    c.repair_round = rnd
                    accepted = True
                    out.evidence = merged

            out.traces.append(
                RepairTrace(
                    claim_index=c.index, round=rnd, original=c.text if not accepted else probe.text,
                    query=q, rewritten=rewritten or None,
                    p_entail_before=p_before, p_entail_after=p_after,
                    accepted=accepted, abstained=abstained,
                    retrieve_s=t_ret, generate_s=t_gen,
                )
            )

    # ---- per-claim abstention for anything still unsupported ----
    for c in out.claims:
        if c.label in ("contradicted", "neutral"):
            c.label = "abstained"

    out.repair_latency_s = time.time() - t_start
    return out
