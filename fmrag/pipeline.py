
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import metrics as M
from .data import Question
from .gate import AnswerabilityGate, extract_features
from .generate import GroundedGenerator, LLM, VanillaGenerator
from .repair import RepairOutcome, repair_answer
from .retrieval import HybridRetriever, Scored
from .verify import Claim, NLIVerifier, decompose_claims

log = logging.getLogger("fmrag.pipeline")


# --------------------------------------------------------------------------
# Configuration of one system variant
# --------------------------------------------------------------------------
@dataclass
class SystemConfig:
    """One row of the ablation table."""
    name: str
    retrieval_mode: str = "medcpt"   # bm25|medcpt|rrf|rrf_bge
    grounded: bool = False           # strict-citation prompt
    verify: bool = False             # run NLI claim verification
    use_repair: bool = False         # claim-level surgical repair
    use_gate: bool = False           # pre-generation answerability gate
    k_context: int = C.PIPE.k_context
    description: str = ""


# The retrieval configuration used from the reranking stage onward.
#
# The proposal specifies BAAI/bge-reranker-base. Experiment 3 measured that
# choice degrading retrieval on this corpus (Recall@10 0.693 vs 0.762 for
# unreranked MedCPT), because bge-reranker-base is general-domain. MedCPT's own
# biomedical cross-encoder instead IMPROVED it substantially (0.819, MRR 0.981).
# Configurations D onward therefore use the domain-matched reranker -- an
# evidence-based revision of the proposed architecture, documented rather than
# silent. Set BEST_RETRIEVAL = "rrf_bge" to reproduce the proposal literally.
BEST_RETRIEVAL = "medcpt_ce"

# The eight configurations named in Part B, Experiment 6.
ABLATION_CONFIGS = [
    SystemConfig("A_vanilla_medrag", "medcpt", False, False, False, False,
                 description="Vanilla MedRAG: dense retrieval + free-form answer."),
    SystemConfig("B_plus_faithfulness_eval", "medcpt", False, True, False, False,
                 description="Same generation, but claims are now verified (measurement only)."),
    SystemConfig("C_plus_hybrid_retrieval", "rrf", False, True, False, False,
                 description="Hybrid MedCPT+BM25 retrieval fused by RRF."),
    SystemConfig("D_plus_reranking", BEST_RETRIEVAL, False, True, False, False,
                 description="Domain-matched cross-encoder reranking (MedCPT-CE). "
                             "Replaces RRF fusion, which Experiment 3 showed harmful."),
    SystemConfig("E_plus_grounded_generation", BEST_RETRIEVAL, True, True, False, False,
                 description="Strict per-sentence citation prompt."),
    SystemConfig("F_plus_claim_repair", BEST_RETRIEVAL, True, True, True, False,
                 description="Adds the claim-level surgical repair loop (Contribution A)."),
    SystemConfig("G_plus_answerability_gate", BEST_RETRIEVAL, True, True, False, True,
                 description="Adds the learned pre-generation gate (Contribution B)."),
    SystemConfig("H_full_faithful_medrag", BEST_RETRIEVAL, True, True, True, True,
                 description="Complete Faithful-MedRAG: all components enabled."),
]


# --------------------------------------------------------------------------
# Per-question result
# --------------------------------------------------------------------------
@dataclass
class ItemResult:
    qid: str
    subset: str
    question: str
    gold_answer: str
    config: str

    predicted: str = ""
    correct: float = 0.0
    answered: bool = True            # False when the gate or the loop abstained
    abstain_reason: str = ""

    gate_prob: float = float("nan")
    retrieved_ids: list[str] = field(default_factory=list)
    gold_doc_ids: list[str] = field(default_factory=list)

    claims: list[dict] = field(default_factory=list)
    faith: dict = field(default_factory=dict)

    rationale: str = ""
    final_answer_text: str = ""

    n_repaired: int = 0
    n_claim_abstained: int = 0
    repair_rounds: int = 0
    n_repair_retrievals: int = 0

    t_retrieve_s: float = 0.0
    t_gate_s: float = 0.0
    t_generate_s: float = 0.0
    t_verify_s: float = 0.0
    t_repair_s: float = 0.0
    t_total_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


# --------------------------------------------------------------------------
# The system
# --------------------------------------------------------------------------
class FaithfulMedRAG:
    """
    The complete system. Heavy components are loaded lazily and shared across
    every configuration, so an 8-way ablation loads each model exactly once.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLM | None = None,
        verifier: NLIVerifier | None = None,
        gate: AnswerabilityGate | None = None,
    ):
        self.retriever = retriever
        self._llm = llm
        self._verifier = verifier
        self.gate = gate
        self._vanilla = None
        self._grounded = None

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    @property
    def verifier(self) -> NLIVerifier:
        if self._verifier is None:
            self._verifier = NLIVerifier()
        return self._verifier

    @property
    def vanilla(self) -> VanillaGenerator:
        if self._vanilla is None:
            self._vanilla = VanillaGenerator(self.llm)
        return self._vanilla

    @property
    def grounded(self) -> GroundedGenerator:
        if self._grounded is None:
            self._grounded = GroundedGenerator(self.llm)
        return self._grounded

    # ---------------------------------------------------------------- gate
    def gate_features(self, question: str, evidence: list[Scored]) -> np.ndarray:
        """Retrieval-only features for the gate. Never sees gold labels."""
        entail = self.verifier.entailment_strength(question, evidence)
        embs = None
        dense = getattr(self.retriever, "_dense", None)
        if dense is not None and getattr(dense, "emb", None) is not None:
            idx = {p.doc_id: i for i, p in enumerate(self.retriever.corpus)}
            rows = [idx[e.passage.doc_id] for e in evidence if e.passage.doc_id in idx]
            if len(rows) >= 2:
                embs = dense.emb[rows]
        return extract_features(evidence, entail=entail, passage_embeddings=embs)

    # ------------------------------------------------------------- one item
    def run_one(self, q: Question, cfg: SystemConfig) -> ItemResult:
        t_start = time.time()
        r = ItemResult(
            qid=q.qid, subset=q.subset, question=q.text,
            gold_answer=q.gold_answer, config=cfg.name,
            gold_doc_ids=list(q.gold_doc_ids),
        )

        # ---- 1. retrieval ----
        t0 = time.time()
        evidence, timing = self.retriever.retrieve(q.text, cfg.retrieval_mode, k=cfg.k_context)
        r.t_retrieve_s = time.time() - t0
        r.retrieved_ids = [e.passage.doc_id for e in evidence]

        # ---- 2. answerability gate, BEFORE generation ----
        if cfg.use_gate and self.gate is not None:
            t0 = time.time()
            feats = self.gate_features(q.text, evidence)
            ok, prob = self.gate.should_answer(feats)
            r.t_gate_s = time.time() - t0
            r.gate_prob = prob
            if not ok:
                r.answered = False
                r.abstain_reason = "gate"
                r.predicted = ""
                r.correct = 0.0
                r.final_answer_text = "Abstained: retrieved evidence judged insufficient."
                r.t_total_s = time.time() - t_start
                return r

        # ---- 3. generation ----
        t0 = time.time()
        gen = (self.grounded if cfg.grounded else self.vanilla).generate(
            q.text, q.options, evidence
        )
        r.t_generate_s = time.time() - t0
        r.predicted = gen.answer_choice
        r.rationale = gen.rationale
        r.prompt_tokens = gen.prompt_tokens
        r.completion_tokens = gen.completion_tokens
        r.correct = M.choice_accuracy(gen.answer_choice, q.gold_answer, q.options)

        if cfg.grounded and gen.abstained:
            r.answered = False
            r.abstain_reason = "generator_insufficient_evidence"

        # ---- 4. claim decomposition + NLI verification ----
        claims: list[Claim] = []
        if cfg.verify:
            t0 = time.time()
            claims = decompose_claims(gen.rationale)
            self.verifier.verify(claims, evidence)
            r.t_verify_s = time.time() - t0

        # ---- 5. claim-level surgical repair ----
        if cfg.verify and cfg.use_repair and claims:
            t0 = time.time()
            outcome: RepairOutcome = repair_answer(
                question=q.text, claims=claims, evidence=evidence,
                retriever=self.retriever, verifier=self.verifier,
                generator=self.grounded, retrieval_mode=cfg.retrieval_mode,
            )
            r.t_repair_s = time.time() - t0
            claims = outcome.claims
            r.n_repaired = outcome.n_repaired
            r.n_claim_abstained = outcome.n_abstained
            r.repair_rounds = outcome.rounds_used
            r.n_repair_retrievals = outcome.n_repair_retrievals
            r.final_answer_text = outcome.final_answer_text()
            if all(c.label == "abstained" for c in claims):
                r.answered = False
                r.abstain_reason = "all_claims_unsupported"
        else:
            r.final_answer_text = gen.rationale

        # ---- 6. faithfulness bookkeeping ----
        if claims:
            fr = M.faithfulness_report(claims)
            r.faith = {
                "n_claims": fr.n_claims, "n_supported": fr.n_supported,
                "n_contradicted": fr.n_contradicted, "n_neutral": fr.n_neutral,
                "n_abstained": fr.n_abstained,
                "faithfulness": fr.faithfulness,
                "hallucination_rate": fr.hallucination_rate,
                "citation_precision": fr.citation_precision,
            }
            r.claims = [
                {
                    "index": c.index, "text": c.text, "label": c.label,
                    "p_entail": round(c.p_entail, 4),
                    "p_contradict": round(c.p_contradict, 4),
                    "cited": c.cited, "citation_ok": c.citation_ok,
                    "repaired": c.repaired, "repair_round": c.repair_round,
                }
                for c in claims
            ]

        r.t_total_s = time.time() - t_start
        return r

    # ------------------------------------------------------------ many items
    def run_batch(
        self, questions: list[Question], cfg: SystemConfig, progress: bool = True
    ) -> list[ItemResult]:
        out: list[ItemResult] = []
        n = len(questions)
        t0 = time.time()
        for i, q in enumerate(questions, start=1):
            try:
                out.append(self.run_one(q, cfg))
            except Exception as exc:                      # keep the run alive
                log.exception("item failed (%s): %s", q.qid, exc)
                out.append(
                    ItemResult(
                        qid=q.qid, subset=q.subset, question=q.text,
                        gold_answer=q.gold_answer, config=cfg.name,
                        answered=False, abstain_reason=f"error:{type(exc).__name__}",
                    )
                )
            if progress and (i % 10 == 0 or i == n):
                el = time.time() - t0
                log.info("  [%s] %d/%d  %.1fs/item  eta %.1f min",
                         cfg.name, i, n, el / i, (n - i) * el / i / 60)
        return out


# --------------------------------------------------------------------------
# Aggregation into one ablation row
# --------------------------------------------------------------------------
def aggregate(results: list[ItemResult], cfg_name: str, ks=(5, 10, 20)) -> dict:
    """Collapse per-item results into a single row of the results table."""
    if not results:
        return {"config": cfg_name, "n": 0}

    correct = np.array([r.correct for r in results], dtype=float)
    answered = np.array([r.answered for r in results], dtype=bool)

    row: dict = {"config": cfg_name, "n": len(results)}
    row.update(M.selective_metrics(correct, answered))

    # accuracy CI
    if correct.size > 1:
        row["accuracy_ci95"] = float(1.96 * correct.std(ddof=1) / np.sqrt(correct.size))

    # faithfulness (claim-weighted, over items that were verified)
    fr = [
        M.FaithfulnessResult(
            n_claims=r.faith["n_claims"], n_supported=r.faith["n_supported"],
            n_contradicted=r.faith["n_contradicted"], n_neutral=r.faith["n_neutral"],
            n_abstained=r.faith["n_abstained"],
            faithfulness=r.faith["faithfulness"],
            hallucination_rate=r.faith["hallucination_rate"],
            unsupported_rate=r.faith["hallucination_rate"],
            citation_precision=r.faith.get("citation_precision", float("nan")),
        )
        for r in results if r.faith
    ]
    if fr:
        row.update(M.aggregate_faithfulness(fr))

    # retrieval quality, where gold labels exist
    runs = [(r.retrieved_ids, set(r.gold_doc_ids)) for r in results if r.gold_doc_ids]
    if runs:
        rr = M.retrieval_report(runs, ks=ks)
        row.update({f"ret_{k}": v for k, v in rr.items()})

    # abstention / repair / cost
    row["abstain_rate"] = float((~answered).mean())
    row["n_gate_abstain"] = sum(1 for r in results if r.abstain_reason == "gate")
    row["n_repaired_claims"] = sum(r.n_repaired for r in results)
    row["n_abstained_claims"] = sum(r.n_claim_abstained for r in results)
    row["n_repair_retrievals"] = sum(r.n_repair_retrievals for r in results)
    for t in ["t_retrieve_s", "t_gate_s", "t_generate_s", "t_verify_s", "t_repair_s", "t_total_s"]:
        row[f"mean_{t}"] = float(np.mean([getattr(r, t) for r in results]))
    row["total_completion_tokens"] = int(sum(r.completion_tokens for r in results))
    return row
