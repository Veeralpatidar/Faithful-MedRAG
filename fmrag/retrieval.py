 
from __future__ import annotations

import json
import logging
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C
from .data import Passage

log = logging.getLogger("fmrag.retrieval")


@dataclass
class Scored:
    """A retrieved passage with its score and rank."""
    passage: Passage
    score: float
    rank: int = -1
    retriever: str = ""


# --------------------------------------------------------------------------
# Lexical : BM25
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over the corpus. Index cached with pickle."""

    name = "bm25"

    def __init__(self, corpus: list[Passage], cache_tag: str = "default"):
        from rank_bm25 import BM25Okapi

        self.corpus = corpus
        cache = C.CACHE / f"bm25_{cache_tag}_{len(corpus)}.pkl"
        if cache.exists():
            log.info("loading cached BM25 index (%s)", cache.name)
            with cache.open("rb") as f:
                self.bm25 = pickle.load(f)
        else:
            log.info("building BM25 index over %d passages", len(corpus))
            t0 = time.time()
            tokens = [_tokenize(p.for_index()) for p in corpus]
            self.bm25 = BM25Okapi(tokens)
            log.info("BM25 index built in %.1fs", time.time() - t0)
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as f:
                pickle.dump(self.bm25, f)

    def retrieve(self, query: str, k: int) -> list[Scored]:
        scores = self.bm25.get_scores(_tokenize(query))
        top = np.argsort(-scores)[:k]
        return [
            Scored(self.corpus[i], float(scores[i]), rank=r + 1, retriever=self.name)
            for r, i in enumerate(top)
        ]


# --------------------------------------------------------------------------
# Dense : MedCPT
# --------------------------------------------------------------------------
class MedCPTRetriever:
    """
    MedCPT bi-encoder. The article encoder consumes [title, text] pairs (the
    format MedCPT was trained on); both encoders CLS-pool.
    """

    name = "medcpt"

    def __init__(self, corpus: list[Passage], cache_tag: str = "default"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.corpus = corpus
        self.device = C.DEVICE
        self._torch = torch

        log.info("loading MedCPT query encoder (%s)", C.MEDCPT_QUERY_ENCODER)
        self.q_tok = AutoTokenizer.from_pretrained(C.MEDCPT_QUERY_ENCODER)
        self.q_model = AutoModel.from_pretrained(C.MEDCPT_QUERY_ENCODER).to(self.device).eval()

        self.emb_path = C.CACHE / f"medcpt_emb_{cache_tag}_{len(corpus)}.npy"
        if self.emb_path.exists():
            self.emb = np.load(self.emb_path)
            log.info("loaded cached MedCPT embeddings %s", self.emb.shape)
        else:
            self.emb = self._embed_corpus()
            np.save(self.emb_path, self.emb)
            log.info("saved MedCPT embeddings %s -> %s", self.emb.shape, self.emb_path.name)

    def _embed_corpus(self) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        log.info("loading MedCPT article encoder (%s)", C.MEDCPT_ARTICLE_ENCODER)
        a_tok = AutoTokenizer.from_pretrained(C.MEDCPT_ARTICLE_ENCODER)
        a_model = AutoModel.from_pretrained(C.MEDCPT_ARTICLE_ENCODER).to(self.device).eval()

        bs = C.PIPE.embed_batch_size
        vecs = []
        n = len(self.corpus)
        log.info("embedding %d passages with MedCPT (batch=%d, device=%s)", n, bs, self.device)
        t0 = time.time()
        with torch.no_grad():
            for s in range(0, n, bs):
                chunk = self.corpus[s : s + bs]
                pairs = [[p.title or "", p.text] for p in chunk]
                enc = a_tok(
                    pairs, truncation=True, padding=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                out = a_model(**enc).last_hidden_state[:, 0, :]
                vecs.append(out.cpu().numpy().astype(np.float32))
                if (s // bs) % 50 == 0 and s:
                    el = time.time() - t0
                    log.info(
                        "  %d/%d passages (%.0f/s, eta %.1f min)",
                        s, n, s / el, (n - s) / max(s / el, 1e-9) / 60,
                    )
        del a_model
        log.info("embedded %d passages in %.1f min", n, (time.time() - t0) / 60)
        return np.vstack(vecs)

    def embed_query(self, query: str) -> np.ndarray:
        with self._torch.no_grad():
            enc = self.q_tok(
                [query], truncation=True, padding=True, max_length=64, return_tensors="pt"
            ).to(self.device)
            v = self.q_model(**enc).last_hidden_state[:, 0, :]
        return v.cpu().numpy().astype(np.float32)[0]

    def retrieve(self, query: str, k: int) -> list[Scored]:
        qv = self.embed_query(query)
        scores = self.emb @ qv
        top = np.argsort(-scores)[:k]
        return [
            Scored(self.corpus[i], float(scores[i]), rank=r + 1, retriever=self.name)
            for r, i in enumerate(top)
        ]


# --------------------------------------------------------------------------
# Fusion : Reciprocal Rank Fusion
# --------------------------------------------------------------------------
def rrf_fuse(runs: list[list[Scored]], k: int, rrf_k: int = C.PIPE.rrf_k) -> list[Scored]:
    """
    RRF: score(d) = sum_runs 1 / (rrf_k + rank_run(d)).

    Rank-based, so it needs no score normalisation between the BM25 and
    MedCPT scales -- which is exactly why the proposal selects it.
    """
    agg: dict[str, float] = {}
    keep: dict[str, Passage] = {}
    for run in runs:
        for s in run:
            did = s.passage.doc_id
            agg[did] = agg.get(did, 0.0) + 1.0 / (rrf_k + s.rank)
            keep[did] = s.passage
    order = sorted(agg.items(), key=lambda kv: -kv[1])[:k]
    return [
        Scored(keep[d], float(sc), rank=r + 1, retriever="rrf")
        for r, (d, sc) in enumerate(order)
    ]


# --------------------------------------------------------------------------
# Reranking : BGE cross-encoder
# --------------------------------------------------------------------------
class BGEReranker:
    """
    Cross-encoder reranker. Scores (query, passage) jointly.

    Defaults to BAAI/bge-reranker-base (the proposal's choice). Passing
    `model_name=C.MEDCPT_RERANKER` selects MedCPT's biomedical cross-encoder
    instead, which is the domain-matched comparator.
    """

    def __init__(self, model_name: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = C.DEVICE
        self.model_name = model_name or C.BGE_RERANKER
        self.name = "medcpt_ce" if "MedCPT" in self.model_name else "bge"
        log.info("loading cross-encoder reranker (%s)", self.model_name)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )

    def rerank(self, query: str, cands: list[Scored], k: int) -> list[Scored]:
        if not cands:
            return []
        torch = self._torch
        pairs = [[query, c.passage.for_index()[:2000]] for c in cands]
        scores: list[float] = []
        bs = C.PIPE.rerank_batch_size
        with torch.no_grad():
            for s in range(0, len(pairs), bs):
                enc = self.tok(
                    pairs[s : s + bs],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.view(-1)
                scores.extend(logits.float().cpu().numpy().tolist())
        order = np.argsort(-np.array(scores))[:k]
        return [
            Scored(cands[i].passage, float(scores[i]), rank=r + 1, retriever=self.name)
            for r, i in enumerate(order)
        ]


# --------------------------------------------------------------------------
# Configurable retrieval front-end (the four bake-off arms)
# --------------------------------------------------------------------------
class HybridRetriever:
    """
    The retrieval front-end used by every downstream experiment.

    mode: 'bm25' | 'medcpt' | 'rrf' | 'rrf_bge'
    These are exactly the four arms of the Experiment-3 bake-off, and the
    component toggles the ablation (Experiment 6) switches on and off.
    """

    def __init__(
        self,
        corpus: list[Passage],
        cache_tag: str = "default",
        use_reranker: bool = True,
        lazy: bool = True,
    ):
        self.corpus = corpus
        self.cache_tag = cache_tag
        self._bm25 = None
        self._dense = None
        self._reranker = None
        self._med_reranker = None
        self._want_reranker = use_reranker
        if not lazy:
            _ = self.bm25, self.dense
            if use_reranker:
                _ = self.reranker

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.corpus, self.cache_tag)
        return self._bm25

    @property
    def dense(self) -> MedCPTRetriever:
        if self._dense is None:
            self._dense = MedCPTRetriever(self.corpus, self.cache_tag)
        return self._dense

    @property
    def reranker(self) -> BGEReranker:
        if self._reranker is None:
            self._reranker = BGEReranker()
        return self._reranker

    @property
    def med_reranker(self) -> BGEReranker:
        """MedCPT's biomedical cross-encoder -- the domain-matched comparator."""
        if self._med_reranker is None:
            self._med_reranker = BGEReranker(C.MEDCPT_RERANKER)
        return self._med_reranker

    def retrieve(self, query: str, mode: str, k: int) -> tuple[list[Scored], dict]:
        """Returns (results, timing_dict). Timings feed the latency tables."""
        t0 = time.time()
        timing: dict[str, float] = {}

        if mode == "bm25":
            res = self.bm25.retrieve(query, k)
            timing["lexical_s"] = time.time() - t0

        elif mode == "medcpt":
            res = self.dense.retrieve(query, k)
            timing["dense_s"] = time.time() - t0

        elif mode in ("medcpt_bge", "medcpt_ce"):
            # Dense retrieval reranked directly, with no RRF stage. Isolates the
            # reranker's effect from the fusion's effect.
            t = time.time()
            cands = self.dense.retrieve(query, C.PIPE.k_rerank_in)
            timing["dense_s"] = time.time() - t
            rr = self.reranker if mode == "medcpt_bge" else self.med_reranker
            t = time.time()
            res = rr.rerank(query, cands, k)
            timing["rerank_s"] = time.time() - t

        elif mode in ("rrf", "rrf_bge", "rrf_medcpt_ce"):
            t = time.time()
            lex = self.bm25.retrieve(query, C.PIPE.k_lexical)
            timing["lexical_s"] = time.time() - t
            t = time.time()
            den = self.dense.retrieve(query, C.PIPE.k_dense)
            timing["dense_s"] = time.time() - t
            t = time.time()
            fused = rrf_fuse([lex, den], k=C.PIPE.k_rerank_in)
            timing["fuse_s"] = time.time() - t
            if mode in ("rrf_bge", "rrf_medcpt_ce"):
                rr = self.reranker if mode == "rrf_bge" else self.med_reranker
                t = time.time()
                res = rr.rerank(query, fused, k)
                timing["rerank_s"] = time.time() - t
            else:
                res = fused[:k]
        else:
            raise ValueError(f"unknown retrieval mode: {mode!r}")

        timing["total_s"] = time.time() - t0
        return res, timing


RETRIEVAL_MODES = ["bm25", "medcpt", "rrf", "rrf_bge"]

# Diagnostic arms added after the main bake-off showed both RRF fusion and BGE
# reranking degrading recall relative to MedCPT alone. These separate the two
# candidate explanations: fusion dilution vs. reranker domain mismatch.
DIAGNOSTIC_MODES = ["medcpt_bge", "medcpt_ce", "rrf_medcpt_ce"]
