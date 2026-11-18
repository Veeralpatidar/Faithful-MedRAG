from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from . import config as C

log = logging.getLogger("fmrag.data")

MIRAGE_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"
MIRAGE_SUBSETS = ["medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"]


# --------------------------------------------------------------------------
# Core dataclasses
# --------------------------------------------------------------------------
@dataclass
class Passage:
    doc_id: str
    title: str
    text: str
    source: str = "unknown"      # 'pubmedqa' | 'textbooks'

    def for_index(self) -> str:
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass
class Question:
    qid: str
    text: str
    gold_answer: str
    subset: str                           # medqa | medmcqa | pubmedqa | bioasq | mmlu
    options: dict | None = None           # multiple-choice options where present
    gold_doc_ids: list[str] = field(default_factory=list)
    answerable: bool = True               # gate supervision, where defined
    long_answer: str | None = None        # PubMedQA free-text reference


# --------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------
def _download(url: str, dest: Path, desc: str = "") -> Path:
    """Download with a skip-if-present check (Part C requirement #5)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("cached: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest
    log.info("downloading %s -> %s", desc or url, dest.name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    log.info("saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


# --------------------------------------------------------------------------
# MIRAGE -- the primary benchmark (7,663 questions)
# --------------------------------------------------------------------------
def load_mirage() -> dict[str, list[Question]]:
    
    path = _download(MIRAGE_URL, C.RAW / "mirage_benchmark.json", "MIRAGE benchmark")
    raw = json.loads(path.read_text())

    out: dict[str, list[Question]] = {}
    for subset in MIRAGE_SUBSETS:
        items = raw.get(subset, {})
        qs = []
        for qid, item in items.items():
            qs.append(
                Question(
                    qid=f"{subset}:{qid}",
                    text=item["question"],
                    gold_answer=str(item["answer"]),
                    subset=subset,
                    options=item.get("options"),
                )
            )
        out[subset] = qs
        log.info("MIRAGE/%-8s %d questions", subset, len(qs))
    total = sum(len(v) for v in out.values())
    log.info("MIRAGE total: %d questions", total)
    return out


# --------------------------------------------------------------------------
# PubMedQA -- grounded, with gold context abstracts
# --------------------------------------------------------------------------
def _split_abstract(pubid: str, contexts: list[str], labels: list[str]) -> list[Passage]:
    """One Passage per labelled section of a PubMed structured abstract."""
    out = []
    for i, (sec, lab) in enumerate(zip(contexts, labels or [])):
        sec = (sec or "").strip()
        if len(sec) < 20:
            continue
        out.append(
            Passage(
                doc_id=f"pmqa:{pubid}:{i}",
                title=str(lab or "").strip().title(),
                text=sec,
                source="pubmedqa",
            )
        )
    return out


def load_pubmedqa_labeled() -> tuple[list[Question], list[Passage]]:
   
    from datasets import load_dataset

    cache = C.PROCESSED / "pubmedqa_labeled.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
        qs = [Question(**q) for q in blob["questions"]]
        ps = [Passage(**p) for p in blob["passages"]]
        log.info("cached PubMedQA: %d questions, %d gold passages", len(qs), len(ps))
        return qs, ps

    log.info("loading PubMedQA (qiaojin/PubMedQA, pqa_labeled) from HuggingFace")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    questions: list[Question] = []
    passages: list[Passage] = []
    for row in ds:
        pubid = str(row["pubid"])
        ctx = row["context"]
        secs = ctx.get("contexts") or []
        labs = ctx.get("labels") or []
        gold = _split_abstract(pubid, secs, labs)
        if not gold:
            continue
        passages.extend(gold)
        questions.append(
            Question(
                qid=f"pubmedqa_src:{pubid}",
                text=row["question"],
                gold_answer=str(row["final_decision"]).strip().lower(),
                subset="pubmedqa_grounded",
                options={"A": "yes", "B": "no", "C": "maybe"},
                gold_doc_ids=[p.doc_id for p in gold],
                answerable=True,
                long_answer=row.get("long_answer"),
            )
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "questions": [asdict(q) for q in questions],
                "passages": [asdict(p) for p in passages],
            }
        )
    )
    log.info("PubMedQA: %d questions, %d gold passages", len(questions), len(passages))
    return questions, passages


# --------------------------------------------------------------------------
# MedRAG textbook corpus -- in-domain distractors
# --------------------------------------------------------------------------
def load_medrag_textbooks(n_chunks: int | None) -> list[Passage]:
    """
    MedRAG/textbooks: 125,847 chunks from 18 medical textbooks -- a genuine
    component of the MedRAG corpus named in the proposal.

    `n_chunks=None` loads all of them; an integer takes a seeded random sample
    (sampling method recorded in the dataset documentation).
    """
    from datasets import load_dataset

    tag = "all" if n_chunks is None else str(n_chunks)
    cache = C.PROCESSED / f"medrag_textbooks_{tag}.json"
    if cache.exists():
        ps = [Passage(**p) for p in json.loads(cache.read_text())]
        log.info("cached MedRAG/textbooks: %d chunks", len(ps))
        return ps

    log.info("loading MedRAG/textbooks from HuggingFace (125,847 chunks)")
    ds = load_dataset("MedRAG/textbooks", split="train")
    total = len(ds)

    idx = list(range(total))
    if n_chunks is not None and n_chunks < total:
        rng = random.Random(C.SEED)
        idx = sorted(rng.sample(idx, n_chunks))
        log.info("seeded sample: %d of %d chunks (seed=%d)", n_chunks, total, C.SEED)

    passages = []
    for i in idx:
        row = ds[i]
        text = (row.get("content") or row.get("contents") or "").strip()
        if len(text) < 50:
            continue
        passages.append(
            Passage(
                doc_id=f"tb:{row.get('id', i)}",
                title=(row.get("title") or "").strip(),
                text=text,
                source="textbooks",
            )
        )

    cache.write_text(json.dumps([asdict(p) for p in passages]))
    log.info("MedRAG/textbooks: %d usable chunks", len(passages))
    return passages


# --------------------------------------------------------------------------
# Corpus assembly
# --------------------------------------------------------------------------
def build_corpus() -> tuple[list[Passage], list[Question]]:
    
    prof = C.CORPUS_PROFILES[C.CORPUS_PROFILE]
    corpus: list[Passage] = []
    grounded: list[Question] = []

    if prof.get("pubmedqa_contexts"):
        gq, gp = load_pubmedqa_labeled()
        grounded = gq
        corpus.extend(gp)

    if prof.get("medrag_textbooks"):
        corpus.extend(load_medrag_textbooks(prof.get("n_textbook_chunks")))

    if prof.get("medrag_pubmed"):
        raise NotImplementedError(
            "CORPUS_PROFILE='full_pubmed' requires the 23.9M-chunk MedRAG PubMed "
            "corpus (~100GB on disk, ~70GB of precomputed MedCPT embeddings). "
            "Run on hardware with >=200GB free disk; see "
            "Shared/Reproducibility_Instructions/SCALING.md"
        )

    # de-duplicate on normalised text (proposal: quality control step)
    seen: set[str] = set()
    deduped = []
    for p in corpus:
        h = hashlib.md5(re.sub(r"\s+", " ", p.text.lower()).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        deduped.append(p)

    log.info(
        "corpus assembled: %d passages (%d after dedup) | %d grounded questions",
        len(corpus), len(deduped), len(grounded),
    )
    return deduped, grounded


# --------------------------------------------------------------------------
# Seeded stratified sampling + splits
# --------------------------------------------------------------------------
def stratified_sample(
    by_subset: dict[str, list[Question]], n_total: int, seed: int = C.SEED
) -> list[Question]:
    """
    Seeded stratified sample across MIRAGE subsets, proportional to each
    subset's true size. Reproducible and recorded in the results provenance.
    """
    pool = {k: v for k, v in by_subset.items() if v}
    grand = sum(len(v) for v in pool.values())
    if n_total >= grand:
        out = [q for v in pool.values() for q in v]
        return sorted(out, key=lambda q: q.qid)

    rng = random.Random(seed)
    picked: list[Question] = []
    for subset, qs in sorted(pool.items()):
        share = max(1, round(n_total * len(qs) / grand))
        share = min(share, len(qs))
        picked.extend(rng.sample(qs, share))

    # trim / pad deterministically to land exactly on n_total
    picked = sorted(picked, key=lambda q: q.qid)
    if len(picked) > n_total:
        rng.shuffle(picked)
        picked = sorted(picked[:n_total], key=lambda q: q.qid)
    return picked


def split_train_val_test(
    items: list, ratios=(0.6, 0.2, 0.2), seed: int = C.SEED
) -> tuple[list, list, list]:
    """
    Reproducible train/val/test split. Thresholds are tuned on val only --
    test is touched once, at the end (Part E requirements #1-#3).
    """
    xs = sorted(items, key=lambda x: getattr(x, "qid", str(x)))
    rng = random.Random(seed)
    rng.shuffle(xs)
    n = len(xs)
    a = int(n * ratios[0])
    b = a + int(n * ratios[1])
    return xs[:a], xs[a:b], xs[b:]


# --------------------------------------------------------------------------
# CLI: acquire everything and report real counts
# --------------------------------------------------------------------------
def acquire_all() -> dict:
    C.ensure_dirs()
    mirage = load_mirage()
    corpus, grounded = build_corpus()
    report = {
        "mirage_counts": {k: len(v) for k, v in mirage.items()},
        "mirage_total": sum(len(v) for v in mirage.values()),
        "corpus_passages": len(corpus),
        "corpus_by_source": {},
        "grounded_questions": len(grounded),
        "corpus_profile": C.CORPUS_PROFILE,
    }
    for p in corpus:
        report["corpus_by_source"][p.source] = report["corpus_by_source"].get(p.source, 0) + 1
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rep = acquire_all()
    print(json.dumps(rep, indent=2))
