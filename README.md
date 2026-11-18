# Faithful-MedRAG

**Claim-Level Verification and Surgical Repair for Faithful Biomedical Retrieval-Augmented Generation**

Veeral Patidar · Dept. of Mathematics and Computing, IIT (ISM) Dhanbad

## Overview

Medical RAG benchmarks usually score systems by answer accuracy alone. But an
answer can pick the correct option while its explanation asserts things the
retrieved evidence does not support. **Faithful-MedRAG** is a pipeline and
evaluation harness that measures and improves this gap at the level of
individual claims.

## What it does

1. **Retrieve**: MedCPT top-30, reranked with the MedCPT cross-encoder, top-5 passages to the generator.
2. **Generate**: grounded explanation with per-sentence `[n]` citations (Qwen2.5-7B-Instruct).
3. **Decompose and verify**: split the explanation into sentence-level claims and label each as supported / contradicted / neutral with a DeBERTa-v3 NLI model. Citation correctness is tracked separately from entailment.
4. **Repair**: for each unsupported claim, build a targeted query, re-retrieve evidence, rewrite only that sentence, and accept the rewrite only if it is strictly better.
5. **Abstain**: withdraw claims that are still unsupported after two rounds.

## Key results

Evaluated on 200 PubMedQA-derived questions (43,246-passage closed corpus):

| Finding | Result |
|---|---|
| Accuracy of configs D, E, F | Identical (0.645) |
| Micro faithfulness of D / E / F | 0.622 / 0.518 / 0.651 |
| Repair gain over strict-citation config (E→F) | +13.3 pts (95% CI 10.2–16.7) |
| Retrieval: MedCPT alone vs. RRF fusion (Recall@10) | 0.782 vs. 0.742 |
| Reranker diagnosis (Recall@10) | General-domain BGE 0.762→0.693; MedCPT cross-encoder 0.762→0.819 |

**Takeaways**
- Accuracy cannot distinguish configurations whose claim faithfulness differs by over 13 points.
- A general-domain reranker lowered recall; a domain-matched biomedical cross-encoder raised it.
- Uniform BM25+MedCPT fusion lowered recall on this corpus (negative result).

## Costs and caveats (please read)

- 195 of 558 claims (34.9%) are **withdrawn**, so the 0 residual hallucination rate reflects withdrawal, not correction.
- Answer coverage drops to 0.895.
- Mean latency rises from 7.04 s to 16.81 s per question.
- Results come from small samples, a closed corpus containing the gold passages, a single 7B generator, and an **NLI verifier not yet validated against human judgments**.
- Learned answerability gate, OOD evaluation, human evaluation, and the full MIRAGE run are **not executed**.

## Models

| Role | Checkpoint |
|---|---|
| Generator | `Qwen/Qwen2.5-7B-Instruct` |
| Dense retriever | `ncbi/MedCPT-Query-Encoder`, `ncbi/MedCPT-Article-Encoder` |
| Lexical retriever | Okapi BM25 (`rank_bm25`) |
| Reranker (adopted) | `ncbi/MedCPT-Cross-Encoder` |
| Reranker (tested) | `BAAI/bge-reranker-base` |
| NLI verifier | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |

## Reproducibility

- Seed 42, greedy decoding, max 256 new tokens
- Run on Google Colab (NVIDIA T4)
- Hyperparameters and prompt templates are in the paper appendices

## Repository structure

<!-- Update to match your actual layout -->
