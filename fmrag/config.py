
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42


def set_all_seeds(seed: int = SEED) -> None:
    """Fix every RNG we touch. Called at the top of every experiment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Environment detection
# --------------------------------------------------------------------------
def detect_env() -> str:
    """Return one of 'colab', 'kaggle', 'local'."""
    if "google.colab" in sys.modules:
        return "colab"
    if os.path.exists("/kaggle/input") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "local"


ENV = detect_env()


def detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


DEVICE = detect_device()


def _default_root() -> Path:
    if ENV == "colab":
        return Path("/content/Faithful_MedRAG")
    if ENV == "kaggle":
        return Path("/kaggle/working/Faithful_MedRAG")
    return Path(__file__).resolve().parent.parent


ROOT = Path(os.environ.get("FMRAG_ROOT", _default_root()))

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"

PHASE1 = ROOT / "Phase_1_Mid_Evaluation"
PHASE2 = ROOT / "Phase_2_End_Evaluation"
SHARED = ROOT / "Shared"


def ensure_dirs() -> None:
    for p in [
        DATA, RAW, PROCESSED, CACHE,
        PHASE1 / "Results", PHASE1 / "Graphs", PHASE1 / "Checkpoints", PHASE1 / "Config",
        PHASE2 / "Results", PHASE2 / "Graphs", PHASE2 / "Checkpoints", PHASE2 / "Config",
        SHARED / "Dataset_Documentation", SHARED / "Model_Documentation",
        SHARED / "References", SHARED / "Reproducibility_Instructions",
    ]:
        p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Model identifiers -- all public HuggingFace checkpoints
# --------------------------------------------------------------------------
# Retrieval (proposal: MedCPT dense + BM25 lexical, fused by RRF)
MEDCPT_QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"

# Reranking (proposal: BGE cross-encoder)
BGE_RERANKER = "BAAI/bge-reranker-base"

# Domain-matched alternative. The BGE reranker is general-domain; MedCPT ships
# its own biomedical cross-encoder, trained on PubMed search logs. Added to test
# whether the observed reranking regression is a domain-mismatch effect.
MEDCPT_RERANKER = "ncbi/MedCPT-Cross-Encoder"

# Verification (proposal: DeBERTa-NLI)
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Generation.
#   'local_small' -- compute-constrained substitution, runs on 8GB M1 / CPU.
#   'proposal_7b' -- the generator named in the proposal; needs a 16GB+ CUDA GPU.
GEN_PROFILES = {
    "local_small": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "quantize_4bit": False,
        "note": "Compute-constrained substitution for 8GB Apple M1 (no CUDA).",
    },
    "proposal_7b": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "quantize_4bit": True,
        "note": "Generator specified in the thesis proposal. Needs CUDA + bitsandbytes.",
    },
    "proposal_llama8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "quantize_4bit": True,
        "note": "Alternate proposal generator. Gated on HuggingFace.",
    },
}

# Auto-select: a CUDA box gets the proposal model, anything else the small one.
GEN_PROFILE = os.environ.get(
    "FMRAG_GEN_PROFILE",
    "proposal_7b" if DEVICE == "cuda" else "local_small",
)

# --------------------------------------------------------------------------
# Corpus configuration
# --------------------------------------------------------------------------
# The proposal targets the 23.9M-doc MedRAG PubMed corpus. That corpus's
# precomputed MedCPT embeddings are ~70GB, which exceeds the disk available on
# the development machine. We therefore build a documented corpus subset:
#   - every PubMedQA context abstract (carries GOLD relevance labels), plus
#   - MedRAG/textbooks chunks as hard in-domain distractors.
# `CORPUS_PROFILE = 'full_pubmed'` restores the proposal corpus on a machine
# with the disk and RAM to hold it.
CORPUS_PROFILES = {
    "subset_textbooks": {
        "medrag_textbooks": True,
        "n_textbook_chunks": 40_000,
        "pubmedqa_contexts": True,
        "note": "Documented subset: PubMedQA gold abstracts + MedRAG textbook distractors.",
    },
    "full_pubmed": {
        "medrag_textbooks": True,
        "n_textbook_chunks": None,     # all 125,847
        "pubmedqa_contexts": True,
        "medrag_pubmed": True,         # 23.9M chunks -- needs ~100GB disk
        "note": "Proposal-scale corpus. Requires >=200GB disk and >=64GB RAM.",
    },
}
CORPUS_PROFILE = os.environ.get("FMRAG_CORPUS_PROFILE", "subset_textbooks")

# --------------------------------------------------------------------------
# Experiment scale -- tiered by computational cost
# --------------------------------------------------------------------------
@dataclass
class ScaleConfig:
    """
    Evaluation sample sizes. Retrieval and gate experiments are cheap and run
    on large samples; generation-heavy experiments run on a smaller seeded
    stratified subset. All sampling is seeded and recorded.
    """
    # Experiment 3: retriever bake-off (cheap -- no LLM in the loop)
    n_retrieval_eval: int = 1000
    # Experiments 1,2,4,6: generation + NLI verification (expensive)
    n_generation_eval: int = 200
    # Experiment 5: answerability-gate training set
    n_gate_total: int = 1000
    # Experiment 7: out-of-distribution MedQA/USMLE
    n_ood_eval: int = 200
    # Experiment 8: human spot-check sheet
    n_human_eval: int = 300


SCALE = ScaleConfig()

# A 'smoke' profile for verifying the pipeline runs before committing hours.
SMOKE = ScaleConfig(
    n_retrieval_eval=40,
    n_generation_eval=12,
    n_gate_total=80,
    n_ood_eval=12,
    n_human_eval=20,
)

if os.environ.get("FMRAG_SMOKE") == "1":
    SCALE = SMOKE

# --------------------------------------------------------------------------
# Retrieval / verification / repair hyperparameters
# --------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    # retrieval
    k_dense: int = 30            # candidates from MedCPT
    k_lexical: int = 30          # candidates from BM25
    rrf_k: int = 60              # RRF smoothing constant (Cormack et al.)
    k_rerank_in: int = 30        # candidates fed to the cross-encoder
    k_context: int = 5           # passages handed to the generator

    # verification
    nli_support_threshold: float = 0.50   # P(entailment) above this = supported
    nli_contradict_threshold: float = 0.50

    # repair loop
    max_repair_rounds: int = 2
    k_repair_retrieve: int = 5

    # generation
    max_new_tokens: int = 256
    temperature: float = 0.0     # greedy -- deterministic for reproducibility

    # answerability gate
    gate_threshold: float = 0.50   # tuned on validation, not test

    # Batching. Measured on Apple M1 MPS with a 40-pair NLI workload:
    # batch 16 -> 4.93s, batch 32 -> 8.09s. Smaller batches win on constrained
    # unified memory because peak activation memory, not kernel launch count,
    # is the binding constraint. On a CUDA GPU raise these substantially.
    embed_batch_size: int = 64
    rerank_batch_size: int = 16
    nli_batch_size: int = 16


PIPE = PipelineConfig()


# --------------------------------------------------------------------------
# Provenance -- recorded into every results file
# --------------------------------------------------------------------------
def provenance() -> dict:
    """Hardware / version / config snapshot for reproducibility."""
    import platform

    info = {
        "seed": SEED,
        "env": ENV,
        "device": DEVICE,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gen_profile": GEN_PROFILE,
        "gen_model": GEN_PROFILES[GEN_PROFILE]["model"],
        "corpus_profile": CORPUS_PROFILE,
        "medcpt_query": MEDCPT_QUERY_ENCODER,
        "medcpt_article": MEDCPT_ARTICLE_ENCODER,
        "reranker": BGE_RERANKER,
        "nli_model": NLI_MODEL,
        "scale": asdict(SCALE),
        "pipeline": asdict(PIPE),
    }
    for mod in ["torch", "transformers", "sklearn", "numpy", "pandas"]:
        try:
            m = __import__(mod)
            info[f"ver_{mod}"] = getattr(m, "__version__", "?")
        except ImportError:
            info[f"ver_{mod}"] = "absent"
    return info


def describe() -> str:
    p = provenance()
    lines = [
        "Faithful-MedRAG configuration",
        f"  environment      : {p['env']}  ({p['platform']})",
        f"  device           : {p['device']}",
        f"  generator        : {p['gen_model']}  [{GEN_PROFILE}]",
        f"                     {GEN_PROFILES[GEN_PROFILE]['note']}",
        f"  corpus profile   : {CORPUS_PROFILE}",
        f"                     {CORPUS_PROFILES[CORPUS_PROFILE]['note']}",
        f"  retriever        : {MEDCPT_QUERY_ENCODER} + BM25 -> RRF",
        f"  reranker         : {BGE_RERANKER}",
        f"  verifier         : {NLI_MODEL}",
        f"  seed             : {SEED}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ensure_dirs()
    print(describe())
