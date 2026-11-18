 
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from . import config as C
from .retrieval import Scored

log = logging.getLogger("fmrag.generate")


@dataclass
class Generation:
    raw: str
    answer_choice: str          # parsed 'Answer:' line (letter or text)
    rationale: str              # the cited explanation -- what gets verified
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    abstained: bool = False


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
VANILLA_SYSTEM = (
    "You are a medical expert answering a multiple-choice medical question. "
    "Use the provided documents if they are helpful. "
    "Give a brief explanation, then your final choice."
)

VANILLA_USER = """Documents:
{context}

Question: {question}
Options:
{options}

Explain briefly, then end with a line of exactly the form:
Answer: <letter>"""

# The strict-citation prompt. Three rules carry the grounded condition:
# per-sentence citation, no uncited assertions, and an explicit abstain option.
GROUNDED_SYSTEM = (
    "You are a careful medical expert. You answer ONLY from the numbered "
    "documents you are given.\n"
    "Rules you must follow exactly:\n"
    "1. Every sentence of your explanation must end with a citation marker "
    "[n] naming the document that supports that exact sentence.\n"
    "2. Never state anything the documents do not support. Do not add outside "
    "medical knowledge.\n"
    "3. If the documents do not contain enough information, write "
    "'Insufficient evidence' as your explanation.\n"
    "4. Keep each sentence to a single factual claim."
)

GROUNDED_USER = """Numbered documents:
{context}

Question: {question}
Options:
{options}

Write 2-4 sentences, each supported by and ending with a [n] citation.
Then end with a line of exactly the form:
Answer: <letter>"""

REPAIR_SYSTEM = (
    "You are a careful medical expert revising ONE unsupported sentence of an "
    "explanation.\n"
    "Rewrite the sentence so it states only what the numbered documents "
    "support, and end it with the correct [n] citation.\n"
    "If no document supports any version of the claim, reply with exactly: "
    "Insufficient evidence"
)

REPAIR_USER = """Numbered documents:
{context}

Question: {question}

This sentence was NOT supported by the evidence:
"{claim}"

Rewrite it as a single sentence grounded in the documents above, ending with
its [n] citation. Reply with the sentence only."""


def format_context(evidence: list[Scored]) -> str:
    """Number passages from 1 so the [n] citation markers line up."""
    out = []
    for i, e in enumerate(evidence, start=1):
        t = e.passage.title or e.passage.source
        out.append(f"[{i}] ({t}) {e.passage.text[:900]}")
    return "\n\n".join(out) if out else "(no documents retrieved)"


def format_options(options: dict | None) -> str:
    if not options:
        return "(open-ended question -- answer in a few words)"
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


_ANSWER_LINE = re.compile(r"answer\s*:\s*(.+)", re.I)


def parse_answer(raw: str, options: dict | None) -> tuple[str, str]:
    """
    Split a completion into (answer_choice, rationale).

    The rationale is everything before the 'Answer:' line -- that is the text
    whose claims get verified.
    """
    text = (raw or "").strip()
    m = None
    for m_ in _ANSWER_LINE.finditer(text):
        m = m_                       # take the last 'Answer:' line
    if m:
        choice = m.group(1).strip().split("\n")[0].strip()
        rationale = text[: m.start()].strip()
    else:
        choice, rationale = "", text
        # fall back to a leading bare letter, e.g. "B. beta cells"
        m2 = re.match(r"^\(?([A-Ea-e])\)?[\.\): ]", text)
        if m2:
            choice = m2.group(1).upper()
    choice = re.sub(r"^\**|\**$", "", choice).strip()
    m3 = re.match(r"^\(?([A-Ea-e])\)?\b", choice)
    if m3 and options and m3.group(1).upper() in options:
        choice = m3.group(1).upper()
    return choice, rationale


# --------------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------------
class LLM:
    """
    One loaded causal LM, shared by both generators and the repair loop.

    Greedy decoding (temperature 0) so every experiment is reproducible.
    """

    def __init__(self, profile: str = C.GEN_PROFILE):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        spec = C.GEN_PROFILES[profile]
        self.model_name = spec["model"]
        self.profile = profile
        log.info("loading generator %s [%s] on %s", self.model_name, profile, C.DEVICE)

        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        kwargs: dict = {"dtype": torch.float16 if C.DEVICE != "cpu" else torch.float32}

        if spec.get("quantize_4bit") and C.DEVICE == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
                log.info("4-bit NF4 quantization enabled")
            except ImportError:
                log.warning("bitsandbytes unavailable -- loading in fp16 instead")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if "device_map" not in kwargs:
            self.model = self.model.to(C.DEVICE)
        self.model.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def chat(self, system: str, user: str, max_new_tokens: int | None = None) -> tuple[str, int, int]:
        torch = self._torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(
            self.model.device
        )
        n_in = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens or C.PIPE.max_new_tokens,
                do_sample=False,                 # greedy -> deterministic
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.pad_token_id,
            )
        gen_ids = out[0][n_in:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return text, n_in, int(gen_ids.shape[0])


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------
class VanillaGenerator:
    """MedRAG-style baseline: evidence in context, no citation discipline."""

    name = "vanilla"

    def __init__(self, llm: LLM):
        self.llm = llm

    def generate(self, question: str, options: dict | None, evidence: list[Scored]) -> Generation:
        t0 = time.time()
        user = VANILLA_USER.format(
            context=format_context(evidence),
            question=question,
            options=format_options(options),
        )
        raw, n_in, n_out = self.llm.chat(VANILLA_SYSTEM, user)
        choice, rationale = parse_answer(raw, options)
        return Generation(
            raw=raw, answer_choice=choice, rationale=rationale,
            latency_s=time.time() - t0, prompt_tokens=n_in, completion_tokens=n_out,
        )


class GroundedGenerator:
    """Strict-citation generation with an explicit abstain option."""

    name = "grounded"

    def __init__(self, llm: LLM):
        self.llm = llm

    def generate(self, question: str, options: dict | None, evidence: list[Scored]) -> Generation:
        t0 = time.time()
        user = GROUNDED_USER.format(
            context=format_context(evidence),
            question=question,
            options=format_options(options),
        )
        raw, n_in, n_out = self.llm.chat(GROUNDED_SYSTEM, user)
        choice, rationale = parse_answer(raw, options)
        from .verify import is_abstention

        return Generation(
            raw=raw, answer_choice=choice, rationale=rationale,
            latency_s=time.time() - t0, prompt_tokens=n_in, completion_tokens=n_out,
            abstained=is_abstention(rationale),
        )

    def repair_claim(self, question: str, claim_text: str, evidence: list[Scored]) -> tuple[str, float]:
        """Rewrite one unsupported claim against freshly retrieved evidence."""
        t0 = time.time()
        user = REPAIR_USER.format(
            context=format_context(evidence), question=question, claim=claim_text
        )
        raw, _, _ = self.llm.chat(REPAIR_SYSTEM, user, max_new_tokens=96)
        line = (raw or "").strip().split("\n")[0].strip()
        return line, time.time() - t0
