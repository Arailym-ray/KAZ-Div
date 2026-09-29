# %% [markdown]
# # KAZ-Div — Experiment 2: Diverse Question Generation
# 
# Reproducible Colab notebook for **grounded diverse question generation in Kazakh**.
# 
# ## Final model set
# 
# | Role | Model |
# |---|---|
# | Proprietary general-purpose | `gpt-5.6-sol` |
# | Proprietary general-purpose | `claude-sonnet-4-6` |
# | Kazakh-specialised, Llama family | `issai/LLama-3.1-KazLLM-1.0-8B` |
# | Kazakh-specialised, different architecture | `astanahub/alemllm` |
# 
# ## Final experimental design
# 
# - **100 source contexts**: recommended 20 each from `general`, `education`, `science`, `technology`, `news`
# - **4 models**
# - **2 methods**: Baseline and KAZ-Div
# - **5 repetitions** in the main experiment
# - **4000 final observations** = 100 × 4 × 2 × 5
# - KAZ-Div generates **K = 12 candidates** per request
# - answers must be **extractive spans from the source context**
# - BGE-M3 embeddings use **CLS pooling + FP32 L2 normalization**
# - checkpoint/resume is enabled
# - pilot and main outputs are physically separated
# 
# ## Important runtime note
# 
# `issai/LLama-3.1-KazLLM-1.0-8B` is loaded locally from Hugging Face and requires an approved/gated Hugging Face token (`HF_TOKEN`).
# 
# `astanahub/alemllm` is a very large MoE model and is **not loaded into a normal single-GPU Colab runtime**. This notebook evaluates exactly `astanahub/alemllm` through an **OpenAI-compatible inference endpoint**. Add `ALEMLLM_BASE_URL` to Colab Secrets. If the endpoint requires authentication, also add `ALEMLLM_API_KEY`.
# 
# ## Workflow
# 
# 1. Add Colab Secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `ALEMLLM_BASE_URL`, and optionally `ALEMLLM_API_KEY`.
# 2. Make sure your Hugging Face account has access to the gated KazLLM repository.
# 3. Start with `RUN_MODE = "pilot"` and `THRESHOLD_FROZEN = False`.
# 4. Run Cells 0–14. Pilot design = **2 contexts/domain × 5 domains × 4 models = 40 KAZ-Div observations**.
# 5. Run Cell 15 and inspect threshold calibration.
# 6. Freeze the selected threshold in Cell 3, set `THRESHOLD_FROZEN = True`, then set `RUN_MODE = "full"`.
# 7. Rerun Cell 3 onward. Main design = **4000 observations**.
# 8. Run Cells 16–17 for metrics/statistics and Cell 18 to export/download results.

# %%
# Cell 0 is a Colab dependency-install cell.
# In a Python environment, install these once before running:
# python -m pip install -U openai anthropic transformers accelerate huggingface_hub nltk sentencepiece safetensors
print("Dependency installation is handled by Cell 0 in the Colab notebook.")

# %%
# ============================================================
# Cell 1. Imports and reproducibility
# ============================================================

import os
import re
import csv
import ast
import gc
import json
import math
import time
import random
import hashlib
import shutil
import traceback
import unicodedata

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import transformers

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from openai import OpenAI
import anthropic

from google.colab import files, userdata
from IPython.display import display


RANDOM_SEED = 20260922

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("Python reproducibility seed:", RANDOM_SEED)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU total memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )

# %%
# ============================================================
# Cell 2. Load Experiment 2 contexts
# ============================================================
#
# CSV columns:
#   id
#   text
#   domain
#
# Recommended full design:
#   100 contexts
#   20 each:
#       general
#       education
#       science
#       technology
#       news
# ============================================================

EXPECTED_DOMAINS = [
    "general",
    "education",
    "science",
    "technology",
    "news",
]


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u00A0", " ")

    for char in ["\u200B", "\u200C", "\u200D", "\u2060", "\uFEFF"]:
        text = text.replace(char, "")

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_question_source_csv() -> pd.DataFrame:
    print("Upload CSV with columns: id, text, domain")
    uploaded = files.upload()

    if not uploaded:
        raise RuntimeError("No dataset uploaded.")

    filename = next(iter(uploaded.keys()))

    if not filename.lower().endswith(".csv"):
        raise ValueError("Upload a CSV file.")

    df = pd.read_csv(filename)

    required = {"id", "text", "domain"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[["id", "text", "domain"]].copy()

    df["text"] = df["text"].map(normalize_text)
    df["domain"] = df["domain"].astype(str).str.strip().str.lower()

    if df["id"].duplicated().any():
        raise ValueError("Source IDs must be unique.")

    if (df["text"].str.len() == 0).any():
        raise ValueError("Empty context detected.")

    return df


if "sources" in globals():
    question_sources = sources[["id", "text", "domain"]].copy()
    question_sources["text"] = question_sources["text"].map(normalize_text)
    question_sources["domain"] = (
        question_sources["domain"].astype(str).str.strip().str.lower()
    )
    print("Reused existing `sources` DataFrame.")
else:
    question_sources = load_question_source_csv()


question_sources = question_sources.sort_values("id").reset_index(drop=True)

question_sources["word_count"] = (
    question_sources["text"].str.split().map(len)
)

print("Rows:", len(question_sources))
print(question_sources["domain"].value_counts().sort_index())

display(question_sources.head())


if len(question_sources) == 100:
    observed = question_sources["domain"].value_counts().to_dict()

    if set(observed) == set(EXPECTED_DOMAINS):
        for domain in EXPECTED_DOMAINS:
            assert observed[domain] == 20

        print("Balanced 100-context design verified.")
    else:
        print("WARNING: domain labels differ from recommended design.")
else:
    print("Pilot/non-standard dataset size detected.")

# %%
# ============================================================
# Cell 3. Experiment 2 configuration
# ============================================================

EXPERIMENT_NAME = "Experiment_2_Diverse_Question_Generation"

# ------------------------------------------------------------
# RUN MODE
# ------------------------------------------------------------
# First stage:
#     RUN_MODE = "pilot"
#     THRESHOLD_FROZEN = False
#
# Main collection, ONLY after Cell 15 calibration:
#     RUN_MODE = "full"
#     THRESHOLD_FROZEN = True
# ------------------------------------------------------------

RUN_MODE = "pilot"          # "pilot" or "full"
THRESHOLD_FROZEN = False     # must be True for full collection

if RUN_MODE not in {"pilot", "full"}:
    raise ValueError("RUN_MODE must be 'pilot' or 'full'.")


# ------------------------------------------------------------
# Output folders — pilot and main are physically separated.
# ------------------------------------------------------------

OUTPUT_ROOT = Path(
    "/content/KAZ_Div/Experiment_2_Question_Generation_4Models"
)

PILOT_OUTPUT_DIR = OUTPUT_ROOT / "pilot"
MAIN_OUTPUT_DIR = OUTPUT_ROOT / "main"

OUTPUT_DIR = (
    PILOT_OUTPUT_DIR
    if RUN_MODE == "pilot"
    else MAIN_OUTPUT_DIR
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Final model set
# ------------------------------------------------------------

MODEL_IDS = {
    "openai": "gpt-5.6-sol",
    "anthropic": "claude-sonnet-4-6",
    "kazllm": "issai/LLama-3.1-KazLLM-1.0-8B",
    "alemllm": "astanahub/alemllm",
}

MODEL_ROLES = {
    "openai": "proprietary_general_purpose",
    "anthropic": "proprietary_general_purpose",
    "kazllm": "kazakh_specialised_llama_family",
    "alemllm": "kazakh_specialised_moe",
}

ENABLED_PROVIDERS = [
    "openai",
    "anthropic",
    "kazllm",
    "alemllm",
]

METHODS = [
    "baseline",
    "kazdiv",
]


# ------------------------------------------------------------
# Main experimental design
# 100 × 4 × 2 × 5 = 4000 final observations
# ------------------------------------------------------------

N_REPEATS = 5
K_CANDIDATES = 12
MAX_KAZDIV_CALLS = 2
MAX_OUTPUT_TOKENS = 1600


# ------------------------------------------------------------
# Balanced pilot
# 2 sources/domain × 5 domains × 4 models × KAZ-Div × 1 rep
# = 40 pilot observations
# ------------------------------------------------------------

PILOT_SOURCES_PER_DOMAIN = 2
PILOT_REPETITION = 1


# ------------------------------------------------------------
# Quality constraints
# ------------------------------------------------------------

MIN_QUESTION_WORDS = 3
MAX_QUESTION_WORDS = 35

MIN_ANSWER_WORDS = 1
MAX_ANSWER_WORDS = 20


# ------------------------------------------------------------
# Semantic near-duplicate threshold
# ------------------------------------------------------------
# Pilot starting value only. Cell 15 evaluates alternatives.
# Freeze this value BEFORE main data collection.
# ------------------------------------------------------------

QUESTION_NEAR_DUPLICATE_THRESHOLD = 0.95
QUESTION_RELEVANCE_MIN = None


# ------------------------------------------------------------
# BGE-M3 semantic representation
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_LENGTH = 512


# ------------------------------------------------------------
# Local KazLLM generation
# ------------------------------------------------------------

KAZLLM_DO_SAMPLE = True
KAZLLM_TEMPERATURE = 0.6
KAZLLM_TOP_P = 0.9


# ------------------------------------------------------------
# AlemLLM endpoint generation
# ------------------------------------------------------------
# The OpenAI-compatible endpoint should serve this exact model.
# If your server exposes a different served-model alias, edit
# ALEMLLM_SERVED_MODEL but keep MODEL_IDS["alemllm"] canonical.
# ------------------------------------------------------------

ALEMLLM_SERVED_MODEL = "astanahub/alemllm"
ALEMLLM_TEMPERATURE = 0.6
ALEMLLM_TOP_P = 0.9


# ------------------------------------------------------------
# Block main collection until pilot threshold is frozen.
# ------------------------------------------------------------

if RUN_MODE == "full" and not THRESHOLD_FROZEN:
    raise RuntimeError(
        "Full experiment is blocked because THRESHOLD_FROZEN=False. "
        "Run pilot + Cell 15 first, freeze the selected "
        "QUESTION_NEAR_DUPLICATE_THRESHOLD, and then set "
        "THRESHOLD_FROZEN=True."
    )


CONFIG = {
    "experiment": EXPERIMENT_NAME,
    "run_mode": RUN_MODE,
    "random_seed": RANDOM_SEED,
    "model_ids": MODEL_IDS,
    "model_roles": MODEL_ROLES,
    "providers": ENABLED_PROVIDERS,
    "methods": METHODS,
    "n_repeats": N_REPEATS,
    "k_candidates": K_CANDIDATES,
    "max_kazdiv_calls": MAX_KAZDIV_CALLS,
    "pilot_sources_per_domain": PILOT_SOURCES_PER_DOMAIN,
    "question_word_range": [
        MIN_QUESTION_WORDS,
        MAX_QUESTION_WORDS,
    ],
    "answer_word_range": [
        MIN_ANSWER_WORDS,
        MAX_ANSWER_WORDS,
    ],
    "question_near_duplicate_threshold": (
        QUESTION_NEAR_DUPLICATE_THRESHOLD
    ),
    "threshold_frozen": THRESHOLD_FROZEN,
    "question_relevance_min": QUESTION_RELEVANCE_MIN,
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_pooling": "CLS",
    "embedding_normalization_dtype": "float32",
    "kazllm_generation": {
        "do_sample": KAZLLM_DO_SAMPLE,
        "temperature": KAZLLM_TEMPERATURE,
        "top_p": KAZLLM_TOP_P,
    },
    "alemllm_generation": {
        "served_model": ALEMLLM_SERVED_MODEL,
        "temperature": ALEMLLM_TEMPERATURE,
        "top_p": ALEMLLM_TOP_P,
        "backend": "openai_compatible_endpoint",
    },
    "stateless": True,
}

with open(
    OUTPUT_DIR / "experiment2_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        CONFIG,
        f,
        ensure_ascii=False,
        indent=2,
    )

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

question_sources.to_csv(
    OUTPUT_ROOT / "experiment2_sources.csv",
    index=False,
    encoding="utf-8-sig",
)

question_sources.to_csv(
    OUTPUT_DIR / "experiment2_sources.csv",
    index=False,
    encoding="utf-8-sig",
)

print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
print("OUTPUT_DIR:", OUTPUT_DIR)

# %%
# ============================================================
# Cell 4. API clients + local KazLLM + AlemLLM endpoint + BGE-M3
# ============================================================
# Required Colab Secrets:
#   OPENAI_API_KEY
#   ANTHROPIC_API_KEY
#   HF_TOKEN
#   ALEMLLM_BASE_URL
# Optional:
#   ALEMLLM_API_KEY
#
# IMPORTANT:
# - `issai/LLama-3.1-KazLLM-1.0-8B` is gated on Hugging Face.
#   Your HF account must have access before this cell can load it.
# - `astanahub/alemllm` is evaluated through a remote
#   OpenAI-compatible endpoint because it is too large for a
#   normal single-GPU Colab runtime.
# ============================================================


def get_secret(name: str) -> str:
    value = ""

    try:
        value = userdata.get(name) or ""
    except Exception:
        value = ""

    if not value:
        value = os.environ.get(name, "")

    return str(value).strip()


def normalize_openai_base_url(url: str) -> str:
    url = str(url).strip().rstrip("/")
    if not url:
        return ""
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
HF_TOKEN = get_secret("HF_TOKEN")
ALEMLLM_BASE_URL = normalize_openai_base_url(
    get_secret("ALEMLLM_BASE_URL")
)
ALEMLLM_API_KEY = get_secret("ALEMLLM_API_KEY")


openai_client = (
    OpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
)

anthropic_client = (
    anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    if ANTHROPIC_API_KEY
    else None
)

alemllm_client = (
    OpenAI(
        api_key=(ALEMLLM_API_KEY or "EMPTY"),
        base_url=ALEMLLM_BASE_URL,
    )
    if ALEMLLM_BASE_URL
    else None
)

print("OpenAI key:", "OK" if OPENAI_API_KEY else "MISSING")
print("Anthropic key:", "OK" if ANTHROPIC_API_KEY else "MISSING")
print("HF token:", "OK" if HF_TOKEN else "MISSING")
print("AlemLLM endpoint:", "OK" if ALEMLLM_BASE_URL else "MISSING")

if openai_client is None:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add it in Colab Secrets."
    )

if anthropic_client is None:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is missing. Add it in Colab Secrets."
    )

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is missing. Add a Hugging Face token in Colab "
        "Secrets and make sure your account has access to "
        "issai/LLama-3.1-KazLLM-1.0-8B."
    )

if alemllm_client is None:
    raise RuntimeError(
        "ALEMLLM_BASE_URL is missing. Provide an OpenAI-compatible "
        "endpoint serving astanahub/alemllm, e.g. https://HOST/v1."
    )

ALEMLLM_ENDPOINT_FINGERPRINT = hashlib.sha256(
    ALEMLLM_BASE_URL.encode("utf-8")
).hexdigest()


# ------------------------------------------------------------
# GPU requirement for local KazLLM + BGE-M3
# ------------------------------------------------------------

if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is required for the local KazLLM experiment."
    )

GPU_TOTAL_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1024**3
)

print("GPU:", torch.cuda.get_device_name(0))
print("GPU total memory:", round(GPU_TOTAL_GB, 2), "GB")

if GPU_TOTAL_GB < 22:
    print(
        "WARNING: <22 GB GPU memory detected. Full-precision/BF16 "
        "KazLLM-8B + BGE-M3 may not fit reliably. For a publication "
        "experiment, prefer an L4/A100/H100-class runtime rather than "
        "silently changing the model with quantization."
    )


# ------------------------------------------------------------
# Local KazLLM-8B
# ------------------------------------------------------------

KAZLLM_MODEL_ID = MODEL_IDS["kazllm"]

_reuse_kazllm = (
    "kazllm_model" in globals()
    and "kazllm_tokenizer" in globals()
    and getattr(
        getattr(kazllm_model, "config", None),
        "_name_or_path",
        None,
    ) == KAZLLM_MODEL_ID
)

if _reuse_kazllm:
    print("Reusing already loaded KazLLM:", KAZLLM_MODEL_ID)
else:
    if "kazllm_model" in globals():
        del kazllm_model
    if "kazllm_tokenizer" in globals():
        del kazllm_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    kazllm_tokenizer = AutoTokenizer.from_pretrained(
        KAZLLM_MODEL_ID,
        token=HF_TOKEN,
        use_fast=True,
    )

    if kazllm_tokenizer.pad_token is None:
        kazllm_tokenizer.pad_token = kazllm_tokenizer.eos_token

    kazllm_tokenizer.padding_side = "left"

    KAZLLM_DTYPE = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    def load_kazllm_model(model_id: str):
        kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "token": HF_TOKEN,
        }

        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=KAZLLM_DTYPE,
                **kwargs,
            )
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=KAZLLM_DTYPE,
                **kwargs,
            )

    kazllm_model = load_kazllm_model(
        KAZLLM_MODEL_ID
    )
    kazllm_model.eval()

KAZLLM_COMMIT_HASH = getattr(
    kazllm_model.config,
    "_commit_hash",
    None,
)

print("Local KazLLM ready:", KAZLLM_MODEL_ID)
print("KazLLM commit:", KAZLLM_COMMIT_HASH)


# ------------------------------------------------------------
# BGE-M3
# CLS pooling + FP32 L2 normalization is implemented in Cell 8.
# ------------------------------------------------------------

BGE_DEVICE = "cuda"

_reuse_bge = (
    "bge_model" in globals()
    and "bge_tokenizer" in globals()
    and getattr(
        getattr(bge_model, "config", None),
        "_name_or_path",
        None,
    ) == EMBEDDING_MODEL_NAME
)

if _reuse_bge:
    print("Reusing already loaded BGE-M3:", EMBEDDING_MODEL_NAME)
else:
    if "bge_model" in globals():
        del bge_model
    if "bge_tokenizer" in globals():
        del bge_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    bge_tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_NAME
    )

    BGE_DTYPE = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    def load_embedding_model(model_id: str):
        try:
            model = AutoModel.from_pretrained(
                model_id,
                dtype=BGE_DTYPE,
            )
        except TypeError:
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=BGE_DTYPE,
            )

        return model.to(BGE_DEVICE)

    bge_model = load_embedding_model(
        EMBEDDING_MODEL_NAME
    )
    bge_model.eval()

BGE_COMMIT_HASH = getattr(
    bge_model.config,
    "_commit_hash",
    None,
)

EMBEDDING_DIMENSION = int(
    bge_model.config.hidden_size
)

print("BGE-M3 loaded:", EMBEDDING_MODEL_NAME)
print("BGE-M3 commit:", BGE_COMMIT_HASH)
print("Embedding dimension:", EMBEDDING_DIMENSION)

print(
    "GPU allocated:",
    round(torch.cuda.memory_allocated(0) / 1024**3, 2),
    "GB",
)


# ------------------------------------------------------------
# Runtime metadata — no secret values are written.
# ------------------------------------------------------------

RUNTIME_METADATA = {
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_total_gb": round(GPU_TOTAL_GB, 3),
    "kazllm_model_id": KAZLLM_MODEL_ID,
    "kazllm_commit_hash": KAZLLM_COMMIT_HASH,
    "bge_model_id": EMBEDDING_MODEL_NAME,
    "bge_commit_hash": BGE_COMMIT_HASH,
    "alemllm_model_id": MODEL_IDS["alemllm"],
    "alemllm_served_model": ALEMLLM_SERVED_MODEL,
    "alemllm_endpoint_fingerprint": ALEMLLM_ENDPOINT_FINGERPRINT,
}

with open(
    OUTPUT_DIR / "experiment2_runtime_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        RUNTIME_METADATA,
        f,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

print("Cell 4 model/runtime setup: PASS")

# %%
# ============================================================
# Cell 5. Unified LLM calls
# OpenAI / Anthropic / KazLLM / AlemLLM
# ============================================================

@dataclass
class LLMResult:
    text: str
    provider: str
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_sec: float
    request_id: Optional[str] = None
    finish_reason: Optional[str] = None
    attempts: int = 1


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def validate_model_id(
    provider: str,
    model_id: str,
) -> None:
    if not model_id:
        raise ValueError(
            f"Model ID for {provider} is empty."
        )

    if "YOUR_" in str(model_id).upper():
        raise ValueError(
            f"Model ID for {provider} is not configured: {model_id}"
        )


def normalize_prompt_text(text: Any) -> str:
    """Normalize Unicode while preserving prompt line structure."""
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u00A0", " ")

    for char in ["\u200B", "\u200C", "\u200D", "\u2060", "\uFEFF"]:
        text = text.replace(char, "")

    lines = [
        re.sub(r"[ \t]+", " ", line).rstrip()
        for line in text.splitlines()
    ]

    return "\n".join(lines).strip()


def is_retryable_api_error(exc: Exception) -> bool:
    """Return True only for likely transient API/network errors."""

    error_text = str(exc).strip().lower()

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "code", None)

    try:
        if status_code is not None and int(status_code) in {
            408, 409, 429, 500, 502, 503, 504
        }:
            return True
    except Exception:
        pass

    non_retryable_markers = [
        "invalid api key",
        "authentication",
        "unauthorized",
        "permission denied",
        "forbidden",
        "invalid_argument",
        "invalid argument",
        "model not found",
        "not configured",
        "unsupported provider",
    ]

    if any(marker in error_text for marker in non_retryable_markers):
        return False

    retryable_markers = [
        "408",
        "409",
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "too many requests",
        "resource_exhausted",
        "unavailable",
        "overloaded",
        "server error",
        "timeout",
        "timed out",
        "deadline",
        "connection reset",
        "connection error",
        "connection aborted",
        "service unavailable",
        "empty response",
    ]

    return any(
        marker in error_text
        for marker in retryable_markers
    )


# ------------------------------------------------------------
# OpenAI
# ------------------------------------------------------------

def call_openai_once(
    prompt: str,
    model_id: str,
) -> LLMResult:

    if openai_client is None:
        raise RuntimeError(
            "OpenAI client not configured."
        )

    validate_model_id("openai", model_id)

    start = time.perf_counter()

    response = openai_client.responses.create(
        model=model_id,
        input=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    latency = time.perf_counter() - start

    text = normalize_text(
        getattr(response, "output_text", "")
    )

    if not text:
        raise RuntimeError(
            "OpenAI returned empty response."
        )

    usage = getattr(response, "usage", None)

    input_tokens = safe_int(
        getattr(usage, "input_tokens", 0)
        if usage is not None else 0
    )

    output_tokens = safe_int(
        getattr(usage, "output_tokens", 0)
        if usage is not None else 0
    )

    total_tokens = safe_int(
        getattr(usage, "total_tokens", None)
        if usage is not None else None,
        default=input_tokens + output_tokens,
    )

    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    return LLMResult(
        text=text,
        provider="openai",
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_sec=float(latency),
        request_id=getattr(response, "id", None),
        finish_reason=getattr(response, "status", None),
    )


# ------------------------------------------------------------
# Anthropic
# ------------------------------------------------------------

def call_anthropic_once(
    prompt: str,
    model_id: str,
) -> LLMResult:

    if anthropic_client is None:
        raise RuntimeError(
            "Anthropic client not configured."
        )

    validate_model_id("anthropic", model_id)

    start = time.perf_counter()

    response = anthropic_client.messages.create(
        model=model_id,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    latency = time.perf_counter() - start

    parts = []
    for block in getattr(response, "content", []):
        block_text = getattr(block, "text", None)
        if block_text:
            parts.append(str(block_text))

    text = normalize_text("\n".join(parts))

    if not text:
        raise RuntimeError(
            "Anthropic returned empty response."
        )

    usage = getattr(response, "usage", None)

    input_tokens = safe_int(
        getattr(usage, "input_tokens", 0)
        if usage is not None else 0
    )

    output_tokens = safe_int(
        getattr(usage, "output_tokens", 0)
        if usage is not None else 0
    )

    return LLMResult(
        text=text,
        provider="anthropic",
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_sec=float(latency),
        request_id=getattr(response, "id", None),
        finish_reason=getattr(response, "stop_reason", None),
    )


# ------------------------------------------------------------
# Local KazLLM
# ------------------------------------------------------------

def call_kazllm_once(
    prompt: str,
    model_id: str,
    generation_seed: Optional[int] = None,
) -> LLMResult:

    validate_model_id("kazllm", model_id)

    if generation_seed is not None:
        generation_seed = int(generation_seed)
        torch.manual_seed(generation_seed)
        torch.cuda.manual_seed_all(generation_seed)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    model_inputs = kazllm_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = next(
        kazllm_model.parameters()
    ).device

    model_inputs = {
        key: value.to(device)
        for key, value in model_inputs.items()
    }

    input_tokens = int(
        model_inputs["input_ids"].shape[1]
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        generated = kazllm_model.generate(
            **model_inputs,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            do_sample=KAZLLM_DO_SAMPLE,
            temperature=KAZLLM_TEMPERATURE,
            top_p=KAZLLM_TOP_P,
            pad_token_id=kazllm_tokenizer.pad_token_id,
            eos_token_id=kazllm_tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    suffix = generated[
        0,
        input_tokens:
    ]

    output_tokens = int(
        suffix.shape[0]
    )

    text = normalize_text(
        kazllm_tokenizer.decode(
            suffix,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )

    if not text:
        raise RuntimeError(
            "Local KazLLM returned empty response."
        )

    finish_reason = (
        "max_tokens"
        if output_tokens >= MAX_OUTPUT_TOKENS
        else "completed"
    )

    return LLMResult(
        text=text,
        provider="kazllm",
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_sec=float(latency),
        request_id=None,
        finish_reason=finish_reason,
    )


# ------------------------------------------------------------
# AlemLLM — OpenAI-compatible endpoint
# ------------------------------------------------------------

def call_alemllm_once(
    prompt: str,
    model_id: str,
) -> LLMResult:

    if alemllm_client is None:
        raise RuntimeError(
            "AlemLLM endpoint client not configured."
        )

    validate_model_id("alemllm", model_id)

    start = time.perf_counter()

    response = alemllm_client.chat.completions.create(
        model=ALEMLLM_SERVED_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=ALEMLLM_TEMPERATURE,
        top_p=ALEMLLM_TOP_P,
    )

    latency = time.perf_counter() - start

    choices = getattr(response, "choices", None) or []

    if not choices:
        raise RuntimeError(
            "AlemLLM endpoint returned no choices."
        )

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message is not None else ""

    text = normalize_text(content)

    if not text:
        raise RuntimeError(
            "AlemLLM returned empty response."
        )

    usage = getattr(response, "usage", None)

    input_tokens = safe_int(
        getattr(usage, "prompt_tokens", 0)
        if usage is not None else 0
    )

    output_tokens = safe_int(
        getattr(usage, "completion_tokens", 0)
        if usage is not None else 0
    )

    total_tokens = safe_int(
        getattr(usage, "total_tokens", None)
        if usage is not None else None,
        default=input_tokens + output_tokens,
    )

    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    return LLMResult(
        text=text,
        provider="alemllm",
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_sec=float(latency),
        request_id=getattr(response, "id", None),
        finish_reason=getattr(choices[0], "finish_reason", None),
    )


# ------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------

def call_llm_once(
    provider: str,
    prompt: str,
    model_id: str,
    generation_seed: Optional[int] = None,
) -> LLMResult:

    provider = str(provider).strip().lower()

    if provider == "openai":
        return call_openai_once(
            prompt,
            model_id,
        )

    if provider == "anthropic":
        return call_anthropic_once(
            prompt,
            model_id,
        )

    if provider == "kazllm":
        return call_kazllm_once(
            prompt,
            model_id,
            generation_seed=generation_seed,
        )

    if provider == "alemllm":
        return call_alemllm_once(
            prompt,
            model_id,
        )

    raise ValueError(
        f"Unsupported provider: {provider}"
    )


def provider_max_attempts(provider: str) -> int:
    provider = str(provider).strip().lower()

    if provider == "kazllm":
        return 1

    if provider in {
        "openai",
        "anthropic",
        "alemllm",
    }:
        return 5

    return 1


def get_retry_delay(attempt: int) -> float:
    delays = [2, 4, 8, 16, 30]
    index = min(
        max(attempt - 1, 0),
        len(delays) - 1,
    )
    return float(delays[index]) + random.uniform(0.0, 1.5)


def call_llm(
    provider: str,
    prompt: str,
    model_id: Optional[str] = None,
    generation_seed: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> LLMResult:

    provider = str(provider).strip().lower()

    if provider not in ENABLED_PROVIDERS:
        raise ValueError(
            f"Unsupported/disabled provider: {provider}"
        )

    if provider not in MODEL_IDS:
        raise ValueError(
            f"Provider '{provider}' is missing from MODEL_IDS."
        )

    if model_id is None:
        model_id = MODEL_IDS[provider]

    validate_model_id(provider, model_id)

    prompt = normalize_prompt_text(prompt)
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    attempts = (
        provider_max_attempts(provider)
        if max_attempts is None
        else max(1, int(max_attempts))
    )

    if provider == "kazllm":
        attempts = 1

    overall_start = time.perf_counter()
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            result = call_llm_once(
                provider=provider,
                prompt=prompt,
                model_id=model_id,
                generation_seed=generation_seed,
            )

            result.text = normalize_text(result.text)
            if not result.text:
                raise RuntimeError(
                    f"{provider} returned empty response."
                )

            result.attempts = attempt

            if attempt > 1:
                result.latency_sec = float(
                    time.perf_counter() - overall_start
                )

            return result

        except Exception as exc:
            last_error = exc

            if provider == "kazllm":
                raise

            retryable = is_retryable_api_error(exc)

            print(
                f"\n[{provider}] Attempt {attempt}/{attempts} failed:"
            )
            print(
                f"{type(exc).__name__}: {exc}"
            )

            if not retryable or attempt >= attempts:
                break

            delay = get_retry_delay(attempt)

            print(
                f"[{provider}] Temporary API/endpoint error. "
                f"Retrying after {delay:.1f} sec..."
            )

            time.sleep(delay)

    raise RuntimeError(
        f"{provider} call failed after {attempts} attempt(s): "
        f"{last_error}"
    )


assert provider_max_attempts("openai") == 5
assert provider_max_attempts("anthropic") == 5
assert provider_max_attempts("kazllm") == 1
assert provider_max_attempts("alemllm") == 5

for _provider in ENABLED_PROVIDERS:
    validate_model_id(
        _provider,
        MODEL_IDS[_provider],
    )

print("Cell 5 unified LLM calls: PASS")
print("Providers:", ENABLED_PROVIDERS)

# %%
# ============================================================
# Cell 6. Fixed prompts
# ============================================================

BASELINE_PROMPT_TEMPLATE = """
Тапсырма:

Берілген қазақ тіліндегі мәтін бойынша бір табиғи және мазмұнды сұрақ құрастырыңыз.

Талаптар:

1. Сұраққа жауап берілген мәтіннің өзінен табылуы тиіс.
2. Сұрақ жаңа немесе мәтінде жоқ ақпаратты талап етпеуі тиіс.
3. Сұрақ қазақ тілінде грамматикалық және орфографиялық тұрғыдан дұрыс болуы тиіс.
4. Сұрақ мәтінді жай ғана сұраулы сөйлемге айналдырумен шектелмеуі тиіс.
5. Жауап мәтіндегі дәл кездесетін қысқа үзінді болуы тиіс.
6. Жауапты өзгертпеңіз және мәтінде жоқ сөздерді қоспаңыз.
7. Тек төмендегі JSON форматында жауап беріңіз.
8. Ешқандай түсіндірме немесе қосымша мәтін қоспаңыз.

JSON форматы:

{{
  "question": "Сұрақ?",
  "answer": "мәтіндегі дәл жауап"
}}

Мәтін:

{source_text}
""".strip()


KAZDIV_PROMPT_TEMPLATE = """
Тапсырма:

Берілген қазақ тіліндегі мәтін бойынша {k_candidates} түрлі, табиғи және мазмұнды сұрақ құрастырыңыз.

Талаптар:

1. Әр сұраққа жауап берілген мәтіннің өзінен табылуы тиіс.
2. Әр сұрақ жаңа немесе мәтінде жоқ ақпаратты талап етпеуі тиіс.
3. Сұрақтар қазақ тілінде грамматикалық және орфографиялық тұрғыдан дұрыс болуы тиіс.
4. Сұрақтар бір-бірінен лексикалық және мүмкін болған жағдайда синтаксистік тұрғыдан ерекшеленуі тиіс.
5. Бірдей немесе өте ұқсас сұрақтарды қайталамаңыз.
6. Сұрақтарды тек сөздердің орнын ауыстыру арқылы өзгертпеңіз.
7. Әр сұрақ үшін жауап мәтіндегі дәл кездесетін қысқа үзінді болуы тиіс.
8. Жауапты өзгертпеңіз және мәтінде жоқ сөздерді қоспаңыз.
9. Жауаптың өзін сұрақ ішінде толық бермеңіз.
10. Дәл {k_candidates} сұрақ жасаңыз.
11. Тек төмендегі JSON форматында жауап беріңіз.
12. Ешқандай түсіндірме немесе қосымша мәтін қоспаңыз.

JSON форматы:

{{
  "candidates": [
    {{
      "question": "1-сұрақ?",
      "answer": "мәтіндегі дәл жауап"
    }},
    {{
      "question": "2-сұрақ?",
      "answer": "мәтіндегі дәл жауап"
    }}
  ]
}}

Мәтін:

{source_text}
""".strip()


def build_prompt(
    method: str,
    source_text: str,
    k_candidates: int = K_CANDIDATES,
) -> str:

    source_text = normalize_text(
        source_text
    )

    method = str(method).strip().lower()

    if method == "baseline":
        return BASELINE_PROMPT_TEMPLATE.format(
            source_text=source_text
        )

    if method == "kazdiv":
        return KAZDIV_PROMPT_TEMPLATE.format(
            source_text=source_text,
            k_candidates=k_candidates,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


with open(
    OUTPUT_DIR /
    "experiment2_prompt_templates.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "baseline": BASELINE_PROMPT_TEMPLATE,
            "kazdiv": KAZDIV_PROMPT_TEMPLATE,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("Prompt templates ready.")

# %%
# ============================================================
# Cell 7. Robust JSON parsing and exact deduplication
# ============================================================

@dataclass
class QuestionCandidate:
    question: str
    answer: str


@dataclass
class QuestionParseResult:
    candidates: List[QuestionCandidate]
    parse_success: bool
    parse_method: str
    raw_candidate_count: int
    cleaned_candidate_count: int
    exact_duplicate_count: int
    invalid_candidate_count: int
    expected_candidate_count: int
    exact_candidate_count_match: bool
    error_message: Optional[str] = None


def remove_code_fences(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    return text.strip()


def extract_balanced_structure(
    text: str,
    opening_char: str,
    closing_char: str,
) -> Optional[str]:

    if not text:
        return None

    start = text.find(opening_char)

    if start == -1:
        return None

    depth = 0
    quote_char = None
    escape = False

    for index in range(
        start,
        len(text),
    ):
        char = text[index]

        if escape:
            escape = False
            continue

        if (
            char == "\\"
            and quote_char is not None
        ):
            escape = True
            continue

        if quote_char is not None:
            if char == quote_char:
                quote_char = None
            continue

        if char in ['"', "'"]:
            quote_char = char
            continue

        if char == opening_char:
            depth += 1

        elif char == closing_char:
            depth -= 1

            if depth == 0:
                return text[
                    start:
                    index + 1
                ]

    return None


def normalize_for_exact_match(
    text: Any
) -> str:

    text = normalize_text(
        text
    ).lower()

    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("«", '"')
        .replace("»", '"')
    )

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[.!?…]+$",
        "",
        text,
    )

    return text.strip()


def candidate_from_any(
    item: Any
) -> Optional[QuestionCandidate]:

    if not isinstance(
        item,
        dict,
    ):
        return None

    lower = {
        str(key).strip().lower():
            value
        for key, value
        in item.items()
    }

    question = (
        lower.get("question") or
        lower.get("сұрақ") or
        lower.get("q")
    )

    answer = (
        lower.get("answer") or
        lower.get("жауап") or
        lower.get("a")
    )

    question = normalize_text(
        question
    )

    answer = normalize_text(
        answer
    )

    if not question or not answer:
        return None

    return QuestionCandidate(
        question=question,
        answer=answer,
    )


def extract_candidate_list(
    obj: Any
) -> Optional[List[Any]]:

    if isinstance(
        obj,
        list,
    ):
        return obj

    if not isinstance(
        obj,
        dict,
    ):
        return None

    lower_obj = {
        str(key).strip().lower():
            value
        for key, value
        in obj.items()
    }

    if (
        (
            "question" in lower_obj or
            "сұрақ" in lower_obj
        ) and
        (
            "answer" in lower_obj or
            "жауап" in lower_obj
        )
    ):
        return [obj]

    for key in [
        "candidates",
        "questions",
        "items",
        "outputs",
        "responses",
    ]:
        value = lower_obj.get(
            key
        )

        if isinstance(
            value,
            list,
        ):
            return value

    return None


def parse_json_like(
    raw_text: Any
) -> Tuple[Optional[List[Any]], str]:

    if raw_text is None:
        return None, "none"

    text = str(
        raw_text
    ).strip()

    if not text:
        return None, "empty"

    attempts = [
        (
            "direct_json",
            text,
        ),
        (
            "code_fence_json",
            remove_code_fences(
                text
            ),
        ),
    ]

    object_text = (
        extract_balanced_structure(
            text,
            "{",
            "}",
        )
    )

    if object_text:
        attempts.append(
            (
                "extracted_json_object",
                object_text,
            )
        )

    array_text = (
        extract_balanced_structure(
            text,
            "[",
            "]",
        )
    )

    if array_text:
        attempts.append(
            (
                "extracted_json_array",
                array_text,
            )
        )

    for method, candidate_text in attempts:
        try:
            obj = json.loads(
                candidate_text
            )

            items = extract_candidate_list(
                obj
            )

            if items is not None:
                return items, method

        except Exception:
            pass

    literal_source = (
        object_text or
        array_text or
        remove_code_fences(
            text
        )
    )

    try:
        obj = ast.literal_eval(
            literal_source
        )

        items = extract_candidate_list(
            obj
        )

        if items is not None:
            return items, "python_literal"

    except Exception:
        pass

    return None, "failed"


def clean_and_deduplicate_question_candidates(
    raw_candidates: List[Any]
) -> Dict[str, Any]:

    cleaned = []
    seen_questions = set()

    duplicates = 0
    invalid = 0

    for item in raw_candidates:
        candidate = candidate_from_any(
            item
        )

        if candidate is None:
            invalid += 1
            continue

        key = normalize_for_exact_match(
            candidate.question
        )

        if not key:
            invalid += 1
            continue

        if key in seen_questions:
            duplicates += 1
            continue

        seen_questions.add(
            key
        )

        cleaned.append(
            candidate
        )

    return {
        "candidates":
            cleaned,

        "raw_candidate_count":
            len(
                raw_candidates
            ),

        "cleaned_candidate_count":
            len(
                cleaned
            ),

        "exact_duplicate_count":
            duplicates,

        "invalid_candidate_count":
            invalid,
    }


def parse_question_response(
    raw_text: Any,
    expected_k: int,
) -> QuestionParseResult:

    raw_candidates, method = (
        parse_json_like(
            raw_text
        )
    )

    if raw_candidates is None:
        return QuestionParseResult(
            candidates=[],
            parse_success=False,
            parse_method=method,
            raw_candidate_count=0,
            cleaned_candidate_count=0,
            exact_duplicate_count=0,
            invalid_candidate_count=0,
            expected_candidate_count=expected_k,
            exact_candidate_count_match=False,
            error_message=(
                "Could not parse "
                "question/answer JSON."
            ),
        )

    result = (
        clean_and_deduplicate_question_candidates(
            raw_candidates
        )
    )

    n = len(
        result[
            "candidates"
        ]
    )

    return QuestionParseResult(
        candidates=result[
            "candidates"
        ],
        parse_success=(
            n >= 1
        ),
        parse_method=method,
        raw_candidate_count=(
            result[
                "raw_candidate_count"
            ]
        ),
        cleaned_candidate_count=(
            result[
                "cleaned_candidate_count"
            ]
        ),
        exact_duplicate_count=(
            result[
                "exact_duplicate_count"
            ]
        ),
        invalid_candidate_count=(
            result[
                "invalid_candidate_count"
            ]
        ),
        expected_candidate_count=(
            expected_k
        ),
        exact_candidate_count_match=(
            n == expected_k
        ),
        error_message=(
            None
            if n == expected_k
            else
            f"Expected {expected_k}, "
            f"recovered {n} unique candidates."
        ),
    )


# Internal parser test.
TEST_Q_JSON = """
{
  "candidates": [
    {
      "question": "Зерттеудің негізгі мақсаты қандай?",
      "answer": "білім беру сапасын арттыру"
    },
    {
      "question": "Зерттеу қандай нәтижеге бағытталған?",
      "answer": "білім беру сапасын арттыру"
    }
  ]
}
"""

test_parse = (
    parse_question_response(
        TEST_Q_JSON,
        expected_k=2,
    )
)

assert test_parse.parse_success
assert len(test_parse.candidates) == 2

print("Cell 7 parser tests: PASS")

# %%
# ============================================================
# Cell 8. BGE-M3 embeddings — CLS pooling
# ============================================================


def prepare_embedding_texts(
    texts: Union[
        str,
        List[str],
        Tuple[str, ...],
    ]
) -> List[str]:

    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(texts, (list, tuple)):
        raise TypeError(
            "texts must be str/list/tuple"
        )

    cleaned = []

    for text in texts:
        text = normalize_text(text)

        if not text:
            raise ValueError(
                "Empty text cannot be embedded."
            )

        cleaned.append(text)

    return cleaned


def encode_texts(
    texts: Union[
        str,
        List[str],
        Tuple[str, ...],
    ],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:

    texts = prepare_embedding_texts(texts)
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[
            start:
            start + batch_size
        ]

        encoded = bge_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=EMBEDDING_MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(BGE_DEVICE)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = bge_model(**encoded)

            # Dense representation: CLS token.
            embeddings = outputs.last_hidden_state[:, 0, :]

            # Critical numerical fix:
            # normalize in FP32 even when model runs in BF16/FP16.
            embeddings = embeddings.float()

            if not torch.isfinite(embeddings).all():
                raise RuntimeError(
                    "BGE-M3 produced NaN or Inf before normalization."
                )

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1,
                eps=1e-12,
            )

            if not torch.isfinite(embeddings).all():
                raise RuntimeError(
                    "BGE-M3 produced NaN or Inf after normalization."
                )

        batch_embeddings = (
            embeddings
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        all_embeddings.append(batch_embeddings)

    result = np.vstack(all_embeddings).astype(
        np.float32,
        copy=False,
    )

    if (
        result.ndim != 2
        or result.shape[1] != EMBEDDING_DIMENSION
    ):
        raise RuntimeError(
            f"Unexpected embedding shape: {result.shape}. "
            f"Expected (*, {EMBEDDING_DIMENSION})."
        )

    if not np.isfinite(result).all():
        raise RuntimeError(
            "Final embeddings contain NaN or Inf."
        )

    return result


def semantic_similarity(
    text_a: str,
    text_b: str,
) -> float:

    embeddings = encode_texts(
        [text_a, text_b]
    )

    similarity = np.dot(
        embeddings[0],
        embeddings[1],
    )

    return float(
        np.clip(similarity, -1.0, 1.0)
    )


def question_semantic_analysis(
    source_text: str,
    questions: List[str],
) -> Dict[str, Any]:

    if not questions:
        return {
            "relevance_scores": [],
            "similarity_matrix": np.empty(
                (0, 0),
                dtype=np.float32,
            ),
            "semantic_diversity": np.nan,
        }

    all_texts = [source_text] + list(questions)
    embeddings = encode_texts(all_texts)

    source_embedding = embeddings[0]
    question_embeddings = embeddings[1:]

    relevance = (
        question_embeddings @ source_embedding
    )
    relevance = np.clip(
        relevance,
        -1.0,
        1.0,
    )

    similarity_matrix = (
        question_embeddings @ question_embeddings.T
    )
    similarity_matrix = np.clip(
        similarity_matrix,
        -1.0,
        1.0,
    ).astype(np.float32)

    np.fill_diagonal(
        similarity_matrix,
        1.0,
    )

    if len(questions) >= 2:
        upper_indices = np.triu_indices(
            len(questions),
            k=1,
        )

        pairwise_similarities = (
            similarity_matrix[upper_indices]
        )

        diversity = float(
            np.mean(
                1.0 - pairwise_similarities
            )
        )
    else:
        diversity = np.nan

    return {
        "relevance_scores": [
            float(value)
            for value in relevance
        ],
        "similarity_matrix": similarity_matrix,
        "semantic_diversity": diversity,
    }


# ------------------------------------------------------------
# Internal tests
# ------------------------------------------------------------

TEST_EMBED_TEXT = (
    "Жасанды интеллект білім беру "
    "саласында қолданылады."
)

test_embedding = encode_texts(
    [TEST_EMBED_TEXT]
)

print(
    "Embedding shape:",
    test_embedding.shape,
)

test_norm = float(
    np.linalg.norm(test_embedding[0])
)

print(
    "Embedding norm:",
    test_norm,
)

assert test_embedding.shape == (
    1,
    EMBEDDING_DIMENSION,
)

assert np.isfinite(test_embedding).all()

assert np.isclose(
    test_norm,
    1.0,
    atol=1e-5,
), (
    f"Embedding is not properly L2-normalized. Norm={test_norm}"
)

same_text_similarity = semantic_similarity(
    TEST_EMBED_TEXT,
    TEST_EMBED_TEXT,
)

print(
    "Identical-text similarity:",
    same_text_similarity,
)

assert same_text_similarity > 0.999

print(
    "Cell 8 BGE-M3 CLS embeddings: PASS"
)

# %%
# ============================================================
# Cell 9. Task-specific quality filtering
# ============================================================

@dataclass
class QuestionQualityRecord:
    candidate_index: int
    question: str
    answer: str
    question_word_count: int
    answer_word_count: int
    relevance_score: Optional[float]
    has_question_mark: bool
    answer_in_context: bool
    answer_leakage: bool
    same_as_context: bool
    question_length_pass: bool
    answer_length_pass: bool
    quality_pass: bool
    rejection_reasons: List[str]


@dataclass
class QuestionQualityResult:
    source_text: str
    input_candidate_count: int
    valid_candidate_count: int
    rejected_candidate_count: int
    valid_candidates: List[QuestionCandidate]
    valid_original_indices: List[int]
    valid_relevance_scores: List[float]
    records: List[QuestionQualityRecord]
    filter_success: bool


def count_words(
    text: Any
) -> int:

    text = normalize_text(
        text
    )

    if not text:
        return 0

    return len(
        text.split()
    )


def normalize_for_span_match(
    text: Any
) -> str:

    text = normalize_text(
        text
    ).lower()

    text = (
        text
        .replace("«", '"')
        .replace("»", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def answer_occurs_in_context(
    source_text: str,
    answer: str,
) -> bool:

    source_norm = (
        normalize_for_span_match(
            source_text
        )
    )

    answer_norm = (
        normalize_for_span_match(
            answer
        )
    )

    if not source_norm or not answer_norm:
        return False

    return (
        answer_norm in
        source_norm
    )


def answer_is_leaked_in_question(
    question: str,
    answer: str,
) -> bool:

    question_norm = (
        normalize_for_span_match(
            question
        )
    )

    answer_norm = (
        normalize_for_span_match(
            answer
        )
    )

    if not question_norm or not answer_norm:
        return False

    if len(
        answer_norm
    ) < 2:
        return False

    return (
        answer_norm in
        question_norm
    )


def evaluate_question_candidate(
    source_text: str,
    candidate: QuestionCandidate,
    relevance_score: Optional[float],
    candidate_index: int,
) -> QuestionQualityRecord:

    question = normalize_text(
        candidate.question
    )

    answer = normalize_text(
        candidate.answer
    )

    reasons = []

    q_words = count_words(
        question
    )

    a_words = count_words(
        answer
    )

    has_question_mark = (
        question.endswith(
            "?"
        )
    )

    if not has_question_mark:
        reasons.append(
            "missing_question_mark"
        )

    answer_in_context = (
        answer_occurs_in_context(
            source_text,
            answer,
        )
    )

    if not answer_in_context:
        reasons.append(
            "answer_not_in_context"
        )

    answer_leakage = (
        answer_is_leaked_in_question(
            question,
            answer,
        )
    )

    if answer_leakage:
        reasons.append(
            "answer_leakage"
        )

    same_as_context = (
        normalize_for_exact_match(
            question
        ) ==
        normalize_for_exact_match(
            source_text
        )
    )

    if same_as_context:
        reasons.append(
            "same_as_context"
        )

    question_length_pass = (
        MIN_QUESTION_WORDS <=
        q_words <=
        MAX_QUESTION_WORDS
    )

    if not question_length_pass:
        reasons.append(
            "question_length_out_of_range"
        )

    answer_length_pass = (
        MIN_ANSWER_WORDS <=
        a_words <=
        MAX_ANSWER_WORDS
    )

    if not answer_length_pass:
        reasons.append(
            "answer_length_out_of_range"
        )

    relevance_pass = True

    if (
        QUESTION_RELEVANCE_MIN
        is not None
    ):
        relevance_pass = (
            relevance_score is not None and
            relevance_score >=
            QUESTION_RELEVANCE_MIN
        )

        if not relevance_pass:
            reasons.append(
                "low_question_context_relevance"
            )

    quality_pass = (
        has_question_mark and
        answer_in_context and
        not answer_leakage and
        not same_as_context and
        question_length_pass and
        answer_length_pass and
        relevance_pass
    )

    return QuestionQualityRecord(
        candidate_index=(
            candidate_index
        ),
        question=question,
        answer=answer,
        question_word_count=(
            q_words
        ),
        answer_word_count=(
            a_words
        ),
        relevance_score=(
            float(
                relevance_score
            )
            if relevance_score
            is not None
            else None
        ),
        has_question_mark=(
            has_question_mark
        ),
        answer_in_context=(
            answer_in_context
        ),
        answer_leakage=(
            answer_leakage
        ),
        same_as_context=(
            same_as_context
        ),
        question_length_pass=(
            question_length_pass
        ),
        answer_length_pass=(
            answer_length_pass
        ),
        quality_pass=(
            quality_pass
        ),
        rejection_reasons=(
            reasons
        ),
    )


def filter_question_candidates(
    source_text: str,
    candidates: List[QuestionCandidate],
) -> QuestionQualityResult:

    source_text = normalize_text(
        source_text
    )

    if not candidates:
        return QuestionQualityResult(
            source_text=source_text,
            input_candidate_count=0,
            valid_candidate_count=0,
            rejected_candidate_count=0,
            valid_candidates=[],
            valid_original_indices=[],
            valid_relevance_scores=[],
            records=[],
            filter_success=False,
        )

    questions = [
        candidate.question
        for candidate
        in candidates
    ]

    analysis = (
        question_semantic_analysis(
            source_text,
            questions,
        )
    )

    relevance_scores = (
        analysis[
            "relevance_scores"
        ]
    )

    records = []

    for index, (
        candidate,
        relevance,
    ) in enumerate(
        zip(
            candidates,
            relevance_scores,
        )
    ):
        records.append(
            evaluate_question_candidate(
                source_text=(
                    source_text
                ),
                candidate=(
                    candidate
                ),
                relevance_score=(
                    relevance
                ),
                candidate_index=(
                    index
                ),
            )
        )

    valid_records = [
        record
        for record
        in records
        if record.quality_pass
    ]

    valid_candidates = [
        QuestionCandidate(
            question=record.question,
            answer=record.answer,
        )
        for record
        in valid_records
    ]

    return QuestionQualityResult(
        source_text=(
            source_text
        ),
        input_candidate_count=(
            len(
                candidates
            )
        ),
        valid_candidate_count=(
            len(
                valid_candidates
            )
        ),
        rejected_candidate_count=(
            len(
                candidates
            ) -
            len(
                valid_candidates
            )
        ),
        valid_candidates=(
            valid_candidates
        ),
        valid_original_indices=[
            record.candidate_index
            for record
            in valid_records
        ],
        valid_relevance_scores=[
            float(
                record.relevance_score
            )
            for record
            in valid_records
            if record.relevance_score
            is not None
        ],
        records=(
            records
        ),
        filter_success=(
            len(
                valid_candidates
            )
            >= 1
        ),
    )


TEST_CONTEXT = (
    "Астана Қазақстанның астанасы болып табылады. "
    "Қала Есіл өзенінің жағасында орналасқан."
)

TEST_CANDIDATES = [
    QuestionCandidate(
        question=(
            "Қазақстанның астанасы "
            "қай қала?"
        ),
        answer="Астана",
    ),
    QuestionCandidate(
        question=(
            "Астана қай өзеннің "
            "жағасында орналасқан?"
        ),
        answer="Есіл өзенінің",
    ),
    QuestionCandidate(
        question=(
            "Қазақстанның астанасы "
            "Алматы ма?"
        ),
        answer="Алматы",
    ),
]

test_quality = (
    filter_question_candidates(
        TEST_CONTEXT,
        TEST_CANDIDATES,
    )
)

assert (
    test_quality
    .records[
        2
    ]
    .answer_in_context
    is False
)

print(
    "Cell 9 quality filtering: PASS"
)

# %%
# ============================================================
# Cell 10. Semantic near-deduplication and clustering
# ============================================================

@dataclass
class QuestionCluster:
    cluster_id: int
    member_local_indices: List[int]
    original_candidate_indices: List[int]
    questions: List[str]
    answers: List[str]
    relevance_scores: List[float]
    representative_local_index: int
    representative_original_index: int
    representative_question: str
    representative_answer: str
    representative_relevance: float
    cluster_size: int


@dataclass
class QuestionClusteringResult:
    source_text: str
    input_candidate_count: int
    cluster_count: int
    semantic_duplicates_removed: int
    edge_count: int
    threshold: float
    clusters: List[QuestionCluster]
    representative_candidates: List[QuestionCandidate]
    representative_relevance_scores: List[float]
    representative_original_indices: List[int]
    similarity_matrix: np.ndarray
    success: bool


def build_duplicate_graph(
    similarity_matrix: np.ndarray,
    threshold: float,
) -> Dict[int, List[int]]:

    if not (
        0 <
        threshold <=
        1
    ):
        raise ValueError(
            "threshold must be in (0, 1]"
        )

    matrix = np.asarray(
        similarity_matrix,
        dtype=np.float32,
    )

    if (
        matrix.ndim !=
        2 or
        matrix.shape[0] !=
        matrix.shape[1]
    ):
        raise ValueError(
            "Similarity matrix must be square."
        )

    n = matrix.shape[0]

    graph = {
        index:
            []
        for index
        in range(
            n
        )
    }

    for i in range(
        n
    ):
        for j in range(
            i + 1,
            n,
        ):
            if (
                float(
                    matrix[
                        i,
                        j
                    ]
                ) >=
                threshold
            ):
                graph[
                    i
                ].append(
                    j
                )

                graph[
                    j
                ].append(
                    i
                )

    return graph


def connected_components(
    graph: Dict[int, List[int]]
) -> List[List[int]]:

    visited = set()
    components = []

    for start in sorted(
        graph
    ):
        if start in visited:
            continue

        stack = [
            start
        ]

        component = []

        while stack:
            node = stack.pop()

            if node in visited:
                continue

            visited.add(
                node
            )

            component.append(
                node
            )

            for neighbor in sorted(
                graph[
                    node
                ],
                reverse=True,
            ):
                if neighbor not in visited:
                    stack.append(
                        neighbor
                    )

        components.append(
            sorted(
                component
            )
        )

    return sorted(
        components,
        key=lambda component:
            min(
                component
            ),
    )


def cluster_quality_result(
    quality_result: QuestionQualityResult,
    threshold: float = QUESTION_NEAR_DUPLICATE_THRESHOLD,
) -> QuestionClusteringResult:

    candidates = (
        quality_result
        .valid_candidates
    )

    relevance_scores = (
        quality_result
        .valid_relevance_scores
    )

    original_indices = (
        quality_result
        .valid_original_indices
    )

    n = len(
        candidates
    )

    if n == 0:
        return QuestionClusteringResult(
            source_text=(
                quality_result
                .source_text
            ),
            input_candidate_count=0,
            cluster_count=0,
            semantic_duplicates_removed=0,
            edge_count=0,
            threshold=(
                float(
                    threshold
                )
            ),
            clusters=[],
            representative_candidates=[],
            representative_relevance_scores=[],
            representative_original_indices=[],
            similarity_matrix=(
                np.empty(
                    (0, 0),
                    dtype=np.float32,
                )
            ),
            success=False,
        )

    if not (
        len(
            relevance_scores
        ) ==
        len(
            original_indices
        ) ==
        n
    ):
        raise RuntimeError(
            "Inconsistent quality result lengths."
        )

    questions = [
        candidate.question
        for candidate
        in candidates
    ]

    analysis = (
        question_semantic_analysis(
            quality_result
            .source_text,
            questions,
        )
    )

    matrix = (
        analysis[
            "similarity_matrix"
        ]
    )

    graph = (
        build_duplicate_graph(
            matrix,
            threshold,
        )
    )

    edge_count = (
        sum(
            len(
                neighbors
            )
            for neighbors
            in graph.values()
        ) //
        2
    )

    components = (
        connected_components(
            graph
        )
    )

    clusters = []
    representatives = []
    rep_relevance = []
    rep_original_indices = []

    for cluster_id, component in enumerate(
        components
    ):

        representative_local_index = max(
            component,
            key=lambda index: (
                float(
                    relevance_scores[
                        index
                    ]
                ),
                -int(
                    index
                ),
            ),
        )

        representative = (
            candidates[
                representative_local_index
            ]
        )

        representative_original_index = int(
            original_indices[
                representative_local_index
            ]
        )

        representative_score = float(
            relevance_scores[
                representative_local_index
            ]
        )

        cluster = QuestionCluster(
            cluster_id=(
                cluster_id
            ),
            member_local_indices=[
                int(
                    index
                )
                for index
                in component
            ],
            original_candidate_indices=[
                int(
                    original_indices[
                        index
                    ]
                )
                for index
                in component
            ],
            questions=[
                candidates[
                    index
                ].question
                for index
                in component
            ],
            answers=[
                candidates[
                    index
                ].answer
                for index
                in component
            ],
            relevance_scores=[
                float(
                    relevance_scores[
                        index
                    ]
                )
                for index
                in component
            ],
            representative_local_index=(
                int(
                    representative_local_index
                )
            ),
            representative_original_index=(
                representative_original_index
            ),
            representative_question=(
                representative
                .question
            ),
            representative_answer=(
                representative
                .answer
            ),
            representative_relevance=(
                representative_score
            ),
            cluster_size=(
                len(
                    component
                )
            ),
        )

        clusters.append(
            cluster
        )

        representatives.append(
            QuestionCandidate(
                question=(
                    representative
                    .question
                ),
                answer=(
                    representative
                    .answer
                ),
            )
        )

        rep_relevance.append(
            representative_score
        )

        rep_original_indices.append(
            representative_original_index
        )

    return QuestionClusteringResult(
        source_text=(
            quality_result
            .source_text
        ),
        input_candidate_count=(
            n
        ),
        cluster_count=(
            len(
                clusters
            )
        ),
        semantic_duplicates_removed=(
            n -
            len(
                clusters
            )
        ),
        edge_count=(
            int(
                edge_count
            )
        ),
        threshold=(
            float(
                threshold
            )
        ),
        clusters=(
            clusters
        ),
        representative_candidates=(
            representatives
        ),
        representative_relevance_scores=(
            rep_relevance
        ),
        representative_original_indices=(
            rep_original_indices
        ),
        similarity_matrix=(
            matrix
        ),
        success=(
            len(
                clusters
            ) >=
            1
        ),
    )


SYNTHETIC_MATRIX = np.array(
    [
        [1.00, 0.97, 0.20, 0.10],
        [0.97, 1.00, 0.96, 0.10],
        [0.20, 0.96, 1.00, 0.15],
        [0.10, 0.10, 0.15, 1.00],
    ],
    dtype=np.float32,
)

graph = (
    build_duplicate_graph(
        SYNTHETIC_MATRIX,
        threshold=0.95,
    )
)

components = (
    connected_components(
        graph
    )
)

assert components == [
    [0, 1, 2],
    [3],
]

print(
    "Cell 10 semantic clustering: PASS"
)

# %%
# ============================================================
# Cell 11. Stateless diversity-aware selection
# ============================================================

@dataclass
class QuestionSelectionResult:
    source_id: Any
    repetition: int
    provider: str
    model_id: str
    method: str
    selection_seed: int
    cluster_count: int
    selection_probability: float
    selected_cluster_id: int
    selected_original_candidate_index: int
    selected_question: str
    selected_answer: str
    selected_relevance: float
    success: bool
    reason: str


def make_selection_seed(
    provider: str,
    model_id: str,
    source_id: Any,
    repetition: int,
    method: str = "kazdiv",
    global_seed: int = RANDOM_SEED,
) -> int:

    material = (
        f"{global_seed}|"
        f"{provider}|"
        f"{model_id}|"
        f"{source_id}|"
        f"{repetition}|"
        f"{method}"
    )

    digest = hashlib.sha256(
        material.encode(
            "utf-8"
        )
    ).hexdigest()

    return int(
        digest[
            :16
        ],
        16,
    )


def select_question_from_clusters(
    clustering_result: QuestionClusteringResult,
    provider: str,
    model_id: str,
    source_id: Any,
    repetition: int,
) -> QuestionSelectionResult:

    cluster_count = (
        clustering_result
        .cluster_count
    )

    if cluster_count < 1:
        return QuestionSelectionResult(
            source_id=source_id,
            repetition=repetition,
            provider=provider,
            model_id=model_id,
            method="kazdiv",
            selection_seed=-1,
            cluster_count=0,
            selection_probability=0.0,
            selected_cluster_id=-1,
            selected_original_candidate_index=-1,
            selected_question="",
            selected_answer="",
            selected_relevance=float("nan"),
            success=False,
            reason="no_clusters",
        )

    seed = make_selection_seed(
        provider=provider,
        model_id=model_id,
        source_id=source_id,
        repetition=repetition,
    )

    if cluster_count == 1:
        selected_position = 0

    else:
        rng = np.random.default_rng(
            seed
        )

        selected_position = int(
            rng.integers(
                low=0,
                high=cluster_count,
            )
        )

    cluster = (
        clustering_result
        .clusters[
            selected_position
        ]
    )

    return QuestionSelectionResult(
        source_id=source_id,
        repetition=repetition,
        provider=provider,
        model_id=model_id,
        method="kazdiv",
        selection_seed=(
            seed
        ),
        cluster_count=(
            cluster_count
        ),
        selection_probability=(
            float(
                1.0 /
                cluster_count
            )
        ),
        selected_cluster_id=(
            int(
                cluster
                .cluster_id
            )
        ),
        selected_original_candidate_index=(
            int(
                cluster
                .representative_original_index
            )
        ),
        selected_question=(
            cluster
            .representative_question
        ),
        selected_answer=(
            cluster
            .representative_answer
        ),
        selected_relevance=(
            float(
                cluster
                .representative_relevance
            )
        ),
        success=True,
        reason=(
            "single_available_cluster"
            if cluster_count == 1
            else
            "uniform_cluster_selection"
        ),
    )


_seed_a = make_selection_seed(
    "kazllm",
    MODEL_IDS[
        "kazllm"
    ],
    1,
    1,
)

_seed_b = make_selection_seed(
    "kazllm",
    MODEL_IDS[
        "kazllm"
    ],
    1,
    1,
)

assert (
    _seed_a ==
    _seed_b
)

print(
    "Cell 11 stateless selection: PASS"
)

# %%
# ============================================================
# Cell 12. End-to-end Baseline / KAZ-Div runner
# ============================================================

@dataclass
class Experiment2Result:
    experiment: str
    source_id: Any
    source_text: str
    domain: Optional[str]
    provider: str
    model_id: str
    method: str
    repetition: int
    run_success: bool
    final_question: str
    final_answer: str
    final_relevance_score: Optional[float]
    final_quality_pass: Optional[bool]
    answer_in_context: Optional[bool]
    llm_call_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_llm_latency_sec: float
    total_pipeline_latency_sec: float
    raw_candidate_count: Optional[int]
    parsed_candidate_count: Optional[int]
    quality_valid_candidate_count: Optional[int]
    semantic_cluster_count: Optional[int]
    semantic_duplicates_removed: Optional[int]
    selected_cluster_id: Optional[int]
    selected_original_candidate_index: Optional[int]
    selection_seed: Optional[int]
    parse_metadata: Optional[Dict[str, Any]]
    quality_metadata: Optional[Dict[str, Any]]
    clustering_metadata: Optional[Dict[str, Any]]
    selection_metadata: Optional[Dict[str, Any]]
    failure_stage: Optional[str]
    error_message: Optional[str]


def to_json_safe(
    value: Any
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        value = float(
            value
        )

    if isinstance(
        value,
        float,
    ):
        if (
            math.isnan(
                value
            ) or
            math.isinf(
                value
            )
        ):
            return None

        return value

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ):
                to_json_safe(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            to_json_safe(
                item
            )
            for item
            in value
        ]

    if hasattr(
        value,
        "__dataclass_fields__",
    ):
        return to_json_safe(
            asdict(
                value
            )
        )

    return value


def make_generation_seed(
    provider: str,
    model_id: str,
    source_id: Any,
    repetition: int,
    method: str,
    call_index: int,
) -> int:

    material = (
        f"{RANDOM_SEED}|"
        f"{provider}|"
        f"{model_id}|"
        f"{source_id}|"
        f"{repetition}|"
        f"{method}|"
        f"{call_index}"
    )

    digest = hashlib.sha256(
        material.encode(
            "utf-8"
        )
    ).hexdigest()

    return int(
        digest[
            :8
        ],
        16,
    )


def aggregate_llm_results(
    results: List[LLMResult]
) -> Dict[str, Any]:

    return {
        "llm_call_count":
            len(
                results
            ),

        "total_input_tokens":
            sum(
                result.input_tokens
                for result
                in results
            ),

        "total_output_tokens":
            sum(
                result.output_tokens
                for result
                in results
            ),

        "total_tokens":
            sum(
                result.total_tokens
                for result
                in results
            ),

        "total_llm_latency_sec":
            float(
                sum(
                    result.latency_sec
                    for result
                    in results
                )
            ),
    }


def baseline_quality_from_candidate(
    source_text: str,
    candidate: QuestionCandidate,
) -> QuestionQualityRecord:

    analysis = (
        question_semantic_analysis(
            source_text,
            [
                candidate.question
            ],
        )
    )

    relevance = (
        analysis[
            "relevance_scores"
        ][
            0
        ]
    )

    return evaluate_question_candidate(
        source_text=source_text,
        candidate=candidate,
        relevance_score=relevance,
        candidate_index=0,
    )


def failed_result(
    source_id: Any,
    source_text: str,
    domain: Optional[str],
    provider: str,
    model_id: str,
    method: str,
    repetition: int,
    pipeline_start: float,
    llm_results: List[LLMResult],
    failure_stage: str,
    error_message: str,
    parse_metadata=None,
    quality_metadata=None,
    clustering_metadata=None,
) -> Experiment2Result:

    totals = aggregate_llm_results(
        llm_results
    )

    parse_candidate_count = None
    raw_candidate_count = None

    if parse_metadata is not None:
        raw_candidate_count = getattr(
            parse_metadata,
            "raw_candidate_count",
            None,
        )

        parsed = getattr(
            parse_metadata,
            "candidates",
            None,
        )

        if parsed is not None:
            parse_candidate_count = len(
                parsed
            )

    quality_valid = (
        getattr(
            quality_metadata,
            "valid_candidate_count",
            None,
        )
        if quality_metadata
        is not None
        else None
    )

    cluster_count = (
        getattr(
            clustering_metadata,
            "cluster_count",
            None,
        )
        if clustering_metadata
        is not None
        else None
    )

    duplicates_removed = (
        getattr(
            clustering_metadata,
            "semantic_duplicates_removed",
            None,
        )
        if clustering_metadata
        is not None
        else None
    )

    return Experiment2Result(
        experiment=EXPERIMENT_NAME,
        source_id=source_id,
        source_text=source_text,
        domain=domain,
        provider=provider,
        model_id=model_id,
        method=method,
        repetition=repetition,
        run_success=False,
        final_question="",
        final_answer="",
        final_relevance_score=None,
        final_quality_pass=False,
        answer_in_context=None,
        llm_call_count=totals[
            "llm_call_count"
        ],
        total_input_tokens=totals[
            "total_input_tokens"
        ],
        total_output_tokens=totals[
            "total_output_tokens"
        ],
        total_tokens=totals[
            "total_tokens"
        ],
        total_llm_latency_sec=totals[
            "total_llm_latency_sec"
        ],
        total_pipeline_latency_sec=(
            time.perf_counter() -
            pipeline_start
        ),
        raw_candidate_count=(
            raw_candidate_count
        ),
        parsed_candidate_count=(
            parse_candidate_count
        ),
        quality_valid_candidate_count=(
            quality_valid
        ),
        semantic_cluster_count=(
            cluster_count
        ),
        semantic_duplicates_removed=(
            duplicates_removed
        ),
        selected_cluster_id=None,
        selected_original_candidate_index=None,
        selection_seed=None,
        parse_metadata=(
            to_json_safe(
                parse_metadata
            )
        ),
        quality_metadata=(
            to_json_safe(
                quality_metadata
            )
        ),
        clustering_metadata=(
            to_json_safe(
                clustering_metadata
            )
        ),
        selection_metadata=None,
        failure_stage=(
            failure_stage
        ),
        error_message=(
            error_message
        ),
    )


def run_baseline_question(
    source_id: Any,
    source_text: str,
    provider: str,
    repetition: int,
    domain: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Experiment2Result:

    pipeline_start = (
        time.perf_counter()
    )

    provider = (
        provider
        .lower()
        .strip()
    )

    if model_id is None:
        model_id = MODEL_IDS[
            provider
        ]

    generation_seed = (
        make_generation_seed(
            provider=provider,
            model_id=model_id,
            source_id=source_id,
            repetition=repetition,
            method="baseline",
            call_index=1,
        )
    )

    prompt = build_prompt(
        "baseline",
        source_text,
    )

    llm_results = []

    try:
        llm = call_llm(
            provider=provider,
            prompt=prompt,
            model_id=model_id,
            generation_seed=(
                generation_seed
            ),
        )

        llm_results.append(
            llm
        )

    except Exception as exc:
        return failed_result(
            source_id,
            source_text,
            domain,
            provider,
            model_id,
            "baseline",
            repetition,
            pipeline_start,
            llm_results,
            "llm_call",
            str(
                exc
            ),
        )

    parse_result = (
        parse_question_response(
            llm.text,
            expected_k=1,
        )
    )

    if not parse_result.candidates:
        return failed_result(
            source_id,
            source_text,
            domain,
            provider,
            model_id,
            "baseline",
            repetition,
            pipeline_start,
            llm_results,
            "parsing",
            (
                parse_result
                .error_message or
                "No parsed baseline candidate."
            ),
            parse_metadata=(
                parse_result
            ),
        )

    # Baseline is one direct output.
    # If multiple objects are returned, keep only the first.
    candidate = (
        parse_result
        .candidates[
            0
        ]
    )

    quality = (
        baseline_quality_from_candidate(
            source_text,
            candidate,
        )
    )

    totals = (
        aggregate_llm_results(
            llm_results
        )
    )

    # IMPORTANT:
    # Baseline is completed even if quality_pass=False.
    # It is not regenerated.
    return Experiment2Result(
        experiment=EXPERIMENT_NAME,
        source_id=source_id,
        source_text=source_text,
        domain=domain,
        provider=provider,
        model_id=model_id,
        method="baseline",
        repetition=repetition,
        run_success=True,
        final_question=(
            candidate.question
        ),
        final_answer=(
            candidate.answer
        ),
        final_relevance_score=(
            quality.relevance_score
        ),
        final_quality_pass=(
            quality.quality_pass
        ),
        answer_in_context=(
            quality.answer_in_context
        ),
        llm_call_count=totals[
            "llm_call_count"
        ],
        total_input_tokens=totals[
            "total_input_tokens"
        ],
        total_output_tokens=totals[
            "total_output_tokens"
        ],
        total_tokens=totals[
            "total_tokens"
        ],
        total_llm_latency_sec=totals[
            "total_llm_latency_sec"
        ],
        total_pipeline_latency_sec=(
            time.perf_counter() -
            pipeline_start
        ),
        raw_candidate_count=(
            parse_result
            .raw_candidate_count
        ),
        parsed_candidate_count=(
            len(
                parse_result
                .candidates
            )
        ),
        quality_valid_candidate_count=(
            1
            if quality.quality_pass
            else 0
        ),
        semantic_cluster_count=None,
        semantic_duplicates_removed=None,
        selected_cluster_id=None,
        selected_original_candidate_index=None,
        selection_seed=None,
        parse_metadata=(
            to_json_safe(
                parse_result
            )
        ),
        quality_metadata=(
            to_json_safe(
                quality
            )
        ),
        clustering_metadata=None,
        selection_metadata=None,
        failure_stage=None,
        error_message=None,
    )


def run_kazdiv_question(
    source_id: Any,
    source_text: str,
    provider: str,
    repetition: int,
    domain: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Experiment2Result:

    pipeline_start = (
        time.perf_counter()
    )

    provider = (
        provider
        .lower()
        .strip()
    )

    if model_id is None:
        model_id = MODEL_IDS[
            provider
        ]

    llm_results = []

    last_parse = None
    last_quality = None
    last_cluster = None

    selection = None

    last_error = None
    failure_stage = None

    for call_index in range(
        1,
        MAX_KAZDIV_CALLS +
        1,
    ):

        generation_seed = (
            make_generation_seed(
                provider=provider,
                model_id=model_id,
                source_id=source_id,
                repetition=repetition,
                method="kazdiv",
                call_index=call_index,
            )
        )

        prompt = build_prompt(
            "kazdiv",
            source_text,
            k_candidates=(
                K_CANDIDATES
            ),
        )

        try:
            llm = call_llm(
                provider=provider,
                prompt=prompt,
                model_id=model_id,
                generation_seed=(
                    generation_seed
                ),
            )

            llm_results.append(
                llm
            )

        except Exception as exc:
            last_error = str(
                exc
            )
            failure_stage = (
                "llm_call"
            )
            continue

        parse_result = (
            parse_question_response(
                llm.text,
                expected_k=(
                    K_CANDIDATES
                ),
            )
        )

        last_parse = (
            parse_result
        )

        if not parse_result.candidates:
            last_error = (
                parse_result
                .error_message or
                "No candidates parsed."
            )
            failure_stage = (
                "parsing"
            )
            continue

        quality_result = (
            filter_question_candidates(
                source_text,
                parse_result
                .candidates,
            )
        )

        last_quality = (
            quality_result
        )

        if (
            quality_result
            .valid_candidate_count ==
            0
        ):
            last_error = (
                "No candidate passed "
                "quality filtering."
            )
            failure_stage = (
                "quality_filtering"
            )
            continue

        clustering_result = (
            cluster_quality_result(
                quality_result,
                threshold=(
                    QUESTION_NEAR_DUPLICATE_THRESHOLD
                ),
            )
        )

        last_cluster = (
            clustering_result
        )

        if (
            clustering_result
            .cluster_count ==
            0
        ):
            last_error = (
                "No semantic clusters."
            )
            failure_stage = (
                "semantic_clustering"
            )
            continue

        selection = (
            select_question_from_clusters(
                clustering_result=(
                    clustering_result
                ),
                provider=(
                    provider
                ),
                model_id=(
                    model_id
                ),
                source_id=(
                    source_id
                ),
                repetition=(
                    repetition
                ),
            )
        )

        if selection.success:
            last_error = None
            failure_stage = None
            break

        last_error = (
            "Selection failed."
        )
        failure_stage = (
            "selection"
        )

    if (
        selection is None or
        not selection.success
    ):
        return failed_result(
            source_id,
            source_text,
            domain,
            provider,
            model_id,
            "kazdiv",
            repetition,
            pipeline_start,
            llm_results,
            (
                failure_stage or
                "unknown"
            ),
            (
                last_error or
                "KAZ-Div failed."
            ),
            parse_metadata=(
                last_parse
            ),
            quality_metadata=(
                last_quality
            ),
            clustering_metadata=(
                last_cluster
            ),
        )

    final_candidate = (
        QuestionCandidate(
            question=(
                selection
                .selected_question
            ),
            answer=(
                selection
                .selected_answer
            ),
        )
    )

    final_quality = (
        baseline_quality_from_candidate(
            source_text,
            final_candidate,
        )
    )

    totals = (
        aggregate_llm_results(
            llm_results
        )
    )

    return Experiment2Result(
        experiment=EXPERIMENT_NAME,
        source_id=source_id,
        source_text=source_text,
        domain=domain,
        provider=provider,
        model_id=model_id,
        method="kazdiv",
        repetition=repetition,
        run_success=True,
        final_question=(
            selection
            .selected_question
        ),
        final_answer=(
            selection
            .selected_answer
        ),
        final_relevance_score=(
            selection
            .selected_relevance
        ),
        final_quality_pass=(
            final_quality
            .quality_pass
        ),
        answer_in_context=(
            final_quality
            .answer_in_context
        ),
        llm_call_count=totals[
            "llm_call_count"
        ],
        total_input_tokens=totals[
            "total_input_tokens"
        ],
        total_output_tokens=totals[
            "total_output_tokens"
        ],
        total_tokens=totals[
            "total_tokens"
        ],
        total_llm_latency_sec=totals[
            "total_llm_latency_sec"
        ],
        total_pipeline_latency_sec=(
            time.perf_counter() -
            pipeline_start
        ),
        raw_candidate_count=(
            last_parse
            .raw_candidate_count
        ),
        parsed_candidate_count=(
            len(
                last_parse
                .candidates
            )
        ),
        quality_valid_candidate_count=(
            last_quality
            .valid_candidate_count
        ),
        semantic_cluster_count=(
            last_cluster
            .cluster_count
        ),
        semantic_duplicates_removed=(
            last_cluster
            .semantic_duplicates_removed
        ),
        selected_cluster_id=(
            selection
            .selected_cluster_id
        ),
        selected_original_candidate_index=(
            selection
            .selected_original_candidate_index
        ),
        selection_seed=(
            selection
            .selection_seed
        ),
        parse_metadata=(
            to_json_safe(
                last_parse
            )
        ),
        quality_metadata=(
            to_json_safe(
                last_quality
            )
        ),
        clustering_metadata=(
            to_json_safe(
                last_cluster
            )
        ),
        selection_metadata=(
            to_json_safe(
                selection
            )
        ),
        failure_stage=None,
        error_message=None,
    )


def run_experiment2_request(
    source_id: Any,
    source_text: str,
    provider: str,
    method: str,
    repetition: int,
    domain: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Experiment2Result:

    method = (
        method
        .lower()
        .strip()
    )

    if method == "baseline":
        return run_baseline_question(
            source_id=source_id,
            source_text=source_text,
            provider=provider,
            repetition=repetition,
            domain=domain,
            model_id=model_id,
        )

    if method == "kazdiv":
        return run_kazdiv_question(
            source_id=source_id,
            source_text=source_text,
            provider=provider,
            repetition=repetition,
            domain=domain,
            model_id=model_id,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


print(
    "Cell 12 end-to-end runner: READY"
)

# %%
# ============================================================
# Cell 13. Provider smoke test
# ============================================================
# Runs one short Baseline-format request for all four final models.
# This is intentionally enabled in pilot mode and disabled in
# full mode to avoid unnecessary extra API calls.
# ============================================================

RUN_PROVIDER_SMOKE_TEST = (
    RUN_MODE == "pilot"
)

SMOKE_SOURCE_INDEX = 0


if RUN_PROVIDER_SMOKE_TEST:
    smoke_source = normalize_text(
        question_sources.iloc[
            SMOKE_SOURCE_INDEX
        ]["text"]
    )

    smoke_prompt = build_prompt(
        "baseline",
        smoke_source,
    )

    smoke_results = {}

    for provider in ENABLED_PROVIDERS:
        print("\n" + "=" * 70)
        print("TEST:", provider)
        print("MODEL:", MODEL_IDS[provider])

        try:
            result = call_llm(
                provider=provider,
                prompt=smoke_prompt,
                model_id=MODEL_IDS[provider],
                generation_seed=RANDOM_SEED,
            )

            smoke_results[provider] = True
            print("STATUS: PASS")
            print("OUTPUT:", result.text[:500])

        except Exception as exc:
            smoke_results[provider] = False
            print("STATUS: FAILED")
            print(
                "ERROR:",
                type(exc).__name__,
                str(exc),
            )

    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)

    for provider in ENABLED_PROVIDERS:
        status = (
            "PASS"
            if smoke_results.get(provider)
            else "FAILED"
        )
        print(f"{provider:12s}: {status}")

    ALL_PROVIDERS_READY = all(
        smoke_results.get(provider, False)
        for provider in ENABLED_PROVIDERS
    )

    print(
        "\nALL PROVIDERS READY:",
        ALL_PROVIDERS_READY,
    )

    if not ALL_PROVIDERS_READY:
        raise RuntimeError(
            "At least one provider failed the smoke test. "
            "Do not start pilot/main collection until all "
            "enabled providers pass."
        )

else:
    print(
        "Provider smoke test skipped in full mode."
    )

# %%
# ============================================================
# Cell 14. Experiment 2 runner — pilot / full
# Checkpoint + resume + provider isolation
# ============================================================
# Pilot:
#   2 source contexts/domain × 5 domains × 4 models
#   × KAZ-Div only × 1 repetition = 40 observations.
#
# Full:
#   100 source contexts × 4 models × 2 methods
#   × 5 repetitions = 4000 observations.
#
# Pilot and full use different OUTPUT_DIR folders.
# ============================================================

MAX_CONSECUTIVE_FAILURES_PER_PROVIDER = 3
AUTO_ZIP_EVERY = 25


# ------------------------------------------------------------
# Mode-specific task design
# ------------------------------------------------------------

if RUN_MODE == "pilot":
    pilot_parts = []

    for domain in EXPECTED_DOMAINS:
        subset = (
            question_sources[
                question_sources["domain"] == domain
            ]
            .sort_values("id")
            .head(PILOT_SOURCES_PER_DOMAIN)
            .copy()
        )

        if len(subset) < PILOT_SOURCES_PER_DOMAIN:
            raise RuntimeError(
                f"Pilot requires at least {PILOT_SOURCES_PER_DOMAIN} "
                f"source(s) for domain '{domain}', found {len(subset)}."
            )

        pilot_parts.append(subset)

    TASK_SOURCES = (
        pd.concat(
            pilot_parts,
            ignore_index=True,
        )
        .sort_values(["domain", "id"])
        .reset_index(drop=True)
    )

    PROVIDERS_TO_RUN = list(ENABLED_PROVIDERS)
    METHODS_TO_RUN = ["kazdiv"]
    REPETITIONS_TO_RUN = [PILOT_REPETITION]

else:
    TASK_SOURCES = question_sources.copy()
    PROVIDERS_TO_RUN = list(ENABLED_PROVIDERS)
    METHODS_TO_RUN = list(METHODS)
    REPETITIONS_TO_RUN = list(
        range(1, N_REPEATS + 1)
    )


# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

RESULTS_JSONL_FILE = (
    OUTPUT_DIR / "experiment2_results.jsonl"
)
RESULTS_CSV_FILE = (
    OUTPUT_DIR / "experiment2_results_summary.csv"
)
FAILURES_JSONL_FILE = (
    OUTPUT_DIR / "experiment2_failures.jsonl"
)
RUNNER_ERRORS_JSONL_FILE = (
    OUTPUT_DIR / "experiment2_runner_errors.jsonl"
)
CHECKPOINT_FILE = (
    OUTPUT_DIR / "experiment2_checkpoint.json"
)
SIGNATURE_FILE = (
    OUTPUT_DIR / "experiment2_signature.json"
)
COMPLETION_REPORT_FILE = (
    OUTPUT_DIR / "experiment2_completion_report.json"
)

LOCAL_ZIP_FILE = Path(
    f"/content/KAZ_Div_Experiment2_{RUN_MODE}_latest.zip"
)


CSV_COLUMNS = [
    "experiment",
    "task_key",
    "source_id",
    "domain",
    "provider",
    "model_id",
    "method",
    "repetition",
    "run_success",
    "final_question",
    "final_answer",
    "final_relevance_score",
    "final_quality_pass",
    "answer_in_context",
    "llm_call_count",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "total_llm_latency_sec",
    "total_pipeline_latency_sec",
    "raw_candidate_count",
    "parsed_candidate_count",
    "quality_valid_candidate_count",
    "semantic_cluster_count",
    "semantic_duplicates_removed",
    "selected_cluster_id",
    "selected_original_candidate_index",
    "selection_seed",
    "failure_stage",
    "error_message",
]


def write_json_atomic(
    path: Path,
    data: Dict[str, Any],
):
    temp_path = Path(str(path) + ".tmp")

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, path)


def append_jsonl(
    path: Path,
    record: Dict[str, Any],
):
    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def make_task_key(
    source_id: Any,
    provider: str,
    model_id: str,
    method: str,
    repetition: int,
) -> str:
    material = (
        f"{source_id}|{provider}|{model_id}|{method}|{repetition}"
    )
    return hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()


def build_tasks() -> List[Dict[str, Any]]:
    tasks = []

    for _, row in TASK_SOURCES.iterrows():
        for repetition in REPETITIONS_TO_RUN:
            for provider in PROVIDERS_TO_RUN:
                model_id = MODEL_IDS[provider]

                for method in METHODS_TO_RUN:
                    task_key = make_task_key(
                        source_id=row["id"],
                        provider=provider,
                        model_id=model_id,
                        method=method,
                        repetition=repetition,
                    )

                    tasks.append(
                        {
                            "task_key": task_key,
                            "source_id": row["id"],
                            "source_text": row["text"],
                            "domain": row["domain"],
                            "provider": provider,
                            "model_id": model_id,
                            "method": method,
                            "repetition": repetition,
                        }
                    )

    return tasks


ALL_TASKS = build_tasks()
TOTAL_TASK_COUNT = len(ALL_TASKS)


# ------------------------------------------------------------
# Design verification
# ------------------------------------------------------------

if RUN_MODE == "pilot":
    expected_pilot = (
        len(EXPECTED_DOMAINS)
        * PILOT_SOURCES_PER_DOMAIN
        * len(PROVIDERS_TO_RUN)
        * 1
        * 1
    )

    assert TOTAL_TASK_COUNT == expected_pilot, (
        f"Unexpected pilot task count: {TOTAL_TASK_COUNT}; "
        f"expected {expected_pilot}."
    )

    print(
        "Pilot design verified:",
        TOTAL_TASK_COUNT,
        "observations",
    )

else:
    if len(question_sources) != 100:
        raise RuntimeError(
            "Full experiment requires exactly 100 source contexts."
        )

    assert len(PROVIDERS_TO_RUN) == 4
    assert len(METHODS_TO_RUN) == 2
    assert len(REPETITIONS_TO_RUN) == 5
    assert TOTAL_TASK_COUNT == 4000

    print(
        "Full design verified: 4000 observations"
    )


# ------------------------------------------------------------
# Signature — scoped to pilot OR main directory
# ------------------------------------------------------------

SIGNATURE_CONFIG = {
    "experiment": EXPERIMENT_NAME,
    "run_mode": RUN_MODE,
    "random_seed": RANDOM_SEED,
    "model_ids": {
        provider: MODEL_IDS[provider]
        for provider in PROVIDERS_TO_RUN
    },
    "providers": PROVIDERS_TO_RUN,
    "methods": METHODS_TO_RUN,
    "repetitions": REPETITIONS_TO_RUN,
    "source_ids": [
        str(value)
        for value in TASK_SOURCES["id"].tolist()
    ],
    "k_candidates": K_CANDIDATES,
    "max_kazdiv_calls": MAX_KAZDIV_CALLS,
    "near_duplicate_threshold": (
        QUESTION_NEAR_DUPLICATE_THRESHOLD
    ),
    "threshold_frozen": THRESHOLD_FROZEN,
    "question_word_range": [
        MIN_QUESTION_WORDS,
        MAX_QUESTION_WORDS,
    ],
    "answer_word_range": [
        MIN_ANSWER_WORDS,
        MAX_ANSWER_WORDS,
    ],
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_pooling": "CLS",
    "embedding_normalization_dtype": "float32",
    "kazllm_do_sample": KAZLLM_DO_SAMPLE,
    "kazllm_temperature": KAZLLM_TEMPERATURE,
    "kazllm_top_p": KAZLLM_TOP_P,
    "kazllm_commit_hash": KAZLLM_COMMIT_HASH,
    "alemllm_served_model": ALEMLLM_SERVED_MODEL,
    "alemllm_temperature": ALEMLLM_TEMPERATURE,
    "alemllm_top_p": ALEMLLM_TOP_P,
    "alemllm_endpoint_fingerprint": ALEMLLM_ENDPOINT_FINGERPRINT,
    "bge_commit_hash": BGE_COMMIT_HASH,
    "baseline_prompt_template": BASELINE_PROMPT_TEMPLATE,
    "kazdiv_prompt_template": KAZDIV_PROMPT_TEMPLATE,
}

SIGNATURE_TEXT = json.dumps(
    SIGNATURE_CONFIG,
    ensure_ascii=False,
    sort_keys=True,
    default=str,
)

EXPERIMENT_SIGNATURE = hashlib.sha256(
    SIGNATURE_TEXT.encode("utf-8")
).hexdigest()


def validate_signature():
    current = {
        "signature": EXPERIMENT_SIGNATURE,
        "updated_at": datetime.now().isoformat(),
        "configuration": SIGNATURE_CONFIG,
    }

    if not SIGNATURE_FILE.exists():
        write_json_atomic(
            SIGNATURE_FILE,
            current,
        )
        print(
            f"{RUN_MODE.capitalize()} signature created."
        )
        return

    with open(
        SIGNATURE_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        saved = json.load(f)

    if saved.get("signature") != EXPERIMENT_SIGNATURE:
        raise RuntimeError(
            f"{RUN_MODE.capitalize()} configuration changed. "
            f"Existing results are in {OUTPUT_DIR}. "
            "Archive/reset this run directory before collecting "
            "data with a different configuration."
        )


def archive_and_reset_current_run():
    """Manual helper. Call only when intentionally restarting this mode."""

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    archive_root = OUTPUT_ROOT / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    backup_dir = archive_root / (
        f"{RUN_MODE}_{timestamp}"
    )

    shutil.copytree(
        OUTPUT_DIR,
        backup_dir,
    )

    shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "Archived old run to:",
        backup_dir,
    )
    print(
        "Current run directory reset:",
        OUTPUT_DIR,
    )

    return backup_dir


# ------------------------------------------------------------
# Resume helpers
# ------------------------------------------------------------


def load_successful_results():
    successful = {}

    if not RESULTS_JSONL_FILE.exists():
        return successful

    with open(
        RESULTS_JSONL_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception:
                continue

            key = record.get("task_key")

            if (
                key
                and record.get("run_success") is True
            ):
                successful[key] = record

    return successful


def csv_row(
    task_key: str,
    result: Dict[str, Any],
):
    row = {}

    for column in CSV_COLUMNS:
        value = (
            task_key
            if column == "task_key"
            else result.get(column)
        )

        if isinstance(value, (dict, list)):
            value = json.dumps(
                value,
                ensure_ascii=False,
            )

        row[column] = value

    return row


def rebuild_csv(
    successful_results: Dict[str, Dict[str, Any]],
):
    with open(
        RESULTS_CSV_FILE,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
        )
        writer.writeheader()

        for task in ALL_TASKS:
            key = task["task_key"]
            if key in successful_results:
                writer.writerow(
                    csv_row(
                        key,
                        successful_results[key],
                    )
                )


def append_csv(
    task_key: str,
    result: Dict[str, Any],
):
    exists = (
        RESULTS_CSV_FILE.exists()
        and RESULTS_CSV_FILE.stat().st_size > 0
    )

    with open(
        RESULTS_CSV_FILE,
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            csv_row(task_key, result)
        )
        f.flush()
        os.fsync(f.fileno())


def save_checkpoint(
    completed: int,
    success_session: int,
    failure_session: int,
    current_task=None,
    status="running",
    blocked_providers=None,
):
    write_json_atomic(
        CHECKPOINT_FILE,
        {
            "experiment": EXPERIMENT_NAME,
            "run_mode": RUN_MODE,
            "signature": EXPERIMENT_SIGNATURE,
            "updated_at": datetime.now().isoformat(),
            "status": status,
            "total_tasks": TOTAL_TASK_COUNT,
            "completed": completed,
            "remaining": max(
                0,
                TOTAL_TASK_COUNT - completed,
            ),
            "success_this_session": success_session,
            "failure_this_session": failure_session,
            "blocked_providers": sorted(
                blocked_providers or []
            ),
            "current_task": current_task,
        },
    )


def create_zip_snapshot():
    base_name = str(
        LOCAL_ZIP_FILE.with_suffix("")
    )

    if LOCAL_ZIP_FILE.exists():
        LOCAL_ZIP_FILE.unlink()

    zip_path = shutil.make_archive(
        base_name=base_name,
        format="zip",
        root_dir=str(OUTPUT_DIR.parent),
        base_dir=OUTPUT_DIR.name,
    )

    return Path(zip_path)


def save_completion_report(
    successful_results,
    blocked_providers=None,
):
    completed = len(successful_results)

    report = {
        "experiment": EXPERIMENT_NAME,
        "run_mode": RUN_MODE,
        "signature": EXPERIMENT_SIGNATURE,
        "created_at": datetime.now().isoformat(),
        "expected": TOTAL_TASK_COUNT,
        "successful": completed,
        "remaining": max(
            0,
            TOTAL_TASK_COUNT - completed,
        ),
        "complete": (
            completed == TOTAL_TASK_COUNT
        ),
        "blocked_providers": sorted(
            blocked_providers or []
        ),
    }

    write_json_atomic(
        COMPLETION_REPORT_FILE,
        report,
    )

    return report


# ------------------------------------------------------------
# Validate + resume
# ------------------------------------------------------------

validate_signature()

successful_results = load_successful_results()
rebuild_csv(successful_results)
COMPLETED_KEYS = set(successful_results.keys())


# ------------------------------------------------------------
# Main runner
# ------------------------------------------------------------


def run_experiment2():
    global successful_results
    global COMPLETED_KEYS

    pending = [
        task
        for task in ALL_TASKS
        if task["task_key"] not in COMPLETED_KEYS
    ]

    session_success = 0
    session_failure = 0

    provider_failure_streak = defaultdict(int)
    blocked_providers = set()
    skipped_blocked = 0

    print("\n" + "=" * 72)
    print("EXPERIMENT 2 RUN")
    print("=" * 72)
    print("Mode:", RUN_MODE)
    print("Output:", OUTPUT_DIR)
    print("Total task count:", TOTAL_TASK_COUNT)
    print("Already completed:", len(COMPLETED_KEYS))
    print("Pending at session start:", len(pending))

    for session_index, task in enumerate(
        pending,
        start=1,
    ):
        provider = task["provider"]

        if provider in blocked_providers:
            skipped_blocked += 1
            continue

        current_task = {
            "session_index": session_index,
            "session_total": len(pending),
            "source_id": task["source_id"],
            "provider": provider,
            "model_id": task["model_id"],
            "method": task["method"],
            "repetition": task["repetition"],
            "task_key": task["task_key"],
        }

        save_checkpoint(
            completed=len(COMPLETED_KEYS),
            success_session=session_success,
            failure_session=session_failure,
            current_task=current_task,
            blocked_providers=blocked_providers,
        )

        print(
            f"\n[{session_index}/{len(pending)}] "
            f"source={task['source_id']} | "
            f"{provider} | {task['method']} | "
            f"rep={task['repetition']}"
        )

        try:
            result = run_experiment2_request(
                source_id=task["source_id"],
                source_text=task["source_text"],
                provider=provider,
                method=task["method"],
                repetition=task["repetition"],
                domain=task["domain"],
                model_id=task["model_id"],
            )

            result_dict = to_json_safe(result)

        except Exception as exc:
            session_failure += 1
            provider_failure_streak[provider] += 1

            append_jsonl(
                RUNNER_ERRORS_JSONL_FILE,
                {
                    "timestamp": datetime.now().isoformat(),
                    "task": current_task,
                    "exception_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

            print(
                "RUNNER ERROR:",
                type(exc).__name__,
                str(exc),
            )

            if (
                provider_failure_streak[provider]
                >= MAX_CONSECUTIVE_FAILURES_PER_PROVIDER
            ):
                blocked_providers.add(provider)
                print(
                    f"Provider '{provider}' blocked for the rest of "
                    "this session after consecutive failures. "
                    "Other providers will continue. Rerun Cell 14 "
                    "later to resume its unfinished tasks."
                )

            continue

        result_dict["task_key"] = task["task_key"]
        result_dict["saved_at"] = datetime.now().isoformat()
        result_dict["experiment_signature"] = EXPERIMENT_SIGNATURE
        result_dict["run_mode"] = RUN_MODE

        if result_dict.get("run_success") is True:
            append_jsonl(
                RESULTS_JSONL_FILE,
                result_dict,
            )

            append_csv(
                task["task_key"],
                result_dict,
            )

            successful_results[
                task["task_key"]
            ] = result_dict

            COMPLETED_KEYS.add(
                task["task_key"]
            )

            session_success += 1
            provider_failure_streak[provider] = 0

            print(
                "SUCCESS | "
                f"{len(COMPLETED_KEYS)}/{TOTAL_TASK_COUNT}"
            )

            if (
                AUTO_ZIP_EVERY is not None
                and len(COMPLETED_KEYS) % AUTO_ZIP_EVERY == 0
            ):
                zip_path = create_zip_snapshot()
                print("ZIP:", zip_path)

        else:
            session_failure += 1
            provider_failure_streak[provider] += 1

            append_jsonl(
                FAILURES_JSONL_FILE,
                result_dict,
            )

            print(
                "FAILED:",
                result_dict.get("failure_stage"),
                result_dict.get("error_message"),
            )

            if (
                provider_failure_streak[provider]
                >= MAX_CONSECUTIVE_FAILURES_PER_PROVIDER
            ):
                blocked_providers.add(provider)
                print(
                    f"Provider '{provider}' blocked for this session."
                )

        save_checkpoint(
            completed=len(COMPLETED_KEYS),
            success_session=session_success,
            failure_session=session_failure,
            current_task=current_task,
            blocked_providers=blocked_providers,
        )

    rebuild_csv(successful_results)

    report = save_completion_report(
        successful_results,
        blocked_providers=blocked_providers,
    )

    save_checkpoint(
        completed=len(COMPLETED_KEYS),
        success_session=session_success,
        failure_session=session_failure,
        current_task=None,
        status=(
            "complete"
            if report["complete"]
            else "partial"
        ),
        blocked_providers=blocked_providers,
    )

    zip_path = create_zip_snapshot()

    print("\n" + "=" * 72)
    print("SESSION FINISHED")
    print("=" * 72)
    print("Mode:", RUN_MODE)
    print("Successful this session:", session_success)
    print("Failed this session:", session_failure)
    print("Skipped because provider blocked:", skipped_blocked)
    print("Blocked providers:", sorted(blocked_providers))
    print("Completed overall:", len(COMPLETED_KEYS))
    print("Expected:", TOTAL_TASK_COUNT)
    print(
        "Remaining:",
        TOTAL_TASK_COUNT - len(COMPLETED_KEYS),
    )
    print("CSV:", RESULTS_CSV_FILE)
    print("JSONL:", RESULTS_JSONL_FILE)
    print("ZIP:", zip_path)

    return report


print("\n" + "=" * 72)
print("EXPERIMENT 2 CONFIGURATION")
print("=" * 72)
print("Mode:", RUN_MODE)
print("Providers:", PROVIDERS_TO_RUN)
print("Methods:", METHODS_TO_RUN)
print("Repetitions:", REPETITIONS_TO_RUN)
print("Task sources:", len(TASK_SOURCES))
print("Expected task count:", TOTAL_TASK_COUNT)
print("Already completed:", len(COMPLETED_KEYS))

FINAL_EXPERIMENT2_REPORT = run_experiment2()

# %%
# ============================================================
# Cell 15. Pilot threshold calibration
# ============================================================
# Always reads from PILOT_OUTPUT_DIR, even if the current
# RUN_MODE has later been changed to "full".
#
# This cell DOES NOT choose a threshold automatically.
# Inspect the table, justify the choice, then freeze it in
# Cell 3 before the main experiment.
# ============================================================

CALIBRATION_THRESHOLDS = [
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
]

PILOT_RESULTS_JSONL_FILE = (
    PILOT_OUTPUT_DIR /
    "experiment2_results.jsonl"
)


def load_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:
    records = []

    if not path.exists():
        return records

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(record, dict):
                records.append(record)

    return records


pilot_records = load_jsonl(
    PILOT_RESULTS_JSONL_FILE
)

kazdiv_records = [
    record
    for record in pilot_records
    if (
        str(record.get("method", "")).strip().lower()
        == "kazdiv"
        and record.get("run_success") is True
    )
]

print(
    "Pilot successful records:",
    len(pilot_records),
)
print(
    "Successful KAZ-Div pilot records:",
    len(kazdiv_records),
)

if not kazdiv_records:
    raise RuntimeError(
        "No successful KAZ-Div pilot results were found in: "
        f"{PILOT_RESULTS_JSONL_FILE}. Run Cell 14 in pilot mode first."
    )


calibration_rows = []

for record in kazdiv_records:
    quality_metadata = (
        record.get("quality_metadata") or {}
    )

    valid_candidates = (
        quality_metadata.get("valid_candidates") or []
    )

    questions = []

    for item in valid_candidates:
        if isinstance(item, dict):
            question = normalize_text(
                item.get("question", "")
            )
        elif isinstance(item, str):
            question = normalize_text(item)
        else:
            question = ""

        if question:
            questions.append(question)

    # Preserve order while removing exact duplicate strings.
    questions = list(dict.fromkeys(questions))

    if len(questions) < 2:
        continue

    source_text = normalize_text(
        record.get("source_text", "")
    )

    if not source_text:
        continue

    analysis = question_semantic_analysis(
        source_text,
        questions,
    )

    matrix = analysis["similarity_matrix"]

    upper_values = matrix[
        np.triu_indices(
            len(questions),
            k=1,
        )
    ]

    for threshold in CALIBRATION_THRESHOLDS:
        graph = build_duplicate_graph(
            matrix,
            threshold,
        )

        components = connected_components(
            graph
        )

        n_candidates = len(questions)
        n_clusters = len(components)
        duplicates_removed = (
            n_candidates - n_clusters
        )

        near_duplicate_pairs = int(
            np.sum(
                upper_values >= threshold
            )
        )

        calibration_rows.append(
            {
                "source_id": record.get("source_id"),
                "domain": record.get("domain"),
                "provider": record.get("provider"),
                "repetition": record.get("repetition"),
                "threshold": float(threshold),
                "valid_candidates": n_candidates,
                "clusters": n_clusters,
                "duplicates_removed": duplicates_removed,
                "retention_rate": (
                    n_clusters / n_candidates
                ),
                "near_duplicate_pairs": near_duplicate_pairs,
                "semantic_diversity": analysis.get(
                    "semantic_diversity"
                ),
            }
        )


calibration_df = pd.DataFrame(
    calibration_rows
)

if calibration_df.empty:
    raise RuntimeError(
        "Pilot records exist, but no record had at least two "
        "usable quality-valid candidates for calibration."
    )


# Overall threshold summary.
calibration_summary = (
    calibration_df
    .groupby("threshold")
    .agg(
        pools=("source_id", "count"),
        mean_valid_candidates=(
            "valid_candidates",
            "mean",
        ),
        mean_clusters=("clusters", "mean"),
        mean_duplicates_removed=(
            "duplicates_removed",
            "mean",
        ),
        mean_retention_rate=(
            "retention_rate",
            "mean",
        ),
        mean_near_duplicate_pairs=(
            "near_duplicate_pairs",
            "mean",
        ),
        mean_semantic_diversity=(
            "semantic_diversity",
            "mean",
        ),
    )
    .reset_index()
)

calibration_summary[
    "mean_retention_percent"
] = (
    calibration_summary[
        "mean_retention_rate"
    ]
    * 100.0
)


# Provider-specific summary for robustness checks.
calibration_by_provider = (
    calibration_df
    .groupby(
        ["provider", "threshold"]
    )
    .agg(
        pools=("source_id", "count"),
        mean_valid_candidates=(
            "valid_candidates",
            "mean",
        ),
        mean_clusters=("clusters", "mean"),
        mean_duplicates_removed=(
            "duplicates_removed",
            "mean",
        ),
        mean_retention_rate=(
            "retention_rate",
            "mean",
        ),
    )
    .reset_index()
)

print("\nOVERALL CALIBRATION")
display(calibration_summary)

print("\nCALIBRATION BY PROVIDER")
display(calibration_by_provider)

calibration_df.to_csv(
    PILOT_OUTPUT_DIR /
    "experiment2_threshold_calibration_detailed.csv",
    index=False,
    encoding="utf-8-sig",
)

calibration_summary.to_csv(
    PILOT_OUTPUT_DIR /
    "experiment2_threshold_calibration_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

calibration_by_provider.to_csv(
    PILOT_OUTPUT_DIR /
    "experiment2_threshold_calibration_by_provider.csv",
    index=False,
    encoding="utf-8-sig",
)

print(
    "\nCalibration files saved to:",
    PILOT_OUTPUT_DIR,
)

print(
    "\nNEXT STEP: choose and justify the threshold, then in Cell 3 set:\n"
    "  QUESTION_NEAR_DUPLICATE_THRESHOLD = <frozen value>\n"
    "  THRESHOLD_FROZEN = True\n"
    "  RUN_MODE = 'full'\n"
    "Then rerun Cell 3 onward. Pilot files remain separate."
)

# %%
# ============================================================
# Cell 16. Experiment 2 metrics
# ============================================================

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction,
)


def tokenize_simple(
    text: str
) -> List[str]:

    text = normalize_text(
        text
    ).lower()

    return re.findall(
        r"\b[\wәіңғүұқөһ]+\b",
        text,
        flags=re.UNICODE,
    )


def unique_rate(
    texts: List[str]
) -> float:

    if not texts:
        return np.nan

    canonical = [
        normalize_for_exact_match(
            text
        )
        for text
        in texts
    ]

    return float(
        len(
            set(
                canonical
            )
        ) /
        len(
            canonical
        )
    )


def distinct_n(
    texts: List[str],
    n: int,
) -> float:

    ngrams = []

    for text in texts:
        tokens = tokenize_simple(
            text
        )

        if len(
            tokens
        ) < n:
            continue

        ngrams.extend(
            tuple(
                tokens[
                    index:
                    index + n
                ]
            )
            for index
            in range(
                len(
                    tokens
                ) -
                n +
                1
            )
        )

    if not ngrams:
        return np.nan

    return float(
        len(
            set(
                ngrams
            )
        ) /
        len(
            ngrams
        )
    )


def pairwise_semantic_diversity(
    texts: List[str]
) -> float:

    if len(
        texts
    ) < 2:
        return np.nan

    embeddings = encode_texts(
        texts
    )

    matrix = (
        embeddings @
        embeddings.T
    )

    values = (
        matrix[
            np.triu_indices(
                len(
                    texts
                ),
                k=1,
            )
        ]
    )

    return float(
        np.mean(
            1.0 -
            values
        )
    )


def self_bleu(
    texts: List[str]
) -> float:

    if len(
        texts
    ) < 2:
        return np.nan

    tokenized = [
        tokenize_simple(
            text
        )
        for text
        in texts
    ]

    smoothing = (
        SmoothingFunction()
        .method1
    )

    scores = []

    for index, hypothesis in enumerate(
        tokenized
    ):
        references = [
            tokenized[
                other
            ]
            for other
            in range(
                len(
                    tokenized
                )
            )
            if other !=
            index
        ]

        if (
            not hypothesis or
            not references
        ):
            continue

        score = sentence_bleu(
            references,
            hypothesis,
            weights=(
                0.5,
                0.5,
                0.0,
                0.0,
            ),
            smoothing_function=(
                smoothing
            ),
        )

        scores.append(
            float(
                score
            )
        )

    return (
        float(
            np.mean(
                scores
            )
        )
        if scores
        else np.nan
    )


QUESTION_TYPE_PREFIXES = {
    "who":
        [
            "кім",
            "кімдер",
        ],

    "what":
        [
            "не",
            "нені",
            "ненің",
            "немен",
        ],

    "where":
        [
            "қайда",
            "қай жерде",
        ],

    "when":
        [
            "қашан",
        ],

    "why":
        [
            "неліктен",
            "не себепті",
            "неге",
        ],

    "how":
        [
            "қалай",
        ],

    "which":
        [
            "қай",
            "қандай",
        ],

    "how_many":
        [
            "қанша",
            "неше",
        ],
}


def detect_question_type(
    question: str
) -> str:

    question_norm = (
        normalize_text(
            question
        )
        .lower()
    )

    for label, prefixes in (
        QUESTION_TYPE_PREFIXES
        .items()
    ):
        for prefix in prefixes:
            if (
                question_norm.startswith(
                    prefix +
                    " "
                ) or
                (
                    " " +
                    prefix +
                    " "
                ) in
                question_norm
            ):
                return label

    return "other"


def entropy_from_labels(
    labels: List[str]
) -> float:

    if not labels:
        return np.nan

    counts = Counter(
        labels
    )

    probabilities = np.array(
        [
            count /
            len(
                labels
            )
            for count
            in counts.values()
        ],
        dtype=float,
    )

    return float(
        -np.sum(
            probabilities *
            np.log2(
                probabilities
            )
        )
    )


if not RESULTS_CSV_FILE.exists():
    raise FileNotFoundError(
        "Run Experiment 2 first."
    )


results_df = pd.read_csv(
    RESULTS_CSV_FILE
)


results_df = (
    results_df[
        results_df[
            "run_success"
        ] ==
        True
    ]
    .copy()
)


metric_rows = []


for (
    source_id,
    provider,
    method,
), group in results_df.groupby(
    [
        "source_id",
        "provider",
        "method",
    ]
):

    questions = [
        normalize_text(
            value
        )
        for value
        in group[
            "final_question"
        ]
        .dropna()
        .tolist()
        if normalize_text(
            value
        )
    ]

    if not questions:
        continue

    labels = [
        detect_question_type(
            question
        )
        for question
        in questions
    ]

    answer_validity = (
        group[
            "answer_in_context"
        ]
        .astype(
            str
        )
        .str
        .lower()
        .map(
            {
                "true":
                    1.0,

                "false":
                    0.0,
            }
        )
        .mean()
    )

    relevance = (
        pd.to_numeric(
            group[
                "final_relevance_score"
            ],
            errors="coerce",
        )
        .mean()
    )

    metric_rows.append(
        {
            "source_id":
                source_id,

            "provider":
                provider,

            "method":
                method,

            "n_outputs":
                len(
                    questions
                ),

            "unique_rate":
                unique_rate(
                    questions
                ),

            "distinct_1":
                distinct_n(
                    questions,
                    1,
                ),

            "distinct_2":
                distinct_n(
                    questions,
                    2,
                ),

            "semantic_diversity":
                pairwise_semantic_diversity(
                    questions
                ),

            "self_bleu":
                self_bleu(
                    questions
                ),

            "question_type_entropy":
                entropy_from_labels(
                    labels
                ),

            "answer_span_validity":
                answer_validity,

            "mean_relevance":
                relevance,

            "mean_total_tokens":
                pd.to_numeric(
                    group[
                        "total_tokens"
                    ],
                    errors="coerce",
                )
                .mean(),

            "mean_latency_sec":
                pd.to_numeric(
                    group[
                        "total_pipeline_latency_sec"
                    ],
                    errors="coerce",
                )
                .mean(),
        }
    )


source_level_metrics = pd.DataFrame(
    metric_rows
)


source_level_metrics.to_csv(
    OUTPUT_DIR /
    "experiment2_source_level_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)


if len(
    source_level_metrics
):
    summary_metrics = (
        source_level_metrics
        .groupby(
            [
                "provider",
                "method",
            ]
        )
        .agg(
            sources=(
                "source_id",
                "count",
            ),

            unique_rate_mean=(
                "unique_rate",
                "mean",
            ),

            distinct_1_mean=(
                "distinct_1",
                "mean",
            ),

            distinct_2_mean=(
                "distinct_2",
                "mean",
            ),

            semantic_diversity_mean=(
                "semantic_diversity",
                "mean",
            ),

            self_bleu_mean=(
                "self_bleu",
                "mean",
            ),

            question_type_entropy_mean=(
                "question_type_entropy",
                "mean",
            ),

            answer_span_validity_mean=(
                "answer_span_validity",
                "mean",
            ),

            relevance_mean=(
                "mean_relevance",
                "mean",
            ),

            total_tokens_mean=(
                "mean_total_tokens",
                "mean",
            ),

            latency_mean=(
                "mean_latency_sec",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_metrics.to_csv(
        OUTPUT_DIR /
        "experiment2_summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display(
        summary_metrics
    )

else:
    print(
        "No successful results available."
    )

# %%
# ============================================================
# Cell 17. Paired statistical comparison
# NumPy-only bootstrap + sign-flip permutation
# ============================================================

STAT_METRICS = [
    "unique_rate",
    "distinct_1",
    "distinct_2",
    "semantic_diversity",
    "self_bleu",
    "question_type_entropy",
    "answer_span_validity",
    "mean_relevance",
]


N_BOOTSTRAP = 10000
N_PERMUTATIONS = 20000


def paired_bootstrap_ci(
    differences: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(
            differences
        )
    ]

    if len(
        differences
    ) == 0:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    n = len(
        differences
    )

    boot_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for index in range(
        n_bootstrap
    ):
        sample = rng.choice(
            differences,
            size=n,
            replace=True,
        )

        boot_means[
            index
        ] = np.mean(
            sample
        )

    return (
        float(
            np.quantile(
                boot_means,
                0.025,
            )
        ),
        float(
            np.quantile(
                boot_means,
                0.975,
            )
        ),
    )


def paired_sign_flip_pvalue(
    differences: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> float:

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(
            differences
        )
    ]

    if len(
        differences
    ) == 0:
        return np.nan

    observed = abs(
        np.mean(
            differences
        )
    )

    rng = np.random.default_rng(
        seed
    )

    extreme = 0

    for _ in range(
        n_permutations
    ):
        signs = rng.choice(
            [
                -1.0,
                1.0,
            ],
            size=len(
                differences
            ),
        )

        permuted = abs(
            np.mean(
                differences *
                signs
            )
        )

        if permuted >= observed:
            extreme += 1

    return float(
        (
            extreme +
            1
        ) /
        (
            n_permutations +
            1
        )
    )


def holm_adjust(
    pvalues: List[float]
) -> List[float]:

    values = np.asarray(
        pvalues,
        dtype=float,
    )

    adjusted = np.full(
        len(
            values
        ),
        np.nan,
        dtype=float,
    )

    valid = np.where(
        np.isfinite(
            values
        )
    )[0]

    if len(
        valid
    ) == 0:
        return adjusted.tolist()

    order = valid[
        np.argsort(
            values[
                valid
            ]
        )
    ]

    total = len(
        order
    )

    running_max = 0.0

    for rank, index in enumerate(
        order
    ):
        candidate = (
            (
                total -
                rank
            ) *
            values[
                index
            ]
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[
            index
        ] = min(
            1.0,
            running_max,
        )

    return adjusted.tolist()


if "source_level_metrics" not in globals():
    raise RuntimeError(
        "Run Cell 16 metrics first."
    )


stat_rows = []


for provider in sorted(
    source_level_metrics[
        "provider"
    ]
    .dropna()
    .unique()
):

    provider_df = (
        source_level_metrics[
            source_level_metrics[
                "provider"
            ] ==
            provider
        ]
    )

    for metric in (
        STAT_METRICS
    ):
        pivot = provider_df.pivot(
            index="source_id",
            columns="method",
            values=metric,
        )

        if not {
            "baseline",
            "kazdiv",
        }.issubset(
            pivot.columns
        ):
            continue

        paired = (
            pivot[
                [
                    "baseline",
                    "kazdiv",
                ]
            ]
            .dropna()
        )

        if len(
            paired
        ) == 0:
            continue

        differences = (
            paired[
                "kazdiv"
            ]
            .to_numpy() -
            paired[
                "baseline"
            ]
            .to_numpy()
        )

        ci_low, ci_high = (
            paired_bootstrap_ci(
                differences
            )
        )

        p_value = (
            paired_sign_flip_pvalue(
                differences
            )
        )

        stat_rows.append(
            {
                "provider":
                    provider,

                "metric":
                    metric,

                "n_pairs":
                    len(
                        paired
                    ),

                "baseline_mean":
                    float(
                        paired[
                            "baseline"
                        ]
                        .mean()
                    ),

                "kazdiv_mean":
                    float(
                        paired[
                            "kazdiv"
                        ]
                        .mean()
                    ),

                "mean_difference_kazdiv_minus_baseline":
                    float(
                        np.mean(
                            differences
                        )
                    ),

                "bootstrap_95_ci_low":
                    ci_low,

                "bootstrap_95_ci_high":
                    ci_high,

                "permutation_p":
                    p_value,
            }
        )


stats_df = pd.DataFrame(
    stat_rows
)


if len(
    stats_df
):
    stats_df[
        "holm_adjusted_p"
    ] = np.nan

    for provider, group in (
        stats_df
        .groupby(
            "provider",
            sort=False,
        )
    ):
        adjusted = holm_adjust(
            group[
                "permutation_p"
            ]
            .tolist()
        )

        for index, value in zip(
            group.index,
            adjusted,
        ):
            stats_df.loc[
                index,
                "holm_adjusted_p"
            ] = value

    stats_df.to_csv(
        OUTPUT_DIR /
        "experiment2_paired_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display(
        stats_df
    )

else:
    print(
        "No paired statistics available."
    )

# %%
# ============================================================
# Cell 18. Export / download Experiment 2 results
# ============================================================


def export_experiment2_zip(
    scope: str = "all",
    download: bool = True,
):
    """
    scope:
        'current' -> current pilot/main folder only
        'pilot'   -> pilot folder only
        'main'    -> main folder only
        'all'     -> complete Experiment 2 root
    """

    scope = str(scope).strip().lower()

    if scope == "current":
        source_dir = OUTPUT_DIR
        zip_name = (
            f"KAZ_Div_Experiment2_{RUN_MODE}_final.zip"
        )

    elif scope == "pilot":
        source_dir = PILOT_OUTPUT_DIR
        zip_name = "KAZ_Div_Experiment2_pilot.zip"

    elif scope == "main":
        source_dir = MAIN_OUTPUT_DIR
        zip_name = "KAZ_Div_Experiment2_main.zip"

    elif scope == "all":
        source_dir = OUTPUT_ROOT
        zip_name = "KAZ_Div_Experiment2_ALL.zip"

    else:
        raise ValueError(
            "scope must be one of: current, pilot, main, all"
        )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Nothing to export: {source_dir}"
        )

    zip_file = Path("/content") / zip_name

    if zip_file.exists():
        zip_file.unlink()

    zip_created = shutil.make_archive(
        base_name=str(
            zip_file.with_suffix("")
        ),
        format="zip",
        root_dir=str(source_dir.parent),
        base_dir=source_dir.name,
    )

    zip_created = Path(zip_created)

    print("ZIP created:", zip_created)
    print(
        "Size:",
        round(
            zip_created.stat().st_size / 1024**2,
            2,
        ),
        "MB",
    )

    if download:
        files.download(str(zip_created))

    return zip_created


print("Experiment 2 export helper ready.")
print("Download everything:")
print("export_experiment2_zip(scope='all', download=True)")
print("Download only main results:")
print("export_experiment2_zip(scope='main', download=True)")

# %% [markdown]
# ## Run checklist
# 
# **Colab Secrets:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `ALEMLLM_BASE_URL`, and optionally `ALEMLLM_API_KEY`.
# 
# **Pilot:** keep `RUN_MODE = "pilot"` and `THRESHOLD_FROZEN = False` in Cell 3 → run through Cell 15. Expected pilot = **40 KAZ-Div observations**.
# 
# **Main:** after reviewing Cell 15, set the frozen threshold in Cell 3, set `THRESHOLD_FROZEN = True` and `RUN_MODE = "full"` → rerun Cell 3 onward → Cell 14 collects **4000 observations** → Cells 16–17 analyze → Cell 18 downloads results.
# 
# **AlemLLM:** the endpoint must serve exactly `astanahub/alemllm` (or set `ALEMLLM_SERVED_MODEL` to the endpoint alias while preserving the canonical model ID in `MODEL_IDS`).

