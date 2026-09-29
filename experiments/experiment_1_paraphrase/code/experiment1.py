# %% [markdown]
# # KAZ-Div — Experiment 1 FINAL
# ## Kazakh Paraphrase Generation
# 
# This notebook is the final corrected implementation of Experiment 1.
# 
# ### Fixed model set
# 
# ```python
# MODEL_IDS = {
#     "openai": "gpt-5.6-sol",
#     "anthropic": "claude-sonnet-4-6",
#     "kazllm": "issai/LLama-3.1-KazLLM-1.0-8B",
#     "llama": "NousResearch/Meta-Llama-3.1-8B-Instruct",
# }
# ```
# 
# Execution:
# - OpenAI GPT-5.6 Sol → API
# - Claude Sonnet 4.6 → API
# - KazLLM-8B → local Transformers
# - Llama-3.1-8B-Instruct → local Transformers
# 
# The two local 8B models are **loaded lazily and one at a time** to reduce GPU memory pressure.
# 
# ### Main experimental design
# 
# \[
# 100\ sources \times 5\ repetitions \times 4\ models \times 2\ methods
# = 4000\ finalized\ observations
# \]
# 
# Methods:
# - **Baseline**: one direct paraphrase per request; output quality is measured but the request is not regenerated because of low quality.
# - **KAZ-Div**: generate \(K=12\) candidates → exact deduplication → quality filtering → BGE-M3 semantic clustering → highest-fidelity representative per cluster → stateless cluster-balanced final selection.
# 
# ### Important corrections in this version
# 
# - BGE-M3 uses **CLS pooling + L2 normalization**.
# - No `sentence-transformers` dependency.
# - No AlemLLM.
# - Full KAZ-Div candidate pools and selection metadata are saved.
# - No hidden fallback from failed KAZ-Div to Baseline.
# - Pilot and main runs are separated.
# - Semantic near-duplicate threshold is calibrated on pilot data and must be frozen before the main run.
# - Technical failures remain pending for resume; genuine method failures are finalized and retained for failure-rate analysis.
# - Token usage, latency and API cost metadata are recorded.
# - Primary statistical comparison is source-level and paired.

# %%
# ============================================================
# Cell 0. Install dependencies
# ============================================================

!pip -q install -U \
    openai \
    anthropic \
    "transformers>=4.45.1,<6" \
    accelerate \
    pandas \
    nltk \
    scikit-learn \
    matplotlib

# %%
# ============================================================
# Cell 1. Imports, reproducibility, helpers
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
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from openai import OpenAI
import anthropic

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction,
)

import matplotlib.pyplot as plt

from google.colab import files

try:
    from google.colab import userdata
except Exception:
    userdata = None


RANDOM_SEED = 20260922

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def normalize_text(text: Any) -> str:

    if text is None:
        return ""

    text = (
        unicodedata
        .normalize(
            "NFC",
            str(text),
        )
        .replace(
            "\u00A0",
            " ",
        )
    )

    for ch in [
        "\u200B",
        "\u200C",
        "\u200D",
        "\u2060",
        "\uFEFF",
    ]:
        text = text.replace(
            ch,
            "",
        )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalize_for_exact_match(
    text: Any
) -> str:

    text = (
        normalize_text(
            text
        )
        .lower()
    )

    for a, b in [
        ("’", "'"),
        ("‘", "'"),
        ("“", '"'),
        ("”", '"'),
        ("«", '"'),
        ("»", '"'),
    ]:
        text = text.replace(
            a,
            b,
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

    return re.sub(
        r"[.!?…]+$",
        "",
        text,
    ).strip()


def count_words(
    text: Any
) -> int:

    text = normalize_text(
        text
    )

    return (
        len(
            text.split()
        )
        if text
        else 0
    )


def deterministic_seed(
    *parts: Any
) -> int:

    material = "|".join(
        str(x)
        for x in parts
    )

    return int(
        hashlib
        .sha256(
            material.encode(
                "utf-8"
            )
        )
        .hexdigest()[:16],
        16,
    )


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
        return (
            None
            if (
                math.isnan(
                    value
                )
                or
                math.isinf(
                    value
                )
            )
            else value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k):
                to_json_safe(
                    v
                )
            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            to_json_safe(
                v
            )
            for v in value
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


def get_secret(
    name: str
) -> str:

    # 1) Google Colab Secrets
    if userdata is not None:
        try:
            value = userdata.get(
                name
            )
            if value:
                return str(
                    value
                )
        except Exception:
            pass

    # 2) Environment variable
    value = os.getenv(
        name,
        "",
    )

    return str(
        value
    ).strip()


print(
    "Torch:",
    torch.__version__,
)

print(
    "Transformers import: OK"
)

print(
    "CUDA:",
    torch.cuda.is_available(),
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

# %%
# ============================================================
# Cell 2. Upload and validate source dataset
# ============================================================
EXPECTED_DOMAINS = ["general","education","science","technology","news"]

print("Upload CSV with columns: id, text, domain")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No source CSV uploaded.")
SOURCE_FILE = next(iter(uploaded.keys()))
if not SOURCE_FILE.lower().endswith(".csv"):
    raise ValueError("Source dataset must be CSV.")

sources = pd.read_csv(SOURCE_FILE)
required = {"id","text","domain"}
missing = required - set(sources.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

sources = sources[["id","text","domain"]].copy()
sources["text"] = sources["text"].map(normalize_text)
sources["domain"] = sources["domain"].astype(str).str.strip().str.lower()

if sources["id"].duplicated().any():
    raise ValueError("Source IDs must be unique.")
if (sources["text"].str.len() == 0).any():
    raise ValueError("Empty source sentence detected.")

sources = sources.sort_values("id").reset_index(drop=True)
sources["word_count"] = sources["text"].str.split().map(len)
sources["char_count"] = sources["text"].str.len()

print("Rows:", len(sources))
print(sources["domain"].value_counts().sort_index())

if len(sources) == 100:
    observed = sources["domain"].value_counts().to_dict()
    if set(observed) == set(EXPECTED_DOMAINS):
        assert all(observed[d] == 20 for d in EXPECTED_DOMAINS)
        print("Balanced 100-source design: VERIFIED")
    else:
        print("WARNING: domain labels differ from the planned five-domain design.")
else:
    print("Pilot/non-standard size detected. Main paper design expects 100 sources.")

display(sources.head())

# %%
# ============================================================
# Cell 3. Experiment configuration
# ============================================================

EXPERIMENT_NAME = (
    "Experiment_1_Kazakh_Paraphrase_Generation_FINAL"
)

ROOT_OUTPUT_DIR = Path(
    "/content/KAZ_Div/Experiment_1_FINAL"
)

PILOT_OUTPUT_DIR = (
    ROOT_OUTPUT_DIR
    /
    "pilot"
)

MAIN_OUTPUT_DIR = (
    ROOT_OUTPUT_DIR
    /
    "main"
)

FIGURE_DIR = (
    ROOT_OUTPUT_DIR
    /
    "figures"
)

for directory in [
    ROOT_OUTPUT_DIR,
    PILOT_OUTPUT_DIR,
    MAIN_OUTPUT_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ------------------------------------------------------------
# FINAL MODEL SET
# ------------------------------------------------------------

MODEL_IDS = {
    "openai":
        "gpt-5.6-sol",

    "anthropic":
        "claude-sonnet-4-6",

    "kazllm":
        "issai/LLama-3.1-KazLLM-1.0-8B",

    "llama":
        "NousResearch/Meta-Llama-3.1-8B-Instruct",
}


ENABLED_PROVIDERS = [
    "openai",
    "anthropic",
    "kazllm",
    "llama",
]


LOCAL_PROVIDERS = {
    "kazllm",
    "llama",
}


METHODS = [
    "baseline",
    "kazdiv",
]


# ------------------------------------------------------------
# Main experimental design
# ------------------------------------------------------------

N_REPEATS = 5

K_CANDIDATES = 12

MAX_KAZDIV_CALLS = 2

MAX_OUTPUT_TOKENS = 1200


# ------------------------------------------------------------
# Shared generation settings
# ------------------------------------------------------------

GENERATION_TEMPERATURE = 0.6

LOCAL_TOP_P = 1.0

LOCAL_DO_SAMPLE = True

OPENAI_REASONING_EFFORT = "none"


# ------------------------------------------------------------
# Paraphrase-quality constraints
# ------------------------------------------------------------

FIDELITY_MIN = 0.75

MIN_LENGTH_RATIO = 0.45

MAX_LENGTH_RATIO = 2.20


# ------------------------------------------------------------
# Semantic near-duplicate threshold
# ------------------------------------------------------------
#
# First:
#   run pilot
#   inspect calibration CSV
#   manually label boundary pairs
#
# Then set:
#
#   FROZEN_NEAR_DUPLICATE_THRESHOLD = <chosen value>
#   THRESHOLD_STATUS = "FROZEN"
#
# ------------------------------------------------------------

PILOT_NEAR_DUPLICATE_THRESHOLD = 0.95

FROZEN_NEAR_DUPLICATE_THRESHOLD = None

THRESHOLD_STATUS = "PILOT"


# ------------------------------------------------------------
# BGE-M3
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = (
    "BAAI/bge-m3"
)

EMBEDDING_BATCH_SIZE = 32

EMBEDDING_MAX_LENGTH = 512


# ------------------------------------------------------------
# Statelessness requirements
# ------------------------------------------------------------

STATELESS_MODE = True

USE_CONVERSATION_HISTORY = False

USE_PREVIOUS_RESPONSES = False

USE_CUMULATIVE_STATISTICS = False


# ------------------------------------------------------------
# API pricing metadata
#
# USD per 1M text tokens.
# Local models have zero API cost; GPU compute cost is not
# represented here.
# ------------------------------------------------------------

MODEL_PRICING_USD_PER_1M = {
    "openai": {
        "input": 4.0,
        "output": 20.0,
    },

    "anthropic": {
        "input": 3.0,
        "output": 15.0,
    },

    "kazllm": {
        "input": 0.0,
        "output": 0.0,
    },

    "llama": {
        "input": 0.0,
        "output": 0.0,
    },
}


CONFIG = {
    "experiment":
        EXPERIMENT_NAME,

    "random_seed":
        RANDOM_SEED,

    "model_ids":
        MODEL_IDS,

    "providers":
        ENABLED_PROVIDERS,

    "local_providers":
        sorted(
            LOCAL_PROVIDERS
        ),

    "methods":
        METHODS,

    "n_repeats":
        N_REPEATS,

    "k_candidates":
        K_CANDIDATES,

    "max_kazdiv_calls":
        MAX_KAZDIV_CALLS,

    "max_output_tokens":
        MAX_OUTPUT_TOKENS,

    "generation_temperature":
        GENERATION_TEMPERATURE,

    "local_top_p":
        LOCAL_TOP_P,

    "openai_reasoning_effort":
        OPENAI_REASONING_EFFORT,

    "fidelity_min":
        FIDELITY_MIN,

    "length_ratio":
        [
            MIN_LENGTH_RATIO,
            MAX_LENGTH_RATIO,
        ],

    "pilot_near_duplicate_threshold":
        PILOT_NEAR_DUPLICATE_THRESHOLD,

    "embedding_model":
        EMBEDDING_MODEL_NAME,

    "embedding_pooling":
        "CLS + L2 normalization",

    "stateless":
        STATELESS_MODE,

    "uses_conversation_history":
        USE_CONVERSATION_HISTORY,

    "uses_previous_responses":
        USE_PREVIOUS_RESPONSES,

    "uses_cumulative_statistics":
        USE_CUMULATIVE_STATISTICS,

    "local_model_loading":
        "lazy; one local generative model at a time",
}


with open(
    ROOT_OUTPUT_DIR
    /
    "experiment1_final_config.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        CONFIG,
        f,
        ensure_ascii=False,
        indent=2,
    )


sources.to_csv(
    ROOT_OUTPUT_DIR
    /
    "experiment1_final_sources.csv",
    index=False,
    encoding="utf-8-sig",
)


print(
    json.dumps(
        CONFIG,
        ensure_ascii=False,
        indent=2,
    )
)

# %%
# ============================================================
# Cell 4. API clients + BGE-M3 + lazy local model manager
# ============================================================
#
# API keys:
#   recommended: Colab Secrets named
#       OPENAI_API_KEY
#       ANTHROPIC_API_KEY
#
# Local generative models are NOT loaded here.
# They are loaded lazily when their provider is first called.
# Only one local generative model is retained at a time.
# ============================================================


# ------------------------------------------------------------
# API clients
# ------------------------------------------------------------

OPENAI_API_KEY = get_secret(
    "OPENAI_API_KEY"
)

ANTHROPIC_API_KEY = get_secret(
    "ANTHROPIC_API_KEY"
)


openai_client = (
    OpenAI(
        api_key=(
            OPENAI_API_KEY
        )
    )
    if OPENAI_API_KEY
    else None
)


anthropic_client = (
    anthropic.Anthropic(
        api_key=(
            ANTHROPIC_API_KEY
        )
    )
    if ANTHROPIC_API_KEY
    else None
)


# ------------------------------------------------------------
# BGE-M3
#
# IMPORTANT:
# CLS pooling + L2 normalization.
# ------------------------------------------------------------

BGE_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


bge_tokenizer = (
    AutoTokenizer
    .from_pretrained(
        EMBEDDING_MODEL_NAME,
        use_fast=True,
    )
)


if torch.cuda.is_available():

    BGE_DTYPE = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

else:

    BGE_DTYPE = (
        torch.float32
    )


def load_embedding_model(
    model_id: str
):

    kwargs = {
        "low_cpu_mem_usage":
            True,
    }

    try:

        model = (
            AutoModel
            .from_pretrained(
                model_id,
                dtype=(
                    BGE_DTYPE
                ),
                **kwargs,
            )
        )

    except TypeError:

        model = (
            AutoModel
            .from_pretrained(
                model_id,
                torch_dtype=(
                    BGE_DTYPE
                ),
                **kwargs,
            )
        )

    return model.to(
        BGE_DEVICE
    )


bge_model = (
    load_embedding_model(
        EMBEDDING_MODEL_NAME
    )
)

bge_model.eval()


EMBEDDING_DIMENSION = int(
    bge_model
    .config
    .hidden_size
)


print(
    "BGE-M3 loaded:",
    EMBEDDING_MODEL_NAME,
)

print(
    "Embedding device:",
    BGE_DEVICE,
)

print(
    "Embedding dimension:",
    EMBEDDING_DIMENSION,
)


# ------------------------------------------------------------
# Lazy local generative model manager
# ------------------------------------------------------------

ACTIVE_LOCAL_PROVIDER = None

active_local_tokenizer = None

active_local_model = None


def local_generation_dtype():

    if not torch.cuda.is_available():
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


LOCAL_DTYPE = (
    local_generation_dtype()
)


def unload_active_local_model():

    global ACTIVE_LOCAL_PROVIDER
    global active_local_tokenizer
    global active_local_model

    if active_local_model is not None:

        try:
            del active_local_model
        except Exception:
            pass

    if active_local_tokenizer is not None:

        try:
            del active_local_tokenizer
        except Exception:
            pass

    active_local_model = None
    active_local_tokenizer = None
    ACTIVE_LOCAL_PROVIDER = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_local_provider(
    provider: str
):

    global ACTIVE_LOCAL_PROVIDER
    global active_local_tokenizer
    global active_local_model

    provider = (
        str(
            provider
        )
        .strip()
        .lower()
    )

    if provider not in LOCAL_PROVIDERS:

        raise ValueError(
            f"{provider} is not a local provider."
        )


    if (
        ACTIVE_LOCAL_PROVIDER == provider
        and
        active_local_model is not None
        and
        active_local_tokenizer is not None
    ):

        return (
            active_local_tokenizer,
            active_local_model,
        )


    unload_active_local_model()


    model_id = (
        MODEL_IDS[
            provider
        ]
    )


    print(
        f"Loading local model: "
        f"{provider} -> {model_id}"
    )


    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_id,
            use_fast=True,
        )
    )


    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    tokenizer.padding_side = (
        "left"
    )


    kwargs = {
        "device_map":
            "auto",

        "low_cpu_mem_usage":
            True,
    }


    try:

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_id,
                dtype=(
                    LOCAL_DTYPE
                ),
                **kwargs,
            )
        )

    except TypeError:

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_id,
                torch_dtype=(
                    LOCAL_DTYPE
                ),
                **kwargs,
            )
        )


    model.eval()


    ACTIVE_LOCAL_PROVIDER = (
        provider
    )

    active_local_tokenizer = (
        tokenizer
    )

    active_local_model = (
        model
    )


    if torch.cuda.is_available():

        print(
            "GPU allocated after load:",
            round(
                torch.cuda
                .memory_allocated(
                    0
                )
                /
                1024**3,
                2,
            ),
            "GB",
        )


    return (
        tokenizer,
        model,
    )


# ------------------------------------------------------------
# Provider readiness
# ------------------------------------------------------------

def provider_readiness_error(
    provider: str
) -> Optional[str]:

    provider = (
        str(
            provider
        )
        .strip()
        .lower()
    )


    if provider == "openai":

        if openai_client is None:

            return (
                "OPENAI_API_KEY is missing."
            )

        return None


    if provider == "anthropic":

        if anthropic_client is None:

            return (
                "ANTHROPIC_API_KEY is missing."
            )

        return None


    if provider in LOCAL_PROVIDERS:

        model_id = (
            MODEL_IDS.get(
                provider,
                "",
            )
        )

        if not model_id:

            return (
                f"Model ID missing for {provider}."
            )

        return None


    return (
        f"Unknown provider: {provider}"
    )


print(
    "\nProvider configuration:"
)

for provider in (
    ENABLED_PROVIDERS
):

    error = (
        provider_readiness_error(
            provider
        )
    )

    print(
        f"{provider:<10}:",
        (
            "CONFIGURED"
            if error is None
            else
            f"NOT READY — {error}"
        ),
    )

# %%
# ============================================================
# Cell 5. Unified LLM calls — final four-model version
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


def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:

        return (
            int(
                value
            )
            if value is not None
            else default
        )

    except Exception:

        return default


def estimate_api_cost_usd(
    provider: str,
    input_tokens: int,
    output_tokens: int,
) -> Optional[float]:

    pricing = (
        MODEL_PRICING_USD_PER_1M
        .get(
            provider,
            {},
        )
    )

    input_price = pricing.get(
        "input"
    )

    output_price = pricing.get(
        "output"
    )

    if (
        input_price is None
        or
        output_price is None
    ):
        return None

    return float(
        (
            input_tokens
            /
            1_000_000
        )
        *
        float(
            input_price
        )
        +
        (
            output_tokens
            /
            1_000_000
        )
        *
        float(
            output_price
        )
    )


# ------------------------------------------------------------
# OpenAI — GPT-5.6 Sol
# ------------------------------------------------------------

def call_openai_once(
    prompt: str,
    model_id: str,
) -> LLMResult:

    if openai_client is None:

        raise RuntimeError(
            "OpenAI client not configured."
        )


    start = (
        time.perf_counter()
    )


    response = (
        openai_client
        .responses
        .create(
            model=(
                model_id
            ),

            input=(
                prompt
            ),

            max_output_tokens=(
                MAX_OUTPUT_TOKENS
            ),

            temperature=(
                GENERATION_TEMPERATURE
            ),

            reasoning={
                "effort":
                    OPENAI_REASONING_EFFORT,
            },
        )
    )


    latency = (
        time.perf_counter()
        -
        start
    )


    text = normalize_text(
        getattr(
            response,
            "output_text",
            "",
        )
    )


    usage = getattr(
        response,
        "usage",
        None,
    )


    input_tokens = safe_int(
        getattr(
            usage,
            "input_tokens",
            0,
        )
        if usage is not None
        else 0
    )


    output_tokens = safe_int(
        getattr(
            usage,
            "output_tokens",
            0,
        )
        if usage is not None
        else 0
    )


    if not text:

        raise RuntimeError(
            "OpenAI returned an empty output."
        )


    return LLMResult(
        text=(
            text
        ),

        provider=(
            "openai"
        ),

        model_id=(
            model_id
        ),

        input_tokens=(
            input_tokens
        ),

        output_tokens=(
            output_tokens
        ),

        total_tokens=(
            input_tokens
            +
            output_tokens
        ),

        latency_sec=(
            latency
        ),

        request_id=getattr(
            response,
            "id",
            None,
        ),

        finish_reason=getattr(
            response,
            "status",
            None,
        ),
    )


# ------------------------------------------------------------
# Anthropic — Claude Sonnet 4.6
# ------------------------------------------------------------

def call_anthropic_once(
    prompt: str,
    model_id: str,
) -> LLMResult:

    if anthropic_client is None:

        raise RuntimeError(
            "Anthropic client not configured."
        )


    start = (
        time.perf_counter()
    )


    response = (
        anthropic_client
        .messages
        .create(
            model=(
                model_id
            ),

            max_tokens=(
                MAX_OUTPUT_TOKENS
            ),

            temperature=(
                GENERATION_TEMPERATURE
            ),

            messages=[
                {
                    "role":
                        "user",

                    "content":
                        prompt,
                }
            ],
        )
    )


    latency = (
        time.perf_counter()
        -
        start
    )


    text = normalize_text(
        "\n".join(
            block.text
            for block
            in getattr(
                response,
                "content",
                [],
            )
            if hasattr(
                block,
                "text",
            )
        )
    )


    usage = getattr(
        response,
        "usage",
        None,
    )


    input_tokens = safe_int(
        getattr(
            usage,
            "input_tokens",
            0,
        )
        if usage is not None
        else 0
    )


    output_tokens = safe_int(
        getattr(
            usage,
            "output_tokens",
            0,
        )
        if usage is not None
        else 0
    )


    if not text:

        raise RuntimeError(
            "Anthropic returned an empty output."
        )


    return LLMResult(
        text=(
            text
        ),

        provider=(
            "anthropic"
        ),

        model_id=(
            model_id
        ),

        input_tokens=(
            input_tokens
        ),

        output_tokens=(
            output_tokens
        ),

        total_tokens=(
            input_tokens
            +
            output_tokens
        ),

        latency_sec=(
            latency
        ),

        request_id=getattr(
            response,
            "id",
            None,
        ),

        finish_reason=getattr(
            response,
            "stop_reason",
            None,
        ),
    )


# ------------------------------------------------------------
# Local generation — shared by KazLLM and Llama
# ------------------------------------------------------------

def call_local_once(
    provider: str,
    prompt: str,
    model_id: str,
    generation_seed: Optional[int] = None,
) -> LLMResult:

    provider = (
        str(
            provider
        )
        .strip()
        .lower()
    )


    if provider not in LOCAL_PROVIDERS:

        raise ValueError(
            f"{provider} is not a local provider."
        )


    tokenizer, model = (
        load_local_provider(
            provider
        )
    )


    if generation_seed is not None:

        torch.manual_seed(
            int(
                generation_seed
            )
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                int(
                    generation_seed
                )
            )


    messages = [
        {
            "role":
                "user",

            "content":
                prompt,
        }
    ]


    model_inputs = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    )


    try:

        input_device = (
            model
            .get_input_embeddings()
            .weight
            .device
        )

    except Exception:

        input_device = next(
            model.parameters()
        ).device


    model_inputs = {
        key:
            value.to(
                input_device
            )
        for key, value
        in model_inputs.items()
    }


    input_tokens = int(
        model_inputs[
            "input_ids"
        ]
        .shape[
            1
        ]
    )


    if torch.cuda.is_available():
        torch.cuda.synchronize()


    start = (
        time.perf_counter()
    )


    with torch.inference_mode():

        generated = (
            model
            .generate(
                **model_inputs,

                max_new_tokens=(
                    MAX_OUTPUT_TOKENS
                ),

                do_sample=(
                    LOCAL_DO_SAMPLE
                ),

                temperature=(
                    GENERATION_TEMPERATURE
                ),

                top_p=(
                    LOCAL_TOP_P
                ),

                pad_token_id=(
                    tokenizer
                    .pad_token_id
                ),

                eos_token_id=(
                    tokenizer
                    .eos_token_id
                ),
            )
        )


    if torch.cuda.is_available():
        torch.cuda.synchronize()


    latency = (
        time.perf_counter()
        -
        start
    )


    generated_suffix = (
        generated[
            0,
            input_tokens:
        ]
    )


    output_tokens = int(
        generated_suffix
        .shape[
            0
        ]
    )


    text = normalize_text(
        tokenizer
        .decode(
            generated_suffix,
            skip_special_tokens=True,
        )
    )


    if not text:

        raise RuntimeError(
            f"{provider} returned an empty output."
        )


    return LLMResult(
        text=(
            text
        ),

        provider=(
            provider
        ),

        model_id=(
            model_id
        ),

        input_tokens=(
            input_tokens
        ),

        output_tokens=(
            output_tokens
        ),

        total_tokens=(
            input_tokens
            +
            output_tokens
        ),

        latency_sec=(
            latency
        ),

        finish_reason=(
            "completed"
        ),
    )


# ------------------------------------------------------------
# Unified dispatcher
# ------------------------------------------------------------

def call_llm(
    provider: str,
    prompt: str,
    model_id: Optional[str] = None,
    generation_seed: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> LLMResult:

    provider = (
        str(
            provider
        )
        .strip()
        .lower()
    )


    if provider not in MODEL_IDS:

        raise ValueError(
            f"Unsupported provider: {provider}"
        )


    model_id = (
        model_id
        or
        MODEL_IDS[
            provider
        ]
    )


    readiness_error = (
        provider_readiness_error(
            provider
        )
    )


    if readiness_error is not None:

        raise RuntimeError(
            readiness_error
        )


    if max_attempts is None:

        max_attempts = (
            1
            if provider in LOCAL_PROVIDERS
            else 5
        )


    retry_delays = [
        2,
        4,
        8,
        16,
        30,
    ]


    last_error = None


    for attempt in range(
        1,
        max_attempts
        +
        1,
    ):

        try:

            if provider == "openai":

                result = (
                    call_openai_once(
                        prompt,
                        model_id,
                    )
                )


            elif provider == "anthropic":

                result = (
                    call_anthropic_once(
                        prompt,
                        model_id,
                    )
                )


            elif provider in LOCAL_PROVIDERS:

                result = (
                    call_local_once(
                        provider=(
                            provider
                        ),

                        prompt=(
                            prompt
                        ),

                        model_id=(
                            model_id
                        ),

                        generation_seed=(
                            generation_seed
                        ),
                    )
                )


            else:

                raise ValueError(
                    f"Unsupported provider: {provider}"
                )


            result.attempts = (
                attempt
            )

            return result


        except Exception as exc:

            last_error = (
                exc
            )

            if attempt >= max_attempts:
                break

            time.sleep(
                retry_delays[
                    min(
                        attempt
                        -
                        1,
                        len(
                            retry_delays
                        )
                        -
                        1,
                    )
                ]
            )


    raise RuntimeError(
        f"{provider} call failed after "
        f"{max_attempts} attempt(s): "
        f"{last_error}"
    )


print(
    "Unified four-model LLM dispatcher: READY"
)

# %%
# ============================================================
# Cell 6. Fixed prompts
# ============================================================
BASELINE_PROMPT_TEMPLATE = """
Тапсырма:

Берілген қазақ тіліндегі сөйлемді мағынасын толық сақтай отырып,
табиғи қазақ тілінде басқа сөздермен қайта жазыңыз.

Талаптар:
1. Бастапқы сөйлемнің негізгі мағынасын толық сақтаңыз.
2. Жаңа ақпарат қоспаңыз.
3. Бастапқы ақпаратты алып тастамаңыз.
4. Сөйлемді жай ғана сөздердің орнын ауыстыру арқылы өзгертпеңіз.
5. Лексикалық немесе синтаксистік тұрғыдан табиғи қайта тұжырымдауды қолданыңыз.
6. Қазақ тілінің грамматикалық және орфографиялық нормаларын сақтаңыз.
7. Тек бір қайта жазылған сөйлемді қайтарыңыз.
8. Түсіндірме, нөмірлеу, ескерту немесе қосымша мәтін қоспаңыз.

Бастапқы сөйлем:
{source_text}
""".strip()

KAZDIV_PROMPT_TEMPLATE = """
Тапсырма:

Берілген қазақ тіліндегі сөйлемнің мағынасын сақтай отырып,
қазақ тілінде {k_candidates} түрлі табиғи парафраз жасаңыз.

Талаптар:
1. Әр нұсқа бастапқы сөйлемнің негізгі мағынасын толық сақтауы тиіс.
2. Жаңа ақпарат қоспаңыз.
3. Бастапқы ақпаратты алып тастамаңыз.
4. Нұсқалар бір-бірінен лексикалық және мүмкін болған жағдайда синтаксистік тұрғыдан ерекшеленуі тиіс.
5. Бірдей немесе өте ұқсас нұсқаларды қайталамаңыз.
6. Тек сөздердің орнын ауыстыру арқылы өзгертпеңіз.
7. Қазақ тілінің грамматикалық және орфографиялық нормаларын сақтаңыз.
8. Дәл {k_candidates} нұсқа жасаңыз.
9. Тек төмендегі JSON форматында жауап беріңіз.
10. Түсіндірме, нөмірлеу немесе қосымша мәтін қоспаңыз.

JSON форматы (candidates массивында дәл {k_candidates} жол болуы тиіс):
{{
  "candidates": [
    "парафраз 1",
    "парафраз 2",
    "...",
    "парафраз {k_candidates}"
  ]
}}

Бастапқы сөйлем:
{source_text}
""".strip()

def build_prompt(method: str,source_text: str) -> str:
    method=method.strip().lower()
    source_text=normalize_text(source_text)
    if method=="baseline":
        return BASELINE_PROMPT_TEMPLATE.format(source_text=source_text)
    if method=="kazdiv":
        return KAZDIV_PROMPT_TEMPLATE.format(source_text=source_text,k_candidates=K_CANDIDATES)
    raise ValueError(f"Unknown method: {method}")

with open(ROOT_OUTPUT_DIR/"experiment1_prompts.json","w",encoding="utf-8") as f:
    json.dump({"baseline":BASELINE_PROMPT_TEMPLATE,"kazdiv":KAZDIV_PROMPT_TEMPLATE},
              f,ensure_ascii=False,indent=2)
print("Prompt templates saved.")

# %%
# ============================================================
# Cell 7. Robust parsing + exact surface deduplication
# ============================================================
@dataclass
class CandidateParseResult:
    candidates: List[str]
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
    text="" if text is None else str(text).strip()
    text=re.sub(r"^\s*```(?:json)?\s*","",text,flags=re.IGNORECASE)
    return re.sub(r"\s*```\s*$","",text).strip()

def clean_candidate(text: Any) -> str:
    text=normalize_text(text)
    text=re.sub(r"^\s*(?:[-*•]\s+|\d+\s*[\.\)\-:]\s*)","",text).strip()
    for left,right in [('"','"'),("'","'"),("«","»"),("“","”")]:
        if len(text)>=2 and text.startswith(left) and text.endswith(right):
            text=text[1:-1].strip()
            break
    return normalize_text(text)

def extract_balanced_structure(text: str,opening: str,closing: str) -> Optional[str]:
    start=text.find(opening)
    if start<0:
        return None
    depth=0; quote=None; escape=False
    for i in range(start,len(text)):
        ch=text[i]
        if escape:
            escape=False
            continue
        if ch=="\\" and quote is not None:
            escape=True
            continue
        if quote is not None:
            if ch==quote:
                quote=None
            continue
        if ch in ['"',"'"]:
            quote=ch
            continue
        if ch==opening:
            depth+=1
        elif ch==closing:
            depth-=1
            if depth==0:
                return text[start:i+1]
    return None

def candidates_from_object(obj: Any) -> Optional[List[Any]]:
    if isinstance(obj,list):
        return obj
    if not isinstance(obj,dict):
        return None
    lower={str(k).strip().lower():v for k,v in obj.items()}
    for key in ["candidates","paraphrases","outputs","responses","sentences"]:
        if isinstance(lower.get(key),list):
            return lower[key]
    for key in ["candidate","paraphrase","output","response","text"]:
        if isinstance(lower.get(key),str):
            return [lower[key]]
    return None

def parse_numbered_lines(text: str) -> List[str]:
    return [x for x in (clean_candidate(line) for line in str(text).splitlines()) if x]

def clean_and_deduplicate_candidates(raw_candidates: List[Any]) -> Dict[str,Any]:
    cleaned=[]; seen=set(); duplicates=0; invalid=0
    for item in raw_candidates:
        if isinstance(item,dict):
            item=item.get("text") or item.get("candidate") or item.get("paraphrase") or ""
        if not isinstance(item,str):
            invalid+=1
            continue
        candidate=clean_candidate(item)
        key=normalize_for_exact_match(candidate)
        if not key:
            invalid+=1
            continue
        if key in seen:
            duplicates+=1
            continue
        seen.add(key)
        cleaned.append(candidate)
    return {
        "candidates":cleaned,
        "raw_candidate_count":len(raw_candidates),
        "cleaned_candidate_count":len(cleaned),
        "exact_duplicate_count":duplicates,
        "invalid_candidate_count":invalid,
    }

def parse_kazdiv_response(raw_text: Any,expected_k: int=K_CANDIDATES) -> CandidateParseResult:
    text="" if raw_text is None else str(raw_text).strip()
    attempts=[("direct_json",text),("code_fence_json",remove_code_fences(text))]
    obj_text=extract_balanced_structure(text,"{","}")
    arr_text=extract_balanced_structure(text,"[","]")
    if obj_text:
        attempts.append(("extracted_json_object",obj_text))
    if arr_text:
        attempts.append(("extracted_json_array",arr_text))
    parsed=None; method="failed"
    for m,candidate_text in attempts:
        try:
            values=candidates_from_object(json.loads(candidate_text))
            if values is not None:
                parsed=values; method=m; break
        except Exception:
            pass
    if parsed is None:
        try:
            values=candidates_from_object(ast.literal_eval(obj_text or arr_text or remove_code_fences(text)))
            if values is not None:
                parsed=values; method="python_literal"
        except Exception:
            pass
    if parsed is None:
        numbered=parse_numbered_lines(text)
        if numbered:
            parsed=numbered; method="numbered_lines"
    if parsed is None:
        return CandidateParseResult([],False,"failed",0,0,0,0,expected_k,False,
                                    "Could not parse KAZ-Div response.")
    result=clean_and_deduplicate_candidates(parsed)
    n=len(result["candidates"])
    return CandidateParseResult(
        result["candidates"],n>=1,method,result["raw_candidate_count"],
        result["cleaned_candidate_count"],result["exact_duplicate_count"],
        result["invalid_candidate_count"],expected_k,n==expected_k,
        None if n==expected_k else f"Expected {expected_k}, recovered {n} unique candidates."
    )

def parse_baseline_response(raw_text: Any) -> CandidateParseResult:
    candidate=clean_candidate(raw_text)
    if not candidate:
        return CandidateParseResult([],False,"baseline_direct",0,0,0,1,1,False,"Empty baseline response.")
    return CandidateParseResult([candidate],True,"baseline_direct",1,1,0,0,1,True,None)

_test_text = "```json\\n{\\\"candidates\\\":[\\\"Бірінші.\\\",\\\"Екінші.\\\",\\\"Бірінші.\\\"]}\\n```"
_test=parse_kazdiv_response(_test_text,3)
assert _test.parse_success and len(_test.candidates)==2 and _test.exact_duplicate_count==1
print("Parser tests: PASS")

# %%
# ============================================================
# Cell 8. BGE-M3 CLS embeddings
# ============================================================
EMBEDDING_CACHE: Dict[str,np.ndarray] = {}

def prepare_embedding_texts(texts: Any) -> List[str]:
    if isinstance(texts,str):
        texts=[texts]
    if not isinstance(texts,(list,tuple)):
        raise TypeError("texts must be str/list/tuple")
    cleaned=[]
    for text in texts:
        text=normalize_text(text)
        if not text:
            raise ValueError("Empty text cannot be embedded.")
        cleaned.append(text)
    return cleaned

def _encode_uncached(texts: List[str],batch_size: int=EMBEDDING_BATCH_SIZE) -> np.ndarray:
    all_embeddings=[]
    for start in range(0,len(texts),batch_size):
        batch=texts[start:start+batch_size]
        encoded=bge_tokenizer(
            batch,padding=True,truncation=True,max_length=EMBEDDING_MAX_LENGTH,return_tensors="pt"
        )
        encoded={k:v.to(BGE_DEVICE) for k,v in encoded.items()}
        with torch.inference_mode():
            outputs=bge_model(**encoded)
            # Correct v2 pooling:
            embeddings=outputs.last_hidden_state[:,0]
            embeddings=F.normalize(embeddings,p=2,dim=1)
        all_embeddings.append(embeddings.float().cpu().numpy())
    return np.vstack(all_embeddings).astype(np.float32)

def encode_texts(texts: Any,batch_size: int=EMBEDDING_BATCH_SIZE) -> np.ndarray:
    texts=prepare_embedding_texts(texts)
    missing=[]; seen=set()
    for text in texts:
        if text not in EMBEDDING_CACHE and text not in seen:
            missing.append(text); seen.add(text)
    if missing:
        new_embeddings=_encode_uncached(missing,batch_size)
        for text,embedding in zip(missing,new_embeddings):
            EMBEDDING_CACHE[text]=embedding.astype(np.float32,copy=True)
    return np.vstack([EMBEDDING_CACHE[text] for text in texts]).astype(np.float32)

def semantic_similarity(text_a: str,text_b: str) -> float:
    emb=encode_texts([text_a,text_b])
    return float(np.clip(np.dot(emb[0],emb[1]),-1.0,1.0))

def analyse_candidate_semantics(source_text: str,candidates: List[str]) -> Dict[str,Any]:
    if not candidates:
        return {
            "fidelity_scores":[],
            "similarity_matrix":np.empty((0,0),dtype=np.float32),
            "pairwise_semantic_diversity":np.nan,
        }
    emb=encode_texts([source_text]+list(candidates))
    source_emb=emb[0]
    candidate_emb=emb[1:]
    fidelity=np.clip(candidate_emb@source_emb,-1.0,1.0)
    matrix=np.clip(candidate_emb@candidate_emb.T,-1.0,1.0).astype(np.float32)
    np.fill_diagonal(matrix,1.0)
    if len(candidates)>=2:
        vals=matrix[np.triu_indices(len(candidates),k=1)]
        diversity=float(np.mean(1.0-vals))
    else:
        diversity=np.nan
    return {
        "fidelity_scores":[float(x) for x in fidelity],
        "similarity_matrix":matrix,
        "pairwise_semantic_diversity":diversity,
    }

TEST_EMBED_TEXT="Жасанды интеллект білім беру саласында қолданылады."
embedding=encode_texts([TEST_EMBED_TEXT])
assert embedding.shape==(1,EMBEDDING_DIMENSION)
assert abs(np.linalg.norm(embedding[0])-1.0)<1e-3
assert semantic_similarity(TEST_EMBED_TEXT,TEST_EMBED_TEXT)>0.999
print("BGE-M3 CLS embedding tests: PASS")

# %%
# ============================================================
# Cell 9. Quality filtering
# ============================================================
@dataclass
class CandidateFilterRecord:
    candidate_index: int
    candidate_text: str
    source_word_count: int
    candidate_word_count: int
    length_ratio: float
    semantic_fidelity: float
    same_as_source: bool
    length_pass: bool
    fidelity_pass: bool
    quality_pass: bool
    rejection_reasons: List[str]

@dataclass
class QualityFilterResult:
    source_text: str
    input_candidate_count: int
    valid_candidate_count: int
    rejected_candidate_count: int
    valid_candidates: List[str]
    valid_candidate_indices: List[int]
    valid_fidelity_scores: List[float]
    records: List[CandidateFilterRecord]
    filter_success: bool

def evaluate_candidate_quality(source_text: str,candidate_text: str,candidate_index: int,
                               fidelity: float,fidelity_min: float=FIDELITY_MIN) -> CandidateFilterRecord:
    source_text=normalize_text(source_text)
    candidate_text=normalize_text(candidate_text)
    source_words=max(1,count_words(source_text))
    candidate_words=count_words(candidate_text)
    ratio=candidate_words/source_words
    same=(normalize_for_exact_match(candidate_text)==normalize_for_exact_match(source_text))
    length_pass=MIN_LENGTH_RATIO<=ratio<=MAX_LENGTH_RATIO
    fidelity_pass=bool(np.isfinite(fidelity) and fidelity>=fidelity_min)
    reasons=[]
    if not candidate_text: reasons.append("empty_candidate")
    if same: reasons.append("same_as_source")
    if not length_pass: reasons.append("length_ratio_out_of_range")
    if not fidelity_pass: reasons.append("low_semantic_fidelity")
    quality_pass=bool(candidate_text and not same and length_pass and fidelity_pass)
    return CandidateFilterRecord(
        candidate_index,candidate_text,source_words,candidate_words,float(ratio),float(fidelity),
        bool(same),bool(length_pass),bool(fidelity_pass),quality_pass,reasons
    )

def filter_candidate_pool(source_text: str,candidates: List[str],
                          fidelity_min: float=FIDELITY_MIN) -> QualityFilterResult:
    if not candidates:
        return QualityFilterResult(normalize_text(source_text),0,0,0,[],[],[],[],False)
    semantics=analyse_candidate_semantics(source_text,candidates)
    records=[
        evaluate_candidate_quality(source_text,candidate,i,fidelity,fidelity_min)
        for i,(candidate,fidelity) in enumerate(zip(candidates,semantics["fidelity_scores"]))
    ]
    valid=[r for r in records if r.quality_pass]
    return QualityFilterResult(
        normalize_text(source_text),len(candidates),len(valid),len(candidates)-len(valid),
        [r.candidate_text for r in valid],[r.candidate_index for r in valid],
        [r.semantic_fidelity for r in valid],records,len(valid)>=1
    )

def validate_baseline_output(source_text: str,baseline_text: str) -> CandidateFilterRecord:
    fidelity=analyse_candidate_semantics(source_text,[baseline_text])["fidelity_scores"][0]
    return evaluate_candidate_quality(source_text,baseline_text,0,fidelity,FIDELITY_MIN)

print("Quality filtering: READY")

# %%
# ============================================================
# Cell 10. Semantic near-deduplication and clustering
# ============================================================
@dataclass
class SemanticCluster:
    cluster_id: int
    member_local_indices: List[int]
    original_candidate_indices: List[int]
    member_texts: List[str]
    member_fidelity_scores: List[float]
    representative_local_index: int
    representative_original_index: int
    representative_text: str
    representative_fidelity: float
    cluster_size: int

@dataclass
class SemanticDeduplicationResult:
    source_text: str
    input_candidate_count: int
    cluster_count: int
    semantic_duplicates_removed: int
    near_duplicate_edge_count: int
    threshold: float
    clusters: List[SemanticCluster]
    representative_candidates: List[str]
    representative_fidelity_scores: List[float]
    representative_original_indices: List[int]
    similarity_matrix: np.ndarray
    deduplication_success: bool

def build_semantic_duplicate_graph(similarity_matrix: np.ndarray,threshold: float) -> Dict[int,List[int]]:
    if not (0.0<float(threshold)<=1.0):
        raise ValueError("threshold must be in (0,1].")
    matrix=np.asarray(similarity_matrix,dtype=np.float32)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1]:
        raise ValueError("Similarity matrix must be square.")
    n=matrix.shape[0]
    graph={i:[] for i in range(n)}
    for i in range(n):
        for j in range(i+1,n):
            if float(matrix[i,j])>=threshold:
                graph[i].append(j); graph[j].append(i)
    return graph

def connected_components(graph: Dict[int,List[int]]) -> List[List[int]]:
    visited=set(); components=[]
    for start in sorted(graph):
        if start in visited:
            continue
        stack=[start]; component=[]
        while stack:
            node=stack.pop()
            if node in visited:
                continue
            visited.add(node); component.append(node)
            for neighbor in sorted(graph[node],reverse=True):
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components,key=lambda c:min(c))

def semantic_deduplicate_quality_result(quality_result: QualityFilterResult,
                                        threshold: float) -> SemanticDeduplicationResult:
    candidates=quality_result.valid_candidates
    fidelities=quality_result.valid_fidelity_scores
    original_indices=quality_result.valid_candidate_indices
    n=len(candidates)
    if n==0:
        return SemanticDeduplicationResult(
            quality_result.source_text,0,0,0,0,float(threshold),[],[],[],[],
            np.empty((0,0),dtype=np.float32),False
        )
    matrix=analyse_candidate_semantics(quality_result.source_text,candidates)["similarity_matrix"]
    graph=build_semantic_duplicate_graph(matrix,threshold)
    edge_count=sum(len(v) for v in graph.values())//2
    components=connected_components(graph)
    clusters=[]; reps=[]; rep_scores=[]; rep_orig=[]
    for cluster_id,component in enumerate(components):
        rep_local=max(component,key=lambda idx:(float(fidelities[idx]),-int(idx)))
        cluster=SemanticCluster(
            cluster_id=cluster_id,
            member_local_indices=[int(i) for i in component],
            original_candidate_indices=[int(original_indices[i]) for i in component],
            member_texts=[candidates[i] for i in component],
            member_fidelity_scores=[float(fidelities[i]) for i in component],
            representative_local_index=int(rep_local),
            representative_original_index=int(original_indices[rep_local]),
            representative_text=candidates[rep_local],
            representative_fidelity=float(fidelities[rep_local]),
            cluster_size=len(component),
        )
        clusters.append(cluster)
        reps.append(cluster.representative_text)
        rep_scores.append(cluster.representative_fidelity)
        rep_orig.append(cluster.representative_original_index)
    return SemanticDeduplicationResult(
        quality_result.source_text,n,len(clusters),n-len(clusters),edge_count,float(threshold),
        clusters,reps,rep_scores,rep_orig,matrix,len(clusters)>=1
    )

SYNTHETIC_MATRIX=np.array([
    [1.00,0.97,0.20,0.10],
    [0.97,1.00,0.96,0.10],
    [0.20,0.96,1.00,0.15],
    [0.10,0.10,0.15,1.00],
],dtype=np.float32)
assert connected_components(build_semantic_duplicate_graph(SYNTHETIC_MATRIX,0.95))==[[0,1,2],[3]]
print("Semantic clustering tests: PASS")

# %%
# ============================================================
# Cell 11. Stateless cluster-balanced selection
# ============================================================
@dataclass
class DiversitySelectionResult:
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
    selected_text: str
    selected_fidelity: float
    selection_success: bool
    selection_reason: str

def make_selection_seed(provider: str,model_id: str,source_id: Any,
                        repetition: int,method: str="kazdiv") -> int:
    return deterministic_seed(RANDOM_SEED,provider,model_id,source_id,repetition,method)

def select_kazdiv_response(semantic_result: SemanticDeduplicationResult,
                           provider: str,model_id: str,source_id: Any,
                           repetition: int) -> DiversitySelectionResult:
    m=semantic_result.cluster_count
    if m<1:
        return DiversitySelectionResult(
            source_id,repetition,provider,model_id,"kazdiv",-1,0,0.0,-1,-1,"",float("nan"),False,"no_clusters"
        )
    seed=make_selection_seed(provider,model_id,source_id,repetition)
    if m==1:
        selected_position=0
    else:
        selected_position=int(np.random.default_rng(seed).integers(0,m))
    cluster=semantic_result.clusters[selected_position]
    return DiversitySelectionResult(
        source_id,repetition,provider,model_id,"kazdiv",seed,m,float(1.0/m),
        int(cluster.cluster_id),int(cluster.representative_original_index),
        cluster.representative_text,float(cluster.representative_fidelity),True,
        "single_available_cluster" if m==1 else "uniform_cluster_selection"
    )

assert make_selection_seed("kazllm",MODEL_IDS["kazllm"],1,1)==make_selection_seed("kazllm",MODEL_IDS["kazllm"],1,1)
print("Stateless cluster-balanced selection: READY")

# %%
# ============================================================
# Cell 12. End-to-end Baseline / KAZ-Div runner
# ============================================================
@dataclass
class Experiment1Result:
    experiment: str
    source_id: Any
    source_text: str
    domain: Optional[str]
    provider: str
    model_id: str
    method: str
    repetition: int
    run_success: bool
    technical_failure: bool
    final_text: str
    final_semantic_fidelity: Optional[float]
    final_quality_pass: Optional[bool]
    llm_call_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_llm_latency_sec: float
    total_pipeline_latency_sec: float
    estimated_api_cost_usd: Optional[float]
    raw_candidate_count: Optional[int]
    parsed_candidate_count: Optional[int]
    exact_duplicate_count: Optional[int]
    quality_valid_candidate_count: Optional[int]
    quality_rejected_candidate_count: Optional[int]
    semantic_cluster_count: Optional[int]
    semantic_duplicates_removed: Optional[int]
    selected_cluster_id: Optional[int]
    selected_original_candidate_index: Optional[int]
    selection_seed: Optional[int]
    semantic_threshold_used: Optional[float]
    kazdiv_attempt_used: Optional[int]
    attempt_logs: List[Dict[str,Any]]
    failure_stage: Optional[str]
    error_message: Optional[str]

def make_generation_seed(provider: str,model_id: str,source_id: Any,repetition: int,
                         method: str,call_index: int) -> int:
    return deterministic_seed(
        RANDOM_SEED,provider,model_id,source_id,repetition,method,call_index
    )%(2**32-1)

def aggregate_llm_results(results: List[LLMResult]) -> Dict[str,Any]:
    inp=sum(x.input_tokens for x in results)
    out=sum(x.output_tokens for x in results)
    return {
        "llm_call_count":len(results),
        "total_input_tokens":inp,
        "total_output_tokens":out,
        "total_tokens":inp+out,
        "total_llm_latency_sec":float(sum(x.latency_sec for x in results)),
    }

def _failure_result(source_id,source_text,domain,provider,model_id,method,repetition,
                    pipeline_start,llm_results,technical_failure,failure_stage,error_message,
                    attempt_logs,semantic_threshold=None,parse_result=None,quality_result=None,
                    semantic_result=None) -> Experiment1Result:
    totals=aggregate_llm_results(llm_results)
    cost=estimate_api_cost_usd(provider,totals["total_input_tokens"],totals["total_output_tokens"])
    return Experiment1Result(
        EXPERIMENT_NAME,source_id,source_text,domain,provider,model_id,method,repetition,
        False,bool(technical_failure),"",None,False,
        totals["llm_call_count"],totals["total_input_tokens"],totals["total_output_tokens"],
        totals["total_tokens"],totals["total_llm_latency_sec"],time.perf_counter()-pipeline_start,cost,
        getattr(parse_result,"raw_candidate_count",None),
        len(parse_result.candidates) if parse_result is not None else None,
        getattr(parse_result,"exact_duplicate_count",None),
        getattr(quality_result,"valid_candidate_count",None),
        getattr(quality_result,"rejected_candidate_count",None),
        getattr(semantic_result,"cluster_count",None),
        getattr(semantic_result,"semantic_duplicates_removed",None),
        None,None,None,
        float(semantic_threshold) if semantic_threshold is not None else None,
        None,attempt_logs,failure_stage,error_message
    )

def run_baseline_request(source_id: Any,source_text: str,provider: str,repetition: int,
                         domain: Optional[str],model_id: str) -> Experiment1Result:
    pipeline_start=time.perf_counter()
    prompt=build_prompt("baseline",source_text)
    generation_seed=make_generation_seed(provider,model_id,source_id,repetition,"baseline",1)
    llm_results=[]; attempt_logs=[]
    try:
        llm=call_llm(provider,prompt,model_id,generation_seed)
        llm_results.append(llm)
    except Exception as exc:
        return _failure_result(
            source_id,source_text,domain,provider,model_id,"baseline",repetition,pipeline_start,
            llm_results,True,"llm_call",str(exc),attempt_logs
        )

    parse_result=parse_baseline_response(llm.text)
    attempt_logs.append({
        "attempt":1,"generation_seed":generation_seed,"raw_response":llm.text,
        "llm_metadata":to_json_safe(llm),"parse_metadata":to_json_safe(parse_result),
    })

    if not parse_result.candidates:
        return _failure_result(
            source_id,source_text,domain,provider,model_id,"baseline",repetition,pipeline_start,
            llm_results,False,"parsing",parse_result.error_message or "No baseline output.",
            attempt_logs,parse_result=parse_result
        )

    baseline_text=parse_result.candidates[0]
    quality=validate_baseline_output(source_text,baseline_text)
    attempt_logs[0]["baseline_quality"]=to_json_safe(quality)

    totals=aggregate_llm_results(llm_results)
    cost=estimate_api_cost_usd(provider,totals["total_input_tokens"],totals["total_output_tokens"])

    # Baseline is never regenerated/replaced due to low quality.
    return Experiment1Result(
        EXPERIMENT_NAME,source_id,source_text,domain,provider,model_id,"baseline",repetition,
        True,False,baseline_text,float(quality.semantic_fidelity),bool(quality.quality_pass),
        totals["llm_call_count"],totals["total_input_tokens"],totals["total_output_tokens"],
        totals["total_tokens"],totals["total_llm_latency_sec"],time.perf_counter()-pipeline_start,cost,
        parse_result.raw_candidate_count,len(parse_result.candidates),parse_result.exact_duplicate_count,
        1 if quality.quality_pass else 0,0 if quality.quality_pass else 1,
        None,None,None,None,None,None,None,attempt_logs,None,None
    )

def run_kazdiv_request(source_id: Any,source_text: str,provider: str,repetition: int,
                       domain: Optional[str],model_id: str,semantic_threshold: float) -> Experiment1Result:
    pipeline_start=time.perf_counter()
    prompt=build_prompt("kazdiv",source_text)
    llm_results=[]; attempt_logs=[]
    last_parse=None; last_quality=None; last_semantic=None
    failure_stage=None; error_message=None

    for attempt in range(1,MAX_KAZDIV_CALLS+1):
        generation_seed=make_generation_seed(provider,model_id,source_id,repetition,"kazdiv",attempt)
        try:
            llm=call_llm(provider,prompt,model_id,generation_seed)
            llm_results.append(llm)
        except Exception as exc:
            attempt_logs.append({
                "attempt":attempt,"generation_seed":generation_seed,"technical_error":str(exc)
            })
            return _failure_result(
                source_id,source_text,domain,provider,model_id,"kazdiv",repetition,pipeline_start,
                llm_results,True,"llm_call",str(exc),attempt_logs,semantic_threshold
            )

        parse_result=parse_kazdiv_response(llm.text,K_CANDIDATES)
        last_parse=parse_result
        log={
            "attempt":attempt,"generation_seed":generation_seed,"raw_response":llm.text,
            "llm_metadata":to_json_safe(llm),"parse_metadata":to_json_safe(parse_result),
        }

        if not parse_result.candidates:
            log["stage_result"]="parsing_failed"
            attempt_logs.append(log)
            failure_stage="parsing"
            error_message=parse_result.error_message or "No candidates parsed."
            continue

        quality_result=filter_candidate_pool(source_text,parse_result.candidates,FIDELITY_MIN)
        last_quality=quality_result
        log["quality_metadata"]=to_json_safe(quality_result)

        if quality_result.valid_candidate_count==0:
            log["stage_result"]="quality_filter_failed"
            attempt_logs.append(log)
            failure_stage="quality_filtering"
            error_message="No candidate passed predefined quality constraints."
            continue

        semantic_result=semantic_deduplicate_quality_result(quality_result,semantic_threshold)
        last_semantic=semantic_result
        log["semantic_clustering_metadata"]=to_json_safe(semantic_result)

        if semantic_result.cluster_count==0:
            log["stage_result"]="semantic_clustering_failed"
            attempt_logs.append(log)
            failure_stage="semantic_clustering"
            error_message="No semantic clusters remained."
            continue

        selection=select_kazdiv_response(
            semantic_result,provider,model_id,source_id,repetition
        )
        log["selection_metadata"]=to_json_safe(selection)
        log["stage_result"]="success" if selection.selection_success else "selection_failed"
        attempt_logs.append(log)

        if selection.selection_success:
            totals=aggregate_llm_results(llm_results)
            cost=estimate_api_cost_usd(
                provider,totals["total_input_tokens"],totals["total_output_tokens"]
            )
            return Experiment1Result(
                EXPERIMENT_NAME,source_id,source_text,domain,provider,model_id,"kazdiv",repetition,
                True,False,selection.selected_text,float(selection.selected_fidelity),True,
                totals["llm_call_count"],totals["total_input_tokens"],totals["total_output_tokens"],
                totals["total_tokens"],totals["total_llm_latency_sec"],time.perf_counter()-pipeline_start,cost,
                parse_result.raw_candidate_count,len(parse_result.candidates),parse_result.exact_duplicate_count,
                quality_result.valid_candidate_count,quality_result.rejected_candidate_count,
                semantic_result.cluster_count,semantic_result.semantic_duplicates_removed,
                selection.selected_cluster_id,selection.selected_original_candidate_index,
                selection.selection_seed,float(semantic_threshold),attempt,attempt_logs,None,None
            )

        failure_stage="selection"
        error_message="Final cluster selection failed."

    # Strict method failure: no fallback to baseline or invalid candidate.
    return _failure_result(
        source_id,source_text,domain,provider,model_id,"kazdiv",repetition,pipeline_start,
        llm_results,False,failure_stage or "unknown",
        error_message or "KAZ-Div produced no valid output.",attempt_logs,semantic_threshold,
        last_parse,last_quality,last_semantic
    )

def run_experiment1_request(source_id: Any,source_text: str,provider: str,method: str,
                            repetition: int,domain: Optional[str]=None,
                            model_id: Optional[str]=None,
                            semantic_threshold: Optional[float]=None) -> Experiment1Result:
    provider=provider.strip().lower()
    method=method.strip().lower()
    model_id=model_id or MODEL_IDS[provider]
    if method=="baseline":
        return run_baseline_request(source_id,source_text,provider,repetition,domain,model_id)
    if method=="kazdiv":
        if semantic_threshold is None:
            raise ValueError("semantic_threshold must be explicit for KAZ-Div.")
        return run_kazdiv_request(
            source_id,source_text,provider,repetition,domain,model_id,float(semantic_threshold)
        )
    raise ValueError(f"Unknown method: {method}")

print("End-to-end runner: READY")

# %%
# ============================================================
# Cell 12A. Four-model preflight
# ============================================================

MODEL_PREFLIGHT = []

for provider in (
    ENABLED_PROVIDERS
):

    error = (
        provider_readiness_error(
            provider
        )
    )

    MODEL_PREFLIGHT.append(
        {
            "provider":
                provider,

            "model_id":
                MODEL_IDS[
                    provider
                ],

            "execution":
                (
                    "local/lazy"
                    if provider in LOCAL_PROVIDERS
                    else "API"
                ),

            "configured":
                error is None,

            "error":
                error,
        }
    )


MODEL_PREFLIGHT_DF = pd.DataFrame(
    MODEL_PREFLIGHT
)


display(
    MODEL_PREFLIGHT_DF
)


with open(
    ROOT_OUTPUT_DIR
    /
    "experiment1_final_model_preflight.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        MODEL_PREFLIGHT,
        f,
        ensure_ascii=False,
        indent=2,
    )


print(
    "All four providers configured:",
    bool(
        MODEL_PREFLIGHT_DF[
            "configured"
        ]
        .all()
    ),
)

# %%
# ============================================================
# Cell 13. Optional real smoke test
# ============================================================
#
# Start with one local model to avoid paid API calls.
# Recommended first smoke test: kazllm.
# ============================================================

RUN_REAL_SMOKE_TEST = False

SMOKE_PROVIDER = "kazllm"

SMOKE_SOURCE_INDEX = 0

SMOKE_REPETITION = 1


if RUN_REAL_SMOKE_TEST:

    row = (
        sources
        .iloc[
            SMOKE_SOURCE_INDEX
        ]
    )


    print(
        "SOURCE:",
        row[
            "text"
        ],
    )


    baseline_smoke = (
        run_experiment1_request(
            row[
                "id"
            ],

            row[
                "text"
            ],

            SMOKE_PROVIDER,

            "baseline",

            SMOKE_REPETITION,

            row[
                "domain"
            ],
        )
    )


    print(
        "\nBASELINE"
    )

    print(
        json.dumps(
            to_json_safe(
                baseline_smoke
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


    kazdiv_smoke = (
        run_experiment1_request(
            row[
                "id"
            ],

            row[
                "text"
            ],

            SMOKE_PROVIDER,

            "kazdiv",

            SMOKE_REPETITION,

            row[
                "domain"
            ],

            semantic_threshold=(
                PILOT_NEAR_DUPLICATE_THRESHOLD
            ),
        )
    )


    print(
        "\nKAZ-DIV"
    )

    print(
        json.dumps(
            to_json_safe(
                kazdiv_smoke
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


else:

    print(
        "Smoke test skipped. "
        "Set RUN_REAL_SMOKE_TEST=True."
    )

# %%
# ============================================================
# Cell 14. Local checkpoint/resume utilities
# ============================================================
SUMMARY_CSV_COLUMNS = [
    "experiment","task_key","source_id","domain","provider","model_id","method","repetition",
    "run_success","technical_failure","final_text","final_semantic_fidelity","final_quality_pass",
    "llm_call_count","total_input_tokens","total_output_tokens","total_tokens",
    "total_llm_latency_sec","total_pipeline_latency_sec","estimated_api_cost_usd",
    "raw_candidate_count","parsed_candidate_count","exact_duplicate_count",
    "quality_valid_candidate_count","quality_rejected_candidate_count",
    "semantic_cluster_count","semantic_duplicates_removed","selected_cluster_id",
    "selected_original_candidate_index","selection_seed","semantic_threshold_used",
    "kazdiv_attempt_used","failure_stage","error_message",
]

def write_json_atomic(path: Path,data: Dict[str,Any]):
    temp=Path(str(path)+".tmp")
    with open(temp,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,default=str)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)

def append_jsonl(path: Path,record: Dict[str,Any]):
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False,default=str)+"\n")
        f.flush(); os.fsync(f.fileno())

def load_jsonl(path: Path) -> List[Dict[str,Any]]:
    if not path.exists():
        return []
    records=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

def looks_like_placeholder(value: Any) -> bool:
    if value is None or not str(value).strip():
        return True
    upper=str(value).upper()
    return any(token in upper for token in ["YOUR_","PASTE_","MODEL_ID_HERE","TODO"])

def make_task_key(source_id: Any,provider: str,model_id: str,method: str,repetition: int) -> str:
    material=f"{source_id}|{provider}|{model_id}|{method}|{repetition}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

def build_task_grid(source_frame: pd.DataFrame,providers: List[str],
                    methods: List[str],repetitions: List[int]) -> List[Dict[str,Any]]:
    tasks=[]
    for _,row in source_frame.iterrows():
        for repetition in repetitions:
            for provider in providers:
                model_id=MODEL_IDS[provider]
                for method in methods:
                    tasks.append({
                        "task_key":make_task_key(row["id"],provider,model_id,method,repetition),
                        "source_id":row["id"],"source_text":row["text"],"domain":row["domain"],
                        "provider":provider,"model_id":model_id,"method":method,"repetition":repetition,
                    })
    return tasks

def rebuild_summary_csv(records: List[Dict[str,Any]],path: Path):
    with open(path,"w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column:record.get(column) for column in SUMMARY_CSV_COLUMNS})

def create_run_signature(mode: str,providers: List[str],methods: List[str],
                         repetitions: List[int],semantic_threshold: float,
                         source_frame: pd.DataFrame) -> Dict[str,Any]:
    source_hash=hashlib.sha256(
        source_frame[["id","text","domain"]].to_csv(index=False).encode("utf-8")
    ).hexdigest()
    configuration={
        "mode":mode,"random_seed":RANDOM_SEED,"providers":providers,"methods":methods,
        "repetitions":repetitions,"model_ids":{p:MODEL_IDS[p] for p in providers},
        "k_candidates":K_CANDIDATES,"max_kazdiv_calls":MAX_KAZDIV_CALLS,
        "fidelity_min":FIDELITY_MIN,"min_length_ratio":MIN_LENGTH_RATIO,
        "max_length_ratio":MAX_LENGTH_RATIO,"semantic_threshold":semantic_threshold,
        "embedding_model":EMBEDDING_MODEL_NAME,"embedding_pooling":"CLS",
        "source_hash":source_hash,
    }
    signature=hashlib.sha256(
        json.dumps(configuration,ensure_ascii=False,sort_keys=True,default=str).encode("utf-8")
    ).hexdigest()
    return {"signature":signature,"configuration":configuration}

def create_local_zip(output_dir: Path) -> Path:
    zip_path=output_dir.parent/f"{output_dir.name}_latest.zip"
    if zip_path.exists():
        zip_path.unlink()
    created=shutil.make_archive(
        base_name=str(zip_path.with_suffix("")),format="zip",
        root_dir=str(output_dir.parent),base_dir=output_dir.name
    )
    return Path(created)

print("Checkpoint/resume helpers: READY")

# %%
# ============================================================
# Cell 15. Generic local task-grid runner
# ============================================================
def run_task_grid(tasks: List[Dict[str,Any]],output_dir: Path,semantic_threshold: float,
                  mode: str,max_tasks_this_run: Optional[int]=None,
                  max_consecutive_technical_failures: int=3,
                  auto_zip_every: Optional[int]=25):

    output_dir.mkdir(parents=True,exist_ok=True)

    results_jsonl=output_dir/"experiment1_results.jsonl"
    technical_errors_jsonl=output_dir/"experiment1_technical_errors.jsonl"
    summary_csv=output_dir/"experiment1_results_summary.csv"
    checkpoint_file=output_dir/"experiment1_checkpoint.json"
    signature_file=output_dir/"experiment1_signature.json"

    providers=sorted(set(task["provider"] for task in tasks))

    # Main experiment can be executed provider-by-provider in separate
    # Colab sessions. The saved experiment signature must therefore
    # describe the FULL planned provider set rather than only the
    # providers in the current session.
    signature_providers=(
        list(ENABLED_PROVIDERS)
        if mode=="main"
        else providers
    )

    readiness_errors = {
        provider:
            provider_readiness_error(
                provider
            )
        for provider
        in providers
    }

    readiness_errors = {
        provider:
            error
        for provider, error
        in readiness_errors.items()
        if error is not None
    }

    if readiness_errors:
        raise RuntimeError(
            "Provider preflight failed:\n"
            +
            json.dumps(
                readiness_errors,
                ensure_ascii=False,
                indent=2,
            )
        )

    task_source_ids=set(task["source_id"] for task in tasks)
    signature_sources=sources[sources["id"].isin(task_source_ids)].copy()

    run_signature=create_run_signature(
        mode,signature_providers,sorted(set(t["method"] for t in tasks)),
        sorted(set(int(t["repetition"]) for t in tasks)),
        float(semantic_threshold),signature_sources,
    )

    if signature_file.exists():
        with open(signature_file,"r",encoding="utf-8") as f:
            saved=json.load(f)
        if saved.get("signature")!=run_signature["signature"]:
            raise RuntimeError(
                "Run configuration changed. Do not mix outputs from different settings."
            )
    else:
        write_json_atomic(
            signature_file,{**run_signature,"created_at":datetime.now().isoformat()}
        )

    existing=load_jsonl(results_jsonl)
    finalized={r["task_key"]:r for r in existing if r.get("task_key")}
    completed=set(finalized)

    # Keep providers together. This prevents repeated GPU model swaps
    # between KazLLM and Llama during a run.
    provider_order={
        provider:index
        for index,provider
        in enumerate(ENABLED_PROVIDERS)
    }

    ordered_tasks=sorted(
        tasks,
        key=lambda t:(
            provider_order.get(t["provider"],999),
            str(t["source_id"]),
            int(t["repetition"]),
            str(t["method"]),
        ),
    )

    pending=[
        t for t in ordered_tasks
        if t["task_key"] not in completed
    ]

    if max_tasks_this_run is not None:
        pending=pending[:max_tasks_this_run]

    success_session=0; method_fail_session=0; technical_fail_session=0
    consecutive_technical=0

    print(f"Mode={mode} | total={len(tasks)} | finalized={len(completed)} | pending now={len(pending)}")

    for session_index,task in enumerate(pending,start=1):
        print(
            f"\n[{session_index}/{len(pending)}] "
            f"source={task['source_id']} | {task['provider']} | {task['method']} | rep={task['repetition']}"
        )

        write_json_atomic(
            checkpoint_file,
            {
                "mode":mode,"updated_at":datetime.now().isoformat(),
                "total_tasks":len(tasks),"finalized":len(completed),
                "remaining":len(tasks)-len(completed),"current_task":task,
            },
        )

        try:
            result=run_experiment1_request(
                source_id=task["source_id"],source_text=task["source_text"],
                provider=task["provider"],method=task["method"],repetition=task["repetition"],
                domain=task["domain"],model_id=task["model_id"],
                semantic_threshold=float(semantic_threshold),
            )
            record=to_json_safe(result)
        except Exception as exc:
            technical_fail_session+=1
            consecutive_technical+=1
            append_jsonl(
                technical_errors_jsonl,
                {
                    "timestamp":datetime.now().isoformat(),"task":task,
                    "exception_type":type(exc).__name__,"error_message":str(exc),
                    "traceback":traceback.format_exc(),
                },
            )
            print("TECHNICAL ERROR:",type(exc).__name__,str(exc))
            if consecutive_technical>=max_consecutive_technical_failures:
                print("Fail-safe stop.")
                break
            continue

        record["task_key"]=task["task_key"]
        record["saved_at"]=datetime.now().isoformat()
        record["run_mode"]=mode
        record["experiment_signature"]=run_signature["signature"]

        # Provider/API failure remains pending for a future rerun.
        if record.get("technical_failure") is True:
            technical_fail_session+=1
            consecutive_technical+=1
            append_jsonl(technical_errors_jsonl,record)
            print("TECHNICAL FAILURE — remains pending")
            if consecutive_technical>=max_consecutive_technical_failures:
                print("Fail-safe stop.")
                break
            continue

        # Genuine method failure is finalized and retained for failure-rate analysis.
        append_jsonl(results_jsonl,record)
        finalized[task["task_key"]]=record
        completed.add(task["task_key"])
        consecutive_technical=0

        if record.get("run_success") is True:
            success_session+=1
            print("SUCCESS")
        else:
            method_fail_session+=1
            print("METHOD FAILURE:",record.get("failure_stage"))

        ordered=[finalized[k] for k in sorted(finalized)]
        rebuild_summary_csv(ordered,summary_csv)

        if auto_zip_every and len(completed)%auto_zip_every==0:
            print("ZIP snapshot:",create_local_zip(output_dir))

    ordered=[finalized[k] for k in sorted(finalized)]
    rebuild_summary_csv(ordered,summary_csv)

    completion={
        "mode":mode,"updated_at":datetime.now().isoformat(),
        "total_tasks":len(tasks),"finalized_tasks":len(completed),
        "successful_outputs":sum(r.get("run_success") is True for r in finalized.values()),
        "method_failures":sum(r.get("run_success") is False for r in finalized.values()),
        "remaining_tasks":len(tasks)-len(completed),
        "success_this_session":success_session,
        "method_failures_this_session":method_fail_session,
        "technical_failures_this_session":technical_fail_session,
    }
    write_json_atomic(output_dir/"experiment1_completion.json",completion)
    print(json.dumps(completion,ensure_ascii=False,indent=2))
    return completion

print("Generic local runner: READY")

# %%
# ============================================================
# Cell 16. Pilot experiment
# ============================================================
#
# Recommended pilot:
# 10 sources × 5 repetitions × 1 provider × 2 methods = 100
#
# Use one provider for semantic-threshold calibration.
# KazLLM is the default because it is local and Kazakh-focused.
# ============================================================

RUN_PILOT = False

PILOT_PROVIDER = "kazllm"

PILOT_N_SOURCES = 10

PILOT_MAX_TASKS_THIS_RUN = None


pilot_parts = []

for domain in EXPECTED_DOMAINS:

    part = (
        sources[
            sources[
                "domain"
            ]
            ==
            domain
        ]
        .head(
            2
        )
    )

    if len(
        part
    ):
        pilot_parts.append(
            part
        )


if (
    pilot_parts
    and
    sum(
        len(x)
        for x in pilot_parts
    )
    >=
    PILOT_N_SOURCES
):

    pilot_sources = (
        pd.concat(
            pilot_parts,
            ignore_index=True,
        )
        .head(
            PILOT_N_SOURCES
        )
    )

else:

    pilot_sources = (
        sources
        .head(
            PILOT_N_SOURCES
        )
        .copy()
    )


PILOT_TASKS = (
    build_task_grid(
        pilot_sources,

        [
            PILOT_PROVIDER
        ],

        [
            "baseline",
            "kazdiv",
        ],

        list(
            range(
                1,
                N_REPEATS
                +
                1,
            )
        ),
    )
)


print(
    "Pilot sources:",
    len(
        pilot_sources
    ),
)

print(
    "Pilot tasks:",
    len(
        PILOT_TASKS
    ),
)


display(
    pilot_sources[
        [
            "id",
            "domain",
            "text",
        ]
    ]
)


if RUN_PILOT:

    PILOT_REPORT = (
        run_task_grid(
            PILOT_TASKS,

            PILOT_OUTPUT_DIR,

            PILOT_NEAR_DUPLICATE_THRESHOLD,

            "pilot",

            max_tasks_this_run=(
                PILOT_MAX_TASKS_THIS_RUN
            ),

            auto_zip_every=25,
        )
    )

else:

    print(
        "Pilot NOT started. "
        "Set RUN_PILOT=True and rerun Cell 16."
    )

# %%
# ============================================================
# Cell 17. Pilot threshold calibration
# ============================================================
CALIBRATION_THRESHOLDS = [0.90,0.92,0.94,0.95,0.96,0.97,0.98]

pilot_results_file=PILOT_OUTPUT_DIR/"experiment1_results.jsonl"
pilot_records=load_jsonl(pilot_results_file)
pilot_kazdiv=[
    r for r in pilot_records
    if r.get("method")=="kazdiv" and r.get("run_success") is True
]

calibration_rows=[]
pair_rows=[]

for record in pilot_kazdiv:
    attempts=record.get("attempt_logs") or []
    successful_attempts=[
        a for a in attempts
        if a.get("stage_result")=="success"
    ]
    if not successful_attempts:
        continue

    attempt=successful_attempts[-1]
    quality_metadata=attempt.get("quality_metadata") or {}
    qrecords=quality_metadata.get("records") or []
    valid=[
        item for item in qrecords
        if item.get("quality_pass") is True
    ]
    candidates=[
        normalize_text(item.get("candidate_text"))
        for item in valid
        if normalize_text(item.get("candidate_text"))
    ]
    if len(candidates)<2:
        continue

    matrix=analyse_candidate_semantics(
        record["source_text"],candidates
    )["similarity_matrix"]

    for threshold in CALIBRATION_THRESHOLDS:
        components=connected_components(
            build_semantic_duplicate_graph(matrix,threshold)
        )
        calibration_rows.append({
            "source_id":record["source_id"],
            "provider":record["provider"],
            "repetition":record["repetition"],
            "threshold":threshold,
            "valid_candidates":len(candidates),
            "clusters":len(components),
            "duplicates_removed":len(candidates)-len(components),
            "retention_rate":len(components)/len(candidates),
        })

    for i in range(len(candidates)):
        for j in range(i+1,len(candidates)):
            pair_rows.append({
                "source_id":record["source_id"],
                "source_text":record["source_text"],
                "provider":record["provider"],
                "repetition":record["repetition"],
                "candidate_a":candidates[i],
                "candidate_b":candidates[j],
                "similarity":float(matrix[i,j]),
            })

calibration_df=pd.DataFrame(calibration_rows)
boundary_pairs_df=pd.DataFrame(pair_rows)

if len(calibration_df):
    calibration_summary_df=(
        calibration_df.groupby("threshold")
        .agg(
            pools=("source_id","count"),
            mean_valid_candidates=("valid_candidates","mean"),
            mean_clusters=("clusters","mean"),
            mean_duplicates_removed=("duplicates_removed","mean"),
            mean_retention_rate=("retention_rate","mean"),
        )
        .reset_index()
    )
    calibration_df.to_csv(
        PILOT_OUTPUT_DIR/"threshold_calibration_detailed.csv",
        index=False,encoding="utf-8-sig"
    )
    calibration_summary_df.to_csv(
        PILOT_OUTPUT_DIR/"threshold_calibration_summary.csv",
        index=False,encoding="utf-8-sig"
    )
    display(calibration_summary_df)
else:
    print("No successful KAZ-Div pilot pools yet.")

if len(boundary_pairs_df):
    boundary_pairs_df["distance_to_095"]=(boundary_pairs_df["similarity"]-0.95).abs()
    manual_validation_sample=(
        boundary_pairs_df.sort_values("distance_to_095")
        .head(min(150,len(boundary_pairs_df)))
        .copy()
    )
    manual_validation_sample["human_near_duplicate"]=""
    manual_validation_sample["human_comment"]=""
    manual_validation_sample.to_csv(
        PILOT_OUTPUT_DIR/"threshold_manual_validation_sample.csv",
        index=False,encoding="utf-8-sig"
    )
    print(
        "Manual review file:",
        PILOT_OUTPUT_DIR/"threshold_manual_validation_sample.csv"
    )

# %%
# ============================================================
# Cell 18. Freeze the final semantic threshold
# ============================================================
#
# After pilot + manual review:
#
# 1) Edit Cell 3:
#    FROZEN_NEAR_DUPLICATE_THRESHOLD = <chosen value>
#    THRESHOLD_STATUS = "FROZEN"
#
# 2) Rerun Cell 3 and this cell.
# ============================================================
if FROZEN_NEAR_DUPLICATE_THRESHOLD is None:
    print("Threshold is NOT frozen.")
    print("Do not start the main 4000-observation experiment.")
else:
    if THRESHOLD_STATUS!="FROZEN":
        raise RuntimeError("Set THRESHOLD_STATUS='FROZEN' after pilot calibration.")
    if not (0.0<float(FROZEN_NEAR_DUPLICATE_THRESHOLD)<=1.0):
        raise ValueError("Frozen threshold must be in (0,1].")

    freeze_record={
        "frozen_at":datetime.now().isoformat(),
        "frozen_near_duplicate_threshold":float(FROZEN_NEAR_DUPLICATE_THRESHOLD),
        "fidelity_min":FIDELITY_MIN,
        "embedding_model":EMBEDDING_MODEL_NAME,
        "embedding_pooling":"CLS",
        "note":"Frozen after pilot calibration and manual near-duplicate review.",
    }
    write_json_atomic(
        ROOT_OUTPUT_DIR/"experiment1_frozen_threshold.json",
        freeze_record
    )
    print(json.dumps(freeze_record,ensure_ascii=False,indent=2))

# %%
# ============================================================
# Cell 19. Main 4000-observation experiment
# ============================================================
#
# Full design:
#
# 100 sources × 5 repetitions × 4 providers × 2 methods
# = 4000 observations
#
# IMPORTANT FOR COLAB:
# You may run providers in separate sessions while writing to
# the same MAIN_OUTPUT_DIR. Checkpoint/resume will merge them.
#
# Examples:
#
# MAIN_PROVIDERS_THIS_RUN = ["kazllm"]
# MAIN_PROVIDERS_THIS_RUN = ["llama"]
# MAIN_PROVIDERS_THIS_RUN = ["openai", "anthropic"]
# MAIN_PROVIDERS_THIS_RUN = list(ENABLED_PROVIDERS)
# ============================================================

RUN_MAIN_EXPERIMENT = False


# Full experiment design — DO NOT CHANGE FOR FINAL ANALYSIS.
MAIN_PROVIDERS = list(
    ENABLED_PROVIDERS
)

MAIN_METHODS = [
    "baseline",
    "kazdiv",
]

MAIN_REPETITIONS = list(
    range(
        1,
        N_REPEATS
        +
        1,
    )
)


# Providers executed in the current Colab session.
# Running one local provider per session is recommended.
MAIN_PROVIDERS_THIS_RUN = list(
    MAIN_PROVIDERS
)


# None = all remaining tasks in this session.
# For a small real test, use e.g. 8.
MAIN_MAX_TASKS_THIS_RUN = None


FULL_MAIN_TASKS = (
    build_task_grid(
        sources,

        MAIN_PROVIDERS,

        MAIN_METHODS,

        MAIN_REPETITIONS,
    )
)


SESSION_MAIN_TASKS = [
    task
    for task
    in FULL_MAIN_TASKS
    if task[
        "provider"
    ]
    in MAIN_PROVIDERS_THIS_RUN
]


print(
    "Full main task count:",
    len(
        FULL_MAIN_TASKS
    ),
)

print(
    "Current-session task count:",
    len(
        SESSION_MAIN_TASKS
    ),
)

print(
    "Providers this session:",
    MAIN_PROVIDERS_THIS_RUN,
)


if (
    len(
        sources
    )
    ==
    100
    and
    len(
        MAIN_PROVIDERS
    )
    ==
    4
    and
    len(
        MAIN_METHODS
    )
    ==
    2
    and
    len(
        MAIN_REPETITIONS
    )
    ==
    5
):

    assert (
        len(
            FULL_MAIN_TASKS
        )
        ==
        4000
    )

    print(
        "Design verified: "
        "100 × 5 × 4 × 2 = 4000"
    )


if RUN_MAIN_EXPERIMENT:

    if (
        FROZEN_NEAR_DUPLICATE_THRESHOLD
        is None
    ):

        raise RuntimeError(
            "Main run blocked: "
            "semantic threshold is not frozen."
        )


    if (
        THRESHOLD_STATUS
        !=
        "FROZEN"
    ):

        raise RuntimeError(
            "Main run blocked: "
            "THRESHOLD_STATUS must be 'FROZEN'."
        )


    invalid_session_providers = [
        p
        for p in MAIN_PROVIDERS_THIS_RUN
        if p not in MAIN_PROVIDERS
    ]

    if invalid_session_providers:

        raise ValueError(
            "Unknown providers in "
            "MAIN_PROVIDERS_THIS_RUN: "
            f"{invalid_session_providers}"
        )


    MAIN_REPORT = (
        run_task_grid(
            SESSION_MAIN_TASKS,

            MAIN_OUTPUT_DIR,

            float(
                FROZEN_NEAR_DUPLICATE_THRESHOLD
            ),

            "main",

            max_tasks_this_run=(
                MAIN_MAX_TASKS_THIS_RUN
            ),

            auto_zip_every=25,
        )
    )


else:

    print(
        "Main experiment NOT started."
    )

    print(
        "Freeze the threshold, then set "
        "RUN_MAIN_EXPERIMENT=True."
    )

# %%
# ============================================================
# Cell 20. Main metrics
# ============================================================
MAIN_RESULTS_JSONL=MAIN_OUTPUT_DIR/"experiment1_results.jsonl"
main_records=load_jsonl(MAIN_RESULTS_JSONL)

def tokenize_simple(text: str) -> List[str]:
    return re.findall(
        r"\b[\wәіңғүұқөһ]+\b",
        normalize_text(text).lower(),
        flags=re.UNICODE,
    )

def unique_rate(texts: List[str]) -> float:
    if not texts:
        return np.nan
    canonical=[normalize_for_exact_match(x) for x in texts]
    return float(len(set(canonical))/len(canonical))

def distinct_n(texts: List[str],n: int) -> float:
    ngrams=[]
    for text in texts:
        tokens=tokenize_simple(text)
        for i in range(max(0,len(tokens)-n+1)):
            ngrams.append(tuple(tokens[i:i+n]))
    return float(len(set(ngrams))/len(ngrams)) if ngrams else np.nan

def pairwise_semantic_diversity(texts: List[str]) -> float:
    if len(texts)<2:
        return np.nan
    emb=encode_texts(texts)
    matrix=emb@emb.T
    vals=matrix[np.triu_indices(len(texts),k=1)]
    return float(np.mean(1.0-vals))

def self_bleu(texts: List[str]) -> float:
    if len(texts)<2:
        return np.nan
    tokenized=[tokenize_simple(x) for x in texts]
    smoothing=SmoothingFunction().method1
    scores=[]
    for i,hypothesis in enumerate(tokenized):
        refs=[tokenized[j] for j in range(len(tokenized)) if j!=i]
        if hypothesis and refs:
            scores.append(float(sentence_bleu(
                refs,hypothesis,weights=(0.5,0.5,0.0,0.0),smoothing_function=smoothing
            )))
    return float(np.mean(scores)) if scores else np.nan

if not main_records:
    print("No main results yet.")
else:
    main_df=pd.DataFrame(main_records)

    # Observation-level operational summary.
    operational_summary_df=(
        main_df.groupby(["provider","method"])
        .agg(
            finalized_observations=("task_key","count"),
            successful_outputs=("run_success","sum"),
            mean_llm_calls=("llm_call_count","mean"),
            mean_input_tokens=("total_input_tokens","mean"),
            mean_output_tokens=("total_output_tokens","mean"),
            mean_total_tokens=("total_tokens","mean"),
            mean_llm_latency_sec=("total_llm_latency_sec","mean"),
            mean_pipeline_latency_sec=("total_pipeline_latency_sec","mean"),
            mean_estimated_api_cost_usd=("estimated_api_cost_usd","mean"),
        )
        .reset_index()
    )
    operational_summary_df["success_rate"]=(
        operational_summary_df["successful_outputs"]/
        operational_summary_df["finalized_observations"]
    )
    operational_summary_df.to_csv(
        MAIN_OUTPUT_DIR/"experiment1_operational_summary.csv",
        index=False,encoding="utf-8-sig"
    )

    # Source-level diversity/quality.
    rows=[]
    for (source_id,provider,method),group in main_df.groupby(["source_id","provider","method"]):
        success_group=group[group["run_success"]==True]
        texts=[
            normalize_text(x)
            for x in success_group["final_text"].dropna().tolist()
            if normalize_text(x)
        ]
        fidelity=pd.to_numeric(success_group["final_semantic_fidelity"],errors="coerce")
        quality=(
            success_group["final_quality_pass"].astype(str).str.lower()
            .map({"true":1.0,"false":0.0})
        )
        rows.append({
            "source_id":source_id,
            "provider":provider,
            "method":method,
            "n_finalized":len(group),
            "n_success":len(texts),
            "success_rate":len(texts)/len(group) if len(group) else np.nan,
            "complete_five_outputs":len(texts)==N_REPEATS,
            "unique_rate":unique_rate(texts) if texts else np.nan,
            "distinct_1":distinct_n(texts,1) if texts else np.nan,
            "distinct_2":distinct_n(texts,2) if texts else np.nan,
            "semantic_diversity":pairwise_semantic_diversity(texts) if len(texts)>=2 else np.nan,
            "self_bleu":self_bleu(texts) if len(texts)>=2 else np.nan,
            "mean_fidelity":fidelity.mean(),
            "quality_pass_rate":quality.mean(),
        })

    source_level_metrics_df=pd.DataFrame(rows)
    source_level_metrics_df.to_csv(
        MAIN_OUTPUT_DIR/"experiment1_source_level_metrics.csv",
        index=False,encoding="utf-8-sig"
    )

    # Primary complete-case population: both methods have all five outputs.
    complete_flags=source_level_metrics_df.pivot_table(
        index=["source_id","provider"],columns="method",
        values="complete_five_outputs",aggfunc="first"
    )
    if {"baseline","kazdiv"}.issubset(complete_flags.columns):
        complete_pairs=(
            complete_flags[
                (complete_flags["baseline"]==True)&
                (complete_flags["kazdiv"]==True)
            ]
            .reset_index()[["source_id","provider"]]
        )
        complete_case_metrics_df=source_level_metrics_df.merge(
            complete_pairs,on=["source_id","provider"],how="inner"
        )
    else:
        complete_case_metrics_df=source_level_metrics_df.iloc[0:0].copy()

    complete_case_metrics_df.to_csv(
        MAIN_OUTPUT_DIR/"experiment1_complete_case_metrics.csv",
        index=False,encoding="utf-8-sig"
    )

    if len(complete_case_metrics_df):
        summary_metrics_df=(
            complete_case_metrics_df.groupby(["provider","method"])
            .agg(
                sources=("source_id","count"),
                unique_rate_mean=("unique_rate","mean"),
                distinct_1_mean=("distinct_1","mean"),
                distinct_2_mean=("distinct_2","mean"),
                semantic_diversity_mean=("semantic_diversity","mean"),
                self_bleu_mean=("self_bleu","mean"),
                fidelity_mean=("mean_fidelity","mean"),
                quality_pass_rate_mean=("quality_pass_rate","mean"),
            )
            .reset_index()
        )
        summary_metrics_df.to_csv(
            MAIN_OUTPUT_DIR/"experiment1_primary_summary_metrics.csv",
            index=False,encoding="utf-8-sig"
        )
        print("Operational summary:")
        display(operational_summary_df)
        print("\nPrimary complete-case summary:")
        display(summary_metrics_df)
    else:
        print("No complete paired source-level population yet.")

# %%
# ============================================================
# Cell 21. Paired statistics
# ============================================================
STAT_METRICS=[
    "unique_rate","distinct_1","distinct_2","semantic_diversity",
    "self_bleu","mean_fidelity","quality_pass_rate",
]
N_BOOTSTRAP=10000
N_PERMUTATIONS=20000

def paired_bootstrap_ci(differences: np.ndarray,seed: int,
                        n_bootstrap: int=N_BOOTSTRAP) -> Tuple[float,float]:
    d=np.asarray(differences,dtype=float)
    d=d[np.isfinite(d)]
    if len(d)==0:
        return np.nan,np.nan
    rng=np.random.default_rng(seed)
    means=np.empty(n_bootstrap,dtype=float)
    for i in range(n_bootstrap):
        means[i]=np.mean(rng.choice(d,size=len(d),replace=True))
    return float(np.quantile(means,0.025)),float(np.quantile(means,0.975))

def paired_sign_flip_pvalue(differences: np.ndarray,seed: int,
                            n_permutations: int=N_PERMUTATIONS) -> float:
    d=np.asarray(differences,dtype=float)
    d=d[np.isfinite(d)]
    if len(d)==0:
        return np.nan
    observed=abs(np.mean(d))
    rng=np.random.default_rng(seed)
    extreme=0
    for _ in range(n_permutations):
        signs=rng.choice([-1.0,1.0],size=len(d))
        if abs(np.mean(d*signs))>=observed:
            extreme+=1
    return float((extreme+1)/(n_permutations+1))

def holm_adjust(pvalues: List[float]) -> List[float]:
    values=np.asarray(pvalues,dtype=float)
    adjusted=np.full(len(values),np.nan,dtype=float)
    valid=np.where(np.isfinite(values))[0]
    if len(valid)==0:
        return adjusted.tolist()
    order=valid[np.argsort(values[valid])]
    running=0.0
    m=len(order)
    for rank,index in enumerate(order):
        running=max(running,(m-rank)*values[index])
        adjusted[index]=min(1.0,running)
    return adjusted.tolist()

if "complete_case_metrics_df" not in globals() or not len(complete_case_metrics_df):
    print("Run Cell 20 after main experiment first.")
else:
    stat_rows=[]
    for provider in sorted(complete_case_metrics_df["provider"].unique()):
        provider_df=complete_case_metrics_df[
            complete_case_metrics_df["provider"]==provider
        ]
        for metric in STAT_METRICS:
            pivot=provider_df.pivot(index="source_id",columns="method",values=metric)
            if not {"baseline","kazdiv"}.issubset(pivot.columns):
                continue
            paired=pivot[["baseline","kazdiv"]].dropna()
            if len(paired)<2:
                continue
            differences=paired["kazdiv"].to_numpy()-paired["baseline"].to_numpy()
            ci_low,ci_high=paired_bootstrap_ci(
                differences,deterministic_seed(RANDOM_SEED,provider,metric,"bootstrap")
            )
            p=paired_sign_flip_pvalue(
                differences,deterministic_seed(RANDOM_SEED,provider,metric,"permutation")
            )
            stat_rows.append({
                "provider":provider,"metric":metric,"n_pairs":len(paired),
                "baseline_mean":float(paired["baseline"].mean()),
                "kazdiv_mean":float(paired["kazdiv"].mean()),
                "mean_difference_kazdiv_minus_baseline":float(np.mean(differences)),
                "bootstrap_95_ci_low":ci_low,"bootstrap_95_ci_high":ci_high,
                "permutation_p":p,
            })

    stats_df=pd.DataFrame(stat_rows)
    if len(stats_df):
        stats_df["holm_adjusted_p"]=np.nan
        for provider,group in stats_df.groupby("provider",sort=False):
            adjusted=holm_adjust(group["permutation_p"].tolist())
            for index,value in zip(group.index,adjusted):
                stats_df.loc[index,"holm_adjusted_p"]=value
        stats_df.to_csv(
            MAIN_OUTPUT_DIR/"experiment1_paired_statistics.csv",
            index=False,encoding="utf-8-sig"
        )
        display(stats_df)
    else:
        print("No paired statistics available.")

# %%
# ============================================================
# Cell 22. Human-validation sample
# ============================================================
HUMAN_SAMPLE_PER_PROVIDER_METHOD_DOMAIN = 5

if "main_df" not in globals():
    print("Run Cell 20 first.")
else:
    successful=main_df[main_df["run_success"]==True].copy()
    sample_parts=[]

    for (provider,method,domain),group in successful.groupby(
        ["provider","method","domain"]
    ):
        n=min(HUMAN_SAMPLE_PER_PROVIDER_METHOD_DOMAIN,len(group))
        if n==0:
            continue
        sample_seed=deterministic_seed(
            RANDOM_SEED,provider,method,domain,"human_validation"
        )%(2**32-1)
        sample_parts.append(
            group.sample(n=n,replace=False,random_state=sample_seed)
        )

    if sample_parts:
        human_validation_df=(
            pd.concat(sample_parts,ignore_index=True)[
                [
                    "task_key","source_id","domain","provider","model_id","method",
                    "repetition","source_text","final_text","final_semantic_fidelity"
                ]
            ].copy()
        )
        human_validation_df["meaning_preservation_1_5"]=""
        human_validation_df["grammar_1_5"]=""
        human_validation_df["naturalness_1_5"]=""
        human_validation_df["annotator_id"]=""
        human_validation_df["comment"]=""

        human_validation_df.to_csv(
            MAIN_OUTPUT_DIR/"experiment1_human_validation_sample.csv",
            index=False,encoding="utf-8-sig"
        )
        print("Human-validation sample rows:",len(human_validation_df))
        display(human_validation_df.head())
    else:
        print("No successful outputs available.")

# %%
# ============================================================
# Cell 23. Publication-ready figures
# ============================================================
if "summary_metrics_df" in globals() and len(summary_metrics_df):
    metrics_to_plot=[
        ("semantic_diversity_mean","Mean Pairwise Semantic Diversity"),
        ("unique_rate_mean","Unique Rate"),
        ("fidelity_mean","Mean Source–Response Fidelity"),
        ("quality_pass_rate_mean","Quality Pass Rate"),
    ]

    for metric,label in metrics_to_plot:
        plot_df=summary_metrics_df.copy()
        plot_df["group"]=plot_df["provider"].astype(str)+" | "+plot_df["method"].astype(str)

        fig,ax=plt.subplots(figsize=(11,5))
        ax.bar(plot_df["group"],plot_df[metric])
        ax.set_title(label)
        ax.set_xlabel("Provider | Method")
        ax.set_ylabel(label)
        ax.tick_params(axis="x",rotation=45)
        fig.tight_layout()

        path=FIGURE_DIR/f"experiment1_{metric}.png"
        fig.savefig(path,dpi=300,bbox_inches="tight")
        plt.show()

if "operational_summary_df" in globals() and len(operational_summary_df):
    plot_df=operational_summary_df.copy()
    plot_df["group"]=plot_df["provider"].astype(str)+" | "+plot_df["method"].astype(str)

    fig,ax=plt.subplots(figsize=(11,5))
    ax.bar(plot_df["group"],plot_df["mean_total_tokens"])
    ax.set_title("Mean Token Usage")
    ax.set_xlabel("Provider | Method")
    ax.set_ylabel("Mean total tokens per observation")
    ax.tick_params(axis="x",rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR/"experiment1_token_usage.png",dpi=300,bbox_inches="tight")
    plt.show()

print("Figures directory:",FIGURE_DIR)

# %%
# ============================================================
# Cell 24. Final integrity report + ZIP export
# ============================================================

def export_experiment1_final_zip(
    download: bool = True
):

    integrity = {
        "experiment":
            EXPERIMENT_NAME,

        "created_at":
            datetime.now().isoformat(),

        "models":
            MODEL_IDS,

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "embedding_pooling":
            "CLS + L2 normalization",

        "fidelity_min":
            FIDELITY_MIN,

        "frozen_near_duplicate_threshold":
            FROZEN_NEAR_DUPLICATE_THRESHOLD,

        "threshold_status":
            THRESHOLD_STATUS,

        "k_candidates":
            K_CANDIDATES,

        "max_kazdiv_calls":
            MAX_KAZDIV_CALLS,

        "hidden_fallback":
            False,

        "baseline_regenerated_on_quality_failure":
            False,

        "stateless_cluster_balanced_selection":
            True,

        "previous_outputs_used":
            False,

        "conversation_history_used":
            False,

        "full_candidate_pools_saved":
            True,

        "local_model_policy":
            "lazy sequential loading",

        "main_expected_observations":
            4000,
    }


    write_json_atomic(
        ROOT_OUTPUT_DIR
        /
        "experiment1_final_integrity_report.json",

        integrity,
    )


    zip_file = Path(
        "/content/KAZ_Div_Experiment1_FINAL.zip"
    )


    if zip_file.exists():
        zip_file.unlink()


    created = shutil.make_archive(
        base_name=str(
            zip_file.with_suffix(
                ""
            )
        ),

        format="zip",

        root_dir=str(
            ROOT_OUTPUT_DIR.parent
        ),

        base_dir=(
            ROOT_OUTPUT_DIR.name
        ),
    )


    created = Path(
        created
    )


    print(
        "ZIP created:",
        created,
    )

    print(
        "Size:",
        round(
            created.stat().st_size
            /
            1024**2,
            2,
        ),
        "MB",
    )


    if download:

        files.download(
            str(
                created
            )
        )


    return created


print(
    "="
    *
    76
)

print(
    "EXPERIMENT 1 FINAL — READY"
)

print(
    "="
    *
    76
)

print(
    "Models:"
)

for provider in (
    ENABLED_PROVIDERS
):

    execution = (
        "LOCAL"
        if provider in LOCAL_PROVIDERS
        else "API"
    )

    print(
        f"  {provider:<10} "
        f"{execution:<5} "
        f"{MODEL_IDS[provider]}"
    )


print()

print(
    "BGE-M3 pooling            : CLS + L2"
)

print(
    "sentence-transformers     : NOT USED"
)

print(
    "Local models              : LAZY / ONE AT A TIME"
)

print(
    "Pilot/main separation     : ENABLED"
)

print(
    "Threshold freeze required : YES"
)

print(
    "Hidden KAZ-Div fallback    : NO"
)

print(
    "Full candidate pools      : SAVED"
)

print(
    "Experiment 4 input        : experiment1_results.jsonl"
)

print(
    "Checkpoint/resume         : ENABLED"
)

print(
    "Primary statistics        : SOURCE-LEVEL PAIRED"
)

print()

print(
    "Recommended execution order:"
)

print(
    "Cell 13 smoke"
)

print(
    "-> Cell 16 pilot"
)

print(
    "-> Cell 17 threshold calibration"
)

print(
    "-> Cell 18 freeze threshold"
)

print(
    "-> Cell 19 main run(s)"
)

print(
    "-> Cells 20–24 analysis/export"
)

print()

print(
    "For local models, run one provider per Colab session:"
)

print(
    'MAIN_PROVIDERS_THIS_RUN = ["kazllm"]'
)

print(
    'MAIN_PROVIDERS_THIS_RUN = ["llama"]'
)

print()

print(
    "To download all experiment outputs:"
)

print(
    "export_experiment1_final_zip(download=True)"
)

print(
    "="
    *
    76
)