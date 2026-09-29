# %% [markdown]
# # KAZ-Div — Experiment 3: Synthetic Kazakh Intent Data + Downstream Classification
# 
# This standalone Colab notebook tests whether KAZ-Div produces **more useful synthetic Kazakh intent data** than conventional single-output generation.
# 
# ## Core experimental design
# 
# For each intent and model family:
# 
# **Baseline**
# \[
# z \rightarrow \text{LLM} \rightarrow u
# \]
# 
# **KAZ-Div**
# \[
# z \rightarrow \{u_1,\ldots,u_K\}
# \rightarrow Q
# \rightarrow D
# \rightarrow C
# \rightarrow u^*
# \]
# 
# where:
# 
# - \(z\) = intent definition,
# - \(u\) = synthetic Kazakh user utterance,
# - \(Q\) = intent-consistency and format/length quality filtering,
# - \(D\) = exact + semantic near-deduplication,
# - \(C\) = semantic clusters,
# - \(u^*\) = stateless cluster-based selected utterance.
# 
# The final downstream test is:
# 
# \[
# \text{synthetic training corpus}
# \rightarrow
# \text{fixed classifier}
# \rightarrow
# \text{held-out human test set}
# \]
# 
# The downstream classifier is **independent of BGE-M3**:
# 
# - word TF-IDF
# - character TF-IDF
# - Logistic Regression
# 
# BGE-M3 is used only inside the KAZ-Div filtering/clustering stage.
# 
# ## Required input files
# 
# ### 1. Intent definitions CSV
# 
# Required columns:
# 
# - `intent_id`
# - `intent_name`
# - `description`
# 
# Example:
# 
# | intent_id | intent_name | description |
# |---|---|---|
# | card_block | Карта бұғатталды | Пайдаланушы банк картасының бұғатталғанын немесе жұмыс істемей тұрғанын хабарлайды |
# | balance_check | Балансты тексеру | Пайдаланушы шоттағы немесе картадағы балансын білгісі келеді |
# 
# ### 2. Human evaluation CSV
# 
# Required columns:
# 
# - `text`
# - `intent_id`
# 
# This file must be **real/human-authored held-out data** and must never be included in generation prompts.
# 
# ## Recommended paper design
# 
# - 4 LLM families
# - 2 methods: Baseline / KAZ-Div
# - 50 final synthetic utterances per intent per provider per method
# - KAZ-Div candidate pool: `K=12`
# - 5 downstream classifier seeds
# - evaluation on the exact same human test set
# - equalized training-set size for Baseline and KAZ-Div
# - Accuracy, Macro-F1, Weighted-F1, per-intent F1
# - paired bootstrap + McNemar
# - diversity metrics for generated training corpora
# 
# **Run a pilot first and freeze semantic thresholds before the main experiment.**

# %%
# ============================================================
# Cell 0. Install dependencies
# ============================================================

!pip -q install -U \
    openai \
    anthropic \
    google-genai \
    transformers \
    accelerate \
    pandas \
    scikit-learn

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

from google import genai
from google.genai import types

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from google.colab import files


RANDOM_SEED = 20260922

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %%
# ============================================================
# Cell 2. Text normalization helpers
# ============================================================

def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize(
        "NFC",
        str(text),
    )

    text = text.replace(
        "\u00A0",
        " ",
    )

    for char in [
        "\u200B",
        "\u200C",
        "\u200D",
        "\u2060",
        "\uFEFF",
    ]:
        text = text.replace(
            char,
            "",
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


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

# %%
# ============================================================
# Cell 3. Upload intent definitions
# ============================================================
#
# Required:
#   intent_id
#   intent_name
#   description
# ============================================================

print(
    "Upload INTENT DEFINITIONS CSV"
)

uploaded_intents = files.upload()

if not uploaded_intents:
    raise RuntimeError(
        "No intent-definition file uploaded."
    )

INTENT_FILE = next(
    iter(
        uploaded_intents.keys()
    )
)

if not INTENT_FILE.lower().endswith(
    ".csv"
):
    raise ValueError(
        "Intent definitions must be CSV."
    )

intents_df = pd.read_csv(
    INTENT_FILE
)

required_intent_columns = {
    "intent_id",
    "intent_name",
    "description",
}

missing_intent_columns = (
    required_intent_columns
    -
    set(
        intents_df.columns
    )
)

if missing_intent_columns:
    raise ValueError(
        "Missing intent-definition columns: "
        f"{sorted(missing_intent_columns)}"
    )

intents_df = (
    intents_df[
        [
            "intent_id",
            "intent_name",
            "description",
        ]
    ]
    .copy()
)

for column in [
    "intent_id",
    "intent_name",
    "description",
]:
    intents_df[
        column
    ] = (
        intents_df[
            column
        ]
        .map(
            normalize_text
        )
    )

if intents_df[
    "intent_id"
].duplicated().any():
    raise ValueError(
        "intent_id values must be unique."
    )

if (
    intents_df[
        "intent_id"
    ].str.len()
    ==
    0
).any():
    raise ValueError(
        "Empty intent_id detected."
    )

if (
    intents_df[
        "description"
    ].str.len()
    ==
    0
).any():
    raise ValueError(
        "Empty intent description detected."
    )

INTENT_IDS = (
    intents_df[
        "intent_id"
    ]
    .tolist()
)

INTENT_NAME_MAP = dict(
    zip(
        intents_df[
            "intent_id"
        ],
        intents_df[
            "intent_name"
        ],
    )
)

INTENT_DESCRIPTION_MAP = dict(
    zip(
        intents_df[
            "intent_id"
        ],
        intents_df[
            "description"
        ],
    )
)

print(
    "Number of intents:",
    len(
        intents_df
    ),
)

display(
    intents_df
)

# %%
# ============================================================
# Cell 4. Upload held-out HUMAN evaluation set
# ============================================================
#
# Required:
#   text
#   intent_id
#
# IMPORTANT:
#   This dataset is NEVER inserted into generation prompts.
# ============================================================

print(
    "Upload HELD-OUT HUMAN TEST CSV"
)

uploaded_test = files.upload()

if not uploaded_test:
    raise RuntimeError(
        "No evaluation file uploaded."
    )

TEST_FILE = next(
    iter(
        uploaded_test.keys()
    )
)

if not TEST_FILE.lower().endswith(
    ".csv"
):
    raise ValueError(
        "Evaluation set must be CSV."
    )

human_test_df = pd.read_csv(
    TEST_FILE
)

required_test_columns = {
    "text",
    "intent_id",
}

missing_test_columns = (
    required_test_columns
    -
    set(
        human_test_df.columns
    )
)

if missing_test_columns:
    raise ValueError(
        "Missing test columns: "
        f"{sorted(missing_test_columns)}"
    )

human_test_df = (
    human_test_df[
        [
            "text",
            "intent_id",
        ]
    ]
    .copy()
)

human_test_df[
    "text"
] = (
    human_test_df[
        "text"
    ]
    .map(
        normalize_text
    )
)

human_test_df[
    "intent_id"
] = (
    human_test_df[
        "intent_id"
    ]
    .map(
        normalize_text
    )
)

unknown_labels = (
    set(
        human_test_df[
            "intent_id"
        ]
    )
    -
    set(
        INTENT_IDS
    )
)

if unknown_labels:
    raise ValueError(
        "Human test set contains labels not found "
        "in intent definitions: "
        f"{sorted(unknown_labels)}"
    )

if (
    human_test_df[
        "text"
    ].str.len()
    ==
    0
).any():
    raise ValueError(
        "Empty test utterance detected."
    )

print(
    "Human test rows:",
    len(
        human_test_df
    ),
)

print(
    "\nHuman test distribution:"
)

print(
    human_test_df[
        "intent_id"
    ]
    .value_counts()
    .sort_index()
)

display(
    human_test_df.head()
)

# %%
# ============================================================
# Cell 5. Experiment 3 configuration
# ============================================================

EXPERIMENT_NAME = (
    "Experiment_3_Synthetic_Kazakh_Intent_Data"
)

OUTPUT_DIR = Path(
    "/content/KAZ_Div/Experiment_3_Intent_Data"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_IDS = {
    "openai": "YOUR_OPENAI_MODEL_ID",
    "anthropic": "YOUR_CLAUDE_MODEL_ID",
    "gemini": "YOUR_GEMINI_MODEL_ID",
    "llama": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}


ENABLED_PROVIDERS = [
    "openai",
    "anthropic",
    "gemini",
    "llama",
]

METHODS = [
    "baseline",
    "kazdiv",
]


# Number of FINAL synthetic utterances per intent
# for each provider × method.
N_SYNTHETIC_PER_INTENT = 50


K_CANDIDATES = 12

MAX_KAZDIV_CALLS = 2

MAX_OUTPUT_TOKENS = 1600


# ------------------------------------------------------------
# Synthetic utterance quality
# ------------------------------------------------------------

MIN_UTTERANCE_WORDS = 2

MAX_UTTERANCE_WORDS = 30


# Target intent must be the most similar intent description.
REQUIRE_TARGET_INTENT_TOP1 = True


# PILOT values.
# Freeze after manual/pilot calibration.
INTENT_SCORE_MIN = None

INTENT_MARGIN_MIN = 0.02


# Semantic near-duplicate threshold among candidates
# for the SAME requested intent.
INTENT_NEAR_DUPLICATE_THRESHOLD = 0.95


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

EMBEDDING_BATCH_SIZE = 32

EMBEDDING_MAX_LENGTH = 512


# ------------------------------------------------------------
# Local Llama
# ------------------------------------------------------------

LLAMA_DO_SAMPLE = True

LLAMA_TEMPERATURE = 0.6

LLAMA_TOP_P = 0.9


# ------------------------------------------------------------
# Downstream evaluation
# ------------------------------------------------------------

CLASSIFIER_RANDOM_SEEDS = [
    20260922,
    20260923,
    20260924,
    20260925,
    20260926,
]


# Equalize Baseline/KAZ-Div corpus sizes
# per provider and intent.
EQUALIZE_TRAIN_SIZE = True


# Remove accidental exact train/test overlaps
# before downstream training.
REMOVE_EXACT_TRAIN_TEST_OVERLAP = True


CONFIG = {
    "experiment":
        EXPERIMENT_NAME,

    "random_seed":
        RANDOM_SEED,

    "model_ids":
        MODEL_IDS,

    "providers":
        ENABLED_PROVIDERS,

    "methods":
        METHODS,

    "n_synthetic_per_intent":
        N_SYNTHETIC_PER_INTENT,

    "k_candidates":
        K_CANDIDATES,

    "max_kazdiv_calls":
        MAX_KAZDIV_CALLS,

    "utterance_word_range":
        [
            MIN_UTTERANCE_WORDS,
            MAX_UTTERANCE_WORDS,
        ],

    "require_target_intent_top1":
        REQUIRE_TARGET_INTENT_TOP1,

    "intent_score_min":
        INTENT_SCORE_MIN,

    "intent_margin_min":
        INTENT_MARGIN_MIN,

    "semantic_duplicate_threshold":
        INTENT_NEAR_DUPLICATE_THRESHOLD,

    "embedding_model":
        EMBEDDING_MODEL_NAME,

    "embedding_pooling":
        "CLS",

    "classifier":
        (
            "word+char TF-IDF + "
            "LogisticRegression"
        ),

    "classifier_random_seeds":
        CLASSIFIER_RANDOM_SEEDS,

    "equalize_train_size":
        EQUALIZE_TRAIN_SIZE,

    "remove_exact_train_test_overlap":
        REMOVE_EXACT_TRAIN_TEST_OVERLAP,

    "stateless":
        True,
}


with open(
    OUTPUT_DIR
    /
    "experiment3_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        CONFIG,
        f,
        ensure_ascii=False,
        indent=2,
    )


intents_df.to_csv(
    OUTPUT_DIR
    /
    "experiment3_intent_definitions.csv",
    index=False,
    encoding="utf-8-sig",
)

human_test_df.to_csv(
    OUTPUT_DIR
    /
    "experiment3_human_test_snapshot.csv",
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

print(
    "\nOUTPUT_DIR:",
    OUTPUT_DIR,
)

# %%
# ============================================================
# Cell 6. API clients + local Llama + BGE-M3
# ============================================================

OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
GEMINI_API_KEY = ""


openai_client = (
    OpenAI(
        api_key=OPENAI_API_KEY
    )
    if OPENAI_API_KEY.strip()
    else None
)

anthropic_client = (
    anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY
    )
    if ANTHROPIC_API_KEY.strip()
    else None
)

gemini_client = (
    genai.Client(
        api_key=GEMINI_API_KEY
    )
    if GEMINI_API_KEY.strip()
    else None
)


# ------------------------------------------------------------
# Local Llama
# ------------------------------------------------------------

LLAMA_MODEL_ID = (
    MODEL_IDS[
        "llama"
    ]
)

llama_tokenizer = (
    AutoTokenizer
    .from_pretrained(
        LLAMA_MODEL_ID,
        use_fast=True,
    )
)

if (
    llama_tokenizer
    .pad_token
    is None
):
    llama_tokenizer.pad_token = (
        llama_tokenizer
        .eos_token
    )

llama_tokenizer.padding_side = "left"


if torch.cuda.is_available():
    LLAMA_DTYPE = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
else:
    LLAMA_DTYPE = (
        torch.float32
    )


def load_causal_model(
    model_id: str
):
    kwargs = {
        "device_map":
            "auto",

        "low_cpu_mem_usage":
            True,
    }

    try:
        return (
            AutoModelForCausalLM
            .from_pretrained(
                model_id,
                dtype=(
                    LLAMA_DTYPE
                ),
                **kwargs,
            )
        )

    except TypeError:
        return (
            AutoModelForCausalLM
            .from_pretrained(
                model_id,
                torch_dtype=(
                    LLAMA_DTYPE
                ),
                **kwargs,
            )
        )


llama_model = (
    load_causal_model(
        LLAMA_MODEL_ID
    )
)

llama_model.eval()

print(
    "Local Llama loaded:",
    LLAMA_MODEL_ID,
)


# ------------------------------------------------------------
# BGE-M3 — direct transformers path
# CLS pooling + L2 normalization
# ------------------------------------------------------------

BGE_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

bge_tokenizer = (
    AutoTokenizer
    .from_pretrained(
        EMBEDDING_MODEL_NAME
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
    try:
        model = (
            AutoModel
            .from_pretrained(
                model_id,
                dtype=(
                    BGE_DTYPE
                ),
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
    "Embedding dimension:",
    EMBEDDING_DIMENSION,
)

if torch.cuda.is_available():
    print(
        "GPU allocated:",
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

# %%
# ============================================================
# Cell 7. Unified LLM calls
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
            model=model_id,
            input=prompt,
            max_output_tokens=(
                MAX_OUTPUT_TOKENS
            ),
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

    return LLMResult(
        text=text,
        provider="openai",
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
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
            model=model_id,
            max_tokens=(
                MAX_OUTPUT_TOKENS
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

    parts = []

    for block in getattr(
        response,
        "content",
        [],
    ):
        if hasattr(
            block,
            "text",
        ):
            parts.append(
                block.text
            )

    text = normalize_text(
        "\n".join(
            parts
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

    return LLMResult(
        text=text,
        provider="anthropic",
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
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


def call_gemini_once(
    prompt: str,
    model_id: str,
) -> LLMResult:

    if gemini_client is None:
        raise RuntimeError(
            "Gemini client not configured."
        )

    start = (
        time.perf_counter()
    )

    response = (
        gemini_client
        .models
        .generate_content(
            model=model_id,
            contents=prompt,
            config=(
                types
                .GenerateContentConfig(
                    max_output_tokens=(
                        MAX_OUTPUT_TOKENS
                    ),
                )
            ),
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
            "text",
            "",
        )
    )

    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    input_tokens = safe_int(
        getattr(
            usage,
            "prompt_token_count",
            0,
        )
        if usage is not None
        else 0
    )

    output_tokens = safe_int(
        getattr(
            usage,
            "candidates_token_count",
            0,
        )
        if usage is not None
        else 0
    )

    return LLMResult(
        text=text,
        provider="gemini",
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(
            input_tokens
            +
            output_tokens
        ),
        latency_sec=(
            latency
        ),
    )


def call_llama_once(
    prompt: str,
    model_id: str,
    generation_seed: Optional[int] = None,
) -> LLMResult:

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
        llama_tokenizer
        .apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    )

    device = next(
        llama_model
        .parameters()
    ).device

    model_inputs = {
        key:
            value.to(
                device
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
            llama_model
            .generate(
                **model_inputs,
                max_new_tokens=(
                    MAX_OUTPUT_TOKENS
                ),
                do_sample=(
                    LLAMA_DO_SAMPLE
                ),
                temperature=(
                    LLAMA_TEMPERATURE
                ),
                top_p=(
                    LLAMA_TOP_P
                ),
                pad_token_id=(
                    llama_tokenizer
                    .pad_token_id
                ),
                eos_token_id=(
                    llama_tokenizer
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

    suffix = generated[
        0,
        input_tokens:
    ]

    output_tokens = int(
        suffix.shape[
            0
        ]
    )

    text = normalize_text(
        llama_tokenizer.decode(
            suffix,
            skip_special_tokens=True,
        )
    )

    return LLMResult(
        text=text,
        provider="llama",
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
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

    if model_id is None:
        model_id = MODEL_IDS[
            provider
        ]

    if (
        not model_id
        or
        "YOUR_"
        in
        model_id
    ):
        raise ValueError(
            f"Model ID for {provider} "
            f"is not configured: {model_id}"
        )

    if max_attempts is None:
        max_attempts = (
            1
            if provider
            ==
            "llama"
            else
            5
        )

    delays = [
        2,
        4,
        8,
        16,
        30,
    ]

    last_error = None

    for attempt in range(
        1,
        max_attempts + 1,
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

            elif provider == "gemini":
                result = (
                    call_gemini_once(
                        prompt,
                        model_id,
                    )
                )

            else:
                result = (
                    call_llama_once(
                        prompt,
                        model_id,
                        generation_seed=(
                            generation_seed
                        ),
                    )
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
                delays[
                    min(
                        attempt - 1,
                        len(
                            delays
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

# %%
# ============================================================
# Cell 8. Fixed generation prompts
# ============================================================

BASELINE_PROMPT_TEMPLATE = """
Тапсырма:

Төменде бір intent (пайдаланушы ниеті) берілген.
Осы intent-ке сәйкес қазақ тілінде бір табиғи пайдаланушы хабарламасын құрастырыңыз.

Intent атауы:
{intent_name}

Intent сипаттамасы:
{intent_description}

Талаптар:

1. Хабарлама дәл осы intent-ті білдіруі тиіс.
2. Басқа intent-ке жататын мағына қоспаңыз.
3. Хабарлама күнделікті қолданушының табиғи сөзі сияқты болуы тиіс.
4. Қысқа, бірақ мағыналы сөйлем немесе қысқа хабарлама жазыңыз.
5. Intent атауын механикалық түрде қайталамаңыз.
6. "intent", "класс", "санат", "label" сияқты техникалық сөздерді қолданбаңыз.
7. Түсіндірме, нөмірлеу немесе қосымша мәтін қоспаңыз.
8. Тек төмендегі JSON форматында жауап беріңіз.

JSON:

{{
  "utterance": "қазақ тіліндегі пайдаланушы хабарламасы"
}}
""".strip()


KAZDIV_PROMPT_TEMPLATE = """
Тапсырма:

Төменде бір intent (пайдаланушы ниеті) берілген.
Осы intent-ке сәйкес қазақ тілінде {k_candidates} түрлі табиғи пайдаланушы хабарламасын құрастырыңыз.

Intent атауы:
{intent_name}

Intent сипаттамасы:
{intent_description}

Талаптар:

1. Барлық хабарлама дәл осы intent-ті білдіруі тиіс.
2. Басқа intent-ке жататын мағына қоспаңыз.
3. Хабарламалар күнделікті қолданушының табиғи сөзі сияқты болуы тиіс.
4. Хабарламалар бір-бірінен лексикалық және мүмкін болған жағдайда синтаксистік тұрғыдан ерекшеленуі тиіс.
5. Бірдей немесе өте ұқсас нұсқаларды қайталамаңыз.
6. Тек сөздердің орнын ауыстырумен шектелмеңіз.
7. Intent атауын механикалық түрде қайталамаңыз.
8. "intent", "класс", "санат", "label" сияқты техникалық сөздерді қолданбаңыз.
9. Дәл {k_candidates} хабарлама жасаңыз.
10. Ешқандай түсіндірме немесе қосымша мәтін қоспаңыз.
11. Тек төмендегі JSON форматында жауап беріңіз.

JSON:

{{
  "candidates": [
    {{"utterance": "1-нұсқа"}},
    {{"utterance": "2-нұсқа"}}
  ]
}}
""".strip()


def build_intent_prompt(
    method: str,
    intent_id: str,
    k_candidates: int = K_CANDIDATES,
) -> str:

    if intent_id not in INTENT_DESCRIPTION_MAP:
        raise ValueError(
            f"Unknown intent_id: {intent_id}"
        )

    intent_name = (
        INTENT_NAME_MAP[
            intent_id
        ]
    )

    intent_description = (
        INTENT_DESCRIPTION_MAP[
            intent_id
        ]
    )

    method = (
        method
        .strip()
        .lower()
    )

    if method == "baseline":
        return (
            BASELINE_PROMPT_TEMPLATE
            .format(
                intent_name=(
                    intent_name
                ),
                intent_description=(
                    intent_description
                ),
            )
        )

    if method == "kazdiv":
        return (
            KAZDIV_PROMPT_TEMPLATE
            .format(
                intent_name=(
                    intent_name
                ),
                intent_description=(
                    intent_description
                ),
                k_candidates=(
                    k_candidates
                ),
            )
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


with open(
    OUTPUT_DIR
    /
    "experiment3_prompt_templates.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "baseline":
                BASELINE_PROMPT_TEMPLATE,

            "kazdiv":
                KAZDIV_PROMPT_TEMPLATE,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(
    "Experiment 3 prompts ready."
)

# %%
# ============================================================
# Cell 9. Robust JSON parsing and exact deduplication
# ============================================================

@dataclass
class SyntheticCandidate:
    utterance: str


@dataclass
class SyntheticParseResult:
    candidates: List[SyntheticCandidate]
    parse_success: bool
    parse_method: str
    raw_candidate_count: int
    cleaned_candidate_count: int
    exact_duplicate_count: int
    invalid_candidate_count: int
    expected_candidate_count: int
    exact_candidate_count_match: bool
    error_message: Optional[str] = None


def remove_code_fences(
    text: Any
) -> str:

    if text is None:
        return ""

    text = str(
        text
    ).strip()

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

    start = text.find(
        opening_char
    )

    if start == -1:
        return None

    depth = 0
    quote_char = None
    escape = False

    for index in range(
        start,
        len(
            text
        ),
    ):
        char = text[
            index
        ]

        if escape:
            escape = False
            continue

        if (
            char == "\\"
            and
            quote_char
            is not None
        ):
            escape = True
            continue

        if quote_char is not None:
            if char == quote_char:
                quote_char = None
            continue

        if char in [
            '"',
            "'",
        ]:
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


def utterance_from_any(
    item: Any
) -> Optional[SyntheticCandidate]:

    if isinstance(
        item,
        str,
    ):
        utterance = normalize_text(
            item
        )

        if utterance:
            return SyntheticCandidate(
                utterance=utterance
            )

        return None

    if not isinstance(
        item,
        dict,
    ):
        return None

    lower = {
        str(
            key
        )
        .strip()
        .lower():
            value
        for key, value
        in item.items()
    }

    utterance = (
        lower.get(
            "utterance"
        )
        or
        lower.get(
            "text"
        )
        or
        lower.get(
            "message"
        )
        or
        lower.get(
            "sentence"
        )
    )

    utterance = normalize_text(
        utterance
    )

    if not utterance:
        return None

    return SyntheticCandidate(
        utterance=(
            utterance
        )
    )


def extract_utterance_list(
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
        str(
            key
        )
        .strip()
        .lower():
            value
        for key, value
        in obj.items()
    }

    if any(
        key
        in
        lower_obj
        for key
        in [
            "utterance",
            "text",
            "message",
            "sentence",
        ]
    ):
        return [
            obj
        ]

    for key in [
        "candidates",
        "utterances",
        "messages",
        "outputs",
        "responses",
        "sentences",
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


def parse_json_like_utterances(
    raw_text: Any
) -> Tuple[
    Optional[
        List[Any]
    ],
    str,
]:

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

            items = (
                extract_utterance_list(
                    obj
                )
            )

            if items is not None:
                return items, method

        except Exception:
            pass

    literal_source = (
        object_text
        or
        array_text
        or
        remove_code_fences(
            text
        )
    )

    try:
        obj = ast.literal_eval(
            literal_source
        )

        items = (
            extract_utterance_list(
                obj
            )
        )

        if items is not None:
            return (
                items,
                "python_literal",
            )

    except Exception:
        pass

    return None, "failed"


def clean_and_deduplicate_utterances(
    raw_candidates: List[Any]
) -> Dict[str, Any]:

    cleaned = []
    seen = set()

    duplicates = 0
    invalid = 0

    for item in raw_candidates:
        candidate = utterance_from_any(
            item
        )

        if candidate is None:
            invalid += 1
            continue

        key = (
            normalize_for_exact_match(
                candidate
                .utterance
            )
        )

        if not key:
            invalid += 1
            continue

        if key in seen:
            duplicates += 1
            continue

        seen.add(
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


def parse_synthetic_response(
    raw_text: Any,
    expected_k: int,
) -> SyntheticParseResult:

    raw_candidates, method = (
        parse_json_like_utterances(
            raw_text
        )
    )

    if raw_candidates is None:
        return SyntheticParseResult(
            candidates=[],
            parse_success=False,
            parse_method=(
                method
            ),
            raw_candidate_count=0,
            cleaned_candidate_count=0,
            exact_duplicate_count=0,
            invalid_candidate_count=0,
            expected_candidate_count=(
                expected_k
            ),
            exact_candidate_count_match=False,
            error_message=(
                "Could not parse synthetic "
                "utterance JSON."
            ),
        )

    result = (
        clean_and_deduplicate_utterances(
            raw_candidates
        )
    )

    n = len(
        result[
            "candidates"
        ]
    )

    return SyntheticParseResult(
        candidates=(
            result[
                "candidates"
            ]
        ),
        parse_success=(
            n >= 1
        ),
        parse_method=(
            method
        ),
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
            f"recovered {n} unique utterances."
        ),
    )


TEST_SYNTHETIC_JSON = """
{
  "candidates": [
    {"utterance": "Картам неге жұмыс істемей тұр?"},
    {"utterance": "Банк картам бұғатталып қалды."},
    {"utterance": "Картаммен төлем жасай алмай жатырмын."}
  ]
}
"""

test_parse = (
    parse_synthetic_response(
        TEST_SYNTHETIC_JSON,
        expected_k=3,
    )
)

assert (
    test_parse
    .parse_success
)

assert (
    len(
        test_parse
        .candidates
    )
    ==
    3
)

print(
    "Cell 9 parsing: PASS"
)

# %%
# ============================================================
# Cell 10. BGE-M3 embeddings and intent scoring
# ============================================================

def prepare_embedding_texts(
    texts: Union[
        str,
        List[str],
        Tuple[str, ...],
    ]
) -> List[str]:

    if isinstance(
        texts,
        str,
    ):
        texts = [
            texts
        ]

    if not isinstance(
        texts,
        (list, tuple),
    ):
        raise TypeError(
            "texts must be str/list/tuple"
        )

    cleaned = []

    for text in texts:
        text = normalize_text(
            text
        )

        if not text:
            raise ValueError(
                "Empty text cannot be embedded."
            )

        cleaned.append(
            text
        )

    return cleaned


def encode_texts(
    texts: Union[
        str,
        List[str],
        Tuple[str, ...],
    ],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:

    texts = prepare_embedding_texts(
        texts
    )

    all_embeddings = []

    for start in range(
        0,
        len(
            texts
        ),
        batch_size,
    ):
        batch = texts[
            start:
            start
            +
            batch_size
        ]

        encoded = (
            bge_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=(
                    EMBEDDING_MAX_LENGTH
                ),
                return_tensors="pt",
            )
        )

        encoded = {
            key:
                value.to(
                    BGE_DEVICE
                )
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():
            outputs = (
                bge_model(
                    **encoded
                )
            )

            # CLS pooling.
            embeddings = (
                outputs
                .last_hidden_state[
                    :,
                    0
                ]
            )

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1,
            )

        all_embeddings.append(
            embeddings
            .float()
            .cpu()
            .numpy()
        )

    result = (
        np.vstack(
            all_embeddings
        )
        .astype(
            np.float32
        )
    )

    if (
        result.shape[
            1
        ]
        !=
        EMBEDDING_DIMENSION
    ):
        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{result.shape}"
        )

    return result


# ------------------------------------------------------------
# Precompute intent-description embeddings ONCE
# ------------------------------------------------------------

INTENT_EMBEDDING_TEXTS = []

for _, row in (
    intents_df
    .iterrows()
):
    combined = (
        f"{row['intent_name']}. "
        f"{row['description']}"
    )

    INTENT_EMBEDDING_TEXTS.append(
        combined
    )


INTENT_EMBEDDINGS = (
    encode_texts(
        INTENT_EMBEDDING_TEXTS
    )
)


assert (
    INTENT_EMBEDDINGS
    .shape[
        0
    ]
    ==
    len(
        INTENT_IDS
    )
)


INTENT_INDEX_MAP = {
    intent_id:
        index
    for index, intent_id
    in enumerate(
        INTENT_IDS
    )
}


def score_utterances_against_intents(
    utterances: List[str]
) -> Dict[str, Any]:

    if not utterances:
        return {
            "similarity_matrix":
                np.empty(
                    (
                        0,
                        len(
                            INTENT_IDS
                        ),
                    ),
                    dtype=np.float32,
                ),

            "predicted_intents":
                [],

            "top1_scores":
                [],

            "second_scores":
                [],

            "margins":
                [],
        }

    utterance_embeddings = (
        encode_texts(
            utterances
        )
    )

    similarity_matrix = (
        utterance_embeddings
        @
        INTENT_EMBEDDINGS.T
    )

    similarity_matrix = (
        np.clip(
            similarity_matrix,
            -1.0,
            1.0,
        )
        .astype(
            np.float32
        )
    )

    predicted_intents = []
    top1_scores = []
    second_scores = []
    margins = []

    for row in similarity_matrix:
        order = np.argsort(
            row
        )[
            ::-1
        ]

        best_index = int(
            order[
                0
            ]
        )

        best_score = float(
            row[
                best_index
            ]
        )

        if len(
            order
        ) >= 2:
            second_score = float(
                row[
                    int(
                        order[
                            1
                        ]
                    )
                ]
            )
        else:
            second_score = (
                float(
                    "-inf"
                )
            )

        predicted_intents.append(
            INTENT_IDS[
                best_index
            ]
        )

        top1_scores.append(
            best_score
        )

        second_scores.append(
            second_score
        )

        margins.append(
            float(
                best_score
                -
                second_score
            )
            if np.isfinite(
                second_score
            )
            else float(
                "inf"
            )
        )

    return {
        "similarity_matrix":
            similarity_matrix,

        "predicted_intents":
            predicted_intents,

        "top1_scores":
            top1_scores,

        "second_scores":
            second_scores,

        "margins":
            margins,
    }


def candidate_similarity_matrix(
    utterances: List[str]
) -> np.ndarray:

    if not utterances:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    embeddings = encode_texts(
        utterances
    )

    matrix = (
        embeddings
        @
        embeddings.T
    )

    matrix = (
        np.clip(
            matrix,
            -1.0,
            1.0,
        )
        .astype(
            np.float32
        )
    )

    np.fill_diagonal(
        matrix,
        1.0,
    )

    return matrix


# Smoke test.
_test_embedding = (
    encode_texts(
        [
            "Картам жұмыс істемей тұр."
        ]
    )
)

assert (
    _test_embedding
    .shape
    ==
    (
        1,
        EMBEDDING_DIMENSION,
    )
)

assert (
    abs(
        np.linalg.norm(
            _test_embedding[
                0
            ]
        )
        -
        1.0
    )
    <
    1e-3
)

print(
    "Cell 10 BGE-M3 intent layer: PASS"
)

# %%
# ============================================================
# Cell 11. Intent-consistency quality filtering
# ============================================================

@dataclass
class IntentQualityRecord:
    candidate_index: int
    utterance: str
    target_intent_id: str
    predicted_intent_id: str
    target_score: float
    top1_score: float
    second_score: float
    margin: float
    word_count: int
    length_pass: bool
    target_is_top1: bool
    score_pass: bool
    margin_pass: bool
    metadata_leakage: bool
    quality_pass: bool
    rejection_reasons: List[str]


@dataclass
class IntentQualityResult:
    target_intent_id: str
    input_candidate_count: int
    valid_candidate_count: int
    rejected_candidate_count: int
    valid_candidates: List[SyntheticCandidate]
    valid_original_indices: List[int]
    valid_target_scores: List[float]
    valid_margins: List[float]
    records: List[IntentQualityRecord]
    filter_success: bool


TECHNICAL_WORDS = [
    "intent",
    "label",
    "класс",
    "санат",
    "категория",
]


def contains_metadata_leakage(
    utterance: str
) -> bool:

    normalized = (
        normalize_text(
            utterance
        )
        .lower()
    )

    return any(
        re.search(
            rf"\b{re.escape(word)}\b",
            normalized,
        )
        is not None
        for word
        in TECHNICAL_WORDS
    )


def filter_intent_candidates(
    target_intent_id: str,
    candidates: List[SyntheticCandidate],
) -> IntentQualityResult:

    if target_intent_id not in INTENT_INDEX_MAP:
        raise ValueError(
            f"Unknown target intent: "
            f"{target_intent_id}"
        )

    if not candidates:
        return IntentQualityResult(
            target_intent_id=(
                target_intent_id
            ),
            input_candidate_count=0,
            valid_candidate_count=0,
            rejected_candidate_count=0,
            valid_candidates=[],
            valid_original_indices=[],
            valid_target_scores=[],
            valid_margins=[],
            records=[],
            filter_success=False,
        )

    utterances = [
        candidate.utterance
        for candidate
        in candidates
    ]

    scoring = (
        score_utterances_against_intents(
            utterances
        )
    )

    target_index = (
        INTENT_INDEX_MAP[
            target_intent_id
        ]
    )

    records = []

    for index, candidate in enumerate(
        candidates
    ):
        utterance = (
            normalize_text(
                candidate
                .utterance
            )
        )

        word_count = (
            count_words(
                utterance
            )
        )

        length_pass = (
            MIN_UTTERANCE_WORDS
            <=
            word_count
            <=
            MAX_UTTERANCE_WORDS
        )

        predicted_intent = (
            scoring[
                "predicted_intents"
            ][
                index
            ]
        )

        target_score = float(
            scoring[
                "similarity_matrix"
            ][
                index,
                target_index
            ]
        )

        top1_score = float(
            scoring[
                "top1_scores"
            ][
                index
            ]
        )

        second_score = float(
            scoring[
                "second_scores"
            ][
                index
            ]
        )

        margin = float(
            scoring[
                "margins"
            ][
                index
            ]
        )

        target_is_top1 = (
            predicted_intent
            ==
            target_intent_id
        )

        if INTENT_SCORE_MIN is None:
            score_pass = True
        else:
            score_pass = (
                target_score
                >=
                INTENT_SCORE_MIN
            )

        if INTENT_MARGIN_MIN is None:
            margin_pass = True
        else:
            margin_pass = (
                margin
                >=
                INTENT_MARGIN_MIN
            )

        metadata_leakage = (
            contains_metadata_leakage(
                utterance
            )
        )

        rejection_reasons = []

        if not length_pass:
            rejection_reasons.append(
                "utterance_length_out_of_range"
            )

        if (
            REQUIRE_TARGET_INTENT_TOP1
            and
            not target_is_top1
        ):
            rejection_reasons.append(
                "target_intent_not_top1"
            )

        if not score_pass:
            rejection_reasons.append(
                "low_target_intent_score"
            )

        if not margin_pass:
            rejection_reasons.append(
                "low_intent_margin"
            )

        if metadata_leakage:
            rejection_reasons.append(
                "metadata_leakage"
            )

        quality_pass = (
            bool(
                utterance
            )
            and
            length_pass
            and
            (
                target_is_top1
                if
                REQUIRE_TARGET_INTENT_TOP1
                else True
            )
            and
            score_pass
            and
            margin_pass
            and
            not metadata_leakage
        )

        records.append(
            IntentQualityRecord(
                candidate_index=(
                    index
                ),
                utterance=(
                    utterance
                ),
                target_intent_id=(
                    target_intent_id
                ),
                predicted_intent_id=(
                    predicted_intent
                ),
                target_score=(
                    target_score
                ),
                top1_score=(
                    top1_score
                ),
                second_score=(
                    second_score
                ),
                margin=(
                    margin
                ),
                word_count=(
                    word_count
                ),
                length_pass=(
                    length_pass
                ),
                target_is_top1=(
                    target_is_top1
                ),
                score_pass=(
                    score_pass
                ),
                margin_pass=(
                    margin_pass
                ),
                metadata_leakage=(
                    metadata_leakage
                ),
                quality_pass=(
                    quality_pass
                ),
                rejection_reasons=(
                    rejection_reasons
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
        SyntheticCandidate(
            utterance=(
                record
                .utterance
            )
        )
        for record
        in valid_records
    ]

    return IntentQualityResult(
        target_intent_id=(
            target_intent_id
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
            )
            -
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
        valid_target_scores=[
            record.target_score
            for record
            in valid_records
        ],
        valid_margins=[
            record.margin
            for record
            in valid_records
        ],
        records=(
            records
        ),
        filter_success=(
            len(
                valid_candidates
            )
            >=
            1
        ),
    )


print(
    "Cell 11 intent-quality filter: READY"
)

# %%
# ============================================================
# Cell 12. Semantic near-deduplication and clustering
# ============================================================

@dataclass
class IntentCluster:
    cluster_id: int
    member_local_indices: List[int]
    original_candidate_indices: List[int]
    utterances: List[str]
    target_scores: List[float]
    margins: List[float]
    representative_local_index: int
    representative_original_index: int
    representative_utterance: str
    representative_target_score: float
    representative_margin: float
    cluster_size: int


@dataclass
class IntentClusteringResult:
    target_intent_id: str
    input_candidate_count: int
    cluster_count: int
    semantic_duplicates_removed: int
    edge_count: int
    threshold: float
    clusters: List[IntentCluster]
    representative_candidates: List[SyntheticCandidate]
    representative_original_indices: List[int]
    similarity_matrix: np.ndarray
    success: bool


def build_duplicate_graph(
    similarity_matrix: np.ndarray,
    threshold: float,
) -> Dict[int, List[int]]:

    if not (
        0
        <
        threshold
        <=
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
        matrix.ndim
        !=
        2
        or
        matrix.shape[
            0
        ]
        !=
        matrix.shape[
            1
        ]
    ):
        raise ValueError(
            "Similarity matrix must be square."
        )

    n = matrix.shape[
        0
    ]

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
                )
                >=
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
            node = (
                stack
                .pop()
            )

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


def cluster_intent_candidates(
    quality_result: IntentQualityResult,
    threshold: float = INTENT_NEAR_DUPLICATE_THRESHOLD,
) -> IntentClusteringResult:

    candidates = (
        quality_result
        .valid_candidates
    )

    n = len(
        candidates
    )

    if n == 0:
        return IntentClusteringResult(
            target_intent_id=(
                quality_result
                .target_intent_id
            ),
            input_candidate_count=0,
            cluster_count=0,
            semantic_duplicates_removed=0,
            edge_count=0,
            threshold=float(
                threshold
            ),
            clusters=[],
            representative_candidates=[],
            representative_original_indices=[],
            similarity_matrix=np.empty(
                (0, 0),
                dtype=np.float32,
            ),
            success=False,
        )

    utterances = [
        candidate.utterance
        for candidate
        in candidates
    ]

    matrix = (
        candidate_similarity_matrix(
            utterances
        )
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
        )
        //
        2
    )

    components = (
        connected_components(
            graph
        )
    )

    clusters = []
    representatives = []
    rep_original_indices = []

    scores = (
        quality_result
        .valid_target_scores
    )

    margins = (
        quality_result
        .valid_margins
    )

    original_indices = (
        quality_result
        .valid_original_indices
    )

    for cluster_id, component in enumerate(
        components
    ):
        # Quality-preserving representative:
        # 1) higher target-intent score
        # 2) larger intent margin
        # 3) lower local index as deterministic tie-break
        representative_local_index = max(
            component,
            key=lambda index: (
                float(
                    scores[
                        index
                    ]
                ),
                float(
                    margins[
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

        cluster = IntentCluster(
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
            utterances=[
                candidates[
                    index
                ].utterance
                for index
                in component
            ],
            target_scores=[
                float(
                    scores[
                        index
                    ]
                )
                for index
                in component
            ],
            margins=[
                float(
                    margins[
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
            representative_utterance=(
                representative
                .utterance
            ),
            representative_target_score=(
                float(
                    scores[
                        representative_local_index
                    ]
                )
            ),
            representative_margin=(
                float(
                    margins[
                        representative_local_index
                    ]
                )
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
            SyntheticCandidate(
                utterance=(
                    representative
                    .utterance
                )
            )
        )

        rep_original_indices.append(
            representative_original_index
        )

    return IntentClusteringResult(
        target_intent_id=(
            quality_result
            .target_intent_id
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
            n
            -
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
        representative_original_indices=(
            rep_original_indices
        ),
        similarity_matrix=(
            matrix
        ),
        success=(
            len(
                clusters
            )
            >=
            1
        ),
    )


SYNTHETIC_SIM_MATRIX = np.array(
    [
        [1.00, 0.97, 0.20, 0.10],
        [0.97, 1.00, 0.96, 0.10],
        [0.20, 0.96, 1.00, 0.15],
        [0.10, 0.10, 0.15, 1.00],
    ],
    dtype=np.float32,
)

_test_graph = (
    build_duplicate_graph(
        SYNTHETIC_SIM_MATRIX,
        threshold=0.95,
    )
)

_test_components = (
    connected_components(
        _test_graph
    )
)

assert (
    _test_components
    ==
    [
        [0, 1, 2],
        [3],
    ]
)

print(
    "Cell 12 clustering: PASS"
)

# %%
# ============================================================
# Cell 13. Stateless diversity-aware selection
# ============================================================

@dataclass
class IntentSelectionResult:
    intent_id: str
    repetition: int
    provider: str
    model_id: str
    method: str
    selection_seed: int
    cluster_count: int
    selection_probability: float
    selected_cluster_id: int
    selected_original_candidate_index: int
    selected_utterance: str
    selected_target_score: float
    selected_margin: float
    success: bool
    reason: str


def make_selection_seed(
    provider: str,
    model_id: str,
    intent_id: str,
    repetition: int,
    method: str = "kazdiv",
    global_seed: int = RANDOM_SEED,
) -> int:

    material = (
        f"{global_seed}|"
        f"{provider}|"
        f"{model_id}|"
        f"{intent_id}|"
        f"{repetition}|"
        f"{method}"
    )

    digest = (
        hashlib.sha256(
            material.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )

    return int(
        digest[
            :16
        ],
        16,
    )


def select_intent_candidate(
    clustering_result: IntentClusteringResult,
    provider: str,
    model_id: str,
    intent_id: str,
    repetition: int,
) -> IntentSelectionResult:

    cluster_count = (
        clustering_result
        .cluster_count
    )

    if cluster_count < 1:
        return IntentSelectionResult(
            intent_id=(
                intent_id
            ),
            repetition=(
                repetition
            ),
            provider=(
                provider
            ),
            model_id=(
                model_id
            ),
            method="kazdiv",
            selection_seed=-1,
            cluster_count=0,
            selection_probability=0.0,
            selected_cluster_id=-1,
            selected_original_candidate_index=-1,
            selected_utterance="",
            selected_target_score=float(
                "nan"
            ),
            selected_margin=float(
                "nan"
            ),
            success=False,
            reason="no_clusters",
        )

    seed = (
        make_selection_seed(
            provider=(
                provider
            ),
            model_id=(
                model_id
            ),
            intent_id=(
                intent_id
            ),
            repetition=(
                repetition
            ),
        )
    )

    if cluster_count == 1:
        selected_position = 0

    else:
        rng = (
            np.random.default_rng(
                seed
            )
        )

        selected_position = int(
            rng.integers(
                low=0,
                high=(
                    cluster_count
                ),
            )
        )

    cluster = (
        clustering_result
        .clusters[
            selected_position
        ]
    )

    return IntentSelectionResult(
        intent_id=(
            intent_id
        ),
        repetition=(
            repetition
        ),
        provider=(
            provider
        ),
        model_id=(
            model_id
        ),
        method="kazdiv",
        selection_seed=(
            seed
        ),
        cluster_count=(
            cluster_count
        ),
        selection_probability=(
            float(
                1.0
                /
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
        selected_utterance=(
            cluster
            .representative_utterance
        ),
        selected_target_score=(
            float(
                cluster
                .representative_target_score
            )
        ),
        selected_margin=(
            float(
                cluster
                .representative_margin
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


print(
    "Cell 13 stateless selection: READY"
)

# %%
# ============================================================
# Cell 14. End-to-end synthetic generation runner
# ============================================================

@dataclass
class Experiment3GenerationResult:
    experiment: str
    intent_id: str
    intent_name: str
    provider: str
    model_id: str
    method: str
    repetition: int
    run_success: bool
    final_utterance: str
    bge_predicted_intent: Optional[str]
    target_intent_score: Optional[float]
    intent_margin: Optional[float]
    final_quality_pass: Optional[bool]
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
            )
            or
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
    intent_id: str,
    repetition: int,
    method: str,
    call_index: int,
) -> int:

    material = (
        f"{RANDOM_SEED}|"
        f"{provider}|"
        f"{model_id}|"
        f"{intent_id}|"
        f"{repetition}|"
        f"{method}|"
        f"{call_index}"
    )

    digest = (
        hashlib.sha256(
            material.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )

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


def single_candidate_quality(
    intent_id: str,
    candidate: SyntheticCandidate,
) -> IntentQualityRecord:

    result = (
        filter_intent_candidates(
            target_intent_id=(
                intent_id
            ),
            candidates=[
                candidate
            ],
        )
    )

    return (
        result
        .records[
            0
        ]
    )


def failed_generation_result(
    intent_id: str,
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
) -> Experiment3GenerationResult:

    totals = (
        aggregate_llm_results(
            llm_results
        )
    )

    raw_count = (
        getattr(
            parse_metadata,
            "raw_candidate_count",
            None,
        )
        if parse_metadata
        is not None
        else None
    )

    parsed_count = (
        len(
            getattr(
                parse_metadata,
                "candidates",
                [],
            )
        )
        if parse_metadata
        is not None
        else None
    )

    valid_count = (
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

    return Experiment3GenerationResult(
        experiment=(
            EXPERIMENT_NAME
        ),
        intent_id=(
            intent_id
        ),
        intent_name=(
            INTENT_NAME_MAP[
                intent_id
            ]
        ),
        provider=(
            provider
        ),
        model_id=(
            model_id
        ),
        method=(
            method
        ),
        repetition=(
            repetition
        ),
        run_success=False,
        final_utterance="",
        bge_predicted_intent=None,
        target_intent_score=None,
        intent_margin=None,
        final_quality_pass=False,
        llm_call_count=(
            totals[
                "llm_call_count"
            ]
        ),
        total_input_tokens=(
            totals[
                "total_input_tokens"
            ]
        ),
        total_output_tokens=(
            totals[
                "total_output_tokens"
            ]
        ),
        total_tokens=(
            totals[
                "total_tokens"
            ]
        ),
        total_llm_latency_sec=(
            totals[
                "total_llm_latency_sec"
            ]
        ),
        total_pipeline_latency_sec=(
            time.perf_counter()
            -
            pipeline_start
        ),
        raw_candidate_count=(
            raw_count
        ),
        parsed_candidate_count=(
            parsed_count
        ),
        quality_valid_candidate_count=(
            valid_count
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


def run_baseline_intent_generation(
    intent_id: str,
    provider: str,
    repetition: int,
    model_id: Optional[str] = None,
) -> Experiment3GenerationResult:

    pipeline_start = (
        time.perf_counter()
    )

    provider = (
        provider
        .strip()
        .lower()
    )

    if model_id is None:
        model_id = MODEL_IDS[
            provider
        ]

    llm_results = []

    generation_seed = (
        make_generation_seed(
            provider=(
                provider
            ),
            model_id=(
                model_id
            ),
            intent_id=(
                intent_id
            ),
            repetition=(
                repetition
            ),
            method="baseline",
            call_index=1,
        )
    )

    prompt = (
        build_intent_prompt(
            method="baseline",
            intent_id=(
                intent_id
            ),
        )
    )

    try:
        llm = call_llm(
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

        llm_results.append(
            llm
        )

    except Exception as exc:
        return failed_generation_result(
            intent_id=(
                intent_id
            ),
            provider=(
                provider
            ),
            model_id=(
                model_id
            ),
            method="baseline",
            repetition=(
                repetition
            ),
            pipeline_start=(
                pipeline_start
            ),
            llm_results=(
                llm_results
            ),
            failure_stage=(
                "llm_call"
            ),
            error_message=(
                str(
                    exc
                )
            ),
        )

    parse_result = (
        parse_synthetic_response(
            llm.text,
            expected_k=1,
        )
    )

    if not parse_result.candidates:
        return failed_generation_result(
            intent_id=(
                intent_id
            ),
            provider=(
                provider
            ),
            model_id=(
                model_id
            ),
            method="baseline",
            repetition=(
                repetition
            ),
            pipeline_start=(
                pipeline_start
            ),
            llm_results=(
                llm_results
            ),
            failure_stage=(
                "parsing"
            ),
            error_message=(
                parse_result
                .error_message
                or
                "No baseline utterance parsed."
            ),
            parse_metadata=(
                parse_result
            ),
        )

    # Direct baseline output:
    # preserve first parsed utterance even if
    # BGE quality check fails.
    candidate = (
        parse_result
        .candidates[
            0
        ]
    )

    quality = (
        single_candidate_quality(
            intent_id,
            candidate,
        )
    )

    totals = (
        aggregate_llm_results(
            llm_results
        )
    )

    return Experiment3GenerationResult(
        experiment=(
            EXPERIMENT_NAME
        ),
        intent_id=(
            intent_id
        ),
        intent_name=(
            INTENT_NAME_MAP[
                intent_id
            ]
        ),
        provider=(
            provider
        ),
        model_id=(
            model_id
        ),
        method="baseline",
        repetition=(
            repetition
        ),
        run_success=True,
        final_utterance=(
            candidate
            .utterance
        ),
        bge_predicted_intent=(
            quality
            .predicted_intent_id
        ),
        target_intent_score=(
            quality
            .target_score
        ),
        intent_margin=(
            quality
            .margin
        ),
        final_quality_pass=(
            quality
            .quality_pass
        ),
        llm_call_count=(
            totals[
                "llm_call_count"
            ]
        ),
        total_input_tokens=(
            totals[
                "total_input_tokens"
            ]
        ),
        total_output_tokens=(
            totals[
                "total_output_tokens"
            ]
        ),
        total_tokens=(
            totals[
                "total_tokens"
            ]
        ),
        total_llm_latency_sec=(
            totals[
                "total_llm_latency_sec"
            ]
        ),
        total_pipeline_latency_sec=(
            time.perf_counter()
            -
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


def run_kazdiv_intent_generation(
    intent_id: str,
    provider: str,
    repetition: int,
    model_id: Optional[str] = None,
) -> Experiment3GenerationResult:

    pipeline_start = (
        time.perf_counter()
    )

    provider = (
        provider
        .strip()
        .lower()
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

    failure_stage = None
    last_error = None

    for call_index in range(
        1,
        MAX_KAZDIV_CALLS
        +
        1,
    ):

        generation_seed = (
            make_generation_seed(
                provider=(
                    provider
                ),
                model_id=(
                    model_id
                ),
                intent_id=(
                    intent_id
                ),
                repetition=(
                    repetition
                ),
                method="kazdiv",
                call_index=(
                    call_index
                ),
            )
        )

        prompt = (
            build_intent_prompt(
                method="kazdiv",
                intent_id=(
                    intent_id
                ),
                k_candidates=(
                    K_CANDIDATES
                ),
            )
        )

        try:
            llm = call_llm(
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

            llm_results.append(
                llm
            )

        except Exception as exc:
            failure_stage = (
                "llm_call"
            )

            last_error = str(
                exc
            )

            continue

        parse_result = (
            parse_synthetic_response(
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
            failure_stage = (
                "parsing"
            )

            last_error = (
                parse_result
                .error_message
                or
                "No candidates parsed."
            )

            continue

        quality_result = (
            filter_intent_candidates(
                target_intent_id=(
                    intent_id
                ),
                candidates=(
                    parse_result
                    .candidates
                ),
            )
        )

        last_quality = (
            quality_result
        )

        if (
            quality_result
            .valid_candidate_count
            ==
            0
        ):
            failure_stage = (
                "quality_filtering"
            )

            last_error = (
                "No candidate passed "
                "intent-quality filtering."
            )

            continue

        clustering_result = (
            cluster_intent_candidates(
                quality_result,
                threshold=(
                    INTENT_NEAR_DUPLICATE_THRESHOLD
                ),
            )
        )

        last_cluster = (
            clustering_result
        )

        if (
            clustering_result
            .cluster_count
            ==
            0
        ):
            failure_stage = (
                "semantic_clustering"
            )

            last_error = (
                "No semantic clusters."
            )

            continue

        selection = (
            select_intent_candidate(
                clustering_result=(
                    clustering_result
                ),
                provider=(
                    provider
                ),
                model_id=(
                    model_id
                ),
                intent_id=(
                    intent_id
                ),
                repetition=(
                    repetition
                ),
            )
        )

        if selection.success:
            failure_stage = None
            last_error = None
            break

        failure_stage = (
            "selection"
        )

        last_error = (
            "Final selection failed."
        )

    if (
        selection is None
        or
        not selection.success
    ):
        return failed_generation_result(
            intent_id=(
                intent_id
            ),
            provider=(
                provider
            ),
            model_id=(
                model_id
            ),
            method="kazdiv",
            repetition=(
                repetition
            ),
            pipeline_start=(
                pipeline_start
            ),
            llm_results=(
                llm_results
            ),
            failure_stage=(
                failure_stage
                or
                "unknown"
            ),
            error_message=(
                last_error
                or
                "KAZ-Div generation failed."
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

    # Re-evaluate selected utterance for explicit final metadata.
    final_candidate = (
        SyntheticCandidate(
            utterance=(
                selection
                .selected_utterance
            )
        )
    )

    final_quality = (
        single_candidate_quality(
            intent_id,
            final_candidate,
        )
    )

    totals = (
        aggregate_llm_results(
            llm_results
        )
    )

    return Experiment3GenerationResult(
        experiment=(
            EXPERIMENT_NAME
        ),
        intent_id=(
            intent_id
        ),
        intent_name=(
            INTENT_NAME_MAP[
                intent_id
            ]
        ),
        provider=(
            provider
        ),
        model_id=(
            model_id
        ),
        method="kazdiv",
        repetition=(
            repetition
        ),
        run_success=True,
        final_utterance=(
            selection
            .selected_utterance
        ),
        bge_predicted_intent=(
            final_quality
            .predicted_intent_id
        ),
        target_intent_score=(
            final_quality
            .target_score
        ),
        intent_margin=(
            final_quality
            .margin
        ),
        final_quality_pass=(
            final_quality
            .quality_pass
        ),
        llm_call_count=(
            totals[
                "llm_call_count"
            ]
        ),
        total_input_tokens=(
            totals[
                "total_input_tokens"
            ]
        ),
        total_output_tokens=(
            totals[
                "total_output_tokens"
            ]
        ),
        total_tokens=(
            totals[
                "total_tokens"
            ]
        ),
        total_llm_latency_sec=(
            totals[
                "total_llm_latency_sec"
            ]
        ),
        total_pipeline_latency_sec=(
            time.perf_counter()
            -
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


def run_experiment3_generation(
    intent_id: str,
    provider: str,
    method: str,
    repetition: int,
    model_id: Optional[str] = None,
) -> Experiment3GenerationResult:

    method = (
        method
        .strip()
        .lower()
    )

    if method == "baseline":
        return (
            run_baseline_intent_generation(
                intent_id=(
                    intent_id
                ),
                provider=(
                    provider
                ),
                repetition=(
                    repetition
                ),
                model_id=(
                    model_id
                ),
            )
        )

    if method == "kazdiv":
        return (
            run_kazdiv_intent_generation(
                intent_id=(
                    intent_id
                ),
                provider=(
                    provider
                ),
                repetition=(
                    repetition
                ),
                model_id=(
                    model_id
                ),
            )
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


print(
    "Cell 14 end-to-end generation runner: READY"
)

# %%
# ============================================================
# Cell 15. Optional real smoke test
# ============================================================

RUN_REAL_SMOKE_TEST = False

SMOKE_PROVIDER = "llama"

SMOKE_INTENT_ID = (
    INTENT_IDS[
        0
    ]
)

SMOKE_REPETITION = 1


if RUN_REAL_SMOKE_TEST:

    print(
        "INTENT:",
        SMOKE_INTENT_ID,
        "|",
        INTENT_NAME_MAP[
            SMOKE_INTENT_ID
        ],
    )

    print(
        "\n--- BASELINE ---"
    )

    smoke_baseline = (
        run_experiment3_generation(
            intent_id=(
                SMOKE_INTENT_ID
            ),
            provider=(
                SMOKE_PROVIDER
            ),
            method="baseline",
            repetition=(
                SMOKE_REPETITION
            ),
        )
    )

    print(
        json.dumps(
            to_json_safe(
                smoke_baseline
            ),
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\n--- KAZ-DIV ---"
    )

    smoke_kazdiv = (
        run_experiment3_generation(
            intent_id=(
                SMOKE_INTENT_ID
            ),
            provider=(
                SMOKE_PROVIDER
            ),
            method="kazdiv",
            repetition=(
                SMOKE_REPETITION
            ),
        )
    )

    print(
        json.dumps(
            to_json_safe(
                smoke_kazdiv
            ),
            ensure_ascii=False,
            indent=2,
        )
    )

else:
    print(
        "Real smoke test skipped."
    )
    print(
        "Set RUN_REAL_SMOKE_TEST = True "
        "to test one intent."
    )

# %%
# ============================================================
# Cell 16. LOCAL full generation runner
# Checkpoint / Resume
# ============================================================

RUN_FULL_GENERATION = False

PROVIDERS_TO_RUN = list(
    ENABLED_PROVIDERS
)

METHODS_TO_RUN = list(
    METHODS
)

REPETITIONS_TO_RUN = list(
    range(
        1,
        N_SYNTHETIC_PER_INTENT
        +
        1,
    )
)


# Pilot:
#     MAX_TASKS_THIS_RUN = 20
#
# Full:
#     MAX_TASKS_THIS_RUN = None

MAX_TASKS_THIS_RUN = None

MAX_CONSECUTIVE_FAILURES = 3

AUTO_ZIP_EVERY = 25


RESULTS_JSONL_FILE = (
    OUTPUT_DIR
    /
    "experiment3_generation_results.jsonl"
)

RESULTS_CSV_FILE = (
    OUTPUT_DIR
    /
    "experiment3_generation_summary.csv"
)

FAILURES_JSONL_FILE = (
    OUTPUT_DIR
    /
    "experiment3_generation_failures.jsonl"
)

RUNNER_ERRORS_JSONL_FILE = (
    OUTPUT_DIR
    /
    "experiment3_runner_errors.jsonl"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR
    /
    "experiment3_generation_checkpoint.json"
)

SIGNATURE_FILE = (
    OUTPUT_DIR
    /
    "experiment3_generation_signature.json"
)

COMPLETION_REPORT_FILE = (
    OUTPUT_DIR
    /
    "experiment3_generation_completion.json"
)

LOCAL_ZIP_FILE = Path(
    "/content/KAZ_Div_Experiment3_latest.zip"
)


GENERATION_CSV_COLUMNS = [
    "experiment",
    "task_key",
    "intent_id",
    "intent_name",
    "provider",
    "model_id",
    "method",
    "repetition",
    "run_success",
    "final_utterance",
    "bge_predicted_intent",
    "target_intent_score",
    "intent_margin",
    "final_quality_pass",
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
    temp_path = Path(
        str(
            path
        )
        +
        ".tmp"
    )

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
        os.fsync(
            f.fileno()
        )

    os.replace(
        temp_path,
        path,
    )


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

        f.write(
            "\n"
        )

        f.flush()
        os.fsync(
            f.fileno()
        )


def dataframe_sha256(
    df: pd.DataFrame
) -> str:

    csv_bytes = (
        df
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    return (
        hashlib.sha256(
            csv_bytes
        )
        .hexdigest()
    )


SIGNATURE_CONFIG = {
    "experiment":
        EXPERIMENT_NAME,

    "random_seed":
        RANDOM_SEED,

    "intent_definitions_hash":
        dataframe_sha256(
            intents_df
        ),

    "model_ids":
        {
            provider:
                MODEL_IDS[
                    provider
                ]
            for provider
            in PROVIDERS_TO_RUN
        },

    "providers":
        PROVIDERS_TO_RUN,

    "methods":
        METHODS_TO_RUN,

    "n_synthetic_per_intent":
        N_SYNTHETIC_PER_INTENT,

    "k_candidates":
        K_CANDIDATES,

    "max_kazdiv_calls":
        MAX_KAZDIV_CALLS,

    "min_utterance_words":
        MIN_UTTERANCE_WORDS,

    "max_utterance_words":
        MAX_UTTERANCE_WORDS,

    "require_target_top1":
        REQUIRE_TARGET_INTENT_TOP1,

    "intent_score_min":
        INTENT_SCORE_MIN,

    "intent_margin_min":
        INTENT_MARGIN_MIN,

    "semantic_duplicate_threshold":
        INTENT_NEAR_DUPLICATE_THRESHOLD,

    "embedding_model":
        EMBEDDING_MODEL_NAME,

    "embedding_pooling":
        "CLS",

    "llama_temperature":
        LLAMA_TEMPERATURE,

    "llama_top_p":
        LLAMA_TOP_P,

    "baseline_prompt_template":
        BASELINE_PROMPT_TEMPLATE,

    "kazdiv_prompt_template":
        KAZDIV_PROMPT_TEMPLATE,
}


SIGNATURE_TEXT = json.dumps(
    SIGNATURE_CONFIG,
    ensure_ascii=False,
    sort_keys=True,
    default=str,
)

EXPERIMENT_SIGNATURE = (
    hashlib.sha256(
        SIGNATURE_TEXT.encode(
            "utf-8"
        )
    )
    .hexdigest()
)


def validate_generation_signature():
    current = {
        "signature":
            EXPERIMENT_SIGNATURE,

        "created_or_checked_at":
            datetime.now()
            .isoformat(),

        "configuration":
            SIGNATURE_CONFIG,
    }

    if not SIGNATURE_FILE.exists():
        write_json_atomic(
            SIGNATURE_FILE,
            current,
        )

        return

    with open(
        SIGNATURE_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        saved = json.load(
            f
        )

    if (
        saved.get(
            "signature"
        )
        !=
        EXPERIMENT_SIGNATURE
    ):
        raise RuntimeError(
            "Experiment 3 generation configuration changed. "
            "Do not mix old and new results. "
            "Use a new OUTPUT_DIR or remove pilot files "
            "after you intentionally freeze new settings."
        )


def looks_like_placeholder(
    value: Any
) -> bool:

    if value is None:
        return True

    text = str(
        value
    ).strip()

    if not text:
        return True

    upper = text.upper()

    return any(
        token in upper
        for token
        in [
            "YOUR_",
            "PASTE_",
            "MODEL_ID_HERE",
            "TODO",
        ]
    )


if RUN_FULL_GENERATION:
    unresolved = [
        provider
        for provider
        in PROVIDERS_TO_RUN
        if looks_like_placeholder(
            MODEL_IDS.get(
                provider
            )
        )
    ]

    if unresolved:
        raise RuntimeError(
            "Set exact model IDs before generation: "
            f"{unresolved}"
        )


def make_task_key(
    intent_id: str,
    provider: str,
    model_id: str,
    method: str,
    repetition: int,
) -> str:

    material = (
        f"{intent_id}|"
        f"{provider}|"
        f"{model_id}|"
        f"{method}|"
        f"{repetition}"
    )

    return (
        hashlib.sha256(
            material.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


def build_generation_tasks():
    tasks = []

    for intent_id in INTENT_IDS:
        for repetition in (
            REPETITIONS_TO_RUN
        ):
            for provider in (
                PROVIDERS_TO_RUN
            ):
                model_id = (
                    MODEL_IDS[
                        provider
                    ]
                )

                for method in (
                    METHODS_TO_RUN
                ):
                    task_key = (
                        make_task_key(
                            intent_id=(
                                intent_id
                            ),
                            provider=(
                                provider
                            ),
                            model_id=(
                                model_id
                            ),
                            method=(
                                method
                            ),
                            repetition=(
                                repetition
                            ),
                        )
                    )

                    tasks.append(
                        {
                            "task_key":
                                task_key,

                            "intent_id":
                                intent_id,

                            "provider":
                                provider,

                            "model_id":
                                model_id,

                            "method":
                                method,

                            "repetition":
                                repetition,
                        }
                    )

    return tasks


ALL_GENERATION_TASKS = (
    build_generation_tasks()
)

TOTAL_GENERATION_TASKS = len(
    ALL_GENERATION_TASKS
)


def load_successful_generation_results():
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
                record = json.loads(
                    line
                )
            except Exception:
                continue

            key = record.get(
                "task_key"
            )

            if (
                key
                and
                record.get(
                    "run_success"
                )
                is True
            ):
                successful[
                    key
                ] = record

    return successful


def generation_csv_row(
    task_key: str,
    result: Dict[str, Any],
):
    row = {}

    for column in (
        GENERATION_CSV_COLUMNS
    ):
        if column == "task_key":
            value = task_key
        else:
            value = result.get(
                column
            )

        if isinstance(
            value,
            (dict, list),
        ):
            value = json.dumps(
                value,
                ensure_ascii=False,
            )

        row[
            column
        ] = value

    return row


def rebuild_generation_csv(
    successful_results
):
    with open(
        RESULTS_CSV_FILE,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                GENERATION_CSV_COLUMNS
            ),
        )

        writer.writeheader()

        for task in (
            ALL_GENERATION_TASKS
        ):
            key = task[
                "task_key"
            ]

            if key in successful_results:
                writer.writerow(
                    generation_csv_row(
                        key,
                        successful_results[
                            key
                        ],
                    )
                )


def append_generation_csv(
    task_key: str,
    result: Dict[str, Any],
):
    exists = (
        RESULTS_CSV_FILE.exists()
        and
        RESULTS_CSV_FILE.stat().st_size
        >
        0
    )

    with open(
        RESULTS_CSV_FILE,
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                GENERATION_CSV_COLUMNS
            ),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            generation_csv_row(
                task_key,
                result,
            )
        )

        f.flush()
        os.fsync(
            f.fileno()
        )


def create_experiment3_zip():
    base_name = str(
        LOCAL_ZIP_FILE
        .with_suffix("")
    )

    if LOCAL_ZIP_FILE.exists():
        LOCAL_ZIP_FILE.unlink()

    zip_path = shutil.make_archive(
        base_name=(
            base_name
        ),
        format="zip",
        root_dir=str(
            OUTPUT_DIR.parent
        ),
        base_dir=(
            OUTPUT_DIR.name
        ),
    )

    return Path(
        zip_path
    )


def save_generation_checkpoint(
    completed: int,
    success_session: int,
    failure_session: int,
    current_task=None,
    status="running",
):
    write_json_atomic(
        CHECKPOINT_FILE,
        {
            "experiment":
                EXPERIMENT_NAME,

            "signature":
                EXPERIMENT_SIGNATURE,

            "updated_at":
                datetime.now()
                .isoformat(),

            "status":
                status,

            "total_tasks":
                TOTAL_GENERATION_TASKS,

            "completed":
                completed,

            "remaining":
                max(
                    0,
                    TOTAL_GENERATION_TASKS
                    -
                    completed,
                ),

            "success_this_session":
                success_session,

            "failure_this_session":
                failure_session,

            "current_task":
                current_task,
        },
    )


validate_generation_signature()

successful_generation_results = (
    load_successful_generation_results()
)

rebuild_generation_csv(
    successful_generation_results
)

COMPLETED_GENERATION_KEYS = set(
    successful_generation_results.keys()
)


def run_full_generation():
    global successful_generation_results
    global COMPLETED_GENERATION_KEYS

    pending = [
        task
        for task
        in ALL_GENERATION_TASKS
        if (
            task[
                "task_key"
            ]
            not in
            COMPLETED_GENERATION_KEYS
        )
    ]

    if MAX_TASKS_THIS_RUN is not None:
        pending = pending[
            :
            MAX_TASKS_THIS_RUN
        ]

    session_success = 0
    session_failure = 0
    consecutive_failures = 0

    for session_index, task in enumerate(
        pending,
        start=1,
    ):
        current_task = {
            "session_index":
                session_index,

            "session_total":
                len(
                    pending
                ),

            **task,
        }

        save_generation_checkpoint(
            completed=(
                len(
                    COMPLETED_GENERATION_KEYS
                )
            ),
            success_session=(
                session_success
            ),
            failure_session=(
                session_failure
            ),
            current_task=(
                current_task
            ),
        )

        print(
            f"\n[{session_index}/{len(pending)}] "
            f"intent={task['intent_id']} | "
            f"{task['provider']} | "
            f"{task['method']} | "
            f"rep={task['repetition']}"
        )

        try:
            result = (
                run_experiment3_generation(
                    intent_id=(
                        task[
                            "intent_id"
                        ]
                    ),
                    provider=(
                        task[
                            "provider"
                        ]
                    ),
                    method=(
                        task[
                            "method"
                        ]
                    ),
                    repetition=(
                        task[
                            "repetition"
                        ]
                    ),
                    model_id=(
                        task[
                            "model_id"
                        ]
                    ),
                )
            )

            result_dict = (
                to_json_safe(
                    result
                )
            )

        except Exception as exc:
            session_failure += 1
            consecutive_failures += 1

            append_jsonl(
                RUNNER_ERRORS_JSONL_FILE,
                {
                    "timestamp":
                        datetime.now()
                        .isoformat(),

                    "task":
                        current_task,

                    "exception_type":
                        type(
                            exc
                        ).__name__,

                    "error_message":
                        str(
                            exc
                        ),

                    "traceback":
                        traceback.format_exc(),
                },
            )

            print(
                "RUNNER ERROR:",
                type(
                    exc
                ).__name__,
                str(
                    exc
                ),
            )

            if (
                consecutive_failures
                >=
                MAX_CONSECUTIVE_FAILURES
            ):
                print(
                    "Fail-safe stop."
                )
                break

            continue

        result_dict[
            "task_key"
        ] = task[
            "task_key"
        ]

        result_dict[
            "saved_at"
        ] = (
            datetime.now()
            .isoformat()
        )

        result_dict[
            "experiment_signature"
        ] = (
            EXPERIMENT_SIGNATURE
        )

        if (
            result_dict.get(
                "run_success"
            )
            is True
        ):
            append_jsonl(
                RESULTS_JSONL_FILE,
                result_dict,
            )

            append_generation_csv(
                task[
                    "task_key"
                ],
                result_dict,
            )

            successful_generation_results[
                task[
                    "task_key"
                ]
            ] = result_dict

            COMPLETED_GENERATION_KEYS.add(
                task[
                    "task_key"
                ]
            )

            session_success += 1
            consecutive_failures = 0

            print(
                "SUCCESS | "
                f"{len(COMPLETED_GENERATION_KEYS)}/"
                f"{TOTAL_GENERATION_TASKS}"
            )

            if (
                AUTO_ZIP_EVERY
                is not None
                and
                len(
                    COMPLETED_GENERATION_KEYS
                )
                %
                AUTO_ZIP_EVERY
                ==
                0
            ):
                print(
                    "ZIP:",
                    create_experiment3_zip(),
                )

        else:
            session_failure += 1
            consecutive_failures += 1

            append_jsonl(
                FAILURES_JSONL_FILE,
                result_dict,
            )

            print(
                "FAILED:",
                result_dict.get(
                    "failure_stage"
                ),
                result_dict.get(
                    "error_message"
                ),
            )

        save_generation_checkpoint(
            completed=(
                len(
                    COMPLETED_GENERATION_KEYS
                )
            ),
            success_session=(
                session_success
            ),
            failure_session=(
                session_failure
            ),
            current_task=(
                current_task
            ),
        )

        if (
            consecutive_failures
            >=
            MAX_CONSECUTIVE_FAILURES
        ):
            print(
                "Fail-safe stop after "
                "consecutive failures."
            )
            break

    rebuild_generation_csv(
        successful_generation_results
    )

    completed = len(
        COMPLETED_GENERATION_KEYS
    )

    completion = {
        "experiment":
            EXPERIMENT_NAME,

        "updated_at":
            datetime.now()
            .isoformat(),

        "total_generation_tasks":
            TOTAL_GENERATION_TASKS,

        "completed":
            completed,

        "remaining":
            max(
                0,
                TOTAL_GENERATION_TASKS
                -
                completed,
            ),

        "complete":
            (
                completed
                ==
                TOTAL_GENERATION_TASKS
            ),
    }

    write_json_atomic(
        COMPLETION_REPORT_FILE,
        completion,
    )

    save_generation_checkpoint(
        completed=(
            completed
        ),
        success_session=(
            session_success
        ),
        failure_session=(
            session_failure
        ),
        current_task=None,
        status=(
            "complete"
            if completion[
                "complete"
            ]
            else "partial"
        ),
    )

    zip_path = (
        create_experiment3_zip()
    )

    print(
        "\nGeneration session finished."
    )

    print(
        "Completed:",
        completed,
    )

    print(
        "Expected:",
        TOTAL_GENERATION_TASKS,
    )

    print(
        "Remaining:",
        TOTAL_GENERATION_TASKS
        -
        completed,
    )

    print(
        "CSV:",
        RESULTS_CSV_FILE,
    )

    print(
        "ZIP:",
        zip_path,
    )

    return completion


print(
    "Intents:",
    len(
        INTENT_IDS
    ),
)

print(
    "Expected generation observations:",
    TOTAL_GENERATION_TASKS,
)

print(
    "Formula:",
    f"{len(INTENT_IDS)} intents × "
    f"{N_SYNTHETIC_PER_INTENT} outputs × "
    f"{len(PROVIDERS_TO_RUN)} providers × "
    f"{len(METHODS_TO_RUN)} methods"
)

print(
    "Already completed:",
    len(
        COMPLETED_GENERATION_KEYS
    ),
)


if RUN_FULL_GENERATION:
    FINAL_GENERATION_REPORT = (
        run_full_generation()
    )

else:
    print(
        "\nGeneration NOT started."
    )

    print(
        "Pilot: RUN_FULL_GENERATION=True, "
        "MAX_TASKS_THIS_RUN=20"
    )

    print(
        "Full: RUN_FULL_GENERATION=True, "
        "MAX_TASKS_THIS_RUN=None"
    )

# %%
# ============================================================
# Cell 17. Pilot calibration diagnostics
# ============================================================
#
# Run AFTER a pilot.
#
# This cell does NOT automatically choose thresholds.
# It gives distributions needed to freeze:
#
#   INTENT_MARGIN_MIN
#   INTENT_NEAR_DUPLICATE_THRESHOLD
#
# before the main experiment.
# ============================================================

CALIBRATION_DUPLICATE_THRESHOLDS = [
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
]


def load_jsonl_records(
    path: Path
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
            line = (
                line
                .strip()
            )

            if not line:
                continue

            records.append(
                json.loads(
                    line
                )
            )

    return records


generation_records = (
    load_jsonl_records(
        RESULTS_JSONL_FILE
    )
)

kazdiv_pilot_records = [
    record
    for record
    in generation_records
    if (
        record.get(
            "method"
        )
        ==
        "kazdiv"
        and
        record.get(
            "run_success"
        )
        is True
    )
]


margin_rows = []

threshold_rows = []


for record in kazdiv_pilot_records:

    quality_metadata = (
        record.get(
            "quality_metadata"
        )
        or {}
    )

    all_quality_records = (
        quality_metadata.get(
            "records"
        )
        or []
    )

    for item in all_quality_records:
        if not isinstance(
            item,
            dict,
        ):
            continue

        margin_rows.append(
            {
                "intent_id":
                    record.get(
                        "intent_id"
                    ),

                "provider":
                    record.get(
                        "provider"
                    ),

                "repetition":
                    record.get(
                        "repetition"
                    ),

                "utterance":
                    item.get(
                        "utterance"
                    ),

                "target_is_top1":
                    item.get(
                        "target_is_top1"
                    ),

                "target_score":
                    item.get(
                        "target_score"
                    ),

                "margin":
                    item.get(
                        "margin"
                    ),

                "quality_pass":
                    item.get(
                        "quality_pass"
                    ),
            }
        )

    valid_candidates = (
        quality_metadata.get(
            "valid_candidates"
        )
        or []
    )

    utterances = []

    for item in valid_candidates:
        if isinstance(
            item,
            dict,
        ):
            utterance = normalize_text(
                item.get(
                    "utterance"
                )
            )

            if utterance:
                utterances.append(
                    utterance
                )

    if len(
        utterances
    ) < 2:
        continue

    matrix = (
        candidate_similarity_matrix(
            utterances
        )
    )

    for threshold in (
        CALIBRATION_DUPLICATE_THRESHOLDS
    ):
        graph = (
            build_duplicate_graph(
                matrix,
                threshold,
            )
        )

        components = (
            connected_components(
                graph
            )
        )

        threshold_rows.append(
            {
                "intent_id":
                    record.get(
                        "intent_id"
                    ),

                "provider":
                    record.get(
                        "provider"
                    ),

                "repetition":
                    record.get(
                        "repetition"
                    ),

                "threshold":
                    threshold,

                "valid_candidates":
                    len(
                        utterances
                    ),

                "clusters":
                    len(
                        components
                    ),

                "duplicates_removed":
                    (
                        len(
                            utterances
                        )
                        -
                        len(
                            components
                        )
                    ),

                "retention_rate":
                    (
                        len(
                            components
                        )
                        /
                        len(
                            utterances
                        )
                    ),
            }
        )


margin_calibration_df = pd.DataFrame(
    margin_rows
)

duplicate_calibration_df = pd.DataFrame(
    threshold_rows
)


if len(
    margin_calibration_df
):
    margin_calibration_df.to_csv(
        OUTPUT_DIR
        /
        "experiment3_intent_margin_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Intent-score / margin diagnostics:"
    )

    display(
        margin_calibration_df[
            [
                "target_score",
                "margin",
            ]
        ]
        .describe(
            percentiles=[
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
    )

else:
    print(
        "No quality records available yet."
    )


if len(
    duplicate_calibration_df
):
    duplicate_summary = (
        duplicate_calibration_df
        .groupby(
            "threshold"
        )
        .agg(
            pools=(
                "intent_id",
                "count",
            ),
            mean_valid_candidates=(
                "valid_candidates",
                "mean",
            ),
            mean_clusters=(
                "clusters",
                "mean",
            ),
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

    display(
        duplicate_summary
    )

    duplicate_calibration_df.to_csv(
        OUTPUT_DIR
        /
        "experiment3_duplicate_threshold_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )

else:
    print(
        "No duplicate-threshold diagnostics available yet."
    )

# %%
# ============================================================
# Cell 18. Build equalized downstream training corpora
# ============================================================
#
# IMPORTANT:
#
# 1. Baseline utterances are NOT post-hoc quality filtered.
#    Their quality failures are part of the baseline method.
#
# 2. KAZ-Div final utterances have already passed its
#    method-internal quality control.
#
# 3. Exact train/test overlaps are removed.
#
# 4. Baseline and KAZ-Div get exactly the SAME number of
#    training examples per intent for each provider.
# ============================================================

if not RESULTS_CSV_FILE.exists():
    raise FileNotFoundError(
        "Generation results CSV does not exist. "
        "Run Cell 16 generation first."
    )


generation_df = pd.read_csv(
    RESULTS_CSV_FILE
)


success_mask = (
    generation_df[
        "run_success"
    ]
    .astype(
        str
    )
    .str
    .lower()
    ==
    "true"
)


generation_df = (
    generation_df[
        success_mask
    ]
    .copy()
)


generation_df[
    "final_utterance"
] = (
    generation_df[
        "final_utterance"
    ]
    .map(
        normalize_text
    )
)


generation_df[
    "intent_id"
] = (
    generation_df[
        "intent_id"
    ]
    .map(
        normalize_text
    )
)


# ------------------------------------------------------------
# Exact held-out test overlap removal
# ------------------------------------------------------------

TEST_CANONICAL_SET = set(
    human_test_df[
        "text"
    ]
    .map(
        normalize_for_exact_match
    )
)


generation_df[
    "exact_test_overlap"
] = (
    generation_df[
        "final_utterance"
    ]
    .map(
        normalize_for_exact_match
    )
    .isin(
        TEST_CANONICAL_SET
    )
)


overlap_summary = (
    generation_df
    .groupby(
        [
            "provider",
            "method",
        ]
    )[
        "exact_test_overlap"
    ]
    .sum()
    .reset_index(
        name="exact_train_test_overlap_count"
    )
)


print(
    "Exact train/test overlaps:"
)

display(
    overlap_summary
)


if REMOVE_EXACT_TRAIN_TEST_OVERLAP:
    generation_for_training_df = (
        generation_df[
            ~generation_df[
                "exact_test_overlap"
            ]
        ]
        .copy()
    )

else:
    generation_for_training_df = (
        generation_df
        .copy()
    )


# ------------------------------------------------------------
# Determine common balanced sample size per provider
# ------------------------------------------------------------

provider_balance_rows = []

PROVIDER_COMMON_N = {}


for provider in (
    PROVIDERS_TO_RUN
):

    provider_df = (
        generation_for_training_df[
            generation_for_training_df[
                "provider"
            ]
            ==
            provider
        ]
    )

    counts = []

    for intent_id in (
        INTENT_IDS
    ):
        for method in (
            METHODS_TO_RUN
        ):
            count = int(
                (
                    (
                        provider_df[
                            "intent_id"
                        ]
                        ==
                        intent_id
                    )
                    &
                    (
                        provider_df[
                            "method"
                        ]
                        ==
                        method
                    )
                )
                .sum()
            )

            counts.append(
                count
            )

            provider_balance_rows.append(
                {
                    "provider":
                        provider,

                    "intent_id":
                        intent_id,

                    "method":
                        method,

                    "available_count":
                        count,
                }
            )

    common_n = (
        min(
            counts
        )
        if counts
        else 0
    )

    PROVIDER_COMMON_N[
        provider
    ] = common_n


provider_balance_df = pd.DataFrame(
    provider_balance_rows
)


provider_balance_df.to_csv(
    OUTPUT_DIR
    /
    "experiment3_training_availability.csv",
    index=False,
    encoding="utf-8-sig",
)


print(
    "\nEqualized samples per intent:"
)

for provider, common_n in (
    PROVIDER_COMMON_N
    .items()
):
    print(
        f"{provider:<12}: "
        f"{common_n} per intent per method"
    )


def deterministic_seed(
    *parts: Any
) -> int:

    material = "|".join(
        str(
            part
        )
        for part
        in parts
    )

    digest = (
        hashlib.sha256(
            material.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )

    return int(
        digest[
            :8
        ],
        16,
    )


def build_equalized_training_set(
    provider: str,
    method: str,
    classifier_seed: int,
) -> pd.DataFrame:

    if provider not in PROVIDER_COMMON_N:
        raise ValueError(
            f"Unknown provider: {provider}"
        )

    common_n = (
        PROVIDER_COMMON_N[
            provider
        ]
    )

    if common_n < 1:
        raise RuntimeError(
            f"No complete balanced training corpus "
            f"for provider {provider}."
        )

    rows = []

    for intent_id in (
        INTENT_IDS
    ):
        group = (
            generation_for_training_df[
                (
                    generation_for_training_df[
                        "provider"
                    ]
                    ==
                    provider
                )
                &
                (
                    generation_for_training_df[
                        "method"
                    ]
                    ==
                    method
                )
                &
                (
                    generation_for_training_df[
                        "intent_id"
                    ]
                    ==
                    intent_id
                )
            ]
            .copy()
        )

        if len(
            group
        ) < common_n:
            raise RuntimeError(
                "Training balance changed unexpectedly."
            )

        sample_seed = (
            deterministic_seed(
                RANDOM_SEED,
                classifier_seed,
                provider,
                method,
                intent_id,
            )
        )

        sampled = (
            group
            .sample(
                n=common_n,
                replace=False,
                random_state=(
                    sample_seed
                ),
            )
        )

        rows.append(
            sampled
        )

    training_df = (
        pd.concat(
            rows,
            ignore_index=True,
        )
    )

    assert (
        training_df[
            "intent_id"
        ]
        .value_counts()
        .nunique()
        ==
        1
    )

    return training_df


# Preview training sizes.
training_size_rows = []

for provider in (
    PROVIDERS_TO_RUN
):
    common_n = (
        PROVIDER_COMMON_N[
            provider
        ]
    )

    training_size_rows.append(
        {
            "provider":
                provider,

            "samples_per_intent":
                common_n,

            "number_of_intents":
                len(
                    INTENT_IDS
                ),

            "training_examples_per_method":
                (
                    common_n
                    *
                    len(
                        INTENT_IDS
                    )
                ),
        }
    )


training_size_df = pd.DataFrame(
    training_size_rows
)

display(
    training_size_df
)

training_size_df.to_csv(
    OUTPUT_DIR
    /
    "experiment3_equalized_training_sizes.csv",
    index=False,
    encoding="utf-8-sig",
)

# %%
# ============================================================
# Cell 19. Downstream classifier training and evaluation
# ============================================================
#
# Independent classifier:
#   word TF-IDF
#   + character TF-IDF
#   + Logistic Regression
#
# Same classifier configuration for Baseline and KAZ-Div.
# ============================================================

CLASSIFIER_RESULTS_FILE = (
    OUTPUT_DIR
    /
    "experiment3_classifier_results.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    /
    "experiment3_classifier_predictions.csv"
)

CLASSIFICATION_REPORTS_FILE = (
    OUTPUT_DIR
    /
    "experiment3_classification_reports.json"
)


def build_classifier(
    random_seed: int
) -> Pipeline:

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(
                        1,
                        2,
                    ),
                    lowercase=True,
                    min_df=1,
                    sublinear_tf=True,
                    token_pattern=(
                        r"(?u)\b\w+\b"
                    ),
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(
                        3,
                        5,
                    ),
                    lowercase=True,
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    classifier = (
        LogisticRegression(
            C=4.0,
            max_iter=3000,
            solver="lbfgs",
            random_state=(
                random_seed
            ),
        )
    )

    return Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


test_texts = (
    human_test_df[
        "text"
    ]
    .tolist()
)

test_labels = (
    human_test_df[
        "intent_id"
    ]
    .tolist()
)


classifier_rows = []
prediction_rows = []
classification_reports = {}


for provider in (
    PROVIDERS_TO_RUN
):

    common_n = (
        PROVIDER_COMMON_N[
            provider
        ]
    )

    if common_n < 1:
        print(
            f"Skipping {provider}: "
            "no balanced training corpus."
        )
        continue

    for method in (
        METHODS_TO_RUN
    ):

        for classifier_seed in (
            CLASSIFIER_RANDOM_SEEDS
        ):

            train_df = (
                build_equalized_training_set(
                    provider=(
                        provider
                    ),
                    method=(
                        method
                    ),
                    classifier_seed=(
                        classifier_seed
                    ),
                )
            )

            train_texts = (
                train_df[
                    "final_utterance"
                ]
                .tolist()
            )

            train_labels = (
                train_df[
                    "intent_id"
                ]
                .tolist()
            )

            model = (
                build_classifier(
                    classifier_seed
                )
            )

            train_start = (
                time.perf_counter()
            )

            model.fit(
                train_texts,
                train_labels,
            )

            train_latency = (
                time.perf_counter()
                -
                train_start
            )

            predict_start = (
                time.perf_counter()
            )

            predictions = (
                model.predict(
                    test_texts
                )
            )

            predict_latency = (
                time.perf_counter()
                -
                predict_start
            )

            accuracy = (
                accuracy_score(
                    test_labels,
                    predictions,
                )
            )

            macro_f1 = (
                f1_score(
                    test_labels,
                    predictions,
                    labels=(
                        INTENT_IDS
                    ),
                    average="macro",
                    zero_division=0,
                )
            )

            weighted_f1 = (
                f1_score(
                    test_labels,
                    predictions,
                    labels=(
                        INTENT_IDS
                    ),
                    average="weighted",
                    zero_division=0,
                )
            )

            classifier_rows.append(
                {
                    "provider":
                        provider,

                    "method":
                        method,

                    "classifier_seed":
                        classifier_seed,

                    "samples_per_intent":
                        common_n,

                    "training_size":
                        len(
                            train_df
                        ),

                    "test_size":
                        len(
                            human_test_df
                        ),

                    "accuracy":
                        float(
                            accuracy
                        ),

                    "macro_f1":
                        float(
                            macro_f1
                        ),

                    "weighted_f1":
                        float(
                            weighted_f1
                        ),

                    "train_latency_sec":
                        float(
                            train_latency
                        ),

                    "prediction_latency_sec":
                        float(
                            predict_latency
                        ),
                }
            )

            report = (
                classification_report(
                    test_labels,
                    predictions,
                    labels=(
                        INTENT_IDS
                    ),
                    target_names=[
                        INTENT_NAME_MAP[
                            intent_id
                        ]
                        for intent_id
                        in INTENT_IDS
                    ],
                    output_dict=True,
                    zero_division=0,
                )
            )

            report_key = (
                f"{provider}|"
                f"{method}|"
                f"{classifier_seed}"
            )

            classification_reports[
                report_key
            ] = report

            for test_index, (
                text,
                true_label,
                predicted_label,
            ) in enumerate(
                zip(
                    test_texts,
                    test_labels,
                    predictions,
                )
            ):
                prediction_rows.append(
                    {
                        "provider":
                            provider,

                        "method":
                            method,

                        "classifier_seed":
                            classifier_seed,

                        "test_index":
                            test_index,

                        "text":
                            text,

                        "true_intent":
                            true_label,

                        "predicted_intent":
                            str(
                                predicted_label
                            ),

                        "correct":
                            bool(
                                predicted_label
                                ==
                                true_label
                            ),
                    }
                )

            print(
                f"{provider:<12} | "
                f"{method:<8} | "
                f"seed={classifier_seed} | "
                f"Acc={accuracy:.4f} | "
                f"Macro-F1={macro_f1:.4f}"
            )


classifier_results_df = pd.DataFrame(
    classifier_rows
)

classifier_predictions_df = pd.DataFrame(
    prediction_rows
)


classifier_results_df.to_csv(
    CLASSIFIER_RESULTS_FILE,
    index=False,
    encoding="utf-8-sig",
)

classifier_predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
    encoding="utf-8-sig",
)


with open(
    CLASSIFICATION_REPORTS_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        classification_reports,
        f,
        ensure_ascii=False,
        indent=2,
    )


if len(
    classifier_results_df
):
    classifier_summary_df = (
        classifier_results_df
        .groupby(
            [
                "provider",
                "method",
            ]
        )
        .agg(
            runs=(
                "classifier_seed",
                "count",
            ),

            samples_per_intent=(
                "samples_per_intent",
                "first",
            ),

            accuracy_mean=(
                "accuracy",
                "mean",
            ),

            accuracy_std=(
                "accuracy",
                "std",
            ),

            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),

            macro_f1_std=(
                "macro_f1",
                "std",
            ),

            weighted_f1_mean=(
                "weighted_f1",
                "mean",
            ),

            weighted_f1_std=(
                "weighted_f1",
                "std",
            ),
        )
        .reset_index()
    )

    classifier_summary_df.to_csv(
        OUTPUT_DIR
        /
        "experiment3_classifier_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display(
        classifier_summary_df
    )

else:
    print(
        "No downstream classifier results."
    )

# %%
# ============================================================
# Cell 20. Primary statistical comparison
# McNemar + bootstrap Macro-F1 difference
# ============================================================

PRIMARY_CLASSIFIER_SEED = (
    CLASSIFIER_RANDOM_SEEDS[
        0
    ]
)

N_BOOTSTRAP = 10000


def exact_mcnemar_pvalue(
    b: int,
    c: int,
) -> float:
    """
    Exact two-sided McNemar test.

    b = Baseline correct, KAZ-Div wrong
    c = Baseline wrong, KAZ-Div correct
    """

    n = (
        b
        +
        c
    )

    if n == 0:
        return 1.0

    smaller = min(
        b,
        c,
    )

    tail = sum(
        math.comb(
            n,
            k,
        )
        *
        (
            0.5
            **
            n
        )
        for k
        in range(
            smaller + 1
        )
    )

    return float(
        min(
            1.0,
            2.0
            *
            tail,
        )
    )


def bootstrap_macro_f1_difference(
    true_labels: np.ndarray,
    baseline_predictions: np.ndarray,
    kazdiv_predictions: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:

    true_labels = np.asarray(
        true_labels
    )

    baseline_predictions = np.asarray(
        baseline_predictions
    )

    kazdiv_predictions = np.asarray(
        kazdiv_predictions
    )

    n = len(
        true_labels
    )

    if n == 0:
        return {
            "difference":
                np.nan,

            "ci_low":
                np.nan,

            "ci_high":
                np.nan,
        }

    observed_baseline = (
        f1_score(
            true_labels,
            baseline_predictions,
            labels=(
                INTENT_IDS
            ),
            average="macro",
            zero_division=0,
        )
    )

    observed_kazdiv = (
        f1_score(
            true_labels,
            kazdiv_predictions,
            labels=(
                INTENT_IDS
            ),
            average="macro",
            zero_division=0,
        )
    )

    observed_difference = (
        observed_kazdiv
        -
        observed_baseline
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    differences = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for bootstrap_index in range(
        n_bootstrap
    ):
        indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        y_true = (
            true_labels[
                indices
            ]
        )

        y_baseline = (
            baseline_predictions[
                indices
            ]
        )

        y_kazdiv = (
            kazdiv_predictions[
                indices
            ]
        )

        baseline_f1 = (
            f1_score(
                y_true,
                y_baseline,
                labels=(
                    INTENT_IDS
                ),
                average="macro",
                zero_division=0,
            )
        )

        kazdiv_f1 = (
            f1_score(
                y_true,
                y_kazdiv,
                labels=(
                    INTENT_IDS
                ),
                average="macro",
                zero_division=0,
            )
        )

        differences[
            bootstrap_index
        ] = (
            kazdiv_f1
            -
            baseline_f1
        )

    return {
        "difference":
            float(
                observed_difference
            ),

        "ci_low":
            float(
                np.quantile(
                    differences,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    differences,
                    0.975,
                )
            ),
    }


def holm_adjust(
    pvalues: List[float]
) -> List[float]:

    values = np.asarray(
        pvalues,
        dtype=float,
    )

    order = np.argsort(
        values
    )

    adjusted = np.zeros_like(
        values
    )

    running_max = 0.0

    m = len(
        values
    )

    for rank, index in enumerate(
        order
    ):
        candidate = (
            (
                m
                -
                rank
            )
            *
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


primary_stat_rows = []


for provider in (
    PROVIDERS_TO_RUN
):

    subset = (
        classifier_predictions_df[
            (
                classifier_predictions_df[
                    "provider"
                ]
                ==
                provider
            )
            &
            (
                classifier_predictions_df[
                    "classifier_seed"
                ]
                ==
                PRIMARY_CLASSIFIER_SEED
            )
        ]
        .copy()
    )

    baseline = (
        subset[
            subset[
                "method"
            ]
            ==
            "baseline"
        ]
        .sort_values(
            "test_index"
        )
    )

    kazdiv = (
        subset[
            subset[
                "method"
            ]
            ==
            "kazdiv"
        ]
        .sort_values(
            "test_index"
        )
    )

    if (
        len(
            baseline
        )
        ==
        0
        or
        len(
            kazdiv
        )
        ==
        0
    ):
        continue

    if not np.array_equal(
        baseline[
            "test_index"
        ].to_numpy(),
        kazdiv[
            "test_index"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "Prediction rows are not paired."
        )

    true_labels = (
        baseline[
            "true_intent"
        ]
        .to_numpy()
    )

    baseline_predictions = (
        baseline[
            "predicted_intent"
        ]
        .to_numpy()
    )

    kazdiv_predictions = (
        kazdiv[
            "predicted_intent"
        ]
        .to_numpy()
    )

    baseline_correct = (
        baseline_predictions
        ==
        true_labels
    )

    kazdiv_correct = (
        kazdiv_predictions
        ==
        true_labels
    )

    b = int(
        np.sum(
            baseline_correct
            &
            ~kazdiv_correct
        )
    )

    c = int(
        np.sum(
            ~baseline_correct
            &
            kazdiv_correct
        )
    )

    mcnemar_p = (
        exact_mcnemar_pvalue(
            b,
            c,
        )
    )

    bootstrap = (
        bootstrap_macro_f1_difference(
            true_labels=(
                true_labels
            ),
            baseline_predictions=(
                baseline_predictions
            ),
            kazdiv_predictions=(
                kazdiv_predictions
            ),
            seed=(
                deterministic_seed(
                    RANDOM_SEED,
                    provider,
                    "bootstrap",
                )
            ),
        )
    )

    primary_stat_rows.append(
        {
            "provider":
                provider,

            "classifier_seed":
                PRIMARY_CLASSIFIER_SEED,

            "baseline_correct_kazdiv_wrong_b":
                b,

            "baseline_wrong_kazdiv_correct_c":
                c,

            "mcnemar_exact_p":
                mcnemar_p,

            "macro_f1_difference_kazdiv_minus_baseline":
                bootstrap[
                    "difference"
                ],

            "bootstrap_95_ci_low":
                bootstrap[
                    "ci_low"
                ],

            "bootstrap_95_ci_high":
                bootstrap[
                    "ci_high"
                ],
        }
    )


primary_stats_df = pd.DataFrame(
    primary_stat_rows
)


if len(
    primary_stats_df
):
    primary_stats_df[
        "mcnemar_holm_p"
    ] = holm_adjust(
        primary_stats_df[
            "mcnemar_exact_p"
        ]
        .tolist()
    )

    primary_stats_df.to_csv(
        OUTPUT_DIR
        /
        "experiment3_primary_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display(
        primary_stats_df
    )

else:
    print(
        "No paired primary statistical comparison available."
    )

# %%
# ============================================================
# Cell 21. Synthetic training-data diversity analysis
# ============================================================

def tokenize_kazakh(
    text: str
) -> List[str]:

    text = (
        normalize_text(
            text
        )
        .lower()
    )

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
        )
        /
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
        tokens = (
            tokenize_kazakh(
                text
            )
        )

        if len(
            tokens
        ) < n:
            continue

        for index in range(
            len(
                tokens
            )
            -
            n
            +
            1
        ):
            ngrams.append(
                tuple(
                    tokens[
                        index:
                        index + n
                    ]
                )
            )

    if not ngrams:
        return np.nan

    return float(
        len(
            set(
                ngrams
            )
        )
        /
        len(
            ngrams
        )
    )


def semantic_diversity(
    texts: List[str]
) -> float:

    if len(
        texts
    ) < 2:
        return np.nan

    embeddings = (
        encode_texts(
            texts
        )
    )

    matrix = (
        embeddings
        @
        embeddings.T
    )

    pair_values = (
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
            1.0
            -
            pair_values
        )
    )


diversity_rows = []


for provider in (
    PROVIDERS_TO_RUN
):
    for method in (
        METHODS_TO_RUN
    ):
        for intent_id in (
            INTENT_IDS
        ):
            group = (
                generation_for_training_df[
                    (
                        generation_for_training_df[
                            "provider"
                        ]
                        ==
                        provider
                    )
                    &
                    (
                        generation_for_training_df[
                            "method"
                        ]
                        ==
                        method
                    )
                    &
                    (
                        generation_for_training_df[
                            "intent_id"
                        ]
                        ==
                        intent_id
                    )
                ]
            )

            texts = [
                normalize_text(
                    value
                )
                for value
                in group[
                    "final_utterance"
                ]
                .dropna()
                .tolist()
                if normalize_text(
                    value
                )
            ]

            if not texts:
                continue

            diversity_rows.append(
                {
                    "provider":
                        provider,

                    "method":
                        method,

                    "intent_id":
                        intent_id,

                    "n":
                        len(
                            texts
                        ),

                    "unique_rate":
                        unique_rate(
                            texts
                        ),

                    "distinct_1":
                        distinct_n(
                            texts,
                            1,
                        ),

                    "distinct_2":
                        distinct_n(
                            texts,
                            2,
                        ),

                    "semantic_diversity":
                        semantic_diversity(
                            texts
                        ),

                    "mean_target_intent_score":
                        pd.to_numeric(
                            group[
                                "target_intent_score"
                            ],
                            errors="coerce",
                        )
                        .mean(),

                    "mean_intent_margin":
                        pd.to_numeric(
                            group[
                                "intent_margin"
                            ],
                            errors="coerce",
                        )
                        .mean(),

                    "quality_pass_rate":
                        (
                            group[
                                "final_quality_pass"
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
                        ),
                }
            )


intent_diversity_df = pd.DataFrame(
    diversity_rows
)


intent_diversity_df.to_csv(
    OUTPUT_DIR
    /
    "experiment3_intent_level_diversity.csv",
    index=False,
    encoding="utf-8-sig",
)


if len(
    intent_diversity_df
):
    diversity_summary_df = (
        intent_diversity_df
        .groupby(
            [
                "provider",
                "method",
            ]
        )
        .agg(
            intents=(
                "intent_id",
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

            target_intent_score_mean=(
                "mean_target_intent_score",
                "mean",
            ),

            intent_margin_mean=(
                "mean_intent_margin",
                "mean",
            ),

            quality_pass_rate_mean=(
                "quality_pass_rate",
                "mean",
            ),
        )
        .reset_index()
    )

    diversity_summary_df.to_csv(
        OUTPUT_DIR
        /
        "experiment3_diversity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display(
        diversity_summary_df
    )

else:
    print(
        "No diversity results available."
    )

# %%
# ============================================================
# Cell 22. Primary confusion matrices and per-intent F1
# ============================================================

CONFUSION_DIR = (
    OUTPUT_DIR
    /
    "confusion_matrices"
)

CONFUSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


per_intent_rows = []


for provider in (
    PROVIDERS_TO_RUN
):
    for method in (
        METHODS_TO_RUN
    ):

        subset = (
            classifier_predictions_df[
                (
                    classifier_predictions_df[
                        "provider"
                    ]
                    ==
                    provider
                )
                &
                (
                    classifier_predictions_df[
                        "method"
                    ]
                    ==
                    method
                )
                &
                (
                    classifier_predictions_df[
                        "classifier_seed"
                    ]
                    ==
                    PRIMARY_CLASSIFIER_SEED
                )
            ]
            .sort_values(
                "test_index"
            )
        )

        if len(
            subset
        ) == 0:
            continue

        y_true = (
            subset[
                "true_intent"
            ]
            .tolist()
        )

        y_pred = (
            subset[
                "predicted_intent"
            ]
            .tolist()
        )

        matrix = (
            confusion_matrix(
                y_true,
                y_pred,
                labels=(
                    INTENT_IDS
                ),
            )
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=(
                INTENT_IDS
            ),
            columns=(
                INTENT_IDS
            ),
        )

        matrix_df.index.name = (
            "true_intent"
        )

        matrix_df.to_csv(
            CONFUSION_DIR
            /
            (
                f"confusion_"
                f"{provider}_"
                f"{method}_"
                f"seed{PRIMARY_CLASSIFIER_SEED}.csv"
            ),
            encoding="utf-8-sig",
        )

        report = (
            classification_report(
                y_true,
                y_pred,
                labels=(
                    INTENT_IDS
                ),
                output_dict=True,
                zero_division=0,
            )
        )

        for intent_id in (
            INTENT_IDS
        ):
            metrics = (
                report.get(
                    intent_id,
                    {}
                )
            )

            per_intent_rows.append(
                {
                    "provider":
                        provider,

                    "method":
                        method,

                    "classifier_seed":
                        PRIMARY_CLASSIFIER_SEED,

                    "intent_id":
                        intent_id,

                    "intent_name":
                        INTENT_NAME_MAP[
                            intent_id
                        ],

                    "precision":
                        metrics.get(
                            "precision",
                            np.nan,
                        ),

                    "recall":
                        metrics.get(
                            "recall",
                            np.nan,
                        ),

                    "f1":
                        metrics.get(
                            "f1-score",
                            np.nan,
                        ),

                    "support":
                        metrics.get(
                            "support",
                            np.nan,
                        ),
                }
            )


per_intent_f1_df = pd.DataFrame(
    per_intent_rows
)


per_intent_f1_df.to_csv(
    OUTPUT_DIR
    /
    "experiment3_primary_per_intent_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)


display(
    per_intent_f1_df.head(
        min(
            20,
            len(
                per_intent_f1_df
            ),
        )
    )
)

# %%
# ============================================================
# Cell 23. Optional data-efficiency experiment
# ============================================================
#
# Tests whether KAZ-Div is more useful in low-data settings.
#
# Default = False because this trains many classifiers.
# ============================================================

RUN_DATA_EFFICIENCY_ANALYSIS = False


DATA_EFFICIENCY_SIZES = [
    5,
    10,
    20,
    30,
    40,
    50,
]


def build_training_set_with_n_per_intent(
    provider: str,
    method: str,
    classifier_seed: int,
    n_per_intent: int,
) -> pd.DataFrame:

    available_common = (
        PROVIDER_COMMON_N[
            provider
        ]
    )

    if n_per_intent > available_common:
        raise ValueError(
            f"Requested {n_per_intent} per intent, "
            f"but only {available_common} are "
            f"available for {provider}."
        )

    rows = []

    for intent_id in (
        INTENT_IDS
    ):
        group = (
            generation_for_training_df[
                (
                    generation_for_training_df[
                        "provider"
                    ]
                    ==
                    provider
                )
                &
                (
                    generation_for_training_df[
                        "method"
                    ]
                    ==
                    method
                )
                &
                (
                    generation_for_training_df[
                        "intent_id"
                    ]
                    ==
                    intent_id
                )
            ]
            .copy()
        )

        sample_seed = (
            deterministic_seed(
                RANDOM_SEED,
                "efficiency",
                classifier_seed,
                provider,
                method,
                intent_id,
                n_per_intent,
            )
        )

        sampled = (
            group
            .sample(
                n=(
                    n_per_intent
                ),
                replace=False,
                random_state=(
                    sample_seed
                ),
            )
        )

        rows.append(
            sampled
        )

    return (
        pd.concat(
            rows,
            ignore_index=True,
        )
    )


if RUN_DATA_EFFICIENCY_ANALYSIS:

    efficiency_rows = []

    for provider in (
        PROVIDERS_TO_RUN
    ):

        available_common = (
            PROVIDER_COMMON_N[
                provider
            ]
        )

        valid_sizes = [
            size
            for size
            in DATA_EFFICIENCY_SIZES
            if size
            <=
            available_common
        ]

        for n_per_intent in (
            valid_sizes
        ):

            for method in (
                METHODS_TO_RUN
            ):

                for classifier_seed in (
                    CLASSIFIER_RANDOM_SEEDS
                ):

                    train_df = (
                        build_training_set_with_n_per_intent(
                            provider=(
                                provider
                            ),
                            method=(
                                method
                            ),
                            classifier_seed=(
                                classifier_seed
                            ),
                            n_per_intent=(
                                n_per_intent
                            ),
                        )
                    )

                    model = (
                        build_classifier(
                            classifier_seed
                        )
                    )

                    model.fit(
                        train_df[
                            "final_utterance"
                        ]
                        .tolist(),
                        train_df[
                            "intent_id"
                        ]
                        .tolist(),
                    )

                    predictions = (
                        model.predict(
                            test_texts
                        )
                    )

                    efficiency_rows.append(
                        {
                            "provider":
                                provider,

                            "method":
                                method,

                            "classifier_seed":
                                classifier_seed,

                            "n_per_intent":
                                n_per_intent,

                            "training_size":
                                len(
                                    train_df
                                ),

                            "accuracy":
                                float(
                                    accuracy_score(
                                        test_labels,
                                        predictions,
                                    )
                                ),

                            "macro_f1":
                                float(
                                    f1_score(
                                        test_labels,
                                        predictions,
                                        labels=(
                                            INTENT_IDS
                                        ),
                                        average="macro",
                                        zero_division=0,
                                    )
                                ),

                            "weighted_f1":
                                float(
                                    f1_score(
                                        test_labels,
                                        predictions,
                                        labels=(
                                            INTENT_IDS
                                        ),
                                        average="weighted",
                                        zero_division=0,
                                    )
                                ),
                        }
                    )

    data_efficiency_df = pd.DataFrame(
        efficiency_rows
    )

    data_efficiency_df.to_csv(
        OUTPUT_DIR
        /
        "experiment3_data_efficiency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    data_efficiency_summary_df = (
        data_efficiency_df
        .groupby(
            [
                "provider",
                "method",
                "n_per_intent",
            ]
        )
        .agg(
            accuracy_mean=(
                "accuracy",
                "mean",
            ),

            accuracy_std=(
                "accuracy",
                "std",
            ),

            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),

            macro_f1_std=(
                "macro_f1",
                "std",
            ),
        )
        .reset_index()
    )

    data_efficiency_summary_df.to_csv(
        OUTPUT_DIR
        /
        "experiment3_data_efficiency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display(
        data_efficiency_summary_df
    )

else:
    print(
        "Data-efficiency analysis skipped."
    )
    print(
        "Set RUN_DATA_EFFICIENCY_ANALYSIS = True "
        "after the main downstream evaluation."
    )

# %%
# ============================================================
# Cell 24. Final integrity checks and experiment summary
# ============================================================

print(
    "=" * 76
)

print(
    "EXPERIMENT 3 INTEGRITY SUMMARY"
)

print(
    "=" * 76
)


print(
    "Intent definitions:",
    len(
        INTENT_IDS
    ),
)

print(
    "Human test examples:",
    len(
        human_test_df
    ),
)

print(
    "Generation observations available:",
    len(
        generation_df
    ),
)


if (
    "classifier_results_df"
    in globals()
):
    print(
        "Classifier runs:",
        len(
            classifier_results_df
        ),
    )


print(
    "\nTrain/test separation:"
)

print(
    "  Human test used in prompts      : NO"
)

print(
    "  Exact train/test overlap removal:",
    REMOVE_EXACT_TRAIN_TEST_OVERLAP,
)


print(
    "\nDownstream fairness:"
)

print(
    "  Same classifier architecture : YES"
)

print(
    "  Same human test set          : YES"
)

print(
    "  Equal examples per intent    : YES"
)

print(
    "  Equal Baseline/KAZ-Div size  : YES"
)

print(
    "  Same classifier seeds        : YES"
)


print(
    "\nKAZ-Div statelessness:"
)

print(
    "  Previous synthetic outputs used : NO"
)

print(
    "  Conversation history used       : NO"
)

print(
    "  Global generation dedup used    : NO"
)


print(
    "\nPrimary downstream classifier:"
)

print(
    "  Word TF-IDF + Character TF-IDF "
    "+ Logistic Regression"
)

print(
    "  Independent from BGE-M3: YES"
)


print(
    "\nPrimary metrics:"
)

print(
    "  Accuracy"
)

print(
    "  Macro-F1"
)

print(
    "  Weighted-F1"
)

print(
    "  Per-intent F1"
)

print(
    "  McNemar exact test"
)

print(
    "  Bootstrap Macro-F1 difference"
)

print(
    "=" * 76
)

# %%
# ============================================================
# Cell 25. Export all Experiment 3 results as ZIP
# ============================================================

def export_experiment3_zip(
    download: bool = True
):
    zip_file = Path(
        "/content/KAZ_Div_Experiment3_final.zip"
    )

    if zip_file.exists():
        zip_file.unlink()

    zip_created = (
        shutil.make_archive(
            base_name=str(
                zip_file
                .with_suffix("")
            ),
            format="zip",
            root_dir=str(
                OUTPUT_DIR.parent
            ),
            base_dir=(
                OUTPUT_DIR.name
            ),
        )
    )

    zip_created = Path(
        zip_created
    )

    print(
        "ZIP created:",
        zip_created,
    )

    print(
        "Size:",
        round(
            zip_created
            .stat()
            .st_size
            /
            1024**2,
            2,
        ),
        "MB",
    )

    if download:
        files.download(
            str(
                zip_created
            )
        )

    return zip_created


print(
    "Experiment 3 export helper ready."
)

print(
    "To download all results:"
)

print(
    "export_experiment3_zip(download=True)"
)