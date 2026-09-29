"""
KAZ-Div — Experiment 2: Diverse Grounded Kazakh Question Generation

Exported from the publication reproducibility notebook.
API keys are not embedded; configure credentials via environment variables
or your notebook secrets manager before execution.
"""

# %% [cell 1]
import sys
import importlib.metadata as md

print("Python:", sys.version)

for pkg in [
    "torch",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "accelerate",
]:
    try:
        print(pkg, "=", md.version(pkg))
    except Exception:
        print(pkg, "= NOT INSTALLED")


import torch
import transformers
import huggingface_hub

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from huggingface_hub import (
    notebook_login,
    model_info,
)

print("\nTransformers:", transformers.__version__)
print("HF Hub:", huggingface_hub.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nBASE ENVIRONMENT: PASS")

# %% [cell 2]
# ============================================================
# Anthropic API key setup
# ============================================================

import os
import getpass

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    ANTHROPIC_API_KEY = getpass.getpass(
        "Enter ANTHROPIC_API_KEY: "
    )

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

print(
    "ANTHROPIC_API_KEY configured:",
    bool(ANTHROPIC_API_KEY)
)

# %% [cell 3]
# ============================================================
# Hugging Face login via browser link + code
# ============================================================

from huggingface_hub import notebook_login

notebook_login()

# %% [cell 4]
# Run this in a fresh Colab runtime.
# If Colab asks to restart after installation, restart once and continue with Cell 1.
%pip install -q --upgrade --no-cache-dir \
  "transformers>=4.57,<6" \
  "accelerate>=1.10" \
  "huggingface_hub>=0.34" \
  "safetensors>=0.5" \
  "sentencepiece>=0.2" \
  "openai>=1.100" \
  "anthropic>=0.60" \
  "scipy>=1.14" \
  "scikit-learn>=1.6"

# %% [cell 5]
# ============================================================
# Cell 1. Imports and reproducibility
# ============================================================

import os
import re
import gc
import csv
import json
import math
import time
import random
import hashlib
import unicodedata
import traceback
import shutil

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Optional,
    List,
    Dict,
    Tuple,
    Union,
)

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

from huggingface_hub import (
    model_info,
    whoami,
)

from openai import OpenAI

import anthropic


RANDOM_SEED = 20260922


random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


print(
    "NumPy:",
    np.__version__
)

print(
    "Pandas:",
    pd.__version__
)

print(
    "Torch:",
    torch.__version__
)

print(
    "Transformers:",
    transformers.__version__
)

print(
    "CUDA:",
    torch.cuda.is_available()
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )


try:

    hf_user = whoami()

    print(
        "Hugging Face:",
        hf_user.get(
            "name",
            "authenticated"
        )
    )

except Exception:

    print(
        "Hugging Face: not authenticated"
    )


print(
    "\nCell 1: PASS"
)

# %% [cell 6]
DATASET_PATH = Path("/content/experiment2_question_contexts.csv")

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
    for ch in ["\u200B", "\u200C", "\u200D", "\u2060", "\uFEFF"]:
        text = text.replace(ch, "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_question_sources(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Upload experiment2_question_contexts.csv "
            "to /content using the Colab Files panel."
        )

    df = pd.read_csv(path, encoding="utf-8-sig")

    required = {"id", "text", "domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[["id", "text", "domain"]].copy()
    df["id"] = df["id"].astype(str).str.strip()
    df["text"] = df["text"].map(normalize_text)
    df["domain"] = df["domain"].astype(str).str.strip().str.lower()

    if len(df) != 100:
        raise ValueError(f"Expected 100 rows, found {len(df)}.")

    if df["id"].duplicated().any():
        raise ValueError("Duplicate source IDs detected.")

    if df["text"].duplicated().any():
        raise ValueError("Duplicate source texts detected.")

    if (df["text"].str.len() == 0).any():
        raise ValueError("Empty source text detected.")

    observed = df["domain"].value_counts().to_dict()

    if set(observed) != set(EXPECTED_DOMAINS):
        raise ValueError(
            f"Unexpected domains. Expected {EXPECTED_DOMAINS}, "
            f"observed {sorted(observed)}."
        )

    for d in EXPECTED_DOMAINS:
        if observed.get(d, 0) != 20:
            raise ValueError(
                f"Domain {d!r} must contain 20 contexts; "
                f"found {observed.get(d, 0)}."
            )

    return df.sort_values("id").reset_index(drop=True)

question_sources = load_question_sources(DATASET_PATH)
question_sources["word_count"] = question_sources["text"].str.split().map(len)

print("Rows:", len(question_sources))
print(question_sources["domain"].value_counts().sort_index())
display(question_sources.head())
print("Cell 2: PASS")

# %% [cell 7]
EXPERIMENT_NAME = "Experiment_2_Diverse_Question_Generation"

# ============================================================
# MAIN EXPERIMENT SETTINGS — PILOT ALREADY COMPLETED
# RUN_MODE = "full"
# THRESHOLD_FROZEN = True
# QUESTION_NEAR_DUPLICATE_THRESHOLD = 0.95
# ============================================================

RUN_MODE = "full"
THRESHOLD_FROZEN = True

if RUN_MODE not in {"pilot", "full"}:
    raise ValueError("RUN_MODE must be 'pilot' or 'full'.")

if not isinstance(THRESHOLD_FROZEN, bool):
    raise TypeError("THRESHOLD_FROZEN must be True/False.")

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

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_IDS = {
    "openai": "gpt-5.6-sol",
    "anthropic": "claude-sonnet-4-6",
    "kazllm": "issai/LLama-3.1-KazLLM-1.0-8B",
    "llama": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

ENABLED_PROVIDERS = [
    "openai",
    "anthropic",
    "kazllm",
    "llama",
]

METHODS = ["baseline", "kazdiv"]

N_REPEATS = 5
K_CANDIDATES = 12
MAX_KAZDIV_CALLS = 2
MAX_OUTPUT_TOKENS = 1600

PILOT_SOURCES_PER_DOMAIN = 2
PILOT_REPETITION = 1

MIN_QUESTION_WORDS = 3
MAX_QUESTION_WORDS = 35

MIN_ANSWER_WORDS = 1
MAX_ANSWER_WORDS = 20

# Frozen after pilot calibration.
QUESTION_NEAR_DUPLICATE_THRESHOLD = 0.95

# Relevance is measured but NOT used as a hard filter.
QUESTION_RELEVANCE_MIN = None

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_LENGTH = 512

KAZLLM_DO_SAMPLE = True
KAZLLM_TEMPERATURE = 0.6
KAZLLM_TOP_P = 1.0

LLAMA_DO_SAMPLE = True
LLAMA_TEMPERATURE = 0.6
LLAMA_TOP_P = 1.0

if RUN_MODE == "full" and not THRESHOLD_FROZEN:
    raise RuntimeError(
        "Full experiment blocked: THRESHOLD_FROZEN=False. "
        "Run pilot/calibration first and freeze the threshold."
    )

assert set(ENABLED_PROVIDERS) == {
    "openai", "anthropic", "kazllm", "llama"
}
assert "gemini" not in ENABLED_PROVIDERS

N_FULL_TASKS = (
    len(question_sources)
    * len(ENABLED_PROVIDERS)
    * len(METHODS)
    * N_REPEATS
)

N_PILOT_CONTEXTS = (
    PILOT_SOURCES_PER_DOMAIN
    * len(EXPECTED_DOMAINS)
)

N_PILOT_TASKS = (
    N_PILOT_CONTEXTS
    * len(ENABLED_PROVIDERS)
)

CONFIG = {
    "experiment": EXPERIMENT_NAME,
    "run_mode": RUN_MODE,
    "random_seed": RANDOM_SEED,
    "model_ids": MODEL_IDS,
    "providers": ENABLED_PROVIDERS,
    "methods": METHODS,
    "n_repeats": N_REPEATS,
    "k_candidates": K_CANDIDATES,
    "max_kazdiv_calls": MAX_KAZDIV_CALLS,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "pilot_sources_per_domain": PILOT_SOURCES_PER_DOMAIN,
    "pilot_repetition": PILOT_REPETITION,
    "question_word_range": [MIN_QUESTION_WORDS, MAX_QUESTION_WORDS],
    "answer_word_range": [MIN_ANSWER_WORDS, MAX_ANSWER_WORDS],
    "question_near_duplicate_threshold": QUESTION_NEAR_DUPLICATE_THRESHOLD,
    "threshold_frozen": THRESHOLD_FROZEN,
    "question_relevance_min": QUESTION_RELEVANCE_MIN,
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_pooling": "CLS + L2 normalization",
    "stateless": True,
    "planned_full_observations": N_FULL_TASKS,
    "planned_pilot_tasks": N_PILOT_TASKS,
}

with open(
    OUTPUT_DIR / "experiment2_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(CONFIG, f, ensure_ascii=False, indent=2)

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
print("Cell 3: PASS")

# %% [cell 8]
# ============================================================
# FIX Transformers stack
# Keep existing torch/CUDA
# ============================================================

%pip install -q --no-cache-dir --force-reinstall --no-deps \
    "transformers==5.17.0" \
    "huggingface_hub==1.33.0"

print("Installed.")
print("NOW: Runtime -> Restart session")

# %% [cell 9]
# ============================================================
# VERIFY CLEAN ENVIRONMENT
# ============================================================

import torch
import transformers
import huggingface_hub
import tokenizers

print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("HF Hub:", huggingface_hub.__version__)
print("Tokenizers:", tokenizers.__version__)
print("CUDA:", torch.cuda.is_available())

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    GgufConfig,
)

print("GgufConfig: PASS")
print("AutoModel: PASS")
print("AutoModelForCausalLM: PASS")

print("\nENVIRONMENT CHECK: PASS")

# %% [cell 10]
# ============================================================
# Cell 4. API clients + HF auth + local models + BGE-M3
# Compatible with Transformers 5.17.0
# ============================================================

import os
import gc
import json
from pathlib import Path
from typing import Optional

import torch
import transformers

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from huggingface_hub import (
    model_info,
    whoami,
)

from openai import OpenAI
import anthropic


# ============================================================
# 1. Colab Secrets
# ============================================================

try:
    from google.colab import userdata
except Exception:
    userdata = None


def get_secret(name: str) -> str:

    # First try Colab Secrets
    if userdata is not None:

        try:
            value = userdata.get(name)

            if value is not None:
                value = str(value).strip()

                if value:
                    return value

        except Exception:
            pass

    # Optional environment-variable fallback
    return str(
        os.environ.get(
            name,
            ""
        )
    ).strip()


OPENAI_API_KEY = get_secret(
    "OPENAI_API_KEY"
)

ANTHROPIC_API_KEY = get_secret(
    "ANTHROPIC_API_KEY"
)


# ============================================================
# 2. API clients
# ============================================================

openai_client = (
    OpenAI(
        api_key=OPENAI_API_KEY
    )
    if OPENAI_API_KEY
    else None
)


anthropic_client = (
    anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY
    )
    if ANTHROPIC_API_KEY
    else None
)


print(
    "OpenAI:",
    "configured"
    if openai_client
    else "NOT configured"
)

print(
    "Anthropic:",
    "configured"
    if anthropic_client
    else "NOT configured"
)


# ============================================================
# 3. Hugging Face authentication check
#
# notebook_login() was already performed earlier.
# No HF_TOKEN variable is required here.
# ============================================================

try:

    HF_USER = whoami()

    print(
        "Hugging Face:",
        HF_USER.get(
            "name",
            "authenticated"
        )
    )

    HF_AUTHENTICATED = True

except Exception as exc:

    HF_USER = None

    HF_AUTHENTICATED = False

    print(
        "Hugging Face: not authenticated"
    )

    print(
        "Run notebook_login() before continuing."
    )

    print(
        repr(exc)
    )


# ============================================================
# 4. GPU
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is required "
        "for the local 8B models."
    )


GPU_NAME = (
    torch.cuda.get_device_name(
        0
    )
)


GPU_TOTAL_GB = (
    torch.cuda
    .get_device_properties(
        0
    )
    .total_memory
    /
    1024**3
)


print(
    "GPU:",
    GPU_NAME
)

print(
    "GPU total:",
    round(
        GPU_TOTAL_GB,
        2
    ),
    "GB"
)


# ============================================================
# 5. Dtype
# ============================================================

LOCAL_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)


print(
    "Local dtype:",
    LOCAL_DTYPE
)


# ============================================================
# 6. Local model state
# ============================================================

LOCAL_PROVIDERS = {
    "kazllm",
    "llama",
}


local_model = None
local_tokenizer = None

local_loaded_provider = None
local_loaded_model_id = None


# ============================================================
# 7. Hugging Face commit hash
# ============================================================

def hf_commit(
    model_id: str
) -> Optional[str]:

    try:

        # Cached notebook_login authentication
        # is automatically used.
        info = model_info(
            model_id
        )

        return info.sha

    except Exception as exc:

        print(
            f"Could not read commit for {model_id}:",
            repr(exc)
        )

        return None


# ============================================================
# 8. Release local generative model
# ============================================================

def release_local_model(
    verbose: bool = True
):

    global local_model
    global local_tokenizer
    global local_loaded_provider
    global local_loaded_model_id


    if local_model is not None:

        del local_model


    if local_tokenizer is not None:

        del local_tokenizer


    local_model = None
    local_tokenizer = None

    local_loaded_provider = None
    local_loaded_model_id = None


    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


    if verbose:

        print(
            "Local generative model released."
        )


# ============================================================
# 9. Lazy local model loader
# ============================================================

def load_local_provider(
    provider: str
):

    global local_model
    global local_tokenizer
    global local_loaded_provider
    global local_loaded_model_id


    provider = (
        str(provider)
        .strip()
        .lower()
    )


    if provider not in LOCAL_PROVIDERS:

        raise ValueError(
            f"{provider!r} "
            "is not a local provider."
        )


    model_id = (
        MODEL_IDS[
            provider
        ]
    )


    # Already loaded
    if (
        local_model is not None
        and
        local_tokenizer is not None
        and
        local_loaded_provider
        ==
        provider
        and
        local_loaded_model_id
        ==
        model_id
    ):

        return (
            local_tokenizer,
            local_model,
        )


    # Release previous local model
    if local_model is not None:

        release_local_model(
            verbose=False
        )


    print(
        "\nLoading local model:"
    )

    print(
        provider,
        "->",
        model_id
    )


    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    local_tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_id,
            use_fast=True,
        )
    )


    if (
        local_tokenizer.pad_token
        is None
    ):

        local_tokenizer.pad_token = (
            local_tokenizer.eos_token
        )


    local_tokenizer.padding_side = (
        "left"
    )


    # --------------------------------------------------------
    # Model
    # Transformers 5.x: dtype= is preferred
    # --------------------------------------------------------

    local_model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_id,

            dtype=LOCAL_DTYPE,

            device_map="auto",

            low_cpu_mem_usage=True,
        )
    )


    local_model.eval()


    local_loaded_provider = (
        provider
    )

    local_loaded_model_id = (
        model_id
    )


    print(
        "Loaded:",
        model_id
    )


    return (
        local_tokenizer,
        local_model,
    )


# ============================================================
# 10. BGE-M3
# ============================================================

BGE_DEVICE = "cuda"


BGE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)


print(
    "\nLoading BGE-M3..."
)

print(
    "Model:",
    EMBEDDING_MODEL_NAME
)

print(
    "Device:",
    BGE_DEVICE
)

print(
    "dtype:",
    BGE_DTYPE
)


# ------------------------------------------------------------
# BGE tokenizer
# ------------------------------------------------------------

bge_tokenizer = (
    AutoTokenizer
    .from_pretrained(
        EMBEDDING_MODEL_NAME,
        use_fast=True,
    )
)


# ------------------------------------------------------------
# BGE model
# ------------------------------------------------------------

bge_model = (
    AutoModel
    .from_pretrained(
        EMBEDDING_MODEL_NAME,
        dtype=BGE_DTYPE,
    )
)


bge_model = (
    bge_model
    .to(
        BGE_DEVICE
    )
)


bge_model.eval()


EMBEDDING_DIMENSION = int(
    bge_model.config.hidden_size
)


print(
    "BGE-M3 loaded."
)

print(
    "Embedding dimension:",
    EMBEDDING_DIMENSION
)


# ============================================================
# 11. Runtime metadata
# ============================================================

RUNTIME_METADATA = {

    "torch_version":
        torch.__version__,

    "transformers_version":
        transformers.__version__,

    "gpu":
        GPU_NAME,

    "gpu_total_gb":
        round(
            GPU_TOTAL_GB,
            3
        ),

    "local_dtype":
        str(
            LOCAL_DTYPE
        ),

    "kazllm_model_id":
        MODEL_IDS[
            "kazllm"
        ],

    "kazllm_commit_hash":
        hf_commit(
            MODEL_IDS[
                "kazllm"
            ]
        ),

    "llama_model_id":
        MODEL_IDS[
            "llama"
        ],

    "llama_commit_hash":
        hf_commit(
            MODEL_IDS[
                "llama"
            ]
        ),

    "bge_model_id":
        EMBEDDING_MODEL_NAME,

    "bge_commit_hash":
        hf_commit(
            EMBEDDING_MODEL_NAME
        ),

    "embedding_dimension":
        EMBEDDING_DIMENSION,

    "embedding_pooling":
        "CLS + L2 normalization",

    "hf_authenticated":
        HF_AUTHENTICATED,
}


RUNTIME_METADATA_FILE = (
    OUTPUT_DIR
    /
    "experiment2_runtime_metadata.json"
)


with open(
    RUNTIME_METADATA_FILE,
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


# ============================================================
# 12. Pre-flight validation
# ============================================================

errors = []


if openai_client is None:

    errors.append(
        "OPENAI_API_KEY is missing."
    )


if anthropic_client is None:

    errors.append(
        "ANTHROPIC_API_KEY is missing."
    )


if not HF_AUTHENTICATED:

    errors.append(
        "Hugging Face authentication "
        "is not active."
    )


if not torch.cuda.is_available():

    errors.append(
        "CUDA is unavailable."
    )


if errors:

    print(
        "\nPRE-FLIGHT ISSUES:"
    )

    for error in errors:

        print(
            "-",
            error
        )

else:

    print(
        "\n======================================"
    )

    print(
        "EXPERIMENT 2 PRE-FLIGHT: PASS"
    )

    print(
        "======================================"
    )


# ============================================================
# 13. Final status
# ============================================================

print(
    "\nOpenAI:",
    "configured"
    if openai_client
    else "NOT configured"
)

print(
    "Anthropic:",
    "configured"
    if anthropic_client
    else "NOT configured"
)

print(
    "Hugging Face:",
    "authenticated"
    if HF_AUTHENTICATED
    else "NOT authenticated"
)

print(
    "BGE-M3 dimension:",
    EMBEDDING_DIMENSION
)

print(
    "Runtime metadata:",
    RUNTIME_METADATA_FILE
)

print(
    "\nCell 4: PASS"
)

# %% [cell 11]
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

def call_openai_once(
    prompt: str,
    model_id: str,
) -> LLMResult:
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

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

    usage = getattr(response, "usage", None)

    input_tokens = safe_int(
        getattr(usage, "input_tokens", 0)
        if usage is not None
        else 0
    )
    output_tokens = safe_int(
        getattr(usage, "output_tokens", 0)
        if usage is not None
        else 0
    )

    return LLMResult(
        text=text,
        provider="openai",
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_sec=latency,
        request_id=getattr(response, "id", None),
        finish_reason=getattr(response, "status", None),
    )

def call_anthropic_once(
    prompt: str,
    model_id: str,
) -> LLMResult:
    if anthropic_client is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

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

    parts = [
        block.text
        for block in getattr(response, "content", [])
        if hasattr(block, "text")
    ]

    text = normalize_text("\n".join(parts))

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
        latency_sec=latency,
        request_id=getattr(response, "id", None),
        finish_reason=getattr(response, "stop_reason", None),
    )

def local_generation_settings(
    provider: str,
) -> Dict[str, Any]:
    if provider == "kazllm":
        return {
            "do_sample": KAZLLM_DO_SAMPLE,
            "temperature": KAZLLM_TEMPERATURE,
            "top_p": KAZLLM_TOP_P,
        }

    if provider == "llama":
        return {
            "do_sample": LLAMA_DO_SAMPLE,
            "temperature": LLAMA_TEMPERATURE,
            "top_p": LLAMA_TOP_P,
        }

    raise ValueError(provider)

def call_local_once(
    provider: str,
    prompt: str,
    model_id: str,
    generation_seed: Optional[int] = None,
) -> LLMResult:
    tokenizer, model = load_local_provider(provider)

    if generation_seed is not None:
        seed = int(generation_seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    try:
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    device = next(model.parameters()).device

    model_inputs = {
        k: v.to(device)
        for k, v in model_inputs.items()
    }

    input_tokens = int(
        model_inputs["input_ids"].shape[1]
    )

    settings = local_generation_settings(provider)

    generation_kwargs = {
        "max_new_tokens": MAX_OUTPUT_TOKENS,
        "do_sample": settings["do_sample"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if settings["do_sample"]:
        generation_kwargs["temperature"] = settings["temperature"]
        generation_kwargs["top_p"] = settings["top_p"]

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            **generation_kwargs,
        )

    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    suffix = generated[0, input_tokens:]
    output_tokens = int(suffix.shape[0])

    text = normalize_text(
        tokenizer.decode(
            suffix,
            skip_special_tokens=True,
        )
    )

    return LLMResult(
        text=text,
        provider=provider,
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_sec=latency,
        finish_reason="completed",
    )

def call_llm(
    provider: str,
    prompt: str,
    model_id: Optional[str] = None,
    generation_seed: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> LLMResult:
    provider = provider.strip().lower()

    if provider not in ENABLED_PROVIDERS:
        raise ValueError(f"Provider {provider!r} is not enabled.")

    model_id = model_id or MODEL_IDS[provider]

    if max_attempts is None:
        max_attempts = (
            1 if provider in LOCAL_PROVIDERS else 5
        )

    retry_delays = [2, 4, 8, 16, 30]
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            if provider == "openai":
                result = call_openai_once(
                    prompt,
                    model_id,
                )
            elif provider == "anthropic":
                result = call_anthropic_once(
                    prompt,
                    model_id,
                )
            else:
                result = call_local_once(
                    provider=provider,
                    prompt=prompt,
                    model_id=model_id,
                    generation_seed=generation_seed,
                )

            result.attempts = attempt
            return result

        except Exception as exc:
            last_error = exc
            print(
                f"[{provider}] technical attempt "
                f"{attempt}/{max_attempts} failed: {repr(exc)}"
            )

            if attempt < max_attempts:
                delay = retry_delays[
                    min(attempt - 1, len(retry_delays) - 1)
                ]
                time.sleep(delay)

    raise RuntimeError(
        f"{provider} failed after {max_attempts} technical attempt(s): "
        f"{last_error}"
    )

print("Cell 5: READY")

# %% [cell 12]
BASELINE_PROMPT_TEMPLATE = """
Тапсырма:

Берілген қазақ тіліндегі мәтін бойынша бір табиғи,
мазмұнды және нақты сұрақ құрастырыңыз.

Талаптар:
1. Сұраққа жауап тек берілген мәтіннің өзінен табылуы тиіс.
2. Сыртқы білімді, болжамды немесе мәтінде жоқ ақпаратты пайдаланбаңыз.
3. Сұрақ қазақ тілінде грамматикалық және орфографиялық дұрыс болуы тиіс.
4. Сұрақ мәтіндегі нақты фактіні сұрауы тиіс.
5. Мәтінді жай ғана сұраулы сөйлемге айналдырмаңыз.
6. Жауап мәтінде дәл кездесетін қысқа үзінді болуы тиіс.
7. Жауапты парафраз жасамаңыз.
8. Жауаптың өзін сұрақ ішінде толық бермеңіз.
9. Бір ғана сұрақ және бір ғана жауап жасаңыз.
10. Тек JSON форматында жауап беріңіз.

JSON:
{{
  "question": "Сұрақ?",
  "answer": "мәтіндегі дәл жауап"
}}

Мәтін:
{source_text}
""".strip()

KAZDIV_PROMPT_TEMPLATE = """
Тапсырма:

Берілген қазақ тіліндегі мәтін бойынша дәл {k_candidates}
түрлі, табиғи, мазмұнды және нақты сұрақ құрастырыңыз.
Әр сұрақ үшін мәтінде дәл кездесетін қысқа жауапты көрсетіңіз.

Талаптар:
1. Әр сұраққа жауап тек берілген мәтіннен табылуы тиіс.
2. Сыртқы білімді пайдаланбаңыз.
3. Сұрақтар қазақ тілінде грамматикалық және орфографиялық дұрыс болуы тиіс.
4. Сұрақтар мүмкіндігінше мәтіндегі әртүрлі фактілерге бағытталуы тиіс.
5. Сұрақтар лексикалық және мүмкіндігінше синтаксистік тұрғыдан әртүрлі болуы тиіс.
6. Бірдей немесе мағынасы өте ұқсас сұрақтарды қайталамаңыз.
7. Сөздердің орнын ғана ауыстырып қайталамаңыз.
8. Мүмкіндігінше әртүрлі сұрақ түрлерін пайдаланыңыз.
9. Әр жауап мәтінде дәл кездесетін қысқа үзінді болуы тиіс.
10. Жауапты парафраз жасамаңыз.
11. Жауаптың өзін сұрақ ішінде толық бермеңіз.
12. Мәтінде нақты жауабы жоқ сұрақ жасамаңыз.
13. Дәл {k_candidates} сұрақ жасаңыз.
14. Тек JSON форматында жауап беріңіз.

JSON:
{{
  "candidates": [
    {{
      "question": "1-сұрақ?",
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
    source_text = normalize_text(source_text)
    method = method.strip().lower()

    if method == "baseline":
        return BASELINE_PROMPT_TEMPLATE.format(
            source_text=source_text
        )

    if method == "kazdiv":
        return KAZDIV_PROMPT_TEMPLATE.format(
            source_text=source_text,
            k_candidates=k_candidates,
        )

    raise ValueError(f"Unknown method: {method}")

with open(
    OUTPUT_DIR / "experiment2_prompt_templates.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "baseline": BASELINE_PROMPT_TEMPLATE,
            "kazdiv": KAZDIV_PROMPT_TEMPLATE,
            "k_candidates": K_CANDIDATES,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("Cell 6: PASS")

# %% [cell 13]
@dataclass
class QuestionCandidate:
    question: str
    answer: str

def remove_code_fences(text: Any) -> str:
    text = "" if text is None else str(text).strip()
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()

def extract_balanced(
    text: str,
    opening: str,
    closing: str,
) -> Optional[str]:
    start = text.find(opening)
    if start < 0:
        return None

    depth = 0
    quote = None
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if escaped:
            escaped = False
            continue

        if ch == "\\" and quote is not None:
            escaped = True
            continue

        if quote is not None:
            if ch == quote:
                quote = None
            continue

        if ch in {'"', "'"}:
            quote = ch
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None

def normalize_for_exact_match(text: Any) -> str:
    text = normalize_text(text).casefold()
    text = text.translate(
        str.maketrans({
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "«": '"',
            "»": '"',
        })
    )
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[.!?…]+$", "", text)
    return text.strip()

def candidate_from_any(item: Any) -> Optional[QuestionCandidate]:
    if not isinstance(item, dict):
        return None

    lower = {
        str(k).strip().lower(): v
        for k, v in item.items()
    }

    q = (
        lower.get("question")
        or lower.get("сұрақ")
        or lower.get("q")
    )
    a = (
        lower.get("answer")
        or lower.get("жауап")
        or lower.get("a")
    )

    q = normalize_text(q)
    a = normalize_text(a)

    if not q or not a:
        return None

    return QuestionCandidate(q, a)

def extract_candidate_list(obj: Any) -> Optional[List[Any]]:
    if isinstance(obj, list):
        return obj

    if not isinstance(obj, dict):
        return None

    lower = {
        str(k).strip().lower(): v
        for k, v in obj.items()
    }

    if (
        any(k in lower for k in ["question", "сұрақ", "q"])
        and
        any(k in lower for k in ["answer", "жауап", "a"])
    ):
        return [obj]

    for key in [
        "candidates",
        "questions",
        "items",
        "outputs",
        "responses",
    ]:
        if isinstance(lower.get(key), list):
            return lower[key]

    return None

def parse_response(raw_text: Any) -> Dict[str, Any]:
    text = "" if raw_text is None else str(raw_text).strip()

    if not text:
        return {
            "parse_success": False,
            "candidates": [],
            "raw_count": 0,
            "unique_count": 0,
            "exact_duplicates_removed": 0,
        }

    fenced = remove_code_fences(text)
    obj_text = extract_balanced(fenced, "{", "}")
    arr_text = extract_balanced(fenced, "[", "]")

    attempts = [
        text,
        fenced,
        obj_text,
        arr_text,
    ]

    parsed_items = None

    for candidate_text in attempts:
        if not candidate_text:
            continue
        try:
            obj = json.loads(candidate_text)
            items = extract_candidate_list(obj)
            if items is not None:
                parsed_items = items
                break
        except Exception:
            pass

    if parsed_items is None:
        return {
            "parse_success": False,
            "candidates": [],
            "raw_count": 0,
            "unique_count": 0,
            "exact_duplicates_removed": 0,
        }

    cleaned = []
    seen = set()
    exact_dups = 0

    for item in parsed_items:
        candidate = candidate_from_any(item)
        if candidate is None:
            continue

        key = normalize_for_exact_match(
            candidate.question
        )

        if key in seen:
            exact_dups += 1
            continue

        seen.add(key)
        cleaned.append(candidate)

    return {
        "parse_success": len(cleaned) > 0,
        "candidates": cleaned,
        "raw_count": len(parsed_items),
        "unique_count": len(cleaned),
        "exact_duplicates_removed": exact_dups,
    }

print("Cell 7: READY")

# %% [cell 14]
def embed_texts(
    texts: List[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    texts = [normalize_text(t) for t in texts]

    if not texts:
        return np.empty(
            (0, EMBEDDING_DIMENSION),
            dtype=np.float32,
        )

    batches = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        tokens = bge_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=EMBEDDING_MAX_LENGTH,
            return_tensors="pt",
        )

        tokens = {
            k: v.to(BGE_DEVICE)
            for k, v in tokens.items()
        }

        with torch.inference_mode():
            out = bge_model(**tokens)

        cls = out.last_hidden_state[:, 0, :]
        cls = F.normalize(
            cls.float(),
            p=2,
            dim=1,
        )

        batches.append(
            cls.detach().cpu().numpy()
        )

    return np.vstack(batches).astype(np.float32)

def answer_is_exact_span(
    answer: str,
    source_text: str,
) -> bool:
    return (
        normalize_text(answer).casefold()
        in normalize_text(source_text).casefold()
    )

def answer_leaked_in_question(
    answer: str,
    question: str,
) -> bool:
    a = normalize_text(answer).casefold()
    q = normalize_text(question).casefold()

    if len(a) < 3:
        return False

    return a in q

def validate_candidate(
    candidate: QuestionCandidate,
    source_text: str,
) -> Dict[str, Any]:
    q = normalize_text(candidate.question)
    a = normalize_text(candidate.answer)

    q_wc = len(q.split())
    a_wc = len(a.split())

    checks = {
        "question_nonempty": bool(q),
        "answer_nonempty": bool(a),
        "question_has_qmark": q.endswith("?"),
        "question_length_ok": (
            MIN_QUESTION_WORDS <= q_wc <= MAX_QUESTION_WORDS
        ),
        "answer_length_ok": (
            MIN_ANSWER_WORDS <= a_wc <= MAX_ANSWER_WORDS
        ),
        "answer_exact_span": answer_is_exact_span(
            a,
            source_text,
        ),
        "answer_not_leaked": not answer_leaked_in_question(
            a,
            q,
        ),
        "question_not_context_copy": (
            normalize_for_exact_match(q)
            != normalize_for_exact_match(source_text)
        ),
    }

    checks["hard_quality_pass"] = all(
        checks[k]
        for k in [
            "question_nonempty",
            "answer_nonempty",
            "question_has_qmark",
            "question_length_ok",
            "answer_length_ok",
            "answer_exact_span",
            "answer_not_leaked",
            "question_not_context_copy",
        ]
    )

    return checks

def relevance_scores(
    source_text: str,
    questions: List[str],
) -> np.ndarray:
    if not questions:
        return np.array([], dtype=np.float32)

    emb = embed_texts(
        [source_text] + questions
    )

    source_emb = emb[0]
    q_emb = emb[1:]

    return (q_emb @ source_emb).astype(np.float32)

def evaluate_candidates(
    source_text: str,
    candidates: List[QuestionCandidate],
) -> List[Dict[str, Any]]:
    rows = []

    for i, candidate in enumerate(candidates):
        checks = validate_candidate(
            candidate,
            source_text,
        )

        rows.append({
            "candidate_index": i,
            "question": candidate.question,
            "answer": candidate.answer,
            **checks,
        })

    scores = relevance_scores(
        source_text,
        [row["question"] for row in rows],
    )

    for row, score in zip(rows, scores):
        row["context_question_relevance"] = float(score)
        row["quality_pass"] = bool(
            row["hard_quality_pass"]
        )

    return rows

print("Cell 8: READY")

# %% [cell 15]
def semantic_components(
    questions: List[str],
    threshold: float,
) -> Dict[str, Any]:
    if not questions:
        return {
            "components": [],
            "similarity_matrix": np.empty((0, 0)),
        }

    embeddings = embed_texts(questions)
    sims = embeddings @ embeddings.T

    n = len(questions)
    visited = [False] * n
    components = []

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component = []

        while stack:
            current = stack.pop()
            component.append(current)

            neighbors = np.where(
                sims[current] >= threshold
            )[0]

            for nb in neighbors:
                nb = int(nb)

                if nb != current and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        components.append(sorted(component))

    return {
        "components": components,
        "similarity_matrix": sims,
    }

def select_kazdiv_final(
    valid_rows: List[Dict[str, Any]],
    threshold: float,
    selection_seed: int,
) -> Dict[str, Any]:
    questions = [
        row["question"]
        for row in valid_rows
    ]

    clustered = semantic_components(
        questions,
        threshold,
    )

    components = clustered["components"]

    representative_indices = []

    for component in components:
        best_idx = max(
            component,
            key=lambda idx: (
                valid_rows[idx][
                    "context_question_relevance"
                ],
                -idx,
            ),
        )

        representative_indices.append(
            best_idx
        )

    rng = random.Random(
        int(selection_seed)
    )

    selected_idx = rng.choice(
        representative_indices
    )

    pairwise_sims = []

    sims = clustered[
        "similarity_matrix"
    ]

    for i in range(len(valid_rows)):
        for j in range(i + 1, len(valid_rows)):
            pairwise_sims.append(
                float(sims[i, j])
            )

    return {
        "selected_row": valid_rows[selected_idx],
        "n_semantic_clusters": len(components),
        "semantic_duplicates_absorbed": (
            len(valid_rows)
            - len(components)
        ),
        "components": components,
        "representative_indices": representative_indices,
        "pairwise_similarities": pairwise_sims,
    }

print("Cell 9: READY")

# %% [cell 16]
def stable_seed(*parts: Any) -> int:
    payload = "||".join(str(x) for x in parts)

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16)

def make_task_id(
    source_id: str,
    provider: str,
    method: str,
    repetition: int,
) -> str:
    return (
        f"{source_id}__{provider}__"
        f"{method}__r{repetition}"
    )

def build_pilot_sources() -> pd.DataFrame:
    parts = []

    for domain in EXPECTED_DOMAINS:
        part = (
            question_sources[
                question_sources["domain"] == domain
            ]
            .sort_values("id")
            .head(PILOT_SOURCES_PER_DOMAIN)
        )
        parts.append(part)

    pilot = pd.concat(
        parts,
        ignore_index=True,
    )

    assert len(pilot) == 10

    return pilot

pilot_sources = build_pilot_sources()

print("Pilot rows:", len(pilot_sources))
print(pilot_sources["domain"].value_counts().sort_index())

# %% [cell 17]
def empty_record(
    source_row: pd.Series,
    provider: str,
    method: str,
    repetition: int,
) -> Dict[str, Any]:
    return {
        "task_id": make_task_id(
            str(source_row["id"]),
            provider,
            method,
            repetition,
        ),
        "source_id": str(source_row["id"]),
        "domain": source_row["domain"],
        "provider": provider,
        "model_id": MODEL_IDS[provider],
        "method": method,
        "repetition": int(repetition),
        "success": False,
        "failure_type": None,
        "failure_stage": None,
        "failure_message": None,
        "final_question": None,
        "final_answer": None,
        "final_relevance": np.nan,
        "final_quality_pass": False,
        "llm_calls": 0,
        "technical_attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "llm_latency_sec": 0.0,
        "pipeline_latency_sec": 0.0,
        "raw_candidate_count": 0,
        "parsed_unique_candidate_count": 0,
        "exact_duplicates_removed": 0,
        "quality_valid_candidate_count": 0,
        "n_semantic_clusters": 0,
        "semantic_duplicates_absorbed": 0,
        "quality_valid_questions_json": "[]",
        "quality_valid_answers_json": "[]",
    }

def run_baseline(
    source_row: pd.Series,
    provider: str,
    repetition: int,
) -> Dict[str, Any]:
    record = empty_record(
        source_row,
        provider,
        "baseline",
        repetition,
    )

    start = time.perf_counter()

    try:
        source_text = normalize_text(
            source_row["text"]
        )

        seed = stable_seed(
            RANDOM_SEED,
            source_row["id"],
            provider,
            "baseline",
            repetition,
        )

        result = call_llm(
            provider=provider,
            prompt=build_prompt(
                "baseline",
                source_text,
            ),
            generation_seed=seed,
        )

        record["llm_calls"] = 1
        record["technical_attempts"] = result.attempts
        record["input_tokens"] = result.input_tokens
        record["output_tokens"] = result.output_tokens
        record["total_tokens"] = result.total_tokens
        record["llm_latency_sec"] = result.latency_sec

        parsed = parse_response(
            result.text
        )

        record["raw_candidate_count"] = parsed["raw_count"]
        record["parsed_unique_candidate_count"] = parsed["unique_count"]
        record["exact_duplicates_removed"] = parsed["exact_duplicates_removed"]

        if not parsed["parse_success"]:
            record["failure_type"] = "method_failure"
            record["failure_stage"] = "parsing"
            record["failure_message"] = "Baseline output could not be parsed."
            return record

        candidate = parsed["candidates"][0]

        checks = validate_candidate(
            candidate,
            source_text,
        )

        relevance = float(
            relevance_scores(
                source_text,
                [candidate.question],
            )[0]
        )

        record.update({
            "success": True,
            "final_question": candidate.question,
            "final_answer": candidate.answer,
            "final_relevance": relevance,
            "final_quality_pass": bool(
                checks["hard_quality_pass"]
            ),
            "quality_valid_candidate_count": int(
                checks["hard_quality_pass"]
            ),
        })

        return record

    except Exception as exc:
        record["failure_type"] = "technical_failure"
        record["failure_stage"] = "runtime_or_llm"
        record["failure_message"] = repr(exc)
        return record

    finally:
        record["pipeline_latency_sec"] = (
            time.perf_counter() - start
        )

def run_kazdiv(
    source_row: pd.Series,
    provider: str,
    repetition: int,
    threshold: float,
) -> Dict[str, Any]:
    record = empty_record(
        source_row,
        provider,
        "kazdiv",
        repetition,
    )

    start = time.perf_counter()

    source_text = normalize_text(
        source_row["text"]
    )

    all_candidates: Dict[str, QuestionCandidate] = {}

    total_raw = 0
    total_exact_dups = 0

    try:
        for method_call in range(
            1,
            MAX_KAZDIV_CALLS + 1,
        ):
            seed = stable_seed(
                RANDOM_SEED,
                source_row["id"],
                provider,
                "kazdiv",
                repetition,
                method_call,
            )

            result = call_llm(
                provider=provider,
                prompt=build_prompt(
                    "kazdiv",
                    source_text,
                    K_CANDIDATES,
                ),
                generation_seed=seed,
            )

            record["llm_calls"] += 1
            record["technical_attempts"] += result.attempts
            record["input_tokens"] += result.input_tokens
            record["output_tokens"] += result.output_tokens
            record["total_tokens"] += result.total_tokens
            record["llm_latency_sec"] += result.latency_sec

            parsed = parse_response(
                result.text
            )

            total_raw += parsed["raw_count"]
            total_exact_dups += parsed["exact_duplicates_removed"]

            for candidate in parsed["candidates"]:
                key = normalize_for_exact_match(
                    candidate.question
                )

                if key not in all_candidates:
                    all_candidates[key] = candidate

            evaluated = evaluate_candidates(
                source_text,
                list(all_candidates.values()),
            )

            valid_rows = [
                row
                for row in evaluated
                if row["quality_pass"]
            ]

            if valid_rows:
                selection_seed = stable_seed(
                    RANDOM_SEED,
                    source_row["id"],
                    provider,
                    "select",
                    repetition,
                )

                selected = select_kazdiv_final(
                    valid_rows,
                    threshold,
                    selection_seed,
                )

                final = selected["selected_row"]

                record.update({
                    "success": True,
                    "final_question": final["question"],
                    "final_answer": final["answer"],
                    "final_relevance": final[
                        "context_question_relevance"
                    ],
                    "final_quality_pass": True,
                    "raw_candidate_count": total_raw,
                    "parsed_unique_candidate_count": len(all_candidates),
                    "exact_duplicates_removed": total_exact_dups,
                    "quality_valid_candidate_count": len(valid_rows),
                    "n_semantic_clusters": selected[
                        "n_semantic_clusters"
                    ],
                    "semantic_duplicates_absorbed": selected[
                        "semantic_duplicates_absorbed"
                    ],
                    "quality_valid_questions_json": json.dumps(
                        [r["question"] for r in valid_rows],
                        ensure_ascii=False,
                    ),
                    "quality_valid_answers_json": json.dumps(
                        [r["answer"] for r in valid_rows],
                        ensure_ascii=False,
                    ),
                })

                return record

        record["raw_candidate_count"] = total_raw
        record["parsed_unique_candidate_count"] = len(all_candidates)
        record["exact_duplicates_removed"] = total_exact_dups
        record["failure_type"] = "method_failure"

        if not all_candidates:
            record["failure_stage"] = "parsing"
            record["failure_message"] = (
                "No parseable candidates after allowed KAZ-Div calls."
            )
        else:
            record["failure_stage"] = "quality_filter"
            record["failure_message"] = (
                "No quality-valid candidates after allowed KAZ-Div calls."
            )

        return record

    except Exception as exc:
        record["failure_type"] = "technical_failure"
        record["failure_stage"] = "runtime_or_llm"
        record["failure_message"] = repr(exc)
        return record

    finally:
        record["pipeline_latency_sec"] = (
            time.perf_counter() - start
        )

def run_observation(
    source_row: pd.Series,
    provider: str,
    method: str,
    repetition: int,
    threshold: float,
) -> Dict[str, Any]:
    if method == "baseline":
        return run_baseline(
            source_row,
            provider,
            repetition,
        )

    if method == "kazdiv":
        return run_kazdiv(
            source_row,
            provider,
            repetition,
            threshold,
        )

    raise ValueError(method)

print("Cell 11: READY")

# %% [cell 18]
def save_checkpoint(
    df: pd.DataFrame,
    path: Path,
):
    tmp = path.with_suffix(".tmp.csv")

    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        tmp,
        path,
    )

def run_collection(
    sources_df: pd.DataFrame,
    providers: List[str],
    methods: List[str],
    repetitions: List[int],
    threshold: float,
    checkpoint_path: Path,
    retry_technical_failures: bool = True,
) -> pd.DataFrame:
    if checkpoint_path.exists():
        results = pd.read_csv(
            checkpoint_path
        )
    else:
        results = pd.DataFrame()

    existing = {}

    if len(results):
        for _, row in results.iterrows():
            existing[str(row["task_id"])] = row.to_dict()

    planned = []

    for _, source_row in sources_df.iterrows():
        for provider in providers:
            for method in methods:
                for repetition in repetitions:
                    planned.append(
                        (
                            source_row,
                            provider,
                            method,
                            repetition,
                        )
                    )

    print("Planned tasks:", len(planned))
    print("Existing checkpoint rows:", len(results))

    for index, (
        source_row,
        provider,
        method,
        repetition,
    ) in enumerate(planned, start=1):

        task_id = make_task_id(
            str(source_row["id"]),
            provider,
            method,
            repetition,
        )

        previous = existing.get(task_id)

        if previous is not None:
            success_value = previous.get("success", False)
            success_value = (
                success_value is True
                or str(success_value).lower() == "true"
            )

            failure_type = previous.get(
                "failure_type"
            )

            if success_value:
                continue

            if failure_type == "method_failure":
                continue

            if (
                failure_type == "technical_failure"
                and not retry_technical_failures
            ):
                continue

        print(
            f"[{index}/{len(planned)}] {task_id}"
        )

        record = run_observation(
            source_row,
            provider,
            method,
            repetition,
            threshold,
        )

        if len(results):
            results = results[
                results["task_id"].astype(str)
                != task_id
            ].copy()

        results = pd.concat(
            [
                results,
                pd.DataFrame([record]),
            ],
            ignore_index=True,
        )

        existing[task_id] = record

        save_checkpoint(
            results,
            checkpoint_path,
        )

        print(
            " success=",
            record["success"],
            "| failure=",
            record["failure_type"],
        )

    return results.sort_values(
        [
            "provider",
            "source_id",
            "method",
            "repetition",
        ]
    ).reset_index(drop=True)

print("Cell 12: READY")

# %% [cell 19]
RUN_PROVIDER_SMOKE_TEST = False  # main-only run; pilot already completed

if RUN_PROVIDER_SMOKE_TEST:
    smoke_source = normalize_text(
        question_sources.iloc[0]["text"]
    )

    smoke_prompt = build_prompt(
        "baseline",
        smoke_source,
    )

    smoke_results = {}

    for provider in ENABLED_PROVIDERS:
        print("\nTEST:", provider)

        try:
            result = call_llm(
                provider=provider,
                prompt=smoke_prompt,
                generation_seed=RANDOM_SEED,
            )

            parsed = parse_response(
                result.text
            )

            smoke_results[provider] = bool(
                parsed["parse_success"]
            )

            print(
                "PASS"
                if parsed["parse_success"]
                else "PARSE FAILED"
            )
            print(result.text[:400])

        except Exception as exc:
            smoke_results[provider] = False
            print("FAILED:", repr(exc))

    print("\nSUMMARY:", smoke_results)

    if not all(smoke_results.values()):
        raise RuntimeError(
            "At least one provider failed the smoke test."
        )

else:
    print("Smoke test skipped in full mode.")

# %% [cell 20]
if RUN_MODE == "pilot":
    PILOT_CHECKPOINT = (
        PILOT_OUTPUT_DIR
        / "experiment2_pilot_results.csv"
    )

    pilot_results = run_collection(
        sources_df=pilot_sources,
        providers=ENABLED_PROVIDERS,
        methods=["kazdiv"],
        repetitions=[PILOT_REPETITION],
        threshold=QUESTION_NEAR_DUPLICATE_THRESHOLD,
        checkpoint_path=PILOT_CHECKPOINT,
        retry_technical_failures=True,
    )

    print(
        pilot_results[
            [
                "provider",
                "success",
                "failure_type",
            ]
        ]
        .value_counts(dropna=False)
    )

    display(pilot_results.head())

else:
    print("Pilot skipped in full mode.")

# %% [cell 21]
THRESHOLD_GRID = [
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
]

def cluster_count_for_questions(
    questions: List[str],
    threshold: float,
) -> Tuple[int, int]:
    clustered = semantic_components(
        questions,
        threshold,
    )

    n_valid = len(questions)
    n_clusters = len(
        clustered["components"]
    )

    return (
        n_clusters,
        n_valid - n_clusters,
    )

if RUN_MODE == "pilot":
    if "pilot_results" not in globals():
        pilot_results = pd.read_csv(
            PILOT_OUTPUT_DIR
            / "experiment2_pilot_results.csv"
        )

    successful = pilot_results[
        pilot_results["success"].astype(str).str.lower()
        == "true"
    ].copy()

    rows = []

    for _, row in successful.iterrows():
        questions = json.loads(
            row["quality_valid_questions_json"]
        )

        for threshold in THRESHOLD_GRID:
            n_clusters, absorbed = (
                cluster_count_for_questions(
                    questions,
                    threshold,
                )
            )

            rows.append({
                "provider": row["provider"],
                "source_id": row["source_id"],
                "threshold": threshold,
                "n_quality_valid": len(questions),
                "n_clusters": n_clusters,
                "semantic_duplicates_absorbed": absorbed,
                "cluster_retention_ratio": (
                    n_clusters / len(questions)
                    if questions else np.nan
                ),
                "semantic_redundancy_ratio": (
                    absorbed / len(questions)
                    if questions else np.nan
                ),
            })

    threshold_task_level = pd.DataFrame(
        rows
    )

    threshold_summary = (
        threshold_task_level
        .groupby(
            ["provider", "threshold"],
            as_index=False,
        )
        .agg(
            n_tasks=("source_id", "nunique"),
            mean_quality_valid=(
                "n_quality_valid",
                "mean",
            ),
            mean_clusters=(
                "n_clusters",
                "mean",
            ),
            mean_cluster_retention=(
                "cluster_retention_ratio",
                "mean",
            ),
            mean_semantic_redundancy=(
                "semantic_redundancy_ratio",
                "mean",
            ),
        )
    )

    threshold_task_level.to_csv(
        PILOT_OUTPUT_DIR
        / "threshold_calibration_task_level.csv",
        index=False,
    )

    threshold_summary.to_csv(
        PILOT_OUTPUT_DIR
        / "threshold_calibration_summary.csv",
        index=False,
    )

    display(threshold_summary)

    print(
        "\nIMPORTANT: no threshold is selected automatically."
    )
    print(
        "Inspect these diagnostics and manually freeze the selected value."
    )
    print(
        "Then edit Cell 3: RUN_MODE='full', "
        "THRESHOLD_FROZEN=True, and set the frozen threshold."
    )

else:
    print("Calibration skipped in full mode.")

# %% [cell 22]
from pathlib import Path
import shutil
import pandas as pd

UPLOADED_CHECKPOINT = Path(
    "/content/experiment2_main_results-19.csv"
)

MAIN_OUTPUT_DIR = Path(
    "/content/KAZ_Div/Experiment_2_Question_Generation_4Models/main"
)

MAIN_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAIN_CHECKPOINT = (
    MAIN_OUTPUT_DIR
    /
    "experiment2_main_results.csv"
)

shutil.copy2(
    UPLOADED_CHECKPOINT,
    MAIN_CHECKPOINT,
)

saved = pd.read_csv(MAIN_CHECKPOINT)

print("Rows:", len(saved))
print("Unique task_id:", saved["task_id"].nunique())

# %% [cell 23]
if RUN_MODE == "full":
    if not THRESHOLD_FROZEN:
        raise RuntimeError(
            "Threshold must be frozen before main collection."
        )

    MAIN_CHECKPOINT = (
        MAIN_OUTPUT_DIR
        / "experiment2_main_results.csv"
    )

    main_results = run_collection(
        sources_df=question_sources,
        providers=ENABLED_PROVIDERS,
        methods=METHODS,
        repetitions=list(
            range(1, N_REPEATS + 1)
        ),
        threshold=QUESTION_NEAR_DUPLICATE_THRESHOLD,
        checkpoint_path=MAIN_CHECKPOINT,
        retry_technical_failures=True,
    )

    print(
        main_results[
            [
                "provider",
                "method",
                "success",
                "failure_type",
            ]
        ]
        .value_counts(dropna=False)
    )

else:
    print(
        "Main collection disabled while RUN_MODE='pilot'."
    )

# %% [cell 24]
def simple_tokens(text: str) -> List[str]:
    return re.findall(
        r"\w+",
        normalize_text(text).casefold(),
        flags=re.UNICODE,
    )

def distinct_n(
    texts: List[str],
    n: int,
) -> float:
    grams = []

    for text in texts:
        tokens = simple_tokens(text)

        grams.extend(
            tuple(tokens[i:i+n])
            for i in range(
                max(0, len(tokens) - n + 1)
            )
        )

    if not grams:
        return np.nan

    return len(set(grams)) / len(grams)

def mean_pairwise_semantic_distance(
    texts: List[str],
) -> float:
    if len(texts) < 2:
        return np.nan

    emb = embed_texts(texts)
    sims = emb @ emb.T

    distances = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            distances.append(
                1.0 - float(sims[i, j])
            )

    return float(np.mean(distances))

def bleu2_similarity(
    candidate: str,
    references: List[str],
) -> float:
    cand = simple_tokens(candidate)

    if len(cand) < 2 or not references:
        return np.nan

    def ngrams(tokens, n):
        return [
            tuple(tokens[i:i+n])
            for i in range(
                len(tokens) - n + 1
            )
        ]

    precisions = []

    for n in [1, 2]:
        cand_g = ngrams(cand, n)

        if not cand_g:
            return 0.0

        ref_max = {}

        for ref in references:
            counts = {}

            for g in ngrams(
                simple_tokens(ref),
                n,
            ):
                counts[g] = counts.get(g, 0) + 1

            for g, count in counts.items():
                ref_max[g] = max(
                    ref_max.get(g, 0),
                    count,
                )

        cand_counts = {}

        for g in cand_g:
            cand_counts[g] = (
                cand_counts.get(g, 0) + 1
            )

        clipped = sum(
            min(count, ref_max.get(g, 0))
            for g, count in cand_counts.items()
        )

        precisions.append(
            clipped / len(cand_g)
        )

    if min(precisions) <= 0:
        return 0.0

    return float(
        math.sqrt(
            precisions[0] * precisions[1]
        )
    )

def self_bleu2(texts: List[str]) -> float:
    if len(texts) < 2:
        return np.nan

    scores = []

    for i, text in enumerate(texts):
        refs = [
            t
            for j, t in enumerate(texts)
            if j != i
        ]

        scores.append(
            bleu2_similarity(
                text,
                refs,
            )
        )

    return float(
        np.nanmean(scores)
    )

def heuristic_question_type(
    question: str,
) -> str:
    q = normalize_text(question).casefold()

    patterns = [
        ("who", ["кім"]),
        ("what", ["не ", "не?", "нені", "немен"]),
        ("which", ["қандай", "қай "]),
        ("where", ["қайда", "қай жерде"]),
        ("when", ["қашан"]),
        ("how_many", ["қанша", "неше"]),
        ("why", ["неге", "не үшін"]),
        ("how", ["қалай"]),
    ]

    for label, starts in patterns:
        if any(q.startswith(x) for x in starts):
            return label

    return "other"

def heuristic_question_type_entropy(
    texts: List[str],
) -> float:
    if not texts:
        return np.nan

    labels = [
        heuristic_question_type(t)
        for t in texts
    ]

    counts = pd.Series(labels).value_counts()
    probs = counts.values / counts.values.sum()

    return float(
        -np.sum(
            probs * np.log2(probs)
        )
    )

def build_source_metrics(
    results: pd.DataFrame,
) -> pd.DataFrame:
    success = results[
        results["success"].astype(str).str.lower()
        == "true"
    ].copy()

    source_lookup = (
        question_sources
        .assign(id=question_sources["id"].astype(str))
        .set_index("id")["text"]
        .to_dict()
    )

    rows = []

    for (
        provider,
        method,
        source_id,
    ), group in success.groupby(
        [
            "provider",
            "method",
            "source_id",
        ]
    ):
        questions = (
            group["final_question"]
            .dropna()
            .astype(str)
            .tolist()
        )

        answers = (
            group["final_answer"]
            .dropna()
            .astype(str)
            .tolist()
        )

        unique_q = {
            normalize_for_exact_match(q)
            for q in questions
        }

        source_text = source_lookup[
            str(source_id)
        ]

        rows.append({
            "provider": provider,
            "method": method,
            "source_id": str(source_id),
            "domain": group["domain"].iloc[0],
            "n_successful_repetitions": len(group),
            "complete_5_repetitions": (
                len(group) == N_REPEATS
            ),
            "unique_ratio": (
                len(unique_q) / len(questions)
                if questions else np.nan
            ),
            "distinct_1": distinct_n(
                questions,
                1,
            ),
            "distinct_2": distinct_n(
                questions,
                2,
            ),
            "semantic_diversity": mean_pairwise_semantic_distance(
                questions
            ),
            "self_bleu2": self_bleu2(
                questions
            ),
            "mean_relevance": pd.to_numeric(
                group["final_relevance"],
                errors="coerce",
            ).mean(),
            "quality_pass_rate": (
                group["final_quality_pass"]
                .astype(str)
                .str.lower()
                .eq("true")
                .mean()
            ),
            "answer_span_validity_rate": np.mean([
                answer_is_exact_span(
                    a,
                    source_text,
                )
                for a in answers
            ]) if answers else np.nan,
            "heuristic_question_type_entropy":
                heuristic_question_type_entropy(
                    questions
                ),
        })

    return pd.DataFrame(rows)

if RUN_MODE == "full":
    if "main_results" not in globals():
        main_results = pd.read_csv(
            MAIN_OUTPUT_DIR
            / "experiment2_main_results.csv"
        )

    source_metrics = build_source_metrics(
        main_results
    )

    source_metrics.to_csv(
        MAIN_OUTPUT_DIR
        / "experiment2_source_level_metrics.csv",
        index=False,
    )

    summary_metrics = (
        source_metrics
        .groupby(
            ["provider", "method"],
            as_index=False,
        )
        .agg(
            n_sources=("source_id", "nunique"),
            complete_sources=(
                "complete_5_repetitions",
                "sum",
            ),
            unique_ratio=("unique_ratio", "mean"),
            distinct_1=("distinct_1", "mean"),
            distinct_2=("distinct_2", "mean"),
            semantic_diversity=(
                "semantic_diversity",
                "mean",
            ),
            self_bleu2=("self_bleu2", "mean"),
            mean_relevance=("mean_relevance", "mean"),
            quality_pass_rate=(
                "quality_pass_rate",
                "mean",
            ),
            answer_span_validity_rate=(
                "answer_span_validity_rate",
                "mean",
            ),
            heuristic_question_type_entropy=(
                "heuristic_question_type_entropy",
                "mean",
            ),
        )
    )

    summary_metrics.to_csv(
        MAIN_OUTPUT_DIR
        / "experiment2_primary_summary_metrics.csv",
        index=False,
    )

    display(summary_metrics)

else:
    print("Evaluation runs in full mode.")

# %% [cell 25]
PRIMARY_METRICS = [
    "unique_ratio",
    "distinct_1",
    "distinct_2",
    "semantic_diversity",
    "self_bleu2",
    "mean_relevance",
    "quality_pass_rate",
    "answer_span_validity_rate",
    "heuristic_question_type_entropy",
]

def bootstrap_mean_ci(
    diffs: np.ndarray,
    n_boot: int = 10000,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    diffs = np.asarray(diffs, dtype=float)

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(diffs),
        size=(n_boot, len(diffs)),
    )

    means = diffs[idx].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )

def signflip_pvalue(
    diffs: np.ndarray,
    n_perm: int = 20000,
    seed: int = RANDOM_SEED,
) -> float:
    diffs = np.asarray(diffs, dtype=float)

    observed = abs(
        float(np.mean(diffs))
    )

    rng = np.random.default_rng(seed)

    extreme = 0
    done = 0
    batch_size = 2000

    while done < n_perm:
        batch = min(
            batch_size,
            n_perm - done,
        )

        signs = rng.choice(
            [-1.0, 1.0],
            size=(batch, len(diffs)),
        )

        perm = np.abs(
            (signs * diffs).mean(axis=1)
        )

        extreme += int(
            np.sum(perm >= observed)
        )

        done += batch

    return (
        extreme + 1
    ) / (
        n_perm + 1
    )

def holm_adjust(
    p_values: List[float],
) -> List[float]:
    p = np.asarray(p_values, dtype=float)

    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0

    for rank, idx in enumerate(order):
        value = (
            (m - rank) * p[idx]
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted[idx] = min(
            running_max,
            1.0,
        )

    return adjusted.tolist()

def paired_statistics(
    source_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for provider in ENABLED_PROVIDERS:
        provider_df = source_metrics[
            source_metrics["provider"] == provider
        ].copy()

        baseline = provider_df[
            (provider_df["method"] == "baseline")
            &
            (provider_df["complete_5_repetitions"] == True)
        ].copy()

        kazdiv = provider_df[
            (provider_df["method"] == "kazdiv")
            &
            (provider_df["complete_5_repetitions"] == True)
        ].copy()

        paired_ids = sorted(
            set(baseline["source_id"])
            & set(kazdiv["source_id"])
        )

        for metric in PRIMARY_METRICS:
            b = (
                baseline[
                    baseline["source_id"].isin(paired_ids)
                ]
                .set_index("source_id")[metric]
            )

            k = (
                kazdiv[
                    kazdiv["source_id"].isin(paired_ids)
                ]
                .set_index("source_id")[metric]
            )

            shared = b.index.intersection(k.index)

            b = pd.to_numeric(
                b.loc[shared],
                errors="coerce",
            )
            k = pd.to_numeric(
                k.loc[shared],
                errors="coerce",
            )

            valid = b.notna() & k.notna()

            diffs = (
                k[valid].values
                - b[valid].values
            )

            if len(diffs) == 0:
                continue

            ci_low, ci_high = bootstrap_mean_ci(
                diffs,
                seed=stable_seed(
                    RANDOM_SEED,
                    provider,
                    metric,
                    "bootstrap",
                ),
            )

            p = signflip_pvalue(
                diffs,
                seed=stable_seed(
                    RANDOM_SEED,
                    provider,
                    metric,
                    "signflip",
                ),
            )

            rows.append({
                "provider": provider,
                "metric": metric,
                "n_complete_pairs": len(diffs),
                "baseline_mean": float(
                    b[valid].mean()
                ),
                "kazdiv_mean": float(
                    k[valid].mean()
                ),
                "mean_difference_kazdiv_minus_baseline":
                    float(np.mean(diffs)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "permutation_p": p,
            })

    stats = pd.DataFrame(rows)

    if len(stats):
        stats["holm_p"] = holm_adjust(
            stats["permutation_p"].tolist()
        )

    return stats

if RUN_MODE == "full":
    paired_stats = paired_statistics(
        source_metrics
    )

    paired_stats.to_csv(
        MAIN_OUTPUT_DIR
        / "experiment2_paired_statistics.csv",
        index=False,
    )

    display(paired_stats)

else:
    print("Statistics run in full mode.")

# %% [cell 26]
if RUN_MODE == "full":
    successful_kd = main_results[
        (main_results["method"] == "kazdiv")
        &
        (
            main_results["success"]
            .astype(str)
            .str.lower()
            == "true"
        )
    ].copy()

    candidate_pool_summary = (
        successful_kd
        .groupby(
            "provider",
            as_index=False,
        )
        .agg(
            successful_kazdiv_tasks=(
                "task_id",
                "count",
            ),
            mean_llm_calls=(
                "llm_calls",
                "mean",
            ),
            mean_raw_candidates=(
                "raw_candidate_count",
                "mean",
            ),
            mean_unique_parsed_candidates=(
                "parsed_unique_candidate_count",
                "mean",
            ),
            mean_exact_duplicates_removed=(
                "exact_duplicates_removed",
                "mean",
            ),
            mean_quality_valid_candidates=(
                "quality_valid_candidate_count",
                "mean",
            ),
            mean_semantic_clusters=(
                "n_semantic_clusters",
                "mean",
            ),
            mean_semantic_duplicates_absorbed=(
                "semantic_duplicates_absorbed",
                "mean",
            ),
        )
    )

    candidate_pool_summary[
        "semantic_redundancy_ratio"
    ] = (
        candidate_pool_summary[
            "mean_semantic_duplicates_absorbed"
        ]
        /
        candidate_pool_summary[
            "mean_quality_valid_candidates"
        ]
    )

    operational_summary = (
        main_results
        .groupby(
            ["provider", "method"],
            as_index=False,
        )
        .agg(
            scheduled_tasks=(
                "task_id",
                "count",
            ),
            successful_tasks=(
                "success",
                lambda s: (
                    s.astype(str)
                    .str.lower()
                    .eq("true")
                    .sum()
                ),
            ),
            mean_llm_calls=(
                "llm_calls",
                "mean",
            ),
            mean_input_tokens=(
                "input_tokens",
                "mean",
            ),
            mean_output_tokens=(
                "output_tokens",
                "mean",
            ),
            mean_total_tokens=(
                "total_tokens",
                "mean",
            ),
            mean_llm_latency_sec=(
                "llm_latency_sec",
                "mean",
            ),
            mean_pipeline_latency_sec=(
                "pipeline_latency_sec",
                "mean",
            ),
        )
    )

    operational_summary[
        "success_rate"
    ] = (
        operational_summary[
            "successful_tasks"
        ]
        /
        operational_summary[
            "scheduled_tasks"
        ]
    )

    failures = main_results[
        main_results["success"]
        .astype(str)
        .str.lower()
        != "true"
    ].copy()

    failure_summary = (
        failures
        .groupby(
            [
                "provider",
                "method",
                "failure_type",
                "failure_stage",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            count=("task_id", "count")
        )
    )

    candidate_pool_summary.to_csv(
        MAIN_OUTPUT_DIR
        / "experiment2_candidate_pool_summary.csv",
        index=False,
    )

    operational_summary.to_csv(
        MAIN_OUTPUT_DIR
        / "experiment2_operational_summary.csv",
        index=False,
    )

    failure_summary.to_csv(
        MAIN_OUTPUT_DIR
        / "experiment2_failure_summary.csv",
        index=False,
    )

    print("Candidate pool")
    display(candidate_pool_summary)

    print("Operational")
    display(operational_summary)

    print("Failures")
    display(failure_summary)

else:
    print("Summaries run in full mode.")

# %% [cell 27]
SOURCE_DIR = Path("/content/KAZ_Div")

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"{SOURCE_DIR} does not exist."
    )

ZIP_PATH = shutil.make_archive(
    "/content/KAZ_Div",
    "zip",
    root_dir="/content",
    base_dir="KAZ_Div",
)

print("ZIP created:", ZIP_PATH)

try:
    from google.colab import files
    files.download(ZIP_PATH)
except Exception:
    pass
