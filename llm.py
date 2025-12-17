import os
from functools import lru_cache

from langchain_perplexity import ChatPerplexity
from langchain_core.language_models.chat_models import BaseChatModel


# ============================================================
# CONFIG
# ============================================================

PPLX_MODEL = os.getenv("PPLX_MODEL", "sonar-pro")
PPLX_API_KEY = os.getenv("PPLX_API_KEY")

if not PPLX_API_KEY:
    raise RuntimeError("PPLX_API_KEY is not set in environment variables")


# ============================================================
# SINGLETON LLM (CRITICAL)
# ============================================================

@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Returns a SINGLE cached Perplexity LLM instance.

    Guarantees:
    - No repeated API instantiation
    - No hidden retries
    - Deterministic behavior
    - No LangChain warnings
    """

    return ChatPerplexity(
        api_key=PPLX_API_KEY,
        model=PPLX_MODEL,

        # --- STABILITY CONTROLS ---
        temperature=0.2,
        max_tokens=3000,
        timeout=60,

        # --- MODEL-SPECIFIC PARAMS ---
        model_kwargs={
            "top_p": 0.9
        },

        # --- IMPORTANT ---
        streaming=False,
    )
