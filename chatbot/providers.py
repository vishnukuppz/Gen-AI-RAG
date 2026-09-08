from abc import ABC, abstractmethod
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import torch
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).parent


# =============================================================================
# Abstract Provider Interfaces
# =============================================================================
class BaseEmbeddingProvider(ABC):
    """Abstract base class for swappable embedding providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Return the LangChain Embeddings instance."""
        pass

    @abstractmethod
    def get_index_path(self) -> Path:
        """Return directory path where vector index is saved."""
        pass


class BaseLLMProvider(ABC):
    """Abstract base class for swappable LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    def get_llm(self) -> BaseLanguageModel:
        """Return the LangChain Language Model instance."""
        pass


# =============================================================================
# OpenAI Implementations
# =============================================================================
class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI Embedding implementation."""

    def __init__(self, model: str = None, api_key: str = None):
        self._model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._embeddings = None

    @property
    def name(self) -> str:
        return "OpenAI"

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(
                model=self._model,
                api_key=self._api_key,
            )
        return self._embeddings

    def get_index_path(self) -> Path:
        return BASE_DIR / "faiss_index_openai"


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI Generative LLM implementation."""

    def __init__(self, model_name: str = None, api_key: str = None, temperature: float = 0.0):
        self._model_name = model_name or os.getenv("OPENAI_GENERATIVE_MODEL", "gpt-4o-mini")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._temperature = temperature
        self._llm = None

    @property
    def name(self) -> str:
        return "OpenAI"

    def get_llm(self) -> BaseLanguageModel:
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model_name=self._model_name,
                api_key=self._api_key,
                temperature=self._temperature,
            )
        return self._llm


# =============================================================================
# Hugging Face Implementations
# =============================================================================
class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """Hugging Face Embedding implementation."""

    def __init__(self, model_name: str = None):
        self._model_name = model_name or os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
        self._embeddings = None

    @property
    def name(self) -> str:
        return "HuggingFace"

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(model_name=self._model_name)
        return self._embeddings

    def get_index_path(self) -> Path:
        return BASE_DIR / "faiss_index_hf"


class HuggingFaceLLMProvider(BaseLLMProvider):
    """Hugging Face Generative LLM implementation (running locally on MPS/CPU)."""

    def __init__(self, model_id: str = None):
        self._model_id = model_id or os.getenv("HF_GENERATIVE_MODEL", "Qwen/Qwen3.5-2B")
        self._llm = None

    @property
    def name(self) -> str:
        return "HuggingFace"

    def get_llm(self) -> BaseLanguageModel:
        if self._llm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from langchain_huggingface import HuggingFacePipeline

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "mps" else torch.float32

            tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                dtype=torch_dtype,
            ).to(device)

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                clean_up_tokenization_spaces=False,
            )
            self._llm = HuggingFacePipeline(pipeline=pipe)
        return self._llm


# =============================================================================
# Factory Helpers
# =============================================================================
SUPPORTED_PROVIDERS = ["OpenAI", "HuggingFace"]


def get_embedding_provider(provider_type: str = "OpenAI", **kwargs) -> BaseEmbeddingProvider:
    """Factory to obtain embedding provider instance."""
    normalized = provider_type.strip().lower()
    if "openai" in normalized:
        return OpenAIEmbeddingProvider(**kwargs)
    elif "hugging" in normalized or "hf" in normalized:
        return HuggingFaceEmbeddingProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider_type}. Choose from: {SUPPORTED_PROVIDERS}")


def get_llm_provider(provider_type: str = "OpenAI", **kwargs) -> BaseLLMProvider:
    """Factory to obtain LLM provider instance."""
    normalized = provider_type.strip().lower()
    if "openai" in normalized:
        return OpenAILLMProvider(**kwargs)
    elif "hugging" in normalized or "hf" in normalized:
        return HuggingFaceLLMProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_type}. Choose from: {SUPPORTED_PROVIDERS}")
