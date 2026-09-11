"""
RAG Chatbot core backend engine.
Provides a unified interface using swappable embedding and LLM providers.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from ragchatbot.providers import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    get_embedding_provider,
    get_llm_provider,
)
from ragchatbot.pipelines import (
    DataStoragePipeline,
    DataRetrievalPipeline,
)

BASE_DIR = Path(__file__).parent


class RAGChatbotEngine:
    """
    High-level RAG orchestrator tying together swappable providers and the two pipelines:
      1. Storage Pipeline (TextLoader -> Splitter -> Embeddings -> Vector DB)
      2. Retrieval Pipeline (Vector DB -> Retrieval -> Prompt + Context -> LLM)
    """

    def __init__(
        self,
        embedding_provider_name: str = "OpenAI",
        llm_provider_name: str = "OpenAI",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.embedding_provider: BaseEmbeddingProvider = get_embedding_provider(embedding_provider_name)
        self.llm_provider: BaseLLMProvider = get_llm_provider(llm_provider_name)
        self.storage_pipeline = DataStoragePipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retrieval_pipeline = DataRetrievalPipeline()
        self.vector_store = self.retrieval_pipeline.load_vector_db(self.embedding_provider)

    def set_providers(
        self,
        embedding_provider_name: Optional[str] = None,
        llm_provider_name: Optional[str] = None,
    ):
        """Hot-swap embedding or LLM providers at runtime."""
        if embedding_provider_name and embedding_provider_name != self.embedding_provider.name:
            self.embedding_provider = get_embedding_provider(embedding_provider_name)
            self.vector_store = self.retrieval_pipeline.load_vector_db(self.embedding_provider)

        if llm_provider_name and llm_provider_name != self.llm_provider.name:
            self.llm_provider = get_llm_provider(llm_provider_name)

    def ingest_document(self, file_path: Path | str) -> int:
        """Run storage pipeline: TextLoader -> Splitter -> Embeddings -> FAISS."""
        self.vector_store, chunk_count = self.storage_pipeline.run(
            file_path=file_path,
            embedding_provider=self.embedding_provider,
        )
        return chunk_count

    def ask(self, question: str, k: int = 3) -> Tuple[str, List[Document]]:
        """Run retrieval pipeline: Retrieve from Vector DB -> Prompt + Context -> LLM."""
        if self.vector_store is None:
            self.vector_store = self.retrieval_pipeline.load_vector_db(self.embedding_provider)
            if self.vector_store is None:
                raise ValueError(
                    f"No vector database found for provider '{self.embedding_provider.name}'. "
                    f"Please run the storage pipeline first."
                )

        retrieved_docs, context = self.retrieval_pipeline.retrieve_context(
            vector_store=self.vector_store,
            question=question,
            k=k,
        )
        answer = self.retrieval_pipeline.generate_answer(
            question=question,
            context=context,
            llm_provider=self.llm_provider,
        )
        return answer, retrieved_docs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline Runner")
    parser.add_argument("--provider", choices=["OpenAI", "HuggingFace"], default="OpenAI")
    parser.add_argument("--doc", type=str, default=str(BASE_DIR / "docs" / "schema.txt"))
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip re-ingestion if index already exists")
    args = parser.parse_args()

    engine = RAGChatbotEngine(
        embedding_provider_name=args.provider,
        llm_provider_name=args.provider,
    )

    if not args.skip_ingest or engine.vector_store is None:
        print(f"\n[Pipeline 1: Storage] Ingesting {args.doc} with {args.provider}...")
        chunks = engine.ingest_document(args.doc)
        print(f"Stored {chunks} chunks into {engine.embedding_provider.get_index_path()}")

    print(f"\n[Pipeline 2: Retrieval] Querying: '{args.query}' with {args.provider}...")
    answer, docs = engine.ask(args.query)
    print(f"\nRetrieved {len(docs)} context chunk(s).")
    print("\nAnswer:\n", answer)
