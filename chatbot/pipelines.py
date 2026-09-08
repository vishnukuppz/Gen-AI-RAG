import os
from pathlib import Path
from typing import List, Tuple, Generator, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .providers import BaseEmbeddingProvider, BaseLLMProvider


# =============================================================================
# Pipeline 1: Storage Pipeline (TextLoader -> Splitter -> Embeddings -> FAISS)
# =============================================================================
class DataStoragePipeline:
    """
    Ingestion pipeline that follows:
    TextLoader -> Splitter -> Embeddings generator -> Store data to vector DB.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: Path | str) -> List[Document]:
        """Step 1: Load text document using TextLoader."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Step 2: Split loaded document into smaller chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_documents(documents)

    def create_and_save_vector_store(
        self,
        chunks: List[Document],
        embedding_provider: BaseEmbeddingProvider,
        save_path: Optional[Path] = None,
    ) -> FAISS:
        """Step 3 & 4: Generate embeddings and persist to FAISS vector database."""
        embeddings = embedding_provider.get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)

        target_path = save_path or embedding_provider.get_index_path()
        target_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(target_path))
        return vector_store

    def run(
        self,
        file_path: Path | str,
        embedding_provider: BaseEmbeddingProvider,
        save_path: Optional[Path] = None,
    ) -> Tuple[FAISS, int]:
        """
        Execute complete storage pipeline:
        TextLoader -> Splitter -> Embeddings generator -> Vector DB
        Returns the created FAISS vector store and number of chunks created.
        """
        # Step 1: TextLoader
        documents = self.load_document(file_path)

        # Step 2: Splitter
        chunks = self.split_documents(documents)

        # Step 3 & 4: Embeddings generator & Store data to vector DB
        vector_store = self.create_and_save_vector_store(
            chunks=chunks,
            embedding_provider=embedding_provider,
            save_path=save_path,
        )

        return vector_store, len(chunks)


# =============================================================================
# Pipeline 2: Retrieval Pipeline (Retrieval -> Context + Prompt -> LLM)
# =============================================================================
DEFAULT_RAG_PROMPT = """You are a helpful HR assistant.
Use the following context to answer the question.

Context:
{context}

Question: {question}

Response should answer accurately and specifically based on the context provided."""


class DataRetrievalPipeline:
    """
    Retrieval and QA pipeline that follows:
    Data retrieval from vector DB -> Prompt and context is given to LLM using retrieved vector data.
    """

    def __init__(self, prompt_template: str = DEFAULT_RAG_PROMPT):
        self.prompt_template = prompt_template

    def load_vector_db(
        self,
        embedding_provider: BaseEmbeddingProvider,
        db_path: Optional[Path] = None,
    ) -> Optional[FAISS]:
        """Load vector DB from local storage using matching embedding provider."""
        index_dir = db_path or embedding_provider.get_index_path()
        faiss_file = index_dir / "index.faiss"

        if not index_dir.exists() or not faiss_file.exists():
            return None

        embeddings = embedding_provider.get_embeddings()
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve_context(
        self,
        vector_store: FAISS,
        question: str,
        k: int = 3,
    ) -> Tuple[List[Document], str]:
        """Step 1: Retrieve relevant chunks from vector store and format context."""
        retrieved_docs = vector_store.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return retrieved_docs, context

    def generate_answer(
        self,
        question: str,
        context: str,
        llm_provider: BaseLLMProvider,
    ) -> str:
        """Step 2: Supply context & prompt to the LLM and return generated text."""
        import re

        llm = llm_provider.get_llm()
        prompt = PromptTemplate.from_template(self.prompt_template)
        chain = prompt | llm | StrOutputParser()
        raw_output = chain.invoke({"context": context, "question": question})
        cleaned = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
        return cleaned.strip()

    def stream_answer(
        self,
        question: str,
        context: str,
        llm_provider: BaseLLMProvider,
    ) -> Generator[str, None, None]:
        """Stream the generated answer chunk by chunk."""
        # For OpenAI and streaming models, stream chunk by chunk
        if llm_provider.name == "OpenAI":
            llm = llm_provider.get_llm()
            prompt = PromptTemplate.from_template(self.prompt_template)
            chain = prompt | llm | StrOutputParser()
            for chunk in chain.stream({"context": context, "question": question}):
                yield chunk
        else:
            # For local models producing internal thinking tokens, clean before yielding
            answer = self.generate_answer(question, context, llm_provider)
            # Yield in smaller increments to simulate streaming if needed
            for word in answer.split(" "):
                yield word + " "

