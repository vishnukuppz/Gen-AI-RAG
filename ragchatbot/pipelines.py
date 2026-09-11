import json
import logging
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

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline 1: Storage Pipeline (Loader -> Atomic Splitter -> Embeddings -> FAISS)
# =============================================================================
class DataStoragePipeline:
    """
    Ingestion pipeline that supports both structured schema.json datasets
    and generic text documents with semantic meaning preservation:
    Structured/Text Loader -> Semantic/Adaptive Splitter -> Batch Embeddings -> FAISS DB.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100, batch_size: int = 256):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

    def load_schema_json(self, file_path: Path | str) -> List[Document]:
        """
        Step 1a: Structured JSON loader for schema.json.
        Traverses topics and sub-schemes, constructing atomic semantic Documents
        that bind topic, scheme title, overview description, and portal link together.
        """
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents: List[Document] = []
        for topic_item in data:
            topic = topic_item.get("topic", "").strip()
            category_url = topic_item.get("url", "").strip()
            sub_schemes = topic_item.get("sub_schemes", [])

            if not sub_schemes:
                continue

            for item in sub_schemes:
                title = item.get("title", "").strip()
                link = item.get("link", "").strip()
                description = item.get("description", "").strip()
                index = item.get("index", len(documents) + 1)

                if not title or not description:
                    continue

                # Context-enriched semantic text content
                page_content = (
                    f"Government Scheme: {title}\n"
                    f"Category: {topic}\n"
                    f"Overview: {description}\n"
                    f"Official Portal: {link}"
                )

                metadata = {
                    "topic": topic,
                    "title": title,
                    "link": link,
                    "category_url": category_url,
                    "index": index,
                    "source": str(file_path.name),
                }

                documents.append(Document(page_content=page_content, metadata=metadata))

        logger.info(f"Loaded {len(documents)} atomic scheme documents from {file_path.name}")
        return documents

    def load_document(self, file_path: Path | str) -> List[Document]:
        """
        Step 1: Intelligent document loader.
        Dispatches .json files to load_schema_json and text files to TextLoader.
        """
        path_obj = Path(file_path)
        if path_obj.suffix.lower() == ".json":
            return self.load_schema_json(path_obj)

        loader = TextLoader(str(path_obj), encoding="utf-8")
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Step 2: Adaptive Length Guard Splitter.
        - For atomic scheme documents under chunk_size, keeps them 100% intact.
        - For documents exceeding chunk_size, splits while prepending context headers.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        final_chunks: List[Document] = []
        for doc in documents:
            # If document is already an atomic semantic unit within size limit, preserve it directly
            if len(doc.page_content) <= self.chunk_size:
                final_chunks.append(doc)
            else:
                # Split large documents with context header prefixing
                sub_chunks = splitter.split_documents([doc])
                topic = doc.metadata.get("topic")
                title = doc.metadata.get("title")

                for chunk in sub_chunks:
                    if topic or title:
                        header = f"[Category: {topic or 'General'} | Scheme: {title or 'Overview'}]\n"
                        chunk.page_content = header + chunk.page_content
                    final_chunks.append(chunk)

        logger.info(f"Generated {len(final_chunks)} chunks from {len(documents)} input documents")
        return final_chunks

    def create_and_save_vector_store(
        self,
        chunks: List[Document],
        embedding_provider: BaseEmbeddingProvider,
        save_path: Optional[Path] = None,
    ) -> FAISS:
        """
        Step 3 & 4: Generate embeddings in safe batches and persist to FAISS vector database.
        """
        embeddings = embedding_provider.get_embeddings()
        total_chunks = len(chunks)
        batch_size = self.batch_size

        logger.info(
            f"Embedding {total_chunks} chunks using {embedding_provider.name} in batches of {batch_size}..."
        )

        vector_store: Optional[FAISS] = None
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            logger.info(f"  Processing embedding batch {batch_num}/{total_batches} ({len(batch)} items)...")

            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)

        target_path = save_path or embedding_provider.get_index_path()
        target_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(target_path))
        logger.info(f"FAISS vector store successfully saved to {target_path}")
        return vector_store

    def run(
        self,
        file_path: Path | str,
        embedding_provider: BaseEmbeddingProvider,
        save_path: Optional[Path] = None,
    ) -> Tuple[FAISS, int]:
        """
        Execute complete storage pipeline:
        Structured Loader -> Adaptive Splitter -> Batch Embeddings -> FAISS Vector DB
        Returns the created FAISS vector store and number of chunks created.
        """
        # Step 1: Intelligent Loader
        documents = self.load_document(file_path)

        # Step 2: Adaptive Length Guard Splitter
        chunks = self.split_documents(documents)

        # Step 3 & 4: Batch Embeddings & Store in Vector DB
        vector_store = self.create_and_save_vector_store(
            chunks=chunks,
            embedding_provider=embedding_provider,
            save_path=save_path,
        )

        return vector_store, len(chunks)


# =============================================================================
# Pipeline 2: Retrieval Pipeline (Retrieval -> Context + Prompt -> LLM)
# =============================================================================
DEFAULT_RAG_PROMPT = """You are a knowledgeable assistant specializing in Government of India schemes, welfare programs, and public services.
Use the following context to answer the user's question accurately.

Context:
{context}

Question: {question}

Instructions:
- Provide clear, well-structured answers highlighting relevant scheme names, categories, key benefits/objectives, and official portal links when available in the context.
- If multiple schemes apply, list them clearly with their respective details.
- If the context does not contain sufficient information to answer the question, state that clearly and advise the user to consult the official portal."""


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

