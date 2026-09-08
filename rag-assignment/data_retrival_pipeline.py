import os
from pathlib import Path
import torch
from dotenv import load_dotenv, find_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (.env)
load_dotenv(find_dotenv())


def load_llm(model_id):
    """Load local LLM onto Apple Silicon GPU (MPS) or CPU."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
    ).to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1000,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return HuggingFacePipeline(pipeline=pipe)


def load_vector_db():
    """Load the FAISS vector database from local storage."""
    model_name = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db_path = Path(__file__).parent / "faiss_index"
    vector_store = FAISS.load_local(
        str(db_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def prompt_execution(question, retrieved_docs, llm):
    """Format prompt with retrieved context and stream model answer."""
    # Combine retrieved document chunks
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Define RTCFR Prompt Template
    prompt = PromptTemplate.from_template(
        "Act as a helpful HR assistant.\n"
        "Use the following context to answer the question.\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Response should answer exactly as per question asked"
    )

    # Build LCEL chain
    chain = prompt | llm | StrOutputParser()

    # Stream output
    for chunk in chain.stream({"question": question, "context": context}):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    # 1. Load Vector DB
    print("Loading Vector DB...")
    db = load_vector_db()

    # 2. Load LLM
    print("Loading LLM...")
    hf_llm = load_llm(os.getenv("HF_GENERATIVE_MODEL"))

    # 3. Sample Question
    question = "What is the sick and casual leave allocation for a year?"
    print(f"\nQuestion: {question}\n")

    # 4. Similarity Search in FAISS
    print("Retrieving relevant documents...")
    retrieved_docs = db.similarity_search(question, k=3)

    # 5. Generate and Stream Answer
    print("\n--- Response ---\n")
    prompt_execution(question, retrieved_docs, hf_llm)
