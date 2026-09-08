import os
import re
import time
from pathlib import Path
import torch
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (.env)
load_dotenv(find_dotenv())

# Page configuration
st.set_page_config(page_title="Chatbot", page_icon="💬")

# Paths
BASE_DIR = Path(__file__).parent
FAISS_SAVE_DIR = BASE_DIR / "faiss_index"

# -----------------------------------------------------------------------------
# Cached Models
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    model_name = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
    return HuggingFaceEmbeddings(model_name=model_name)


@st.cache_resource(show_spinner=False)
def get_llm():
    model_id = os.getenv("HF_GENERATIVE_MODEL", "Qwen/Qwen3.5-2B")
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
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


# -----------------------------------------------------------------------------
# Document Processing
# -----------------------------------------------------------------------------
def extract_text(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    elif filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n\n".join(pages)
    return ""


def process_and_store_document(raw_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)
    embeddings = get_embedding_model()
    vector_store = FAISS.from_texts(chunks, embeddings)
    FAISS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_SAVE_DIR))
    return vector_store


def load_saved_vector_store():
    if FAISS_SAVE_DIR.exists() and (FAISS_SAVE_DIR / "index.faiss").exists():
        try:
            embeddings = get_embedding_model()
            return FAISS.load_local(
                str(FAISS_SAVE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# Prompt & Generation
# -----------------------------------------------------------------------------
PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful assistant. You answer questions based on the provided Context from the user's uploaded document.

Instructions:
- If the answer IS in the Context:
Answer the question directly and accurately based only on the Context. Do NOT output any disclaimer or note.
- If the answer IS NOT in the Context (or outside the scope of the personal document):
Output:
⚠️ **Note:** This content is not available in your uploaded document (personal content).

**General Information:**
[Provide general information answering the question here]<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""

def generate_answer(question: str, context: str, llm) -> str:
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"context": context, "question": question})
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    return cleaned.strip()


# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if "screen" not in st.session_state:
    st.session_state.screen = "upload"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_saved_vector_store()


# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Navigation")
    choice = st.radio(
        "Screens",
        ["Upload File", "Chatbot"],
        index=0 if st.session_state.screen == "upload" else 1,
        label_visibility="collapsed",
    )
    if choice == "Upload File" and st.session_state.screen != "upload":
        st.session_state.screen = "upload"
        st.rerun()
    elif choice == "Chatbot" and st.session_state.screen != "chat":
        st.session_state.screen = "chat"
        st.rerun()

    if st.session_state.screen == "chat":
        st.write("")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# -----------------------------------------------------------------------------
# Screen 1: File Upload
# -----------------------------------------------------------------------------
if st.session_state.screen == "upload":
    st.title("Upload File")

    uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("Submit & Process File", type="primary"):
            with st.spinner("Processing file..."):
                raw_text = extract_text(uploaded_file)
                if not raw_text.strip():
                    st.error("The uploaded file does not contain readable text.")
                else:
                    st.session_state.vector_store = process_and_store_document(raw_text)
                    st.session_state.file_processed = True
                    st.success("File uploaded and processed successfully!")

    if st.session_state.vector_store is not None:
        st.write("")
        if st.button("Go to Chatbot ➡️", type="primary"):
            st.session_state.screen = "chat"
            st.rerun()


# -----------------------------------------------------------------------------
# Screen 2: Chatbot
# -----------------------------------------------------------------------------
elif st.session_state.screen == "chat":
    st.title("Chatbot")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = ""
                if st.session_state.vector_store is not None:
                    docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                    context = "\n\n".join([d.page_content for d in docs])
                else:
                    context = "No document uploaded."

                llm = get_llm()
                response = generate_answer(prompt, context, llm)

            # Stream message to UI
            message_placeholder = st.empty()
            streamed_text = ""
            for word in response.split(" "):
                streamed_text += word + " "
                message_placeholder.markdown(streamed_text + "▌")
                time.sleep(0.01)
            message_placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
