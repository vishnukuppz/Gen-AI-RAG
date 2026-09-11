import sys
from pathlib import Path
import tempfile
import streamlit as st

# Add project root to sys.path so modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ragchatbot.providers import (
    SUPPORTED_PROVIDERS,
    get_embedding_provider,
    get_llm_provider,
)
from ragchatbot.pipelines import (
    DataStoragePipeline,
    DataRetrievalPipeline,
)

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"

# =============================================================================
# Streamlit Page Configuration
# =============================================================================
st.set_page_config(
    page_title="India Government Schemes Assistant",
    page_icon="🏛️",
    layout="wide",
)

# =============================================================================
# Session State Initialization
# =============================================================================
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Storage Pipeline"

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "embedding_provider_choice" not in st.session_state:
    st.session_state.embedding_provider_choice = "OpenAI"

if "llm_provider_choice" not in st.session_state:
    st.session_state.llm_provider_choice = "OpenAI"


# =============================================================================
# Cached Pipeline Helpers
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_cached_embedding_provider(name: str):
    return get_embedding_provider(name)


@st.cache_resource(show_spinner=False)
def get_cached_llm_provider(name: str):
    return get_llm_provider(name)


def check_index_status(provider_name: str) -> bool:
    provider = get_cached_embedding_provider(provider_name)
    index_path = provider.get_index_path()
    return (index_path / "index.faiss").exists()


# Hide sidebar completely across the entire app
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SCREEN 1: Data Storage pipeline
# =============================================================================
if st.session_state.current_screen == "Storage Pipeline":
    st.title("🏛️ Government Schemes Data Storage Pipeline")
    st.caption("Ingest and embed government schemes dataset into FAISS vector database.")

    # Swappable embedding provider
    selected_provider = st.selectbox(
        "Select Embedding Provider",
        options=SUPPORTED_PROVIDERS,
        index=SUPPORTED_PROVIDERS.index(st.session_state.embedding_provider_choice),
    )
    st.session_state.embedding_provider_choice = selected_provider

    # Quick Action: Ingest Pre-scraped schema.json
    default_schema_path = DOCS_DIR / "schema.json"
    col_def, col_info = st.columns([1, 2])
    with col_def:
        ingest_default = st.button("📥 Ingest schema.json (3,076 Schemes)", use_container_width=True)

    # Upload custom document (JSON or TXT)
    uploaded_file = st.file_uploader("Or Upload Custom Document (JSON or TXT)", type=["json", "txt"])

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100

    target_process_path = None
    file_sig = None

    if ingest_default and default_schema_path.exists():
        target_process_path = default_schema_path
        file_sig = f"default_schema_{default_schema_path.stat().st_size}_{selected_provider}"
    elif uploaded_file is not None:
        file_sig = f"{uploaded_file.name}_{uploaded_file.size}_{selected_provider}"
        temp_dir = Path(tempfile.gettempdir())
        temp_file_path = temp_dir / uploaded_file.name
        temp_file_path.write_bytes(uploaded_file.getvalue())
        target_process_path = temp_file_path

    # Automatic conversion upon upload or button trigger
    if target_process_path is not None:
        if st.session_state.get("last_processed_sig") != file_sig:
            progress_bar = st.progress(0, text="Initializing conversion...")
            try:
                storage_pipeline = DataStoragePipeline(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                emb_provider = get_cached_embedding_provider(selected_provider)

                # Step 1: Intelligent Loader
                progress_bar.progress(20, text="Loading and validating documents...")
                docs = storage_pipeline.load_document(target_process_path)

                # Step 2: Adaptive Splitter
                progress_bar.progress(40, text=f"Processing {len(docs)} atomic documents...")
                chunks = storage_pipeline.split_documents(docs)

                # Step 3 & 4: Embeddings generator & Vector DB
                progress_bar.progress(60, text=f"Generating embeddings and vector index with {selected_provider}...")
                storage_pipeline.create_and_save_vector_store(chunks, emb_provider)

                progress_bar.progress(100, text="Vector database created and saved successfully!")
                st.session_state.last_processed_sig = file_sig
                st.session_state.chunks_count = len(chunks)
                st.session_state.is_storage_ready = True
                st.success(f"Successfully converted and stored {len(chunks)} scheme chunks into the vector database!")
            except Exception as e:
                progress_bar.empty()
                st.error(f"Error during conversion to vector DB: {e}")
                st.session_state.is_storage_ready = False
        else:
            st.success(f"Document ready: {st.session_state.get('chunks_count', '')} chunks stored in vector database.")
    else:
        st.session_state.is_storage_ready = False

    st.write("")

    # Validation: Enabled only if data is available in the vector DB, otherwise disabled
    is_db_available = check_index_status(selected_provider)

    if st.button("Go to chatbot", type="primary", disabled=not is_db_available):
        st.session_state.current_screen = "Retrieval Pipeline"
        st.rerun()


# =============================================================================
# SCREEN 2: Chatbot
# =============================================================================
elif st.session_state.current_screen == "Retrieval Pipeline":
    st.title("🏛️ India Government Schemes Assistant")
    st.caption("Ask questions about government schemes, financial assistance, and eligibility criteria.")

    # Action buttons: Back to data storage and Clear chat history
    col_back, col_clear = st.columns([1, 1])
    with col_back:
        if st.button("⬅️ Back to data storage", use_container_width=True):
            st.session_state.current_screen = "Storage Pipeline"
            st.rerun()

    with col_clear:
        if st.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    # Model selectors: Vector DB and LLM
    c1, c2 = st.columns(2)
    with c1:
        emb_choice = st.selectbox(
            "Select Vector DB",
            SUPPORTED_PROVIDERS,
            index=SUPPORTED_PROVIDERS.index(st.session_state.embedding_provider_choice),
            key="retrieval_emb_select",
        )
        st.session_state.embedding_provider_choice = emb_choice

    with c2:
        llm_choice = st.selectbox(
            "Select LLM",
            SUPPORTED_PROVIDERS,
            index=SUPPORTED_PROVIDERS.index(st.session_state.llm_provider_choice),
            key="retrieval_llm_select",
        )
        st.session_state.llm_provider_choice = llm_choice

    # Hardcoded top-k (no UI selection)
    TOP_K = 3

    # Load vector database
    emb_provider = get_cached_embedding_provider(emb_choice)
    retrieval_pipeline = DataRetrievalPipeline()
    vector_store = retrieval_pipeline.load_vector_db(emb_provider)

    if vector_store is None:
        st.warning(
            f"Vector DB for {emb_choice} is not initialized. Please go back to data storage and upload a document first."
        )

    # Render Chat History
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if user_query := st.chat_input("Ask about government schemes, eligibility, subsidies, or student loans..."):
        # Append and display user message
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant generation
        with st.chat_message("assistant"):
            if vector_store is None:
                err_msg = (
                    f"Vector DB for {emb_choice} is not initialized. "
                    "Please go back to data storage and upload a document."
                )
                st.error(err_msg)
                st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})
            else:
                try:
                    with st.spinner("Searching relevant schemes..."):
                        retrieved_docs, context = retrieval_pipeline.retrieve_context(
                            vector_store=vector_store,
                            question=user_query,
                            k=TOP_K,
                        )
                        llm_provider = get_cached_llm_provider(llm_choice)

                    # Stream response to UI
                    stream = retrieval_pipeline.stream_answer(
                        question=user_query,
                        context=context,
                        llm_provider=llm_provider,
                    )
                    response_text = st.write_stream(stream)

                    # Show sources expander if metadata has links
                    if retrieved_docs:
                        with st.expander("📚 View Sourced Schemes & Official Links"):
                            for i, doc in enumerate(retrieved_docs, 1):
                                title = doc.metadata.get("title", f"Scheme {i}")
                                topic = doc.metadata.get("topic", "General")
                                link = doc.metadata.get("link")
                                link_text = f"[{link}]({link})" if link else "N/A"
                                st.markdown(f"**{i}. {title}** ({topic})  \n🔗 Official Portal: {link_text}")

                    # Save to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response_text,
                    })
                except Exception as e:
                    err_msg = f"Generation error: {e}"
                    st.error(err_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})
