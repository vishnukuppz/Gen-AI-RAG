from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain_community.document_loaders import TextLoader

from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables (.env)
load_dotenv(find_dotenv())

# Define document path
def read_document():
    file_path = Path(__file__).parent / "docs" / "keka_logs_jameer.txt"
    return TextLoader(file_path).load()
        
# Define embedding model
def get_embedding_model():
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))

# Define text splitter
def split_text(document):
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)  
    return splitter.split_documents(document)


print("Loading document")
document = read_document()
print("Splitting text")
chunks = split_text(document)
print("Loading embedding model")
embeddings = get_embedding_model()
print("Creating vector store")
vector_store = FAISS.from_documents(chunks, embeddings)
save_path = Path(__file__).parent / "faiss_index_openai"
vector_store.save_local(str(save_path))

print("Vector store created successfully!")