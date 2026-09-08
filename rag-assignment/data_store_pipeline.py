import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(find_dotenv())

def read_document():
    file_path = Path(__file__).parent / "docs" / "keka_logs_jameer.txt"
    with open(file_path, "r") as f:
        text_content = f.read()
    return text_content

def text_splitting(content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    text = text_splitter.split_text(content)
    return text

def extract_embeddings(splitted_text):
    model_name = os.getenv("EMBEDDING_MODEL")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def create_vector_db(splitted_text, embeddings):
    vector_store = FAISS.from_texts(splitted_text, embeddings)
    return vector_store

load_dotenv()
print("Data Loading started")

#Read data from the content
content = read_document()

print("Text splitting started")
#Split the content into smaller chunks
splitted_text = text_splitting(content)

print("Embeddings started")
#Extract the embeddings from the splitted text
embeddings = extract_embeddings(splitted_text)

print("Storing data to the Vector DB")
#Creating the vector store
db = create_vector_db(splitted_text,embeddings)

print("Saving the DB to local memory")
save_path = Path(__file__).parent / "faiss_index"
db.save_local(str(save_path))

print("Data loaded and saved successfully")

