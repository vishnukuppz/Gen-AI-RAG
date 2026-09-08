import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from pathlib import Path
load_dotenv(find_dotenv())


def load_generative_model():
    llm = ChatOpenAI(model_name=os.getenv("OPENAI_GENERATIVE_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
    return llm

def load_embedding():
    embedding = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
    return embedding

def load_vector_db(embedding):
    db_path = Path(__file__).parent / "faiss_index_openai"
    vector_store = FAISS.load_local(
        str(db_path),
        embedding,
        allow_dangerous_deserialization=True,
    )
    return vector_store


embeddings = load_embedding()

db = load_vector_db(embeddings)

question = "25 May 2026 related logs"


retrieved_docs = db.similarity_search(question, k=10)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])

print(f"context. {context}")
genLlm = load_generative_model()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Act as log analyser expert"),
    ("user", "Context: {context}\nQuestion: {question}\nAnswer should be exactly as per question asked"),
])

chain = prompt | genLlm | StrOutputParser()

print(chain.invoke({"context": context, "question": question}))