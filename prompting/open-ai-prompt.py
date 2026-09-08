import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


chat = ChatOpenAI(
    model = "o3-mini",
    api_key = os.getenv("OPENAI_API_KEY"),  
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}"),
])

chain = prompt | chat | StrOutputParser()


response = chain.invoke({"question": "Is Lang trace good for tracking?"})

print(response)