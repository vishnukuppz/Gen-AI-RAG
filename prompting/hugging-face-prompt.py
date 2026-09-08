from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Define local model repository
model_id = os.getenv("HF_GENERATIVE_MODEL")

print(f"⏳ Loading {model_id} locally into M2 Pro GPU (MPS)...")

# 2. Load tokenizer and model locally onto Apple Silicon GPU (MPS)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="mps"
)
print("✅ Local model loaded on MPS.\n")

# 3. Create Hugging Face pipeline for local execution
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True
)

# 4. Wrap with LangChain HuggingFacePipeline (enables automatic LangSmith tracing)
hf_llm = HuggingFacePipeline(pipeline=pipe)

# 5. Build LangChain prompt & chain
prompt = PromptTemplate.from_template(
    "Write a python function to compute {topic}:\n\n"
)
chain = prompt | hf_llm | StrOutputParser()

print("🚀 Executing query and sending trace to LangSmith...\n")

# 6. Run query with streaming — tokens are printed to the terminal as they are generated
print("🤖 Model Response:\n")
for chunk in chain.stream({"topic": "fibonacci numbers"}):
    print(chunk, end="", flush=True)
print()