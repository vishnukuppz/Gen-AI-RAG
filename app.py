import runpy
from pathlib import Path

# Launch the main Streamlit application in rag-assignment/rag_chat_bot.py
target_script = Path(__file__).parent / "rag-assignment" / "rag_chat_bot.py"
runpy.run_path(str(target_script), run_name="__main__")
