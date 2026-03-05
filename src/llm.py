from langchain_ollama import ChatOllama
from src.config import LLM_MODEL
from src.logger import get_logger

logger = get_logger(__name__)

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        logger.info(f"Initialising LLM: '{LLM_MODEL}' (temperature=0)")
        _llm = ChatOllama(model=LLM_MODEL, temperature=0)
        logger.info("LLM ready.")
    return _llm