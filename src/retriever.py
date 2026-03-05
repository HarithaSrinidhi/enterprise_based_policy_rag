import chromadb
from langchain_ollama import OllamaEmbeddings
from src.config import EMBEDDING_MODEL, RETRIEVAL_K
from src.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "enterprise_rag"

client = chromadb.PersistentClient(path="vector_db")
embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)


def retrieve(query):

    logger.debug(f"Embedding query: '{query}'")

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Could not load collection '{COLLECTION_NAME}': {e}")
        raise

    try:
        query_embedding = embedding_model.embed_query(query)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RETRIEVAL_K,
        include=["documents", "metadatas", "distances"]
    )

    distances = results["distances"][0]
    logger.debug(f"Retrieved {len(distances)} chunks | Best distance: {round(min(distances), 3)} | Worst: {round(max(distances), 3)}")

    return results