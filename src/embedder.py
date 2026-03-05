import chromadb
from langchain_ollama import OllamaEmbeddings
from src.config import EMBEDDING_MODEL
from src.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "enterprise_rag"
BATCH_SIZE = 100

client = chromadb.PersistentClient(path="vector_db")
embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)


def reset_collection():

    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("Old collection deleted.")
    except Exception:
        logger.info("No previous collection found — creating fresh.")

    return client.get_or_create_collection(name=COLLECTION_NAME)


def create_embeddings(chunks):

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    logger.info(f"Starting batch embedding for {len(texts)} chunks using model '{EMBEDDING_MODEL}'...")

    try:
        embeddings = embedding_model.embed_documents(texts)
        logger.info("Batch embedding complete.")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise

    # Insert into Chroma in batches
    total_inserted = 0
    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))

        try:
            collection.add(
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
                ids=ids[start:end]
            )
            total_inserted += (end - start)
            logger.info(f"Inserted chunks {start+1}–{end} into vector DB.")
        except Exception as e:
            logger.error(f"Failed to insert batch {start}–{end}: {e}")
            raise

    logger.info(f"All {total_inserted} chunks inserted into ChromaDB successfully.")