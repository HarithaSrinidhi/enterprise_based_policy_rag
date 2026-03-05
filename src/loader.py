import os
from langchain_community.document_loaders import PyPDFLoader
from src.config import DATA_PATH
from src.logger import get_logger

logger = get_logger(__name__)


def load_documents():

    logger.info(f"Scanning '{DATA_PATH}' for PDF files...")
    documents = []
    skipped = 0

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file)

            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                documents.extend(pages)
                logger.info(f"Loaded: {file} ({len(pages)} pages)")

            except Exception as e:
                skipped += 1
                logger.warning(f"Skipping '{file}' — could not read: {e}")

    logger.info(f"Load complete. Total pages: {len(documents)} | Skipped files: {skipped}")
    return documents