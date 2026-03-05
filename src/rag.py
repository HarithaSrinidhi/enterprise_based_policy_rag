# import re
# import os
# from src.retriever import retrieve
# from src.llm import get_llm
# from src.config import RELEVANCE_THRESHOLD
# from src.logger import get_logger
# from langchain_core.prompts import ChatPromptTemplate

# logger = get_logger(__name__)


# # -----------------------------
# # Sentence Filtering
# # -----------------------------
# def filter_sentences(context):
#     sentences = re.split(r'(?<=[.!?])\s+', context)
#     filtered = [s.strip() for s in sentences if len(s.strip()) >= 15]
#     removed = len(sentences) - len(filtered)
#     if removed:
#         logger.debug(f"Sentence filter removed {removed} short fragments.")
#     return "\n".join(filtered)


# # -----------------------------
# # LLM Answer Extraction
# # -----------------------------
# def extract_answer(context, question):

#     llm = get_llm()

#     prompt = ChatPromptTemplate.from_template("""
# You are a precise assistant. Answer the question using ONLY the information provided in the Context below.
# Be concise and direct. Do not add any information that is not present in the Context.
# Return the FULL sentence that explains the rule or restriction.
# Never return single words like "Yes." or "No."
# Always include the complete policy statement.

# If the answer cannot be found in the Context, respond with exactly:
# REFUSED: No policy found

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)

#     chain = prompt | llm

#     logger.debug("Sending context + question to LLM...")

#     try:
#         response = chain.invoke({"context": context, "question": question})
#         logger.debug("LLM response received.")
#         return response.content.strip()
#     except Exception as e:
#         logger.error(f"LLM inference failed: {e}")
#         return "REFUSED: No policy found"


# # -----------------------------
# # Extract Source Citations
# # -----------------------------
# def extract_sources(metas):
#     sources = []
#     for meta in metas:
#         if meta is None:
#             continue
#         source = meta.get("source", "Unknown Document")
#         source = os.path.basename(source).replace("-", " ").replace(".pdf", "").title()
#         page = meta.get("page")
#         if page is not None:
#             sources.append(f"{source} (Page {page + 1})")
#         else:
#             sources.append(source)
#     return list(set(sources))


# # -----------------------------
# # RAG Pipeline
# # -----------------------------
# def rag_answer(question):

#     logger.info(f"Question received: \"{question}\"")

#     results = retrieve(question)

#     docs = results["documents"][0]
#     metas = results["metadatas"][0]
#     scores = results["distances"][0]

#     if not docs:
#         logger.warning("No documents returned from retrieval.")
#         return {"answer": "REFUSED: No policy found", "sources": [], "confidence": 0, "distance": None}

#     # Rerank across all retrieved chunks
#     question_words = set(question.lower().split())
#     scored = []
#     for i, (doc, score) in enumerate(zip(docs, scores)):
#         doc_words = set(doc.lower().split())
#         overlap = len(question_words.intersection(doc_words))
#         scored.append((score, -overlap, i, doc, metas[i]))

#     scored.sort(key=lambda x: (x[0], x[1]))
#     best_score = scored[0][0]

#     logger.info(f"Reranking complete | Best distance: {round(best_score, 3)} | Threshold: {RELEVANCE_THRESHOLD}")

#     if best_score > RELEVANCE_THRESHOLD:
#         logger.warning(f"Best distance {round(best_score, 3)} exceeds threshold {RELEVANCE_THRESHOLD} — refusing answer.")
#         return {"answer": "REFUSED: No policy found", "sources": [], "confidence": 0, "distance": round(best_score, 3)}

#     # Build context from top 3 reranked chunks
#     top_chunks = scored[:3]
#     top_docs = [entry[3] for entry in top_chunks]
#     top_metas = [entry[4] for entry in top_chunks]

#     context = "\n\n".join(top_docs)
#     context = filter_sentences(context)

#     answer = extract_answer(context, question)

#     if not answer or answer == "REFUSED: No policy found":
#         logger.warning("LLM returned no valid answer.")
#         return {"answer": "REFUSED: No policy found", "sources": [], "confidence": 0, "distance": round(best_score, 3)}

#     confidence = round(1 - best_score, 2)
#     sources = extract_sources(top_metas)

#     logger.info(f"Answer generated | Confidence: {confidence} | Sources: {sources}")

#     return {
#         "answer": answer,
#         "sources": sources,
#         "confidence": confidence,
#         "distance": round(best_score, 3)
#     }


import re
import os
from src.retriever import retrieve
from src.llm import get_llm
from src.config import RELEVANCE_THRESHOLD
from src.logger import get_logger
from langchain_core.prompts import ChatPromptTemplate

logger = get_logger(__name__)


# -----------------------------
# Sentence Filtering
# -----------------------------
def filter_sentences(context):
    sentences = re.split(r'(?<=[.!?])\s+', context)
    filtered = [s.strip() for s in sentences if len(s.strip()) >= 15]
    removed = len(sentences) - len(filtered)
    if removed:
        logger.debug(f"Sentence filter removed {removed} short fragments.")
    return "\n".join(filtered)


# -----------------------------
# LLM Answer Extraction
# -----------------------------
def extract_answer(context, question):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a policy assistant. Answer the question using ONLY the context provided below.

Rules:
- Answer using only information present in the Context.
- Be concise and direct.
- If the context contains a partial answer, still provide it.
- Only respond with REFUSED: No policy found if the Context contains absolutely no relevant information.

Context:
{context}

Question:
{question}

Answer:
""")

    chain = prompt | llm

    logger.debug("Sending context + question to LLM...")

    try:
        response = chain.invoke({"context": context, "question": question})
        logger.debug("LLM response received.")
        return response.content.strip()
    except Exception as e:
        logger.error(f"LLM inference failed: {e}")
        return "REFUSED: No policy found"


# -----------------------------
# Extract Source Citations
# -----------------------------
def extract_sources(metas):
    sources = []
    for meta in metas:
        if meta is None:
            continue
        source = meta.get("source", "Unknown Document")
        source = os.path.basename(source).replace("-", " ").replace(".pdf", "").title()
        page = meta.get("page")
        if page is not None:
            sources.append(f"{source} (Page {page + 1})")
        else:
            sources.append(source)
    return list(set(sources))


# -----------------------------
# RAG Pipeline
# -----------------------------
def rag_answer(question):

    logger.info(f"Question received: \"{question}\"")

    results = retrieve(question)

    docs   = results["documents"][0]
    metas  = results["metadatas"][0]
    scores = results["distances"][0]

    if not docs:
        logger.warning("No documents returned from retrieval.")
        return {"answer": "REFUSED: No policy found", "sources": [], "confidence": 0, "distance": None}

    # -----------------------------
    # Rerank across ALL retrieved chunks
    # -----------------------------
    question_words = set(question.lower().split())
    scored = []
    for i, (doc, score) in enumerate(zip(docs, scores)):
        doc_words = set(doc.lower().split())
        overlap   = len(question_words.intersection(doc_words))
        scored.append((score, -overlap, i, doc, metas[i]))

    scored.sort(key=lambda x: (x[0], x[1]))
    best_score = scored[0][0]

    # -----------------------------
    # LOG ALL CHUNKS WITH SCORES
    # -----------------------------
    logger.info("=" * 70)
    logger.info(f"ALL RETRIEVED CHUNKS — ranked by distance + keyword overlap")
    logger.info("=" * 70)

    for rank, entry in enumerate(scored):
        chunk_distance   = entry[0]
        chunk_overlap    = -entry[1]
        chunk_index      = entry[2]
        chunk_text       = entry[3]
        chunk_meta       = entry[4]
        chunk_confidence = round(1 - chunk_distance, 2)
        chunk_source     = os.path.basename(chunk_meta.get("source", "Unknown")) if chunk_meta else "Unknown"
        chunk_page       = (chunk_meta.get("page", 0) or 0) + 1 if chunk_meta else "?"
        label            = ">>> CHOSEN" if rank < 3 else "    skipped"

        logger.info(
            f"  Rank {rank+1:02d} | {label} | "
            f"Distance: {round(chunk_distance, 3):5} | "
            f"Confidence: {chunk_confidence} | "
            f"Overlap: {chunk_overlap} words | "
            f"{chunk_source} — Page {chunk_page}"
        )
        # Preview first 120 chars of chunk text
        preview = chunk_text[:120].strip().replace("\n", " ")
        logger.info(f"           Preview: \"{preview}...\"")

    logger.info("=" * 70)
    logger.info(f"Reranking complete | Best distance: {round(best_score, 3)} | Threshold: {RELEVANCE_THRESHOLD}")

    if best_score > RELEVANCE_THRESHOLD:
        logger.warning(f"Best distance {round(best_score, 3)} exceeds threshold {RELEVANCE_THRESHOLD} — refusing answer.")
        return {"answer": "REFUSED: No policy found", "sources": [], "confidence": 0, "distance": round(best_score, 3)}

    # -----------------------------
    # Build context from top 3 reranked chunks
    # -----------------------------
    top_chunks = scored[:3]
    top_docs   = [entry[3] for entry in top_chunks]
    top_metas  = [entry[4] for entry in top_chunks]

    # -----------------------------
    # LOG EXACT CONTEXT SENT TO LLM
    # -----------------------------
    logger.info("-" * 70)
    logger.info("CONTEXT BEING SENT TO LLM (top 3 chunks after reranking):")
    logger.info("-" * 70)
    for i, (doc, meta) in enumerate(zip(top_docs, top_metas)):
        src  = os.path.basename(meta.get("source", "Unknown")) if meta else "Unknown"
        page = (meta.get("page", 0) or 0) + 1 if meta else "?"
        logger.info(f"  [Chunk {i+1}] Source: {src} — Page {page}")
        logger.info(f"  Content: {doc.strip().replace(chr(10), ' ')}")
        logger.info(f"  {'-'*66}")
    logger.info("-" * 70)

    context = "\n\n".join(top_docs)
    context  = filter_sentences(context)

    answer = extract_answer(context, question)

    if not answer or answer == "REFUSED: No policy found":
        # Fallback: if LLM refused but distance is strong, serve best raw chunk
        if best_score < 0.55:
            fallback_answer = scored[0][3][:500]
            logger.warning(f"LLM refused but distance {round(best_score,3)} is strong — using raw chunk as fallback.")
            confidence = round(1 - best_score, 2)
            sources    = extract_sources([scored[0][4]])
            return {"answer": fallback_answer, "sources": sources, "confidence": confidence, "distance": round(best_score, 3)}
        logger.warning("LLM returned no valid answer and distance too weak for fallback.")
        return {"answer": "REFUSED: No policy found", "sources": [], "confidence": 0, "distance": round(best_score, 3)}

    confidence = round(1 - best_score, 2)
    sources    = extract_sources(top_metas)

    logger.info(f"Answer generated | Confidence: {confidence} | Sources: {sources}")

    return {
        "answer":     answer,
        "sources":    sources,
        "confidence": confidence,
        "distance":   round(best_score, 3)
    }