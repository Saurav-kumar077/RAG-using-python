
import os
import re
import json
import glob
import hashlib
import numpy as np
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Config

PDF_FOLDER = "./pdfs"
INDEX_DIR = "./rag_index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")
HASH_PATH = os.path.join(INDEX_DIR, "file_hashes.json")

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 8
KEYWORD_BOOST_WEIGHT = 0.15   # how much keyword overlap nudges the final score

os.makedirs(INDEX_DIR, exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL_NAME)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))



# PDF extraction 

def extract_pages(pdf_path: str):
    """Returns list of dicts: {text, page_number, source_file}"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "text": text,
                    "page_number": i + 1,
                    "source_file": os.path.basename(pdf_path),
                })
    return pages



# 2. Sentence-aware chunking 

def split_into_sentences(text: str):
    # simple, dependency-free sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


SECTION_HEADERS = re.compile(
    r'(?=\n?(?:FEES|SCOPE OF SERVICES|EXPENSES|TERM|RELATIONSHIP BETWEEN THE PARTIES|'
    r'EXCLUSIVITY|OWNERSHIP|CONFIDENTIALITY|GOVERNING LAW|SEVERABILITY|AMENDMENTS|'
    r'FEES AND COMPENSATION|SIGNATURES?|PARTIES|BACKGROUND|APPOINTMENT AND ROLE)[:\s])'
)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    sections = SECTION_HEADERS.split(text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            sentences = split_into_sentences(section)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= chunk_size:
                    current += (" " if current else "") + sent
                else:
                    if current:
                        chunks.append(current)
                    overlap_text = current[-overlap:] if current else ""
                    current = (overlap_text + " " + sent).strip()
            if current:
                chunks.append(current)
    return chunks


def chunk_pages(pages):
    """Turns page dicts into chunk dicts, preserving metadata."""
    all_chunks = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            all_chunks.append({
                "text": chunk,
                "page_number": page["page_number"],
                "source_file": page["source_file"],
            })
    return all_chunks



# 3. Incremental indexing helpers

def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def load_or_init_index(dim):
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatIP(dim)   # cosine sim via normalized inner product
    return index


def build_or_update_index():
    pdf_paths = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {PDF_FOLDER}")

    known_hashes = load_json(HASH_PATH, {})
    metadata = load_json(META_PATH, [])   # list aligned with FAISS vector order

    dim = embedder.get_sentence_embedding_dimension()
    index = load_or_init_index(dim)

    new_hashes = dict(known_hashes)
    changed_files = []

    for path in pdf_paths:
        h = file_hash(path)
        fname = os.path.basename(path)
        if known_hashes.get(fname) != h:
            changed_files.append(path)
            new_hashes[fname] = h

    if not changed_files:
        print("No new or changed PDFs. Using existing index.")
        return index, metadata

    print(f"Indexing {len(changed_files)} new/changed file(s): "
          f"{[os.path.basename(p) for p in changed_files]}")

    new_chunks = []
    for path in changed_files:
        pages = extract_pages(path)
        new_chunks.extend(chunk_pages(pages))

    if new_chunks:
        texts = [c["text"] for c in new_chunks]
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        index.add(np.array(embeddings, dtype="float32"))
        metadata.extend(new_chunks)

    faiss.write_index(index, INDEX_PATH)
    save_json(META_PATH, metadata)
    save_json(HASH_PATH, new_hashes)

    return index, metadata



# 4. Hybrid retrieval: 

def keyword_overlap_score(query: str, text: str) -> float:
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


def retrieve(query: str, index, metadata, top_k=TOP_K):
    query_vec = embedder.encode([query], normalize_embeddings=True)
    # over-fetch, then re-rank with keyword boost
    fetch_k = min(top_k * 3, index.ntotal)
    scores, ids = index.search(np.array(query_vec, dtype="float32"), fetch_k)

    candidates = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = metadata[idx]
        kw_score = keyword_overlap_score(query, chunk["text"])
        final_score = float(score) + KEYWORD_BOOST_WEIGHT * kw_score
        candidates.append((final_score, chunk))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in candidates[:top_k]]



# 5. Answer generation 

def build_prompt(question: str, chunks: list) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(
            f"[Source: {c['source_file']}, Page {c['page_number']}]\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    return f"""You are a precise assistant that answers questions strictly based on the provided context.

Instructions:
- Answer ONLY using information explicitly present in the context below.
- Do not add outside knowledge, assumptions, or generalizations.
- Preserve exact distinctions made in the context (e.g., "X is NOT Y").
- Treat each source document as describing one distinct entity or individual. Before comparing, merging, or attributing information across different source documents, verify that the identifying name, ID, or label matches EXACTLY across all sources — including word order. Similarly-worded names or labels (e.g., words in a different order, minor spelling variants) may refer to different entities; never assume they are the same without explicit confirmation in the text.
- If information about two different-but-similarly-named entities appears together, clearly separate your answer per entity and do not merge their facts into a single narrative.
- Cite the source file and page number for each claim you make, using the format (source_file, p.X).
- If the context does not contain enough information, respond with exactly:
  "I cannot answer this based on the provided context."

Context:
{context}

Question: {question}

Answer:"""

def ask(question: str, index, metadata):
    chunks = retrieve(question, index, metadata)

    print("\n--- Retrieved Chunks ---")
    for i, c in enumerate(chunks):
        print(f"[{i+1}] {c['source_file']} p.{c['page_number']} :: {c['text'][:150]}...")

    if not chunks:
        print("No relevant context found.")
        return

    prompt = build_prompt(question, chunks)

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content
    print("\nAnswer:", answer)
    return answer



# Entry point

if __name__ == "__main__":
    index, metadata = build_or_update_index()
    print(f"\nIndex ready — {index.ntotal} chunks total.\n")

    question = "Where does Anaya Meera reside?"
    ask(question, index, metadata) 