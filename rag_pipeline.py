import os
import json
import glob
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

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 8

os.makedirs(INDEX_DIR, exist_ok=True)

embedder = SentenceTransformer(EMBED_MODEL_NAME)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# 1. Load PDFs

def load_pdfs(folder: str):
    """Returns a list of pages: {text, page_number, source_file}"""
    pages = []
    for pdf_path in glob.glob(os.path.join(folder, "*.pdf")):
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


# 2. Chunking (recursive character text splitter)

# Try to split on paragraphs first, then lines, then sentences, then words,
# and only fall back to raw characters if nothing else works.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def merge_splits(splits, separator, chunk_size, overlap):
    """Combine small pieces into chunks of up to chunk_size, keeping some overlap."""
    chunks = []
    current = []
    total = 0   # length of separator.join(current)

    for piece in splits:
        sep_len = len(separator) if current else 0

        if current and total + sep_len + len(piece) > chunk_size:
            chunk = separator.join(current).strip()
            if chunk:
                chunks.append(chunk)

            # Drop pieces from the front until what's left fits in the overlap
            # and leaves room for the next piece.
            while current and (
                total > overlap
                or total + len(separator) + len(piece) > chunk_size
            ):
                total -= len(current[0]) + (len(separator) if len(current) > 1 else 0)
                current.pop(0)

        current.append(piece)
        total += len(piece) + (len(separator) if len(current) > 1 else 0)

    chunk = separator.join(current).strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, separators=SEPARATORS):
    # Pick the first separator that actually appears in the text
    separator = separators[-1]
    remaining_separators = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_separators = separators[i + 1:]
            break

    splits = text.split(separator) if separator else list(text)

    chunks = []
    small_pieces = []
    for piece in splits:
        if len(piece) <= chunk_size:
            small_pieces.append(piece)
        else:
            # Flush the small pieces collected so far
            if small_pieces:
                chunks.extend(merge_splits(small_pieces, separator, chunk_size, overlap))
                small_pieces = []
            # Piece is too big: split it again with the next separator
            if remaining_separators:
                chunks.extend(chunk_text(piece, chunk_size, overlap, remaining_separators))
            else:
                chunks.append(piece)

    if small_pieces:
        chunks.extend(merge_splits(small_pieces, separator, chunk_size, overlap))

    return chunks


def chunk_pages(pages):
    chunks = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            chunks.append({
                "text": chunk,
                "page_number": page["page_number"],
                "source_file": page["source_file"],
            })
    return chunks


# 3. Build or load the index

def build_or_load_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("Loading existing index...")
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
        return index, metadata

    print("Building new index...")
    pages = load_pdfs(PDF_FOLDER)
    if not pages:
        raise FileNotFoundError(f"No PDFs found in {PDF_FOLDER}")

    metadata = chunk_pages(pages)
    texts = [c["text"] for c in metadata]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])   # cosine similarity via normalized vectors
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    return index, metadata


# 4. Retrieval

def retrieve(query: str, index, metadata, top_k=TOP_K):
    query_vec = embedder.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(query_vec, dtype="float32"), top_k)
    return [metadata[i] for i in ids[0] if i != -1]


# 5. Answer generation

def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n".join(
        f"[Source: {c['source_file']}, Page {c['page_number']}]\n{c['text']}"
        for c in chunks
    )

    return f"""You are a helpful assistant that answers questions using only the provided context.

Instructions:
- Answer using only the information in the context below.
- Cite the source file and page number for each claim, using the format (source_file, p.X).
- If the context does not contain the answer, respond with:
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

    prompt = build_prompt(question, chunks)

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content
    print("\nAnswer:", answer)
    return answer


# Entry point

if __name__ == "__main__":
    index, metadata = build_or_load_index()

    print(f"\nRAG index ready — {index.ntotal} chunks loaded.")
    print("Ask questions about your documents. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        ask(question, index, metadata)