import io, uuid, os
from typing import List, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma")
os.makedirs(PERSIST_DIR, exist_ok=True)

# one persistent client/collection for all docs
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
COLLECTION_NAME = "policies"
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# light, fast local embedding model (free)
_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def pdf_bytes_to_text(pdf_bytes: bytes) -> Tuple[str, List[int]]:
    """Extracts text and page indices from a PDF (bytes)."""
    reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
    pages, page_nums = [], []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            pages.append(t)
            page_nums.append(i + 1)
    full = "\n\n".join(pages)
    return full, page_nums

def chunk_text(txt: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Simple word-chunking with overlap."""
    words = txt.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def ensure_policy_index(policy_path: str) -> str:
    """Create or reuse embeddings for a policy file path."""
    doc_id = f"policy:{os.path.basename(policy_path)}"
    # if already indexed, skip
    if collection.count() > 0:
        existing = collection.get(ids=[doc_id], include=[])
        if existing.get("ids"):
            return doc_id

    with open(policy_path, "rb") as f:
        pdf_bytes = f.read()
    txt, _ = pdf_bytes_to_text(pdf_bytes)
    chunks = chunk_text(txt)
    if not chunks:
        return doc_id

    embs = _embedder.encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=[{"doc_id": doc_id} for _ in ids], embeddings=embs)
    # add a root id to mark presence
    collection.add(ids=[doc_id], documents=["INDEX_ROOT"], metadatas=[{"doc_id": doc_id}], embeddings=[_embedder.encode("INDEX_ROOT").tolist()])
    return doc_id

def index_temp_pdf(pdf_bytes: bytes) -> str:
    """Index an uploaded PDF in-memory (ephemeral)."""
    doc_id = f"temp:{uuid.uuid4().hex}"
    txt, _ = pdf_bytes_to_text(pdf_bytes)
    chunks = chunk_text(txt)
    if not chunks:
        return doc_id
    embs = _embedder.encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=[{"doc_id": doc_id} for _ in ids], embeddings=embs)
    # mark root
    collection.add(ids=[doc_id], documents=["INDEX_ROOT"], metadatas=[{"doc_id": doc_id}], embeddings=[_embedder.encode("INDEX_ROOT").tolist()])
    return doc_id

def retrieve(doc_id: str, question: str, k: int = 5) -> List[Tuple[str, float]]:
    """Return top-k (chunk, score) for doc_id given question."""
    q_emb = _embedder.encode(question, normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k*2,  # get a few more then doc_id filter
        where={"doc_id": {"$eq": doc_id}},
        include=["documents", "distances", "metadatas"]
    )
    docs = res["documents"][0]
    dists = res["distances"][0]
    # convert cosine distance to similarity score (1 - dist)
    scored = sorted([(docs[i], 1 - dists[i]) for i in range(len(docs))], key=lambda x: x[1], reverse=True)
    # filter out the injected INDEX_ROOT if present
    scored = [x for x in scored if x[0] != "INDEX_ROOT"][:k]
    return scored