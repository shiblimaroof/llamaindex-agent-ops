import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
faiss.omp_set_num_threads(1)
from rank_bm25 import BM25Okapi

EMBED_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_POOL_SIZE = 20
FILE_PATH_BOOST = 0.25      # additive boost after normalization, tune later

_embed_model = None
_rerank_model = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True, device="cpu")
    return _embed_model


def _get_rerank_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder(RERANK_MODEL_NAME, device="cpu")
    return _rerank_model

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Pure-numpy row-wise L2 normalize — replaces faiss.normalize_L2,
    which segfaults on this machine once torch has loaded a model in
    the same process (FAISS/PyTorch OpenMP conflict)."""
    x = x.astype("float32")
    x = np.ascontiguousarray(x)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (x / norms).astype("float32")

def _build_faiss_index(chunks: list[dict], source_id: str, created_at: str) -> tuple[faiss.Index, np.ndarray]:
    MAX_CHARS = 6000  # conservative cap, well under jina's 8192-token limit
    CACHE_PATH = f"/tmp/embeddings_cache_{source_id}_{created_at}.npy"

    model = _get_embed_model()
    texts = [c["source"] for c in chunks]
    lengths = [len(t) for t in texts]
    texts = [t[:MAX_CHARS] for t in texts]

    if os.path.exists(CACHE_PATH):
        embeddings = np.load(CACHE_PATH)
    else:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=8)
        embeddings = embeddings.astype("float32")
        np.save(CACHE_PATH, embeddings)

    embeddings = _l2_normalize(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def _build_bm25_index(chunks : list[dict]) ->BM25Okapi:
    corpus = []
    for c in chunks:
        text = f"{c['source']} {c['file_path']} {c['name']} "
        corpus.append(_tokenize(text))
    return BM25Okapi(corpus)

def _normalize_scores(scores : np.ndarray) ->np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) /(hi - lo)


def retrieve(query: dict, chunks :list[dict], top_k:int =5)-> list[dict]:
    if not chunks:
        return []
    
    n = len(chunks)

    #FAISS : full corpus scored against semantic query
    faiss_index, _ = _build_faiss_index(chunks, query["source_id"], query["created_at"])
    model = _get_embed_model()
    q_emb = model.encode([query["semantic"]], convert_to_numpy=True, show_progress_bar=False)
    q_emb = _l2_normalize(q_emb)
    faiss_scores, faiss_idx = faiss_index.search(q_emb, n)
    faiss_scores_full = np.zeros(n, dtype="float32")
    faiss_scores_full[faiss_idx[0]] = faiss_scores[0]

    #BM25 : full corpus scored against identifier/file path/exception query

    bm25_index = _build_bm25_index(chunks)
    bm25_query_terms = (
        query.get("identifiers" ,[]) +
        query.get("file_paths", []) +
        query.get("exception_types", [])
    )
    bm25_query_tokens = _tokenize(" ".join(bm25_query_terms))
    bm25_scores_full = np.array(bm25_index.get_scores(bm25_query_tokens), dtype="float32")

    # fuse: normalize each independently, then combine
    faiss_norm = _normalize_scores(faiss_scores_full)
    bm25_norm = _normalize_scores(bm25_scores_full)
    fused = faiss_norm + bm25_norm

    # explicit file_path boost, applied after fusion normaliation
    query_file_paths = set(query.get("file_paths", []))
    if query_file_paths:
        for i, c in enumerate(chunks):
            if c["file_path"] in query_file_paths:
                fused[i] += FILE_PATH_BOOST

    #top candidate pool before reranking
    pool_size = min(CANDIDATE_POOL_SIZE, n)
    top_pool_idx = np.argsort(fused)[::-1][:pool_size]
    candidates = [chunks[i] for i in top_pool_idx]

    #rerank candidates with cross-encoder against the raw semantic query
    reranker = _get_rerank_model()
    pairs = [(query["semantic"], c["source"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)
    reranked_idx = np.argsort(rerank_scores)[::-1]
    reranked = [candidates[i] for i in reranked_idx]
    return reranked[:top_k]



if __name__ == "__main__":
    import json
    from issue_worker.retrieval.checkout import get_repo_at_commit
    from issue_worker.retrieval.chunker import chunk_repo
    from issue_worker.retrieval.query_builder import build_query

    with open("data/raw_issues.jsonl") as f:
        issue = json.loads(f.readline())

    repo_path = get_repo_at_commit(issue["source_id"], issue["created_at"])
    chunks = chunk_repo(repo_path, issue["source_id"])
    query = build_query(issue, chunks)
    results = retrieve(query, chunks, top_k=5)
    for r in results:
        print(r["chunk_id"])

    

