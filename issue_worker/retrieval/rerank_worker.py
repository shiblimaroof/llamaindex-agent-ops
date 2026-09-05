"""
Isolated reranker worker. Runs in its own process so the CrossEncoder
never shares process state with the SentenceTransformer embed model —
loading both in one process causes native segfaults / silent NaN on
this machine (Apple Silicon, likely an ONNX/OpenMP conflict between
the two model backends).

Reads JSON from stdin: {"query": str, "texts": [str, ...]}
Writes JSON to stdout: {"scores": [float, ...]}
"""

import json
import sys

from sentence_transformers import CrossEncoder

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"


def main():
    payload = json.load(sys.stdin)
    query = payload["query"]
    texts = payload["texts"]

    model = CrossEncoder(RERANK_MODEL_NAME, device="cpu")
    pairs = [(query, t) for t in texts]
    scores = model.predict(pairs)

    json.dump({"scores": [float(s) for s in scores]}, sys.stdout)


if __name__ == "__main__":
    main()