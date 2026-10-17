"""Build and persist FAISS index from maintenance documentation."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import build_faiss_index
from src.utils import MAINTENANCE_DOCS_DIR


def main():
    print("=" * 60)
    print("SentinelAI: Building FAISS RAG Index from Maintenance Docs")
    print("=" * 60)

    try:
        index, chunks = build_faiss_index(MAINTENANCE_DOCS_DIR)
        print(f"\nSuccessfully built and saved index!")
        print(f"Total vector embeddings: {index.ntotal}")
        print(f"Total chunk passages:    {len(chunks)}")
        print("\nIndexed Documents:")
        doc_names = sorted(set(c["doc_name"] for c in chunks))
        for d in doc_names:
            count = sum(1 for c in chunks if c["doc_name"] == d)
            print(f"  - {d} ({count} sections)")

    except Exception as exc:
        print(f"\n[ERROR] Failed to build RAG index: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
