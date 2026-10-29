import argparse
from pathlib import Path
from rag_core import DEFAULT_CORPUS, build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a local FAISS index from approved guideline PDFs.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=Path("index"))
    args = parser.parse_args()
    print(f"Indexed {build_index(args.corpus, args.output)} text chunks.")
