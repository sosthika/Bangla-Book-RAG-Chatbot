
"""Create two FAISS indexes using two chunking strategies."""
import json
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BOOK_FILE = Path("data/book.json")
VECTOR_DIR = Path("vectorstore")
EMBEDDING_MODEL = "BAAI/bge-m3"


def load_documents():
    if not BOOK_FILE.exists():
        raise FileNotFoundError("data/book.json not found. Run: python crawler.py")

    records = json.loads(BOOK_FILE.read_text(encoding="utf-8"))
    documents = []

    for item in records:
        documents.append(Document(
            page_content=item["text"],
            metadata={
                "book": item.get("book", "মেজদিদি"),
                "edition": item.get("edition", "১৯৬৩"),
                "author": item.get("author", "শরৎচন্দ্র চট্টোপাধ্যায়"),
                "section": item["section"],
                "chapter": item["chapter"],
                "title": item["title"],
                "source_url": item["source_url"],
            }
        ))
    return documents


def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} chapter documents.")

    print(f"Loading multilingual embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
    )

    strategies = {
        "strategy_a": {"chunk_size": 500, "chunk_overlap": 100},
        "strategy_b": {"chunk_size": 1000, "chunk_overlap": 200},
    }

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    for name, config in strategies.items():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", "।", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(docs)
        print(
            f"\n{name}: {len(chunks)} chunks "
            f"(size={config['chunk_size']}, overlap={config['chunk_overlap']})"
        )

        output_dir = VECTOR_DIR / name
        if output_dir.exists():
            shutil.rmtree(output_dir)

        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(str(output_dir))
        print(f"Saved FAISS index: {output_dir}")

    print("\nBoth indexes are ready.")


if __name__ == "__main__":
    main()
