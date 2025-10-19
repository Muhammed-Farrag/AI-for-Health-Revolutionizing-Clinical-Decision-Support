import os
import json
import argparse
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from retrieval_engine import entry_to_documents


def load_cleaned_json(json_path: str) -> List[Dict]:
    """Load a cleaned drugs JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of drug entries in the JSON file.")
    return data


def build_faiss_from_documents(documents: List[Document], model_name: str) -> FAISS:
    """Build FAISS store from documents."""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(documents, embedding=embeddings)


def save_faiss(store: FAISS, persist_dir: str) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    store.save_local(persist_dir)


def make_documents_from_json(json_path: str) -> List[Document]:
    entries = load_cleaned_json(json_path)
    all_docs = []
    for entry in entries:
        all_docs.extend(entry_to_documents(entry))
    return all_docs


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from cleaned drug JSON")
    parser.add_argument("--json", default=r"D:\RAG\cleaned_db\drugbank_database.json", help="Path to cleaned JSON file")
    parser.add_argument("--out", default="faiss_medical_db", help="Output directory")
    parser.add_argument("--model", default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    args = parser.parse_args()


    print("Loading cleaned JSON...")
    docs = make_documents_from_json(args.json)
    print(f"Created {len(docs)} documents.")

    print(f"Building FAISS index using model: {args.model}")
    store = build_faiss_from_documents(docs, args.model)

    save_faiss(store, args.out)
    print(f"FAISS index saved to: {args.out}")

    # Cache metadata
    cache_path = os.path.join(args.out, "all_docs_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([d.metadata for d in docs], f, ensure_ascii=False, indent=2)
    print(f"Cached document metadata at: {cache_path}")


if __name__ == "__main__":
    main()
