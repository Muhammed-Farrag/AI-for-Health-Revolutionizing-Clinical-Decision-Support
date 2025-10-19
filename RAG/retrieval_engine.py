import os
import re
import json
from typing import List, Dict, Iterable, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ========= Formatting + Document Creation =========

def _format_known_interactions(known_interactions: Iterable[Dict]) -> str:
    if not known_interactions:
        return ""
    parts = []
    for item in known_interactions:
        other = item.get("drug_name", "")
        desc = item.get("description", "")
        if other and desc:
            parts.append(f"- {other}: {desc}")
    return "\n".join(parts).strip()


def _format_food_interactions(food_interactions: Iterable[str]) -> str:
    if not food_interactions:
        return ""
    return "\n".join(f"- {fi}" for fi in food_interactions if fi)


def entry_to_documents(entry: Dict) -> List[Document]:
    name = entry.get("name", "")
    description = entry.get("description", "")
    synonyms = entry.get("synonyms", [])
    drugbank_id = entry.get("drugbank_id", "")
    known_interactions = entry.get("known_interactions", [])
    food_interactions = entry.get("food_interactions", [])

    base_meta = {
        "drugbank_id": drugbank_id,
        "name": name,
        "synonyms": synonyms,
        "_aliases": [a for a in {name.lower(), *[s.lower() for s in synonyms]} if a],
    }

    docs = []

    docs.append(Document(
        page_content=f"Drug: {name}\n\nDescription: {description}",
        metadata={**base_meta, "doc_type": "profile"}
    ))

    if known_interactions:
        docs.append(Document(
            page_content=f"Drug-drug interactions for {name}:\n\n{_format_known_interactions(known_interactions)}",
            metadata={**base_meta, "doc_type": "drug_drug_interaction"}
        ))

    if food_interactions:
        docs.append(Document(
            page_content=f"Food interactions for {name}:\n\n{_format_food_interactions(food_interactions)}",
            metadata={**base_meta, "doc_type": "drug_food_interaction"}
        ))

    return docs


# ========= Retrieval System =========

def load_faiss(persist_dir: str, model_name: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)


def detect_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["food", "eat", "drink", "alcohol", "vitamin", "juice"]):
        return "food"
    if any(w in q for w in ["interact", "interaction", "combine", "with", "between"]):
        return "ddi"
    return "general"


def extract_candidate_drug_names(query: str) -> List[str]:
    pattern = r"\b[a-z][a-z0-9\-]*(?:\s+[a-z0-9\-]+){0,2}\b"
    tokens = re.findall(pattern, query.lower())
    return [t for t in tokens if len(t) > 2]


def build_synonym_map(docs: List[Document]) -> Dict[str, List[str]]:
    synonym_map = {}
    for d in docs:
        if d.metadata.get("doc_type") == "profile":
            aliases = d.metadata.get("_aliases", [])
            for a in aliases:
                synonym_map[a] = aliases
    return synonym_map


def expand_query_with_synonyms(query: str, synonym_map: Dict[str, List[str]]) -> str:
    candidates = extract_candidate_drug_names(query)
    expanded = set()
    for c in candidates:
        if c in synonym_map:
            expanded.update(synonym_map[c])
    if expanded:
        return f"{query} {' '.join(expanded)}"
    return query


def intent_to_doc_type(intent: str) -> str:
    return {
        "food": "drug_food_interaction",
        "ddi": "drug_drug_interaction",
        "general": "profile",
    }.get(intent, "profile")


def smart_retrieve(store: FAISS, docs: List[Document], synonym_map: Dict[str, List[str]], query: str, k=5):
    intent = detect_intent(query)
    expanded_query = expand_query_with_synonyms(query, synonym_map)
    target_type = intent_to_doc_type(intent)

    results = store.similarity_search(expanded_query, k=k*3)
    filtered = [d for d in results if d.metadata.get("doc_type") == target_type]
    return filtered[:k]


def retrieval_demo(persist_dir: str, model_name: str, query: str):
    store = load_faiss(persist_dir, model_name)

    cache_path = os.path.join(persist_dir, "all_docs_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        all_docs = [Document(page_content="", metadata=m) for m in cached]
    else:
        all_docs = list(getattr(store.docstore, "_dict", {}).values())

    synonym_map = build_synonym_map(all_docs)
    results = smart_retrieve(store, all_docs, synonym_map, query)

    print(f"\nQuery: {query}")
    for idx, doc in enumerate(results, 1):
        print(f"\n{idx}. {doc.metadata.get('name')} [{doc.metadata.get('doc_type')}]")
        preview = doc.page_content[:400]
        print(preview, "..." if len(doc.page_content) > 400 else "")


if __name__ == "__main__":
    retrieval_demo("faiss_medical_db", 
                   "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                   "what are the food interactions with warfarin?")
