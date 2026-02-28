# retrieve_only.py
import os
from glob import glob
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Updated embeddings import (avoids the LangChain deprecation warning)
# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdfs(data_dir: str) -> List:
    pdf_paths = sorted(glob(os.path.join(data_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in '{data_dir}'. Put your PDFs in {data_dir}/"
        )

    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())  # one Document per page
    return docs


def build_or_load_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists(INDEX_DIR):
        # Load existing index
        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )

    # Build new index
    raw_docs = load_pdfs(DATA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_DIR)
    return vs


def pretty_print_results(results: List, query: str):
    print("\n" + "=" * 90)
    print(f"QUERY: {query}")
    print("=" * 90)

    for i, doc in enumerate(results, start=1):
        src = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", None)
        page_str = f"p.{page + 1}" if isinstance(page, int) else "p.?"
        snippet = doc.page_content.strip().replace("\n", " ")

        # Truncate snippet for readability
        if len(snippet) > 700:
            snippet = snippet[:700] + " ..."

        print(f"\n[{i}] SOURCE: {src}, {page_str}")
        print("-" * 90)
        print(snippet)
    print("\n" + "=" * 90)

def save_results_to_file(results, query, filename="retrieval_output.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(f"QUERY: {query}\n")
        f.write("=" * 90 + "\n\n")

        for i, doc in enumerate(results, start=1):
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", None)
            page_str = f"p.{page + 1}" if isinstance(page, int) else "p.?"

            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 700:
                snippet = snippet[:700] + " ..."

            f.write(f"[{i}] SOURCE: {src}, {page_str}\n")
            f.write("-" * 90 + "\n")
            f.write(snippet + "\n\n")

        f.write("=" * 90 + "\n")

    print(f"\n✅ Results saved to {filename}")

def main():
    # Build or load the FAISS vectorstore
    vectorstore = build_or_load_vectorstore()

    # You can tweak k here
    k = 5

    print("Retrieval-only mode (no LLM). Type a query, or 'exit' to quit.")
    print(f"PDF folder: {DATA_DIR}/ | Index folder: {INDEX_DIR}/ | Top-k: {k}")

    while True:
        query = input("\n> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        # Retrieve top-k chunks by semantic similarity
        results = vectorstore.similarity_search(query, k=k)
        pretty_print_results(results, query)
        save_results_to_file(results, query)


if __name__ == "__main__":
    main()