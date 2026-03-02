import os
from glob import glob
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGRetriever:
    """Retrieval-only RAG utility backed by local PDFs and a FAISS index."""

    def __init__(
        self,
        data_dir: str | None = None,
        index_dir: str | None = None,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        base_dir = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir else base_dir / "data"
        self.index_dir = Path(index_dir) if index_dir else base_dir / "faiss_index"
        self.embed_model = embed_model
        self._vectorstore = None

    def load_pdfs(self) -> list:
        pdf_paths = sorted(glob(os.path.join(str(self.data_dir), "*.pdf")))
        if not pdf_paths:
            raise FileNotFoundError(
                f"No PDFs found in '{self.data_dir}'. Put your PDFs in {self.data_dir}/"
            )

        docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        return docs

    def build_or_load_vectorstore(self) -> FAISS:
        embeddings = HuggingFaceEmbeddings(model_name=self.embed_model)
        if self.index_dir.exists():
            self._vectorstore = FAISS.load_local(
                str(self.index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return self._vectorstore

        raw_docs = self.load_pdfs()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        self._vectorstore = FAISS.from_documents(chunks, embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(self.index_dir))
        return self._vectorstore

    def retrieve(self, query: str, k: int = 5) -> list:
        if self._vectorstore is None:
            self.build_or_load_vectorstore()
        return self._vectorstore.similarity_search(query, k=k)

    def pretty_print_results(self, results: list, query: str) -> None:
        print("\n" + "=" * 90)
        print(f"QUERY: {query}")
        print("=" * 90)

        for i, doc in enumerate(results, start=1):
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", None)
            page_str = f"p.{page + 1}" if isinstance(page, int) else "p.?"
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 700:
                snippet = snippet[:700] + " ..."
            print(f"\n[{i}] SOURCE: {src}, {page_str}")
            print("-" * 90)
            print(snippet)
        print("\n" + "=" * 90)

    def save_results_to_file(
        self, results: list, query: str, filename: str | None = None
    ) -> Path:
        output_path = (
            Path(filename)
            if filename
            else Path(__file__).resolve().parent / "retrieval_output.txt"
        )
        with output_path.open("w", encoding="utf-8") as output_file:
            output_file.write("=" * 90 + "\n")
            output_file.write(f"QUERY: {query}\n")
            output_file.write("=" * 90 + "\n\n")

            for i, doc in enumerate(results, start=1):
                src = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", None)
                page_str = f"p.{page + 1}" if isinstance(page, int) else "p.?"

                snippet = doc.page_content.strip().replace("\n", " ")
                if len(snippet) > 700:
                    snippet = snippet[:700] + " ..."

                output_file.write(f"[{i}] SOURCE: {src}, {page_str}\n")
                output_file.write("-" * 90 + "\n")
                output_file.write(snippet + "\n\n")

            output_file.write("=" * 90 + "\n")
        return output_path
