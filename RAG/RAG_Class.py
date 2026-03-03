import os
from glob import glob
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class PDFRetriever:

    def __init__(
        self,
        data_dir: str = "data",
        index_dir: str = "faiss_index",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.embed_model = embed_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self.vectorstore = None

    def _load_pdfs(self) -> List:
        pdf_paths = sorted(glob(os.path.join(self.data_dir, "*.pdf")))

        if not pdf_paths:
            raise FileNotFoundError(
                f"No PDFs found in '{self.data_dir}'. Put PDFs in this folder."
            )

        docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        return docs

    def index(self):
        """
        Build FAISS index if it doesn't exist,
        otherwise load existing index.
        """

        if os.path.exists(self.index_dir):
            self.vectorstore = FAISS.load_local(
                self.index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("Loaded existing FAISS index.")
            return

        print("Building FAISS index...")

        raw_docs = self._load_pdfs()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(raw_docs)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.index_dir)

        print("Index built and saved.")

    def retrieve(self, query: str, k: int = 5):
        """
        Retrieve top-k similar chunks.
        """

        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Run index() first.")

        results = self.vectorstore.similarity_search(query, k=k)

        output = []

        for doc in results:
            output.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                }
            )

        return output