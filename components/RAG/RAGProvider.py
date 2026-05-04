from typing import Any, Optional

from langchain_core.documents import Document


class RAGProvider:
    def __init__(self, retriever: Any):
        # NOTE: In documentation, clearly specify that the passed-in object
        # must implement: `invoke(query: str) -> List[Document]`.
        # This is the only required contract for compatibility.
        if not hasattr(retriever, "invoke"):
            raise TypeError(
                "Retriever must implement `invoke(query: str) -> List[Document]`"
            )
        self.retriever = retriever

    def get_most_relevant_doc(self, query: str) -> Optional[Document]:
        docs = self.retriever.invoke(query)
        return docs[0] if docs else None

    def extract_content(self, query: str) -> Optional[str]:
        doc = self.get_most_relevant_doc(query)
        return doc.page_content if doc else None
