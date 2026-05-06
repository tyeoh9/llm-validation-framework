from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langchain_core.documents import Document


class RAGProvider:
    """Thin adapter that wraps any LangChain-compatible retriever.

    The retriever must implement: invoke(query: str) -> List[Document]
    """

    def __init__(self, retriever: Any):
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
