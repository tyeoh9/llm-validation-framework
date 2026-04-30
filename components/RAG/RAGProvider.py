from typing import List

from langchain_core.documents import Document


class RAGProvider:
    """
    General RAG wrapper.

    This class does not know or care which vector DB is being used.
    It only calls the provider.
    """

    def __init__(self, vector_provider: VectorDBProvider):
        self.vector_provider = vector_provider

    def getRelevantDocs(self, query: str, k: int = 5) -> List[Document]:
        """
        Return the top-k most relevant document chunks.
        """
        return self.vector_provider.getRelevantDocs(query, k)

    def extractDocs(
        self,
        query: str,
        threshold: float,
        k: int = 5,
    ) -> List[Document]:
        """
        Return top-k document chunks that pass the threshold.

        If none pass, return [].
        """
        return self.vector_provider.extractDocs(query, threshold, k)