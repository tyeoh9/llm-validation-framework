"""Accuracy step: RAG + online search + LLM-as-a-judge on (question, answer)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.rag.RAGRetriever import RAGRetriever
from components.onlinedata.OnlineData import OnlineData
from components.accuracy.AccuracyAgent import AccuracyAgent


class AccuracyStep:
    """Step that fetches evidence (RAG + web) and uses AccuracyAgent to score the LLM answer."""

    def __init__(self, rag_k: int = 5, config_path: str | None = None):
        self.rag_k = rag_k
        self.config_path = config_path
        self._rag = None
        self._online = None
        self._accuracy_agent = None

    def evaluate(self, payload: dict) -> dict:
        # payload = {"question": str, "answer": str}
        question = payload.get("question", "")
        answer = payload.get("answer", "")
        if not question or not answer:
            return {"status": "fail", "score": 0.0, "reason": "Missing question or answer"}

        # RAG
        if self._rag is None:
            self._rag = RAGRetriever()
            self._rag.build_or_load_vectorstore()
        rag_docs = self._rag.retrieve(question, k=self.rag_k)
        rag_evidence = "\n\n".join(doc.page_content.strip() for doc in rag_docs).strip()

        # Online search
        if self._online is None:
            self._online = OnlineData()
        try:
            web_body, web_href = self._online.search(question)
            online_evidence = f"[Source: {web_href}]\n{web_body}"
        except Exception as e:
            online_evidence = f"(Online search failed: {e})"

        combined_evidence = f"## Evidence from documents (RAG)\n{rag_evidence}\n\n## Evidence from web\n{online_evidence}"

        # LLM-as-a-judge
        if self._accuracy_agent is None:
            self._accuracy_agent = AccuracyAgent(config_path=self.config_path)
        result = self._accuracy_agent.evaluate(text1=answer, text2=combined_evidence)
        score = result["score"]
        reason = result["reason"]
        status = "success" if score >= 0.5 else "fail"
        return {"status": status, "score": score, "reason": reason}