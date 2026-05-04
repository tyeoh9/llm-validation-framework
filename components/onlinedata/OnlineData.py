from ddgs import DDGS
from rank_bm25 import BM25Okapi

'''
TODO:
- If there are no good results (<threshold), scrape highest ranked (or maybe popular?) page
    - Tutorial: https://codesignal.com/learn/courses/navigating-the-web-for-information/lessons/searching-the-web-with-ddgs-in-python
- Find a way to make sure the facts are accurate
'''

class OnlineData:
    """Retrieves relevant facts/data from the web based on the query."""

    def __init__(self, max_results=10, min_score=1.0):
        self.searcher = DDGS()
        self.max_results = max_results
        self.min_score = min_score

    def rank_results(self, claim: str, pages: list[str]) -> list[tuple[str, float]]:
        tokenized_corpus = [p["body"].lower().split() for p in pages]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(claim.lower().split())
        ranked = sorted(zip(pages, scores), key=lambda x: x[1], reverse=True)
        return ranked

    def simplify(self, query: str, max_words: int = 30) -> str:
        """Condense a query to its key terms by stripping stop words.
        For queries over 30 words, extracts the first sentence first."""
        if len(query.split()) > max_words:
            import re
            first_sentence = re.split(r'(?<=[.!?])\s', query)[0]
            query = first_sentence
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "on",
            "at", "by", "for", "with", "about", "into", "from", "up", "down",
            "it", "its", "this", "that", "these", "those", "i", "me", "my",
            "we", "our", "you", "your", "he", "him", "his", "she", "her",
            "they", "them", "their", "and", "but", "or", "nor", "so", "yet",
            "as", "if", "when", "while", "which", "who", "what", "how",
            "also", "very", "just", "there", "then", "than", "not", "only",
        }
        words = query.split()
        key_words = [w for w in words if w.lower().strip(".,!?;:\"'") not in stop_words]
        return " ".join(key_words[:max_words])

    def search(self, query):
        try:
            search_query = self.simplify(query)
            results = self.searcher.text(search_query, max_results=self.max_results)
            if not results:
                return None, None
            top_result, top_score = self.rank_results(query, results)[0]
            if top_score < self.min_score:
                return None, None
            return top_result["body"], top_result["href"]
        except Exception:
            return None, None
