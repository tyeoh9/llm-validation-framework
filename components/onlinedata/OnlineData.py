from ddgs import DDGS
from rank_bm25 import BM25Okapi

'''
TODO:
- simplify(): condense a sentence into its essence (used for input + output)
- If there are no good results (<threshold), scrape highest ranked (or maybe popular?) page
    - Tutorial: https://codesignal.com/learn/courses/navigating-the-web-for-information/lessons/searching-the-web-with-ddgs-in-python
'''

class OnlineData():

    def __init__(self, max_results=10):
        self.searcher = DDGS()
        self.max_results = max_results

    def rank_results(self, claim: str, pages: list[str]) -> list[tuple[str, float]]:
        tokenized_corpus = [p["body"].lower().split() for p in pages]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(claim.lower().split())
        ranked = sorted(zip(pages, scores), key=lambda x: x[1], reverse=True)
        return ranked

    def search(self, query):
        results = self.searcher.text(query, max_results=self.max_results)
        top_result = self.rank_results(query, results)[0][0]
        body = top_result["body"]
        href = top_result["href"]
        return body, href
