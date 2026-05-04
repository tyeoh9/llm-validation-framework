from RAG import RAG


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class FakeVectorStore:
    def similarity_search(self, query, k=5):
        docs = [
            FakeDocument("Doc 1 content", {"source": "a.pdf", "page": 0}),
            FakeDocument("Doc 2 content", {"source": "b.pdf", "page": 1}),
            FakeDocument("Doc 3 content", {"source": "c.pdf", "page": 2}),
        ]
        return docs[:k]

    def similarity_search_with_score(self, query, k=5):
        scored_docs = [
            (FakeDocument(" First chunk.\n\nHas spaces. ", {"source": "a.pdf", "page": 0}), 0.2),
            (FakeDocument("Second chunk\nwith newline.", {"source": "b.pdf", "page": 1}), 0.8),
            (FakeDocument("Third chunk", {"source": "c.pdf", "page": 2}), 1.5),
        ]
        return scored_docs[:k]


def test_get_relevant_docs_basic():
    retriever = RAG(data_dir="dummy_data", index_dir="dummy_index")
    retriever._vectorstore = FakeVectorStore()

    results = retriever.getRelevantDocs("privacy", k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].page_content == "Doc 1 content"
    assert results[1].metadata["source"] == "b.pdf"
    print("test_get_relevant_docs_basic passed")


def test_get_relevant_docs_k_larger_than_docs():
    retriever = RAG(data_dir="dummy_data", index_dir="dummy_index")
    retriever._vectorstore = FakeVectorStore()

    results = retriever.getRelevantDocs("privacy", k=10)

    assert len(results) == 3
    print("test_get_relevant_docs_k_larger_than_docs passed")


def test_extract_docs_basic():
    retriever = RAG(data_dir="dummy_data", index_dir="dummy_index")
    retriever._vectorstore = FakeVectorStore()

    result = retriever.extractDocs("privacy", threshold=1.0, k=3)

    expected = "First chunk. Has spaces.\n\nSecond chunk with newline."
    assert result == expected
    print("test_extract_docs_basic passed")


def test_extract_docs_filters_by_threshold():
    retriever = RAG(data_dir="dummy_data", index_dir="dummy_index")
    retriever._vectorstore = FakeVectorStore()

    result = retriever.extractDocs("privacy", threshold=0.5, k=3)

    assert result == "First chunk. Has spaces."
    print("test_extract_docs_filters_by_threshold passed")


def test_extract_docs_no_match():
    retriever = RAG(data_dir="dummy_data", index_dir="dummy_index")
    retriever._vectorstore = FakeVectorStore()

    result = retriever.extractDocs("privacy", threshold=0.1, k=3)

    assert result == ""
    print("test_extract_docs_no_match passed")


def test_clean_text():
    retriever = RAG(data_dir="dummy_data", index_dir="dummy_index")

    cleaned = retriever._clean_text("  Hello   world \n this is   a test  ")
    assert cleaned == "Hello world this is a test"
    print("test_clean_text passed")


def test_load_pdfs_no_files():
    retriever = RAG(data_dir="empty_folder_that_does_not_exist", index_dir="dummy_index")

    try:
        retriever.load_pdfs()
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        print("test_load_pdfs_no_files passed")


def run_all_tests():
    test_get_relevant_docs_basic()
    test_get_relevant_docs_k_larger_than_docs()
    test_extract_docs_basic()
    test_extract_docs_filters_by_threshold()
    test_extract_docs_no_match()
    test_clean_text()
    test_load_pdfs_no_files()
    print("\nAll tests passed")


if __name__ == "__main__":
    run_all_tests()