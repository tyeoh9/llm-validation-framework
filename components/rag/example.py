"""Example usage of RAGRetriever."""

from RAGRetriever import RAGRetriever


if __name__ == "__main__":
    retriever = RAGRetriever()
    k = 5

    print("Retrieval-only mode (no LLM). Type a query, or 'exit' to quit.")
    print(
        f"PDF folder: {retriever.data_dir}/ | Index folder: {retriever.index_dir}/ | Top-k: {k}"
    )

    while True:
        query = input("\n> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        results = retriever.retrieve(query, k=k)
        retriever.pretty_print_results(results, query)
        output_path = retriever.save_results_to_file(results, query)
        print(f"\nResults saved to {output_path}")
