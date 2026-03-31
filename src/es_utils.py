import pandas as pd
from ir_measures import iter_calc
from nltk.tokenize import sent_tokenize


def bm25_search(client, index_name: str, text: str, k=10):
    res = client.search(
        index=index_name, size=k, query={"match": {"text": {"query": text}}}
    )
    return {hit["_id"]: hit["_score"] for hit in res["hits"]["hits"]}


def knn_search(client, index_name: str, model, text: str, k=10, num_candidates=50):
    qv = model.encode(text, normalize_embeddings=True).tolist()
    res = client.search(
        index=index_name,
        size=k,
        knn={
            "field": "text_vector",
            "query_vector": qv,
            "k": k,
            "num_candidates": num_candidates,
        },
    )
    return {hit["_id"]: hit["_score"] for hit in res["hits"]["hits"]}


def rrf_search(client, index_name: str, model, text, k=10, num_candidates=50):
    qv = model.encode(text, normalize_embeddings=True).tolist()
    res = client.search(
        index=index_name,
        size=k,
        retriever={
            "rrf": {
                "retrievers": [
                    {"standard": {"query": {"match": {"text": text}}}},
                    {
                        "knn": {
                            "field": "text_vector",
                            "query_vector": qv,
                            "k": k,
                            "num_candidates": num_candidates,
                        }
                    },
                ]
            }
        },
    )
    return {hit["_id"]: hit["_score"] for hit in res["hits"]["hits"]}


def evaluate_benchmarks(benchmark_results, metrics, return_dict=False):
    """
    benchmark_results: The output from run_benchmarks
    metrics: List of ir_measures (e.g., [R@1, R@3])
    return_dict: If True, return the summary dictionary instead of just printing
    """
    all_summaries = {}

    for scenario, data in benchmark_results.items():
        qrels = data["qrels"]

        # Store mean scores for a nice summary table later
        summary = {}

        for model_name, run in data["runs"].items():
            # Calculate metrics using iter_calc
            scores = list(iter_calc(metrics, qrels, run))

            # Group by measure and calculate Mean
            for m in metrics:
                m_str = str(m)
                # Filter scores for this specific measure and average them
                m_values = [s.value for s in scores if str(s.measure) == m_str]
                avg = sum(m_values) / len(m_values) if m_values else 0

                if m_str not in summary:
                    summary[m_str] = {}
                summary[m_str][model_name] = avg

        if not return_dict:
            print(f"--- Scenario: {scenario.upper()} ---")
            print(pd.DataFrame(summary).round(3))
            print("\n")

        all_summaries[scenario] = summary

    if return_dict:
        return all_summaries


def sentence_chunk(
    text: str, max_words: int = 512, overlap: int = 0, language: str = "italian"
):
    sents = [s.strip() for s in sent_tokenize(text, language=language) if s.strip()]
    chunks = []
    start = 0

    while start < len(sents):
        end = start
        word_count = 0
        current = []

        while end < len(sents):
            sent = sents[end]
            sent_words = len(sent.split())

            if max_words and (word_count + sent_words) > max_words:
                if not current:
                    current.append(sent)
                    end += 1
                break

            current.append(sent)
            word_count += sent_words
            end += 1

        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(sents):
            break

        start = max(end - overlap, start + 1) if overlap > 0 else end

    return chunks
