"""
Retrieval benchmarking script for Elasticsearch.
This script test three retrieval models (i.e. BM25, kNN, and RRF) on three
settings:
    - KQ: all words in the query are present in the target document
    - SQ: any words in the query are present in the target document
    - MQ: some words in the query are present in the target document
This script evaluates document-level retrieval over the synthetic finance/PII 
dataset across multiple languages (en, it, fr, de) and embedding models.

Configurable parameters
----------------------------------------------
**Model selection**
- MODELS: list[str]
  SentenceTransformer model identifiers to benchmark (loaded one by one).

**Context / chunking (retrieval context granularity)**
- MAX_WORDS_LIST: list[int]
  Maximum number of words per chunk. Each value triggers a full re-index + eval.
- OVERLAP: int
  Number of overlapping words between consecutive chunks (0 = no overlap).

**Metrics**
- metrics: list
  A list of ir_measures metrics, e.g. [R@10, nDCG@10]. These are passed into
  evaluate_benchmarks. The script currently prints/stores only R@10 for the RRF
  run, but eval_results contains all requested metrics for all retrieval methods.
"""

# TODO save results in a standard json format shared across tests
# TODO allows multiple metrics to be loaded

import os
import nltk
import ast
import sys

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from ir_measures import R, nDCG
from collections import defaultdict
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.es_utils import (
    bm25_search,
    knn_search,
    rrf_search,
    evaluate_benchmarks,
    sentence_chunk,
)

load_dotenv("elastic-start-local/.env")
nltk.download("punkt_tab") # TODO is there another way not to use nltk?

MODELS = [
    "sentence-transformers/distiluse-base-multilingual-cased-v1",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # "intfloat/multilingual-e5-large",
    "google/embeddinggemma-300m",
    # "Qwen/Qwen3-Embedding-0.6B",
]

LANGUAGES = {
    "en": {"name": "Inglese", "nltk": "english"},
    "it": {"name": "Italiano", "nltk": "italian"},
    "fr": {"name": "Francese", "nltk": "french"},
    "de": {"name": "Tedesco", "nltk": "german"},
}

MAX_WORDS_LIST = [100]  # Different chunk sizes to test
METRIC = 'nDCG@10'
R_MODEL = "knn"

N = 100  # Number of queries # TODO is this fixed?
OVERLAP = 0

client = Elasticsearch(
    hosts=os.getenv("ES_LOCAL_URL"), api_key=os.getenv("ES_LOCAL_API_KEY")
)


def make_actions(df, model, lang_code, max_words, index_name, text_col = "generated_text"):
    nltk_lang = LANGUAGES[lang_code]["nltk"]

    for row in df.itertuples(index=False):
        r = row._asdict()
        doc_id = str(r["id"])
        text = r[text_col]

        chunks = sentence_chunk(
            text, max_words=max_words, overlap=OVERLAP, language=nltk_lang
        )
        if not chunks:
            continue

        vectors = model.encode(chunks, normalize_embeddings=True)

        for i, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
            yield {
                "_op_type": "index",
                "_index": index_name,
                "_id": f"{doc_id}::{i}",
                "_source": {
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "language": lang_code,
                    "document_type": r["document_type"],
                    "text": chunk_text,
                    "text_vector": vec.tolist(),
                },
            }


def top_docs_from_chunk_hits(hits, top_docs=10):
    seen = {}
    for chunk_id, score in hits.items():
        doc_id = chunk_id.rsplit("::", 1)[0]
        if doc_id not in seen or score > seen[doc_id]:
            seen[doc_id] = score
        if len(seen) >= top_docs:
            break
    return seen


def build_model_configs(client, index_name, model, selected=None):
    """Select on ly the tested model"""
    configs = {
        "bm25": lambda q, k: bm25_search(client, index_name, q, k=k),
        "knn": lambda q, k: knn_search(client, index_name, model, q, k=k, num_candidates=max(50, k * 5)),
        "rrf": lambda q, k: rrf_search(client, index_name, model, q, k=k, num_candidates=max(50, k * 5)),
    }
    return {k: v for k, v in configs.items() if selected is None or k == selected}


def run_evaluation(model_configs, queries_dict, metrics):
    """
    Run evaluation for a specific model and language
    
        qrels = {
            'Q0': {
                "D0": 0,
                "D1": 1,
            },
            "Q1": {
                "D0": 0,
                "D3": 2
            }
        }

        run = {
            'Q0': {
                "D0": 1.2,
                "D1": 1.0,
            },
            "Q1": {
                "D0": 2.4,
                "D3": 3.6
            }
        }
    """

    results = {}

    for name, df in queries_dict.items():
        qrels = defaultdict(dict)
        runs = {m_name: {} for m_name in model_configs}

        for r in df.itertuples(index=False):
            # Update Relevance Labels
            for doc_id in r.doc_ids:
                qrels[r.id][doc_id] = 1

            # Execute search models
            for m_name, search_call in model_configs.items():
                hits = search_call(r.query, k=10)
                pred_doc_ids = top_docs_from_chunk_hits(hits, top_docs=10)
                runs[m_name][r.id] = pred_doc_ids

        results[name] = {"qrels": qrels, "runs": runs}

    # Get evaluation scores
    eval_results = evaluate_benchmarks(results, metrics, return_dict=True)

    return eval_results


def main():

    # Initialize results structure
    final_results = {
        model_name: {
            LANGUAGES[lang]["name"]: {"KQ": [], "SQ": [], "MQ": []}
            for lang in LANGUAGES
        }
        for model_name in MODELS
    }

    metrics = [R@10, nDCG@10] 

    # Iterate over models
    for model_name in MODELS:
        print(f"\n{'=' * 80}")
        print(f"Processing model: {model_name}")
        print(f"{'=' * 80}")

        model = SentenceTransformer(model_name)
        dims = model.get_sentence_embedding_dimension()

        # Iterate over languages
        for lang_code, lang_info in LANGUAGES.items():
            lang_name = lang_info["name"]
            print(f"\n  Language: {lang_name} ({lang_code})")

            # Load queries
            base = f"data/synthetic_pii_finance/{lang_code}"
            kq = pd.read_csv(f"{base}/keyword_queries_{lang_code}_{N}.csv", converters={"doc_ids": ast.literal_eval})
            sq = pd.read_csv(f"{base}/semantic_queries_{lang_code}_{N}.csv", converters={"doc_ids": ast.literal_eval})
            mq = pd.read_csv(f"{base}/mixed_queries_{lang_code}_{N}.csv", converters={"doc_ids": ast.literal_eval})
            df = pd.read_csv(f"{base}/synthetic_pii_finance_{lang_code}.csv")
            
            # kq, sq, mq, contains doc ids that are use to filter the csv with text
            all_doc_ids = set(
                sq['doc_ids'].explode()) | set(kq['doc_ids'].explode()) | set(mq['doc_ids'].explode()
            )
            df = df[df["id"].isin(all_doc_ids)]

            # Iterate over chunk sizes
            for max_words in MAX_WORDS_LIST:
                print(f"    Chunk size: {max_words} words")

                index_name = f"synthetic_pii_finance_{lang_code}_{model_name.lower().replace('/', '_')}_{max_words}"

                # Delete and create index
                client.indices.delete(index=index_name, ignore_unavailable=True)
                client.indices.create(
                    index=index_name,
                    mappings={
                        "properties": {
                            "doc_id": {"type": "keyword"},
                            "chunk_id": {"type": "integer"},
                            "language": {"type": "keyword"},
                            "document_type": {"type": "keyword"},
                            "text": {"type": "text"},
                            "text_vector": {
                                "type": "dense_vector",
                                "dims": dims,
                                "index": True,
                                "similarity": "cosine",
                            },
                        }
                    },
                )

                # Index documents
                helpers.bulk(
                    client,
                    make_actions(df, model, lang_code, max_words, index_name),
                )
                client.indices.refresh(index=index_name)
                total_chunks = client.count(index=index_name)["count"]
                print(f"      Indexed {total_chunks} chunks")

                # Run evaluation
                queries_dict = {"kq": kq, "sq": sq, "mq": mq}
                model_configs = build_model_configs(client, index_name, model, selected=R_MODEL)
                eval_results = run_evaluation(model_configs, queries_dict, metrics)
                
                kq_score = (
                    eval_results.get("kq", {}).get(METRIC, {}).get(R_MODEL, 0.0)
                )
                sq_score = (
                    eval_results.get("sq", {}).get(METRIC, {}).get(R_MODEL, 0.0)
                )
                mq_score = (
                    eval_results.get("mq", {}).get(METRIC, {}).get(R_MODEL, 0.0)
                )

                print(
                    f"      KQ {METRIC}: {kq_score:.2f}, SQ {METRIC}: {sq_score:.2f}, MQ {METRIC}: {mq_score:.2f}"
                )

                final_results[model_name][lang_name]["KQ"].append(
                    round(kq_score, 2)
                )
                final_results[model_name][lang_name]["SQ"].append(
                    round(sq_score, 2)
                )
                final_results[model_name][lang_name]["MQ"].append(
                    round(mq_score, 2)
                )

                # Clean up index to save space
                client.indices.delete(index=index_name, ignore_unavailable=True)

    # Print final results
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS - {R_MODEL.upper()}")
    print("=" * 80)
    print("\ndata = {")
    for model_name, model_results in final_results.items():
        print(f'    "{model_name}": {{')
        for lang_name, scores in model_results.items():
            print(
                f"        '{lang_name}': {{ 'KQ': {scores['KQ']}, 'SQ': {scores['SQ']}, 'MQ': {scores['MQ']}}},"
            )
        print("    },")
    print("}")

    return final_results


if __name__ == "__main__":
    results = main()
