"""
Quick evaluation harness for a llm served with OpenAI-compatible /v1/chat/completions enpoint.
It tests whether the model follows the provided system instruction (e.g., language/style, “use only
the documents”) and whether the response is **complete** with respect to a set of expected entities.

The input is JSONL file where each line has these keys:
- `id` (int): example identifier
- `question` (str): the user question to answer from the provided documents
- `entities` (List[str]): strings that should appear in a complete answer (used for scoring)
- `free_form_answer` (str): reference/gold answer (used for printing/comparison)
- `paragraphs` (List[List[str]]): document context as groups of paragraphs (each inner list is a
  sequence of text chunks/paragraphs)
"""

import json
import requests
from typing import Any, Dict, List


SYSTEM_PROMPT = """Answer using only the provided resources. If necessary translate them in english.

- Do not add assumptions, reasoning, external knowledge, or unrelated information.
- For fact questions: return only the exact fact.  If necessary translate it in english.
- If the user question require information aggregation: extract and aggregate evidence in a clear and concise way.
- Do not repeat the same information if it appears in multiple sections.
- Maintain a direct, authoritative tone based solely on the provided context.
- Always write in italian
"""

VLLM_BASE_URL = "http://localhost:8000"
# VLLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# VLLM_MODEL = "google/gemma-3n-E2B-it"
# VLLM_MODEL = "Qwen/Qwen3-0.6B"
VLLM_MODEL = "mistralai/Ministral-3-3B-Instruct-2512"
TRIALS = 3

def completeness_score(answer: str, entities: List[str]) -> float:
    """
    Completeness = fraction of entities (case-insensitive substring match) found in answer.
    """

    a = answer.lower().replace('.', '').replace(',', '')
    hit = 0
    for e in entities:
        if e.lower().replace('.', '') in a:
            hit += 1
    return hit / len(entities)


def missing_entities(answer: str, entities: List[str]) -> List[str]:
    a = answer.lower()
    return [e for e in entities if e.lower() not in a]

if __name__ == "__main__":
    # lang = 'en'
    # output_path = f"output/{lang}/results-{VLLM_MODEL.split('/')[1]}.jsonl"

    lang = 'qit_pen'
    output_path = f"output/language_instruction/{lang}/results-{VLLM_MODEL.split('/')[1]}.jsonl"
    out_f = open(output_path, "w", encoding="utf-8")
    
    
    # input_path = f'data/synthetic_pii_finance/{lang}/answerable_10.jsonl'
    input_path = f'data/language_instruction/answerable_10_{lang}.jsonl'
    items = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))

    for j, item in enumerate(items[:3]):
        print(">"*50 + f" ITEM {j+1} " + "<"*50)
        question = item["question"]
        entities = item["entities"]
        paragraphs = '\n'.join(['\n'.join(ps) for ps in item["paragraphs"]])
        free_form_answer = item["free_form_answer"]


        user_content = (
            f"Start documents\n\n{paragraphs}\n\nEnd documents\n\n"
            f"{question}"
        )

        payload = {
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3
        }

        url = VLLM_BASE_URL.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }

        trials_out: List[Dict[str, Any]] = []
        scores: List[float] = []

        for t in range(TRIALS):
            resp = requests.post(url, headers=headers, data=json.dumps(payload))
            resp = resp.json()
            answer = resp["choices"][0]["message"]["content"]
            # score = completeness_score(answer.split('</think>')[1].strip(), entities)
            score = completeness_score(answer, entities)
            miss = missing_entities(answer, entities)
            

            scores.append(score)
            trials_out.append({
                "trial": t,
                "answer": answer,
                "completeness": score,
                "missing_entities": miss,
                "usage": resp.get("usage"),
            })
            print("="*50 + f" TRIAL {t+1} " + "="*50)
            if score!=1:
                print(f"Question: {question}\nGold answer: {free_form_answer}\nEntities: {entities}\nScore: {score}\nMissing entities: {miss}\n\nAnswer:\n{answer}")
            else:
                print(f"Question: {question}\nGold answer: {free_form_answer}\nEntities: {entities}\nScore: {score}\n\nAnswer:\n{answer}")
            print("="*120 + "\n")

        row = {
            "question": question,
            "entities": entities,
            "avg_completeness": sum(scores)/len(scores),
            "gold": item["free_form_answer"],
            "trials": trials_out,
        }

        # out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_f.close()


