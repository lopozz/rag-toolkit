import os
import sys
import json

from tqdm import tqdm
from llama_cpp import Llama

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prompts import SYS_PROMPT
from src.utils import create_retrieval_context_section
from src.chat_defaults import HISTORY

MODEL = "gemma-3-12b-it-Q3_K_M.gguf"
TEST_TYPE = "complex_q"

with open(
    f"/home/lpozzi/Git/RAGghina/benchmark/data/{TEST_TYPE}.json", "r", encoding="utf-8"
) as f:
    data = json.load(f)

llm = Llama(
    model_path=f"/home/lpozzi/models/{MODEL}",
    n_ctx=3000,
    temperature=1,
    n_gpu_layers=50,
    verbose=True,
)

generated_answers = []
messages = [{"role": "system", "content": SYS_PROMPT}]
for turn in HISTORY:
    messages.append(
        {
            "role": "user",
            "content": f"{create_retrieval_context_section(turn['retrieved_contexts'])}\n\n{turn['question']}",
        }
    )
    messages.append({"role": "assistant", "content": turn["answer"]})


for question, context in tqdm(
    zip(data["question"], data["retrieved_contexts"]),
    total=len(data["question"]),
    desc="Generating answers",
):
    messages.append(
        {
            "role": "user",
            "content": f"{create_retrieval_context_section(context)}\n\n{question}",
        }
    )

    response = llm.create_chat_completion(
        messages=messages, max_tokens=1000, temperature=0.1
    )

    answer = response["choices"][0]["message"]["content"].strip()
    generated_answers.append(answer)

    messages.pop()

data["answer"] = generated_answers

result_file = MODEL.replace(".gguf", "")
with open(
    f"/home/lpozzi/Git/RAGghina/benchmark/results/{TEST_TYPE}_{result_file}.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
