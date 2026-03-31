import os
import sys
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Add local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prompts import SYS_PROMPT
from src.utils import create_retrieval_context_section
from src.chat_defaults import HISTORY

MODEL = "google/gemma-3-1b-it"
TEST_TYPE = "simple_q"

with open(f"benchmark/data/{TEST_TYPE}.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize vLLM
llm = LLM(model=MODEL, dtype="auto", gpu_memory_utilization=0.9)

# Set your generation parameters
sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=3000,
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
    messages.append({"role": "assistant", "content": turn["response"]})

for question, context in tqdm(
    zip(data["question"], data["retrieved_contexts"]),
    total=len(data["question"]),
    desc="Generating answers",
):
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {
            "role": "user",
            "content": f"{create_retrieval_context_section(context)}\n\nGenera una risposta di almeno 8 frasi. {question}",
        },
    ]

    response = llm.chat(
        messages=messages, sampling_params=sampling_params, use_tqdm=False
    )

    answer = response[0].outputs[0].text.strip()
    generated_answers.append(answer)

data["response"] = generated_answers

result_file = MODEL.split("/")[-1]
with open(
    f"benchmark/results/{TEST_TYPE}_{result_file}.json", "w", encoding="utf-8"
) as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
