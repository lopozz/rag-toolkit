"""
Minimal driver script to test a query rewriting prompt against a vLLM-served chat model.
Given a conversation log, it builds a short sliding window of recent turns and asks the model
to rewrite the latest user message into a standalone retrieval query (for RAG / hybrid search).

Input format

Reads a JSON file at `data/mtrag/<lang>/conversations_10.json` with structure:
- Top-level: List[Conversation]
- Conversation: {
    "messages": List[Message],
    ...
  }
- Message: {
    "speaker": "user" | "assistant",
    "text": str,
    "enrichments": { ... }   # optional metadata; this script prints some fields for user turns
  }
"""

import json
import requests


SYSTEM_PROMPT = """
You are an expert Query Rewriter for a Retrieval-Augmented Generation system. Your goal is to transform the user's latest query into a standalone search query optimized for a hybrid retriever.

### Task Instructions
1. Analyze the chat history and the latest user query.
2. Perform **Contextual Carryover**: 
    - Replace pronouns or add implicit info from previous context with the specific entities they refer to.
    - If the user asks a continuation question, make explicit the specific topic of the history into the query. *Example: History: "Tell me about Mars." Query: "Is there water?" -> Rewrite: "Is there water on Mars?"*
3. Perform **Friction Reduction**: If the query is ambiguous or seems to contain noise, rewrite it to clearly reflect the user's original intent based on the conversation flow.
4. **Retrieval Optimization**: Ensure the final query is a single, clear and stand-alone question or statement.
5. Handle **Conversational/Chitchat** turns: If the latest query is a greeting, feedback, or a general statement (e.g., "Hello", "I see", "Thanks for explaining"), do NOT adapt it. Return the original text as-is.

### Constraints
- Output ONLY the rewritten/standalone query.
- Do not include explanations, labels, or conversational fillers.
- If the query is already standalone and clear or is not dependent from past queries, return it exactly as-is.
"""


VLLM_BASE_URL = "http://localhost:8000"
# VLLM_MODEL = "Qwen/Qwen3-0.6B"
VLLM_MODEL = "mistralai/Ministral-3-3B-Instruct-2512"

if __name__ == "__main__":
    # lang = 'en'
    # output_path = f"output/{lang}/results-{VLLM_MODEL.split('/')[1]}.jsonl"

    # lang = 'qen_pit'
    # output_path = f"output/language_instruction/{lang}/results-{VLLM_MODEL.split('/')[1]}.jsonl"
    # out_f = open(output_path, "w", encoding="utf-8")
    lang = "it"
    with open(
        f"data/mtrag/{lang}/conversations_10.json", "r", encoding="utf-8"
    ) as file:
        conversations = json.load(file)

    for j, c in enumerate(conversations[5:6]):
        messages = c["messages"]

        for i in list(range(0, len(messages), 2)):
            history = []
            assert c["messages"][i]["speaker"] == "user"

            if i == 0:
                continue
            start_index = max(0, i - 6)
            history_window = messages[start_index : i + 1]

            formatted_history = ""
            debug_history = ""

            for msg in history_window:
                role_label = "USER" if msg["speaker"] == "user" else "ASSISTANT"
                formatted_history += f"{role_label}: {msg['text']}\n"
                debug_history += f"{role_label}: {msg['text']}\n"
                if msg["speaker"] == "user":
                    debug_history += f"   - {msg['enrichments']['Answerability']}\n   - {msg['enrichments']['Multi-Turn']}\n   - {msg['enrichments']['Question Type']}\n"
                history.append({"role": role_label.lower(), "content": msg["text"]})

            print()
            print("=" * 100)
            print(debug_history.strip())

            user_content = (
                f"Chat History\n\n{formatted_history}\n\nChat History\n\n"
                f"Generate the contextualized user query."
            )

            payload = {
                "model": VLLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.3,
            }

            url = VLLM_BASE_URL.rstrip("/") + "/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
            }

            resp = requests.post(url, headers=headers, data=json.dumps(payload))
            resp = resp.json()
            answer = resp["choices"][0]["message"]["content"]

            print(answer)
            print("=" * 100)
            print()
        break
