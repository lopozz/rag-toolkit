import os
import sys
from vllm import SamplingParams
import gradio as gr
from openai import OpenAI


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prompts import SYS_PROMPT
from src.utils import create_retrieval_context_section


client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "google/gemma-3-1b-it"
sparams = SamplingParams(temperature=0.1, max_tokens=3000)

DOCS = [
    "Guglielmo Giovanni Maria Marconi (Bologna, 25 aprile 1874[1] – Roma, 20 luglio 1937) è stato un inventore, imprenditore e politico italiano.",
    "A lui si deve lo sviluppo di un efficace sistema di telecomunicazione a distanza via onde radio, ovvero la telegrafia senza fili o radiotelegrafo",
    "La rivendicazione dell'invenzione della radio di Marconi fu sempre contestata da Nikola Tesla. Nel 1943 una sentenza della Corte suprema degli Stati Uniti d'America riconosce a Nikola Tesla la paternità del brevetto della radio.",
    "Tesla, un inventore e ingegnere elettrotecnico di origini serbe, aveva già gettato le basi per la trasmissione wireless prima che Marconi intraprendesse i suoi esperimenti.",
]


def respond(message, history):
    context = create_retrieval_context_section(DOCS)

    # OpenAI-style messages
    msgs = [{"role": "system", "content": SYS_PROMPT}]
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"{context}\n\n{message}"})
    print(msgs)
    print(history)

    # Request streaming completion
    stream = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        temperature=sparams.temperature,
        max_tokens=sparams.max_tokens,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        partial += delta
        yield partial


chat = gr.ChatInterface(
    fn=respond, examples=["Marconi iniziò prima di Tesla a sperimentare con la radio"]
)

if __name__ == "__main__":
    chat.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        auth=("pco", "pco"),  # ← add a simple login if you like
    )
