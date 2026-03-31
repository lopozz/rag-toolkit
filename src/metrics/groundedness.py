import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.callbacks import Callbacks
from langchain_core.prompt_values import StringPromptValue

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, MetricWithLLM, SingleTurnMetric

logger = logging.getLogger(__name__)


@dataclass
class ResponseGroundednessIT(MetricWithLLM, SingleTurnMetric):
    """Parameters:
    Score the groundedness of the response based on the retrieved contexts.

    Input:
        data: list of Dicts with keys: response, retrieved contexts
    Output:
        0.0: response is not grounded in the retrieved contexts
        0.5: response is partially grounded in the retrieved contexts
        1.0: response is fully grounded in the retrieved contexts
    """

    name: str = field(default="nv_response_groundedness_it", repr=True)  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "response",
                "retrieved_contexts",
            },
        }
    )

    template_groundedness1 = (
        "### Istruzioni\n\n"
        "Sei un esperto di livello mondiale incaricato di valutare il grado di attinenza di un'affermazione al contesto.\n"
        "Ti verranno forniti un'affermazione e un contesto.\n"
        "Il tuo compito è determinare se l'affermazione è supportata dal contesto.\n"
        "Segui le istruzioni seguenti:\n"
        "A. Se non c'è un contesto o un'affermazione, oppure se il contesto o l'affermazione sono vuoti, rispondi con 0.\n"
        "B. Se l'affermazione non è supportata dal contesto, rispondi con 0.\n"
        "C. Se l'affermazione è parzialmente supportata dal contesto, rispondi con 1.\n"
        "D. Se l'affermazione è pienamente supportata dal contesto, rispondi con 2.\n"
        "Devi fornire solo un punteggio: 0, 1 o 2, nient'altro.\n\n"
        "### Contesto:\n"
        "<{context}>\n\n"
        "### Affermazione:\n"
        "<{response}>\n\n"
        "Analizzando Contesto e Affermazione, il punteggio di Aderenza è "
    )
    template_groundedness2 = (
        "As a specialist in assessing the strength of connections between statements and their given contexts, "
        "I will evaluate the level of support an assertion receives from the provided context. Follow these guidelines:\n\n"
        "* If the assertion is not supported or context is empty or assertion is empty, assign a score of 0.\n"
        "* If the assertion is partially supported, assign a score of 1.\n"
        "* If the assertion is fully supported, assign a score of 2.\n\n"
        "I will provide a rating of 0, 1, or 2, without any additional information.\n\n"
        "---\n**Context:**\n[{context}]\n\n"
        "**Assertion:**\n[{response}]\n\n"
        "Do not explain."
        "Based on the provided context and response, the Groundedness score is:"
    )
    template_groundedness2 = (
        "In qualità di specialista nella valutazione della correlazione tra affermazioni e contesti forniti, "
        "valuterò il livello di attinenza che un'affermazione ha sul contesto. Segui queste linee guida:\n\n"
        "* Se l'affermazione non è supportata, oppure il contesto o l'affermazione sono vuoti, assegna un punteggio di 0.\n"
        "* Se l'affermazione è parzialmente supportata, assegna un punteggio di 1.\n"
        "* Se l'affermazione è pienamente supportata, assegna un punteggio di 2.\n\n"
        "Fornirò solo un punteggio tra 0, 1 o 2, senza alcuna spiegazione.\n\n"
        "---\n**Contesto:**\n[{context}]\n\n"
        "**Affermazione:**\n[{response}]\n\n"
        "Non fornire spiegazioni.\n"
        "In base al contesto e all'affermazione forniti, il punteggio di è:"
    )

    retry = 5  # Number of retries if rating is not in the first 8 tokens.

    def process_score(self, response):
        for i in [2, 1, 0]:
            if str(i) in response:
                return i / 2
        return np.nan

    def average_scores(self, score0, score1):
        score = np.nan
        if score0 >= 0 and score1 >= 0:
            score = (score0 + score1) / 2
        else:
            score = max(score0, score1)
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.response is not None, "Response is not set"
        assert sample.retrieved_contexts is not None, "Retrieved Context is not set"

        if (sample.response.strip() == "") or (
            "\n".join(sample.retrieved_contexts).strip().strip() == ""
        ):
            return 0.0
        if sample.response.strip() == "\n".join(sample.retrieved_contexts).strip():
            return 1.0
        if sample.response.strip() in "\n".join(sample.retrieved_contexts).strip():
            return 1.0

        try:
            score0 = score1 = np.nan
            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_groundedness1.format(
                        context="\n".join(sample.retrieved_contexts)[:7000],
                        response=sample.response,
                    )
                )
                req = self.llm.agenerate_text(
                    # req = self.llm.agenerate(
                    formatted_prompt,
                    n=1,
                    # temperature=0.1,
                )
                resp = await req
                score0 = self.process_score(resp.generations[0][0].text)
                if score0 == score0:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_groundedness2.format(
                        context="\n".join(sample.retrieved_contexts)[:7000],
                        response=sample.response,
                    )
                )
                req = self.llm.agenerate_text(
                    # req = self.llm.agenerate(
                    formatted_prompt,
                    n=1,
                    # temperature=0.1,
                )
                resp = await req
                score1 = self.process_score(resp.generations[0][0].text)
                if score1 == score1:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            score = self.average_scores(score0, score1)

        except Exception as e:
            print(
                f"An error occurred: {e}. Skipping a sample by assigning it nan score."
            )
            score = np.nan

        return score


groundedness_it = ResponseGroundednessIT()
