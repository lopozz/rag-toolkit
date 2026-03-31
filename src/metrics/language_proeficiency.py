"""
Custom RAGAS metric: LanguageProficiencyIT
------------------------------------------------
Evaluates the linguistic quality of a response written in Italian and
checks whether it follows a provided *system prompt* (tone, style, level
of formality, etc.).

Metric returns a **continuous score** in the range **0‒1** where:
    • 1.0  → flawless Italian, perfect compliance with system prompt
    • 0.0  → very poor Italian / completely off‑prompt

Designed for *single‑turn* QA evaluations.
"""

import logging
import typing as t
from dataclasses import dataclass, field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt
from langchain_core.callbacks import Callbacks
from pydantic import BaseModel, Field, validator

LOGGER = logging.getLogger(__name__)


class LanguageProficiencyInput(BaseModel):
    """Input given to the LLM for scoring language quality."""

    response: str = Field(..., description="The assistant's answer in Italian.")
    system_prompt: str = Field(
        ..., description="The system prompt that defines expected tone/style."
    )


class ScoreWithReason(BaseModel):
    """LLM output: numeric score ∈ [0,1] + justification."""

    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(...)

    # Clip any float that might be slightly out of range due to LLM fuzziness
    @validator("score", pre=True)
    def _clip_score(cls, v):  # noqa: D401
        try:
            v = float(v)
        except Exception as exc:  # pragma: no cover
            raise ValueError("score must be a float") from exc
        return max(0.0, min(1.0, v))


class LanguageProficiencyPrompt(
    PydanticPrompt[LanguageProficiencyInput, ScoreWithReason]
):
    """LLM eval prompt – returns ScoreWithReason."""

    instruction = (
        "Sei un valutatore linguistico competente in italiano. "
        "Valuta la qualità linguistica e la conformità al prompt di sistema del seguente testo.\n\n"
        "1. *Correttezza grammaticale e sintattica*\n"
        "2. *Ricchezza lessicale e scorrevolezza*\n"
        "3. *Aderenza alle istruzioni del prompt di sistema (tono, formalità, persona, ecc.)*\n\n"
        "Assegna un punteggio complessivo tra 0 e 1 (due decimali sono sufficienti) dove 1 corrisponde "
        "a un italiano impeccabile e perfettamente in conformità con il prompt; e 0 corrisponde a "
        "un italiano molto scarso e completamente fuori luogo.\n"
        "Fornisci una breve spiegazione (<40 parole).\n\n"
        "Restituisci un JSON con le chiavi *score* e *reason*."
    )

    input_model = LanguageProficiencyInput
    output_model = ScoreWithReason

    # A single good/bad example helps steer the LLM output.
    examples = [
        (
            LanguageProficiencyInput(
                response=(
                    "Salve, di seguito troverai le informazioni richieste. "
                    "Il punto di ebollizione dell'acqua è di 100 °C al livello del mare. "
                    "Non esitare a fare ulteriori domande in caso di dubbi."
                ),
                system_prompt=(
                    "Sei un assistente AI professionale che risponde in tono formale e cortese, in italiano."
                ),
            ),
            ScoreWithReason(
                score=0.95, reason="Italiano corretto e formale, aderente al prompt."
            ),
        ),
        (
            LanguageProficiencyInput(
                response="yo bro l'acqua bolle tipo a cento gradi, ciao",
                system_prompt=(
                    "Sei un assistente AI professionale che risponde in tono formale e cortese, in italiano."
                ),
            ),
            ScoreWithReason(
                score=0.15,
                reason="Registro colloquiale e errori di punteggiatura, non aderente al tono richiesto.",
            ),
        ),
    ]


@dataclass
class LanguageProficiencyIT(MetricWithLLM, SingleTurnMetric):
    """Measure Italian language proficiency & prompt‑compliance of a response."""

    name: str = "language_proficiency_it"
    system_prompt: str = "Sei un assistente AI professionale che risponde in tono formale e cortese, in italiano."

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"response"}  # with system_prompt it doesn't work
        }
    )

    output_type: MetricOutputType = MetricOutputType.CONTINUOUS
    proficiency_prompt: PydanticPrompt = field(
        default_factory=LanguageProficiencyPrompt
    )
    max_retries: int = 1

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        """Compute the metric value for one SingleTurnSample."""

        assert self.llm is not None, "LLM must be provided for LanguageProficiencyIT"

        row = sample.to_dict()
        response_text = row.get("response", "").strip()
        if not response_text:
            raise ValueError("'response' is empty, cannot score language proficiency.")

        # Build input for the evaluation LLM
        prompt_input = LanguageProficiencyInput(
            response=response_text,
            system_prompt=self.system_prompt,
        )

        # Query the evaluation LLM via the ragas PydanticPrompt helper
        evaluation: ScoreWithReason = await self.proficiency_prompt.generate(
            llm=self.llm, data=prompt_input, callbacks=callbacks
        )

        score = float(evaluation.score)
        LOGGER.debug(
            "Language proficiency score: %s (reason: %s)", score, evaluation.reason
        )
        return score


language_proficiency_it = LanguageProficiencyIT()
