"""
**FormatComplianceIT** – checks whether the answer respects any *output‑
   format instructions* present in the last user query (e.g. “respond in
   JSON”, “give a Markdown table”).
"""

import logging
import typing as t
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt
from langchain_core.callbacks import Callbacks

LOGGER = logging.getLogger(__name__)


class ScoreWithReason(BaseModel):
    """LLM output: numeric score ∈ [0,1] + short justification."""

    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(...)

    # Clip small numeric overflow/underflow from the LLM.
    @validator("score", pre=True)
    def _clip_score(cls, v):  # noqa: D401
        return max(0.0, min(1.0, float(v)))


class FormatComplianceInput(BaseModel):
    user_query: str
    response: str


class FormatCompliancePrompt(PydanticPrompt[FormatComplianceInput, ScoreWithReason]):
    instruction: str = (
        "Valuta quanto la *risposta* segue *le istruzioni di formato* contenute nell'ultima query dell'utente.\n"
        'Se la query contiene indicazioni esplicite (es. "rispondi in JSON con campi x, y", "usa un elenco puntato Markdown", "scrivi solamente tre frasi"), controlla che la risposta rispetti esattamente il formato richiesto.\n'
        "Rendi un JSON con: `score` (0‒1, due decimali) e `reason` (≤40 parole)."
    )

    input_model = FormatComplianceInput
    output_model = ScoreWithReason

    examples = [
        # Example 1 – compliant JSON
        (
            FormatComplianceInput(
                user_query='Rispondi in JSON: {"temperatura": numero}',
                response='{"temperatura": 100}',
            ),
            ScoreWithReason(score=1.0, reason="Formato JSON esatto come richiesto."),
        ),
        # Example 2 – wrong format
        (
            FormatComplianceInput(
                user_query="Per favore rispondi con una lista puntata Markdown.",
                response="La temperatura è 100 °C al livello del mare.",
            ),
            ScoreWithReason(score=0.0, reason="Manca la lista puntata Markdown."),
        ),
        # Example 3 – partial format
        (
            FormatComplianceInput(
                user_query="Per favore rispondi con una lista puntata Markdown.",
                response="Ci sono diverse cose da sapere sull'acqua, un elemento molto interessante.\n- il punto di ebollizione è 100 gradi\n- ma può variare con l'altitudine",
            ),
            ScoreWithReason(
                score=0.6,
                reason="La risposta contiene l'elenco puntato ma anche una sezione di testo libero.",
            ),
        ),
    ]


@dataclass
class FormatComplianceIT(MetricWithLLM, SingleTurnMetric):
    """Scores adherence to explicit output‑format instructions in the user query."""

    name: str = "format_compliance_it"

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
    )
    output_type: MetricOutputType = MetricOutputType.CONTINUOUS
    compliance_prompt: PydanticPrompt = field(default_factory=FormatCompliancePrompt)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM must be set for FormatComplianceIT"
        user_query = sample.user_input.strip()
        response = sample.response.strip()
        if not response:
            raise ValueError(
                "'response' è vuoto: impossibile valutare la compliance di formato."
            )

        prompt_input = FormatComplianceInput(user_query=user_query, response=response)
        result: ScoreWithReason = await self.compliance_prompt.generate(
            llm=self.llm, data=prompt_input, callbacks=callbacks
        )
        LOGGER.debug("FormatCompliance score %.2f – %s", result.score, result.reason)
        return float(result.score)


format_compliance_it = FormatComplianceIT()
