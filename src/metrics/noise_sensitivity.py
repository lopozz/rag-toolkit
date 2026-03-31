import os
import sys
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np

from ragas.dataset_schema import SingleTurnSample

from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

from langchain_core.callbacks import Callbacks

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from src.metrics.faithfulness import (
    NLIStatementInputIT,
    NLIStatementPromptIT,
    StatementGeneratorInputIT,
    StatementGeneratorPromptIT,
)

logger = logging.getLogger(__name__)


@dataclass
class NoiseSensitivityIT(MetricWithLLM, SingleTurnMetric):
    name: str = "noise_sensitivity_it"
    mode: t.Literal["relevant", "irrelevant"] = "relevant"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "reference",
                "retrieved_contexts",
            }
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS
    nli_statements_prompt: PydanticPrompt = field(default_factory=NLIStatementPromptIT)
    statement_generator_prompt: PydanticPrompt = field(
        default_factory=StatementGeneratorPromptIT
    )
    max_retries: int = 1

    def __post_init__(self):

        if self.mode not in {"relevant", "irrelevant"}:
            raise ValueError(
                f"Invalid argument passed for 'mode': {self.mode}. Must be 'relevant' or 'irrelevant'."
            )

    async def _evaluate_statement_faithfulness(
        self, statements: t.List[str], context: str, callbacks: Callbacks
    ) -> t.List[int]:
        assert self.llm is not None, "LLM is not set"

        verdicts = await self.nli_statements_prompt.generate(
            data=NLIStatementInputIT(context=context, statements=statements),
            llm=self.llm,
            callbacks=callbacks,
        )

        verdict_list = [
            1 if statement.verdict else 0 for statement in verdicts.statements
        ]
        return verdict_list

    async def _decompose_answer_into_statements(
        self, text: str, question: str, callbacks: Callbacks
    ) -> t.List[str]:
        assert self.llm is not None, "LLM is not set"

        statements = await self.statement_generator_prompt.generate(
            llm=self.llm,
            data=StatementGeneratorInputIT(question=question, answer=text),
            callbacks=callbacks,
        )
        statements = statements.statements
        return statements

    def _compute_score(self, answers: t.Dict) -> float:
        # relevant retrievals
        relevant_retrieved = np.max(
            answers["retrieved2ground_truth"], axis=0, keepdims=True
        )
        relevant_faithful = np.max(
            relevant_retrieved & answers["retrieved2answer"], axis=1
        )

        # irrelevant retrievals
        irrelevant_retrieved = ~np.max(
            answers["retrieved2ground_truth"], axis=0, keepdims=True
        )
        irrelevant_faithful = np.max(
            irrelevant_retrieved & answers["retrieved2answer"], axis=1
        )

        # to keep them exclusive
        irrelevant_faithful &= ~relevant_faithful

        incorrect = ~answers["ground_truth2answer"]
        noise_sensitivity_in_relevant = np.mean(relevant_faithful & incorrect)
        noise_sensitivity_in_irrelevant = np.mean(irrelevant_faithful & incorrect)

        if self.mode == "irrelevant":
            return noise_sensitivity_in_irrelevant

        return noise_sensitivity_in_relevant

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        if "reference" not in row or not row["reference"]:
            raise ValueError(
                "reference is missing in the test sample. Please add reference to the test sample."
            )

        if "user_input" not in row or not row["user_input"]:
            raise ValueError(
                "user_input is missing in the test sample. Please add user_input to the test sample."
            )

        if "response" not in row or not row["response"]:
            raise ValueError(
                "response is missing in the test sample. Please add response to the test sample."
            )

        if "retrieved_contexts" not in row or not row["retrieved_contexts"]:
            raise ValueError(
                "retrieved_contexts is missing in the test sample. Please add retrieved_contexts to the test sample."
            )

        gt_statements = await self._decompose_answer_into_statements(
            row["reference"], row["user_input"], callbacks
        )
        ans_statements = await self._decompose_answer_into_statements(
            row["response"], row["user_input"], callbacks
        )
        gt_verdictslist = []
        ans_verdictslist = []

        for ctx in row["retrieved_contexts"]:
            verdicts = await self._evaluate_statement_faithfulness(
                gt_statements, ctx, callbacks
            )
            gt_verdictslist.append(np.array(verdicts))

            verdicts = await self._evaluate_statement_faithfulness(
                ans_statements, ctx, callbacks
            )
            ans_verdictslist.append(np.array(verdicts))

        answers = {}
        answers["retrieved2ground_truth"] = np.array(gt_verdictslist).T
        answers["retrieved2answer"] = np.array(ans_verdictslist).T
        answers["ground_truth2answer"] = np.array(
            await self._evaluate_statement_faithfulness(
                ans_statements, row["reference"], callbacks
            )
        )
        answers["ground_truth2answer"] = np.array([answers["ground_truth2answer"]])
        answers = {k: v.astype(bool) for k, v in answers.items()}
        return self._compute_score(answers)


noise_sensitivity_it = NoiseSensitivityIT(mode="irrelevant")
