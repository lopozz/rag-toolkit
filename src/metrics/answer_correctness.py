import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness import (
    StatementGeneratorInput,
    StatementGeneratorOutput,
    StatementGeneratorPrompt,
)
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.metrics.utils import fbeta_score
from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig

from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class QuestionAnswerGroundTruth(BaseModel):
    question: str
    answer: list[str]
    ground_truth: list[str]


class StatementsWithReason(BaseModel):
    statement: str
    reason: str


class ClassificationWithReason(BaseModel):
    TP: list[StatementsWithReason]
    FP: list[StatementsWithReason]
    FN: list[StatementsWithReason]


class CorrectnessClassifierIT(
    PydanticPrompt[QuestionAnswerGroundTruth, ClassificationWithReason]
):
    instruction = (
        "Data una risposta generata e una verità di riferimento, analizza ciascuna affermazione nelle risposte "
        "e classificale in una delle seguenti categorie:\n"
        "- TP (vero positivo): affermazioni presenti nella risposta generata e supportate direttamente da una o più affermazioni della verità di riferimento;\n"
        "- FP (falso positivo): affermazioni presenti nella risposta generata ma non supportate direttamente da alcuna affermazione della verità di riferimento;\n"
        "- FN (falso negativo): affermazioni presenti nella verità di riferimento ma assenti nella risposta generata.\n"
        "Ogni affermazione può appartenere solo a una categoria. Fornisci una motivazione per ogni classificazione."
    )
    input_model = QuestionAnswerGroundTruth
    output_model = ClassificationWithReason
    examples = [
        (
            QuestionAnswerGroundTruth(
                question="Cosa alimenta il Sole e qual è la sua funzione principale?",
                answer=[
                    "Il Sole è alimentato dalla fissione nucleare, simile ai reattori nucleari sulla Terra.",
                    "La funzione principale del Sole è fornire luce al sistema solare.",
                ],
                ground_truth=[
                    "Il Sole è alimentato dalla fusione nucleare, dove gli atomi di idrogeno si fondono per formare elio.",
                    "Questo processo di fusione nel nucleo del Sole rilascia una grande quantità di energia.",
                    "L'energia del Sole fornisce calore e luce, essenziali per la vita sulla Terra.",
                    "La luce solare ha un ruolo cruciale nel sistema climatico terrestre.",
                    "La luce solare contribuisce a guidare i modelli meteorologici e le correnti oceaniche.",
                ],
            ),
            ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="La funzione principale del Sole è fornire luce al sistema solare.",
                        reason="Questa affermazione è in parte supportata dalla verità di riferimento che menziona la luce del Sole e i suoi ruoli, anche se si concentra maggiornmente sull’energia del Sole.",
                    )
                ],
                FP=[
                    StatementsWithReason(
                        statement="Il Sole è alimentato dalla fissione nucleare, simile ai reattori nucleari sulla Terra.",
                        reason="Questa affermazione è errata e contraddice la verità di riferimento, che afferma che il Sole è alimentato dalla fusione nucleare.",
                    )
                ],
                FN=[
                    StatementsWithReason(
                        statement="Il Sole è alimentato dalla fusione nucleare, dove gli atomi di idrogeno si fondono per formare elio.",
                        reason="Questa descrizione accurata della fonte di energia del Sole non è inclusa nella risposta.",
                    ),
                    StatementsWithReason(
                        statement="Questo processo di fusione nel nucleo del Sole rilascia una grande quantità di energia.",
                        reason="Questo processo e la sua importanza non sono menzionati nella risposta.",
                    ),
                    StatementsWithReason(
                        statement="L'energia del Sole fornisce calore e luce, essenziali per la vita sulla Terra.",
                        reason="La risposta menziona solo la luce, omettendo gli aspetti essenziali del calore e della sua importanza per la vita, che sono presenti nella verità di riferimento.",
                    ),
                    StatementsWithReason(
                        statement="La luce del Sole ha un ruolo cruciale nel sistema climatico terrestre.",
                        reason="Questo impatto più ampio della luce solare sul sistema climatico terrestre non è affrontato nella risposta.",
                    ),
                    StatementsWithReason(
                        statement="La luce solare contribuisce a guidare i modelli meteorologici e le correnti oceaniche.",
                        reason="L'effetto della luce solare sui modelli meteorologici e sulle correnti oceaniche è omesso nella risposta.",
                    ),
                ],
            ),
        ),
        (
            QuestionAnswerGroundTruth(
                question="Qual è il punto di ebollizione dell'acqua?",
                answer=[
                    "Il punto di ebollizione dell'acqua è di 100 gradi Celsius al livello del mare"
                ],
                ground_truth=[
                    "Il punto di ebollizione dell'acqua è di 100 gradi Celsius (212 gradi Fahrenheit) al livello del mare.",
                    "Il punto di ebollizione dell'acqua può variare con l'altitudine.",
                ],
            ),
            ClassificationWithReason(
                TP=[
                    StatementsWithReason(
                        statement="Il punto di ebollizione dell'acqua è di 100 gradi Celsius al livello del mare",
                        reason="Questa affermazione è direttamente supportata dalla verità di riferimento, che specifica che il punto di ebollizione dell'acqua è di 100 gradi Celsius al livello del mare.",
                    )
                ],
                FP=[],
                FN=[
                    StatementsWithReason(
                        statement="Il punto di ebollizione dell'acqua può variare con l'altitudine.",
                        reason="Questa informazione aggiuntiva su come il punto di ebollizione dell'acqua possa variare con l'altitudine non è menzionata nella risposta.",
                    )
                ],
            ),
        ),
    ]


@dataclass
class AnswerCorrectnessIT(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity (in italian)

    Attributes
    ----------
    name: string
        The name of the metrics
    weights:
        a list of two weights corresponding to factuality and semantic similarity
        Defaults [0.75, 0.25]
    answer_similarity:
        The AnswerSimilarity object
    """

    name: str = "answer_correctness_it"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "reference"}
        }
    )
    output_type = MetricOutputType.CONTINUOUS
    correctness_prompt: PydanticPrompt = field(default_factory=CorrectnessClassifierIT)
    statement_generator_prompt: PydanticPrompt = field(
        default_factory=StatementGeneratorPrompt
    )
    weights: list[float] = field(default_factory=lambda: [1.0, 0.0])
    beta: float = 1.0
    answer_similarity: t.Optional[AnswerSimilarity] = None
    max_retries: int = 1

    def __post_init__(self):
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

        if type(self.beta) is not float:
            raise ValueError(
                "Beta must be a float. A beta > 1 gives more weight to recall, while beta < 1 favors precision."
            )

    def init(self, run_config: RunConfig):
        super().init(run_config)
        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(embeddings=self.embeddings)

    def _compute_statement_presence(
        self, prediction: ClassificationWithReason
    ) -> float:
        tp = len(prediction.TP)
        fp = len(prediction.FP)
        fn = len(prediction.FN)
        score = fbeta_score(tp, fp, fn, self.beta)
        return score

    async def _create_simplified_statements(
        self, question: str, text: str, callbacks: Callbacks
    ) -> StatementGeneratorOutput:
        assert self.llm is not None, "llm is not set"

        prompt_input = StatementGeneratorInput(question=question, answer=text)
        statements = await self.statement_generator_prompt.generate(
            llm=self.llm,
            data=prompt_input,
            callbacks=callbacks,
        )

        return statements

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        score = await self._ascore(row, callbacks)
        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM must be set"

        # extract the statements from the answer and the ground truth
        question = row["user_input"]
        statements: t.Dict[str, t.List[str]] = {}
        for item in ["response", "reference"]:
            statements_x = await self._create_simplified_statements(
                question, row[item], callbacks
            )
            statements_x = statements_x.statements
            statements[item] = statements_x

        if not all([val == [] for val in statements.values()]):
            ground_truth = [statement for statement in statements["reference"]]
            answer = [statement for statement in statements["response"]]
            answers = await self.correctness_prompt.generate(
                llm=self.llm,
                data=QuestionAnswerGroundTruth(
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth,
                ),
                callbacks=callbacks,
            )
            if answers is None:
                return np.nan

            f1_score = self._compute_statement_presence(answers)
        else:
            f1_score = 1.0

        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            assert self.answer_similarity is not None, "AnswerSimilarity must be set"

            similarity_score = await self.answer_similarity.ascore(
                row, callbacks=callbacks
            )

        score = np.average(
            [f1_score, similarity_score],
            weights=self.weights,
        )

        return float(score)


answer_correctness_it = AnswerCorrectnessIT()
