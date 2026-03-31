import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class StatementGeneratorInputIT(BaseModel):
    question: str = Field(description="La domanda a cui rispondere")
    answer: str = Field(description="La ripososta alla domanda")


class StatementGeneratorOutputIT(BaseModel):
    statements: t.List[str] = Field(description="L'affermazione generata")


class StatementGeneratorPromptIT(
    PydanticPrompt[StatementGeneratorInputIT, StatementGeneratorOutputIT]
):
    instruction = "Data una domanda e una risposta, analizza la complessità di ogni frase nella risposta. Scomponi ogni frase in una o più affermazioni autonome. Assicurati che nessun'affermazione contenga pronomi. Formatta i risultati in JSON."
    input_model = StatementGeneratorInputIT
    output_model = StatementGeneratorOutputIT
    examples = [
        (
            StatementGeneratorInputIT(
                question="Chi era Albert Einstein e per cosa è maggiormente conosciuto?",
                answer="Era un fisico teorico nato in Germania, ampiamente riconosciuto come uno dei fisici più grandi e influenti di tutti i tempi. Era maggiormente conosciuto per aver sviluppato la teoria della relatività, e contribuì anche in modo significativo allo sviluppo della teoria della meccanica quantistica.",
            ),
            StatementGeneratorOutputIT(
                statements=[
                    "Albert Einstein era un fisico teorico nato in Germania.",
                    "Albert Einstein è riconosciuto come uno dei fisici più grandi e influenti di tutti i tempi.",
                    "Albert Einstein era maggiormente conosciuto per aver sviluppato la teoria della relatività.",
                    "Albert Einstein contribuì anche in modo significativo allo sviluppo della teoria della meccanica quantistica.",
                ]
            ),
        )
    ]


class StatementFaithfulnessAnswerIT(BaseModel):
    statement: str = Field(
        ..., description="l'affermazione originale, parola per parola"
    )
    reason: str = Field(..., description="la motivazione del verdetto")
    verdict: int = Field(
        ..., description="il verdetto (0/1) sull'attendibilità dell'affermazione"
    )


class NLIStatementOutput(BaseModel):
    statements: t.List[StatementFaithfulnessAnswerIT]


class NLIStatementInputIT(BaseModel):
    context: str = Field(..., description="Il contesto della domanda")
    statements: t.List[str] = Field(..., description="Le affermazioni da valutare")


class NLIStatementPromptIT(PydanticPrompt[NLIStatementInputIT, NLIStatementOutput]):
    instruction = "Il tuo compito è valutare l'attendibilità di una serie di affermazioni basandoti su un contesto fornito. Per ciascuna affermazione, restituisci un verdetto pari a 1 se l'affermazione può essere direttamente dedotta dal contesto, oppure 0 se non può essere direttamente dedotta dal contesto."
    input_model = NLIStatementInputIT
    output_model = NLIStatementOutput
    examples = [
        (
            NLIStatementInputIT(
                context="""John è uno studente presso l'Università XYZ. Sta conseguendo una laurea in Informatica. È iscritto a diversi corsi questo semestre, tra cui Strutture Dati, Algoritmi e Gestione di Basi di Dati. John è uno studente diligente e dedica molto tempo allo studio e al completamento dei compiti. Spesso rimane fino a tardi in biblioteca per lavorare ai suoi progetti.""",
                statements=[
                    "John è iscritto al corso di Biologia.",
                    "John sta seguendo un corso di Intelligenza Artificiale.",
                    "John è uno studente dedicato.",
                    "John ha un lavoro part-time.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswerIT(
                        statement="John è iscritto al corso di Biologia.",
                        reason="Il corso di laurea di John è esplicitamente indicato come Informatica. Non c'è alcuna informazione che suggerisca che sia iscritto a Biologia.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswerIT(
                        statement="John sta seguendo un corso di Intelligenza Artificiale.",
                        reason="Il contesto elenca i corsi a cui John è iscritto, e Intelligenza Artificiale non è menzionato. Pertanto, non si può dedurre che stia seguendo quel corso.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswerIT(
                        statement="John è uno studente dedicato.",
                        reason="Il contesto afferma che dedica molto tempo allo studio e ai compiti, e che spesso rimane fino a tardi in biblioteca per lavorare ai suoi progetti, il che implica dedizione.",
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswerIT(
                        statement="John ha un lavoro part-time.",
                        reason="Nel contesto non vi è alcuna informazione sul fatto che John abbia un lavoro part-time.",
                        verdict=0,
                    ),
                ]
            ),
        ),
        (
            NLIStatementInputIT(
                context="La fotosintesi è un processo utilizzato da piante, alghe e alcuni batteri per convertire l'energia luminosa in energia chimica.",
                statements=[
                    "Albert Einstein era un genio.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswerIT(
                        statement="Albert Einstein era un genio.",
                        reason="Il contesto e l'affermazione non sono correlati.",
                        verdict=0,
                    )
                ]
            ),
        ),
    ]


@dataclass
class FaithfulnessIT(MetricWithLLM, SingleTurnMetric):
    name: str = "faithfulness_it"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
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

    async def _create_verdicts(
        self, row: t.Dict, statements: t.List[str], callbacks: Callbacks
    ) -> NLIStatementOutput:
        assert self.llm is not None, "llm must be set to compute score"

        contexts_str: str = "\n".join(row["retrieved_contexts"])
        verdicts = await self.nli_statements_prompt.generate(
            data=NLIStatementInputIT(context=contexts_str, statements=statements),
            llm=self.llm,
            callbacks=callbacks,
        )

        return verdicts

    async def _create_statements(
        self, row: t.Dict, callbacks: Callbacks
    ) -> StatementGeneratorOutputIT:
        assert self.llm is not None, "llm is not set"

        text, question = row["response"], row["user_input"]

        prompt_input = StatementGeneratorInputIT(question=question, answer=text)
        statements = await self.statement_generator_prompt.generate(
            llm=self.llm,
            data=prompt_input,
            callbacks=callbacks,
        )

        return statements

    def _compute_score(self, answers: NLIStatementOutput):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.statements
        )
        num_statements = len(answers.statements)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

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

        statements = await self._create_statements(row, callbacks)
        statements = statements.statements
        if statements == []:
            return np.nan

        verdicts = await self._create_verdicts(row, statements, callbacks)
        return self._compute_score(verdicts)


@dataclass
class FaithfulnesswithHHEM(FaithfulnessIT):
    name: str = "faithfulness_with_hhem"
    device: str = "cpu"
    batch_size: int = 10

    def __post_init__(self):
        try:
            from transformers import AutoModelForSequenceClassification  # type: ignore
        except ImportError:
            raise ImportError(
                "Huggingface transformers must be installed to use this feature, try `pip install transformers`"
            )
        self.nli_classifier = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        )
        self.nli_classifier.to(self.device)
        super().__post_init__()

    def _create_pairs(
        self, row: t.Dict, statements: t.List[str]
    ) -> t.List[t.Tuple[str, str]]:
        """
        create pairs of (question, answer) from the row
        """
        premise = "\n".join(row["retrieved_contexts"])
        pairs = [(premise, statement) for statement in statements]
        return pairs

    def _create_batch(
        self, pairs: t.List[t.Tuple[str, str]]
    ) -> t.Generator[t.List[t.Tuple[str, str]], None, None]:
        length_of_pairs = len(pairs)
        for ndx in range(0, length_of_pairs, self.batch_size):
            yield pairs[ndx : min(ndx + self.batch_size, length_of_pairs)]

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        statements = await self._create_statements(row, callbacks)
        statements = statements.statements
        if statements == []:
            return np.nan

        scores = []
        pairs = self._create_pairs(row, statements)
        for input_pairs in self._create_batch(pairs):  # to avoid OOM
            batch_scores = (
                self.nli_classifier.predict(input_pairs).cpu().detach().round()
            )
            # convert tensor to list of floats
            scores.extend(batch_scores.tolist())

        return sum(scores) / len(scores)


faithfulness_it = FaithfulnessIT()
