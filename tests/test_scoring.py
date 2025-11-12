import pytest
from types_ import EvalTask, ScoringCondition
from scoring import (
    RubricScorer,
    contains_answer,
    answer_at_end,
    no_hallucinated_numbers,
    is_concise,
    mentions_topic,
)


@pytest.fixture
def numeric_task() -> EvalTask:
    return EvalTask(
        task_id="t_num",
        prompt="What is 6 times 7?",
        expected_answer="42",
    )


@pytest.fixture
def text_task() -> EvalTask:
    return EvalTask(
        task_id="t_txt",
        prompt="The capital of France is",
        expected_answer="Paris",
    )


@pytest.fixture
def no_answer_task() -> EvalTask:
    return EvalTask(
        task_id="t_none",
        prompt="Describe the sky.",
        expected_answer=None,
    )


class TestContainsAnswer:
    def test_present(self, numeric_task):
        assert contains_answer("The answer is 42.", numeric_task)

    def test_absent(self, numeric_task):
        assert not contains_answer("The answer is 43.", numeric_task)

    def test_negated_answer_fails(self, numeric_task):
        assert not contains_answer("The answer is not 42.", numeric_task)

    def test_negated_with_never(self, numeric_task):
        assert not contains_answer("This is never 42.", numeric_task)

    def test_un_negated_answer_passes(self, text_task):
        assert contains_answer("Paris is the capital.", text_task)

    def test_case_insensitive(self, text_task):
        assert contains_answer("paris is correct", text_task)

    def test_no_expected_answer(self, no_answer_task):
        assert not contains_answer("some response", no_answer_task)


class TestAnswerAtEnd:
    def test_answer_in_final_third(self, text_task):
        response = "Let me think about this carefully. Paris"
        assert answer_at_end(response, text_task)

    def test_answer_only_at_start(self, text_task):
        response = "Paris. " + "x " * 200
        assert not answer_at_end(response, text_task)

    def test_no_expected_answer(self, no_answer_task):
        assert not answer_at_end("some response", no_answer_task)


class TestNoHallucinatedNumbers:
    def test_only_correct_number(self, numeric_task):
        assert no_hallucinated_numbers("The answer is 42.", numeric_task)

    def test_extra_number_fails(self, numeric_task):
        assert not no_hallucinated_numbers(
            "It could be 41 or 42.", numeric_task
        )

    def test_non_numeric_expected_passes(self, text_task):
        assert no_hallucinated_numbers("Paris 123 456", text_task)

    def test_no_expected_answer(self, no_answer_task):
        assert no_hallucinated_numbers("some text 99", no_answer_task)


class TestIsConcise:
    def test_short(self, numeric_task):
        assert is_concise("42", numeric_task)

    def test_long(self, numeric_task):
        assert not is_concise(" ".join(["word"] * 200), numeric_task)

    def test_boundary_passes(self, numeric_task):
        assert is_concise(" ".join(["word"] * 149), numeric_task)


class TestMentionsTopic:
    def test_keyword_present(self, text_task):
        assert mentions_topic("France is in Europe", text_task)

    def test_completely_off_topic(self, text_task):
        assert not mentions_topic("I like pizza very much", text_task)


class TestRubricScorer:
    def test_score_in_range(self, text_task):
        scorer = RubricScorer()
        score, _ = scorer.score("Paris", text_task)
        assert 0.0 <= score <= 1.0

    def test_negated_answer_scores_low(self, numeric_task):
        scorer = RubricScorer()
        score, details = scorer.score("The answer is not 42.", numeric_task)
        assert details.get("contains_answer", 1.0) == 0.0

    def test_custom_conditions_override_defaults(self):
        task = EvalTask(
            task_id="t_c",
            prompt="Name a color.",
            expected_answer="blue",
            conditions=[
                ScoringCondition(
                    name="contains_answer",
                    weight=1.0,
                    description="Must contain 'blue'",
                )
            ],
        )
        scorer = RubricScorer()
        score, details = scorer.score("The sky is blue", task)
        assert score == 1.0
        assert list(details.keys()) == ["contains_answer"]

    def test_unknown_condition_scores_zero(self):
        task = EvalTask(
            task_id="t_unk",
            prompt="test",
            conditions=[
                ScoringCondition(
                    name="nonexistent_condition",
                    weight=1.0,
                    description="Does not exist",
                )
            ],
        )
        scorer = RubricScorer()
        score, details = scorer.score("test", task)
        assert score == 0.0
        assert details["nonexistent_condition"] == 0.0

    def test_aggregate_condition_scores_shape(self):
        breakdowns = [
            {"contains_answer": 0.5, "is_concise": 0.2},
            {"contains_answer": 0.0, "is_concise": 0.2},
            {"contains_answer": 0.5, "is_concise": 0.0},
        ]
        result = RubricScorer.aggregate_condition_scores(breakdowns)

        assert "contains_answer" in result
        assert "mean_awarded" in result["contains_answer"]
        assert "pass_rate" in result["contains_answer"]

        assert abs(result["contains_answer"]["mean_awarded"] - (1.0 / 3.0)) < 1e-9
        assert abs(result["contains_answer"]["pass_rate"] - (2.0 / 3.0)) < 1e-9
        assert abs(result["is_concise"]["mean_awarded"] - (0.4 / 3.0)) < 1e-9
        assert abs(result["is_concise"]["pass_rate"] - (2.0 / 3.0)) < 1e-9

    def test_aggregate_empty(self):
        assert RubricScorer.aggregate_condition_scores([]) == {}


class TestContainsAnswerNegation:
    """Regression tests for the expanded negation list."""

    @pytest.fixture
    def task(self) -> EvalTask:
        return EvalTask(
            task_id="t",
            prompt="What is 6 times 7?",
            expected_answer="42",
        )

    @pytest.mark.parametrize("response", [
        "He doesn't think the answer is 42.",
        "I don't believe it's 42.",
        "She didn't say it was 42.",
        "It won't be 42.",
        "It wouldn't be 42.",
        "It shouldn't be 42.",
        "It hasn't been 42.",
        "They haven't said 42.",
        "It's wrong to say 42.",
        "It's incorrect that the answer is 42.",
        "That's false; the answer is 42.",
    ])
    def test_negated_forms_all_caught(self, task, response):
        assert not contains_answer(response, task), (
            f"contains_answer should reject negated form: {response!r}"
        )

    def test_un_negated_passes(self, task):
        assert contains_answer("The answer is 42.", task)
