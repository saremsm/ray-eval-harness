import logging

import pytest
from hooks import EarlyStoppingHook, LoggingHook


class TestEarlyStoppingHook:
    def test_stops_on_trigger(self):
        hook = EarlyStoppingHook(triggers=["Paris"])
        hook.on_delta(" Paris", "The capital is Paris")
        assert hook.should_stop("The capital is Paris")
        assert hook.triggered_by == "Paris"

    def test_does_not_stop_without_trigger(self):
        hook = EarlyStoppingHook(triggers=["Paris"])
        hook.on_delta(" London", "The capital is London")
        assert not hook.should_stop("The capital is London")
        assert hook.triggered_by is None

    def test_case_insensitive_default(self):
        hook = EarlyStoppingHook(triggers=["paris"])
        assert hook.should_stop("The answer is Paris")

    def test_case_sensitive_no_match(self):
        hook = EarlyStoppingHook(triggers=["paris"], case_sensitive=True)
        assert not hook.should_stop("The answer is Paris")

    def test_case_sensitive_match(self):
        hook = EarlyStoppingHook(triggers=["Paris"], case_sensitive=True)
        assert hook.should_stop("The answer is Paris")

    def test_multiple_triggers_fires_on_first_match(self):
        hook = EarlyStoppingHook(triggers=["London", "Paris"])
        assert hook.should_stop("Paris is the capital")
        assert hook.triggered_by in {"London", "Paris"}

    def test_triggered_by_none_before_match(self):
        hook = EarlyStoppingHook(triggers=["Paris"])
        assert hook.triggered_by is None

    def test_trigger_split_across_deltas(self):
        """Regression: a trigger straddling the previous scan boundary must still."""
        hook = EarlyStoppingHook(triggers=["Paris"])
        assert not hook.should_stop("The capital is Par")   # scan advances
        assert hook.should_stop("The capital is Paris")     # completes across boundary
        assert hook.triggered_by == "Paris"

    def test_stays_triggered_on_subsequent_calls(self):
        hook = EarlyStoppingHook(triggers=["Paris"])
        assert hook.should_stop("Paris")
        assert hook.should_stop("Paris and then more text")

    def test_empty_triggers_never_stop(self):
        hook = EarlyStoppingHook(triggers=[])
        assert not hook.should_stop("anything at all")


class TestLoggingHook:
    def test_never_stops(self):
        hook = LoggingHook(task_id="t1")
        hook.on_delta("word", "some word")
        assert not hook.should_stop("some word")

    def test_token_count_increments(self):
        hook = LoggingHook(task_id="t1", log_every_n=1)
        for i in range(5):
            hook.on_delta(f"t{i}", f"accumulated {i}")
        assert hook._token_count == 5

    def test_log_every_n_respected(self):
        hook = LoggingHook(task_id="t2", log_every_n=3)
        for i in range(9):
            hook.on_delta(f"t{i}", f"acc {i}")
        assert hook._token_count == 9

    def test_level_is_configurable(self, caplog):
        hook = LoggingHook(task_id="t3", log_every_n=1, level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="hooks"):
            hook.on_delta("tok", "some tok")
        assert any("t3" in rec.message for rec in caplog.records), (
            "An INFO-level hook must emit visible records under an "
            "INFO-configured logger (the demo relies on this)."
        )
