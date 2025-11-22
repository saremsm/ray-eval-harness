from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LoggingHook:
    """Logs generated tokens."""

    def __init__(
        self,
        task_id: str,
        log_every_n: int = 5,
        level: int = logging.DEBUG,
    ) -> None:
        self.task_id = task_id
        self.log_every_n = log_every_n
        self.level = level
        self._token_count = 0

    def on_delta(self, delta: str, accumulated: str) -> None:
        self._token_count += 1
        if self._token_count % self.log_every_n == 0:
            preview = accumulated[-40:].replace("\n", " ")
            logger.log(
                self.level,
                f"[{self.task_id}] step {self._token_count}: ...{preview!r}",
            )

    def should_stop(self, accumulated: str) -> bool:
        return False


class EarlyStoppingHook:
    """Stops generation when any trigger phrase appears in the output."""

    def __init__(
        self,
        triggers: list[str],
        case_sensitive: bool = False,
    ) -> None:
        self.triggers = triggers
        self.case_sensitive = case_sensitive
        self.triggered_by: str | None = None
        self._scanned_upto = 0
        self._max_trigger_len = max((len(t) for t in triggers), default=0)

    def on_delta(self, delta: str, accumulated: str) -> None:
        # detection is in should_stop
        pass

    def should_stop(self, accumulated: str) -> bool:
        if self.triggered_by is not None:
            return True
        if not self.triggers:
            return False

        # overlap window: a trigger can straddle the previous scan boundary
        start = max(0, self._scanned_upto - self._max_trigger_len + 1)
        window = accumulated[start:]
        haystack = window if self.case_sensitive else window.lower()

        for trigger in self.triggers:
            needle = trigger if self.case_sensitive else trigger.lower()
            if needle in haystack:
                self.triggered_by = trigger
                return True

        self._scanned_upto = len(accumulated)
        return False
