"""Shared retry/backoff and call pacing for every LLM provider adapter.

Why this exists: a single `submit()` fires several LLM calls back to back
(triage, retrieval-grounded diagnosis, remediation). On free or low-tier plans
that trips the provider's requests-per-second / tokens-per-minute cap, and the
provider answers HTTP 429. Neither `langchain-mistralai` (it only retries
connection errors, not 429s) nor the raw SDKs handle that consistently, so the
policy lives here, once, and every adapter goes through it.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from ..exceptions import RateLimitError

logger = logging.getLogger("opscopilot")

T = TypeVar("T")

# 429 = rate limited; 5xx = transient upstream trouble. 4xx auth/validation
# errors are deliberately absent: retrying a bad key only wastes time.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status from httpx, openai, or google-genai exceptions."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    value = getattr(getattr(exc, "response", None), "status_code", None)
    return value if isinstance(value, int) else None


# Google puts the wait in the body ("Please retry in 34.7s." / "retryDelay": "34s"),
# not in a Retry-After header, so read both.
_RETRY_IN_RE = re.compile(r"retry(?:Delay)?['\"]?\s*(?:in|:)\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.I)
# A per-day quota or a zero quota cannot recover within a retry window.
_HARD_QUOTA_RE = re.compile(r"per[\s_]?day|limit:\s*0(?![\d.])", re.I)
_QUOTA_ID_RE = re.compile(r"quotaId['\"]?\s*:\s*['\"]([\w.-]+)")
_LIMIT_RE = re.compile(r"limit:\s*(\d+)", re.I)


def _text(exc: BaseException) -> str:
    try:
        return str(exc)
    except Exception:  # pragma: no cover - a broken __str__ must not break retrying
        return ""


def retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds the server asked us to wait: `Retry-After` header, else the body."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        try:
            raw = headers.get("retry-after")
            if raw is not None:
                return float(raw)
        except (TypeError, ValueError):
            pass
    match = _RETRY_IN_RE.search(_text(exc))
    return float(match.group(1)) if match else None


def is_quota_exhausted(exc: BaseException) -> bool:
    """True for a 429 that names a daily or zero quota (waiting a minute won't help)."""
    return status_of(exc) == 429 and bool(_HARD_QUOTA_RE.search(_text(exc)))


def quota_hint(exc: BaseException) -> str:
    """Short, human-readable reason for a 429, pulled from the provider's message."""
    text = _text(exc)
    parts: list[str] = []
    quota_id = _QUOTA_ID_RE.search(text)
    if quota_id:
        parts.append(f"quota={quota_id.group(1)}")
    limit = _LIMIT_RE.search(text)
    if limit:
        parts.append(f"limit={limit.group(1)}")
    if not parts:
        parts.append(" ".join(text.split())[:200])
    return " ".join(parts)


def is_retryable(exc: BaseException) -> bool:
    if status_of(exc) not in _RETRYABLE_STATUS:
        return False
    # OpenAI reports "you are out of credit" as a 429 too. Waiting won't fix it.
    return getattr(exc, "code", None) != "insufficient_quota"


class RetryPolicy:
    """Backoff on retryable failures, plus an optional minimum gap between calls."""

    def __init__(
        self,
        max_attempts: int = 6,
        min_interval: float = 0.0,
        max_delay: float = 60.0,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_attempts = max(1, max_attempts)
        self.min_interval = max(0.0, min_interval)
        self.max_delay = max_delay
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _pace(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:  # reserve a slot, then sleep outside the lock
            now = self._clock()
            start = max(now, self._next_slot)
            self._next_slot = start + self.min_interval
        if start > now:
            self._sleep(start - now)

    def call(self, provider: str, fn: Callable[[], T]) -> T:
        for attempt in range(1, self.max_attempts + 1):
            self._pace()
            try:
                return fn()
            except Exception as exc:
                if not is_retryable(exc):
                    raise
                hint = quota_hint(exc) if status_of(exc) == 429 else ""
                if is_quota_exhausted(exc):
                    # Fail fast: sleeping through 6 backoffs cannot fix a daily quota.
                    raise RateLimitError(provider, attempt, hint, quota_exhausted=True) from exc
                if attempt == self.max_attempts:
                    if status_of(exc) == 429:
                        raise RateLimitError(provider, attempt, hint) from exc
                    raise
                hinted = retry_after_seconds(exc)
                if hinted is not None and hinted > self.max_delay:
                    detail = f"{hint} (server asked us to wait {hinted:.0f}s)".strip()
                    raise RateLimitError(provider, attempt, detail, quota_exhausted=True) from exc
                delay = hinted if hinted is not None else float(2**attempt)
                delay = min(delay, self.max_delay) + random.uniform(0, 1)
                logger.warning(
                    "opscopilot: %s returned HTTP %s (attempt %d/%d)%s; retrying in %.1fs",
                    provider,
                    status_of(exc),
                    attempt,
                    self.max_attempts,
                    f" [{hint}]" if hint else "",
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover
