"""Budget guard — daily request counter with hard cap to prevent accidental Gemini charges.

Safety model:
  - Gemini free tier (no billing account linked) = Google rejects requests past quota.
  - This guard adds a LOCAL hard cap so we stop BEFORE hitting Google's limit,
    avoiding even failed requests and giving clear feedback.
  - The counter resets daily (midnight local time) and persists to a small JSON file
    so restarts don't reset the count mid-day.

If you ever link a billing account to your GCP project, this guard is your
last line of defense against surprise charges.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DAILY_CAP = 50  # conservative: Gemini 2.5 Flash free = 250 RPD
_STATE_FILENAME = ".gemini_usage.json"

_lock = threading.Lock()


class BudgetExceeded(Exception):
    """Raised when the daily Gemini request cap is reached."""


class BudgetGuard:
    """Thread-safe daily request counter with a hard cap.

    Usage:
        guard = BudgetGuard(project_root, daily_cap=50)
        guard.check()    # raises BudgetExceeded if over cap
        guard.record()   # call after a successful API request
        guard.remaining  # how many requests left today
    """

    def __init__(self, project_root: Path, daily_cap: int = _DEFAULT_DAILY_CAP):
        self._path = project_root / _STATE_FILENAME
        self._daily_cap = daily_cap
        self._today: str = ""
        self._count: int = 0
        self._load()

    # -- public API ----------------------------------------------------------

    @property
    def remaining(self) -> int:
        self._maybe_reset()
        return max(0, self._daily_cap - self._count)

    @property
    def count_today(self) -> int:
        self._maybe_reset()
        return self._count

    @property
    def daily_cap(self) -> int:
        return self._daily_cap

    def check(self) -> None:
        """Raise BudgetExceeded if the daily cap is reached."""
        self._maybe_reset()
        if self._count >= self._daily_cap:
            raise BudgetExceeded(
                f"Daily Gemini request cap reached ({self._daily_cap}). "
                f"Increase free_tier.daily_request_cap in pipeline.yaml or wait until tomorrow."
            )

    def record(self) -> None:
        """Record one API request. Call AFTER a successful call."""
        with _lock:
            self._maybe_reset()
            self._count += 1
            self._save()
            logger.info("Gemini usage: %d / %d today", self._count, self._daily_cap)

    def status_line(self) -> str:
        self._maybe_reset()
        return f"{self._count} / {self._daily_cap} Gemini requests today"

    # -- persistence ---------------------------------------------------------

    def _maybe_reset(self) -> None:
        today = date.today().isoformat()
        if self._today != today:
            with _lock:
                if self._today != today:
                    self._today = today
                    self._count = 0
                    self._save()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._today = data.get("date", "")
                self._count = data.get("count", 0)
            except (json.JSONDecodeError, KeyError):
                self._today = ""
                self._count = 0
        self._maybe_reset()

    def _save(self) -> None:
        self._path.write_text(json.dumps({"date": self._today, "count": self._count}))
