"""What the platform is spending, and on what.

Recording happens at the provider seam because that is the only choke point
every model call passes through. Instrumenting individual features would leave
gaps the moment a new one is added, and gaps are worse than no measurement —
they produce a number that looks complete and is not.

Two deliberate limits, stated because a cost figure that hides its assumptions
invites misuse:

  * Token counts come from the provider's own response where it reports them,
    and are estimated from character length where it does not. Estimated rows
    are flagged, never silently mixed in.
  * Prices are a local table, not a live feed. They drift, they vary by region
    and agreement, and a figure derived from them is an indication of relative
    cost between features — not an invoice.

Events are appended to a file per day. Nothing is aggregated at write time, so a
question nobody thought to ask in advance can still be answered later.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config import get_settings

log = logging.getLogger(__name__)

# Indicative prices per million tokens. Override in usage_prices.json inside the
# data directory; the shipped values are conservative placeholders.
DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "default":                    {"input": 1.00, "output": 3.00},
    "gpt-4.1-nano":               {"input": 0.10, "output": 0.40},
    "gpt-4o-mini":                {"input": 0.15, "output": 0.60},
    "gpt-4o":                     {"input": 2.50, "output": 10.00},
    "claude-sonnet":              {"input": 3.00, "output": 15.00},
    "claude-haiku":               {"input": 0.80, "output": 4.00},
    "titan-embed":                {"input": 0.02, "output": 0.00},
    "text-embedding-3-small":     {"input": 0.02, "output": 0.00},
    "text-embedding-3-large":     {"input": 0.13, "output": 0.00},
    "local":                      {"input": 0.00, "output": 0.00},
}


@dataclass
class UsageEvent:
    at: str
    operation: str          # extraction | answering | verification | agent | embedding | training
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_tokens: bool
    cost: float
    duration_ms: int
    domain: str = ""
    principal: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class UsageRecorder:
    def __init__(self, data_dir: Path) -> None:
        self.dir = Path(data_dir) / "usage"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prices: dict[str, dict[str, float]] | None = None

    # ------------------------------------------------------------- prices
    @property
    def prices(self) -> dict[str, dict[str, float]]:
        if self._prices is None:
            path = self.dir.parent / "usage_prices.json"
            prices = dict(DEFAULT_PRICES)
            if path.exists():
                try:
                    prices.update(json.loads(path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    log.warning("usage_prices.json is not valid JSON; using defaults.")
            self._prices = prices
        return self._prices

    def price_for(self, model: str) -> dict[str, float]:
        """Longest matching prefix wins, so 'gpt-4o-mini-2024-07-18' does not
        pick up the far more expensive 'gpt-4o' entry."""
        model_l = (model or "").lower()
        best, best_len = self.prices["default"], -1
        for name, price in self.prices.items():
            if name == "default":
                continue
            if name.lower() in model_l and len(name) > best_len:
                best, best_len = price, len(name)
        return best

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.price_for(model)
        return round(
            input_tokens / 1_000_000 * price["input"]
            + output_tokens / 1_000_000 * price["output"], 6)

    # ------------------------------------------------------------- writing
    def _path(self, day: date) -> Path:
        return self.dir / f"{day.isoformat()}.jsonl"

    def record(self, event: UsageEvent) -> None:
        try:
            with self._lock:
                with open(self._path(date.today()), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:      # never let accounting break the request
            log.warning("Usage event could not be recorded: %s", exc)

    # ------------------------------------------------------------- reading
    def events(self, days: int = 30) -> list[dict[str, Any]]:
        today = date.today()
        out: list[dict[str, Any]] = []
        for offset in range(days):
            path = self._path(today - timedelta(days=offset))
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def summary(self, days: int = 30) -> dict[str, Any]:
        events = self.events(days)
        if not events:
            return {"days": days, "calls": 0, "cost": 0.0, "input_tokens": 0,
                    "output_tokens": 0, "estimated_share": 0.0,
                    "by_operation": {}, "by_model": {}, "by_domain": {},
                    "by_day": [], "note": "Nothing recorded yet."}

        def bucket(key: str) -> dict[str, dict[str, float]]:
            grouped: dict[str, dict[str, float]] = defaultdict(
                lambda: {"calls": 0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0})
            for e in events:
                name = e.get(key) or "unknown"
                g = grouped[name]
                g["calls"] += 1
                g["cost"] += e.get("cost", 0.0)
                g["input_tokens"] += e.get("input_tokens", 0)
                g["output_tokens"] += e.get("output_tokens", 0)
            return {k: {**v, "cost": round(v["cost"], 4)}
                    for k, v in sorted(grouped.items(), key=lambda kv: -kv[1]["cost"])}

        by_day: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "cost": 0.0})
        for e in events:
            day = (e.get("at") or "")[:10]
            by_day[day]["calls"] += 1
            by_day[day]["cost"] += e.get("cost", 0.0)

        estimated = sum(1 for e in events if e.get("estimated_tokens"))
        return {
            "days": days,
            "calls": len(events),
            "cost": round(sum(e.get("cost", 0.0) for e in events), 4),
            "input_tokens": sum(e.get("input_tokens", 0) for e in events),
            "output_tokens": sum(e.get("output_tokens", 0) for e in events),
            "estimated_share": round(estimated / len(events), 3),
            "by_operation": bucket("operation"),
            "by_model": bucket("model"),
            "by_domain": bucket("domain"),
            "by_principal": bucket("principal"),
            "by_day": [{"day": d, "calls": v["calls"], "cost": round(v["cost"], 4)}
                       for d, v in sorted(by_day.items())],
            "note": ("Costs are indicative, from a local price table. Token counts "
                     "are reported by the provider where available and estimated "
                     "otherwise."),
        }


_recorder: UsageRecorder | None = None


def get_recorder() -> UsageRecorder:
    global _recorder
    if _recorder is None:
        _recorder = UsageRecorder(get_settings().data_dir)
    return _recorder


# --------------------------------------------------------------- attribution
# The operation and caller are set by whichever feature is running, so an event
# recorded deep in the provider still knows why it happened. Contextvars rather
# than arguments, because threading them through every call site would touch
# code that has no business knowing about accounting.
import contextvars

_operation: contextvars.ContextVar[str] = contextvars.ContextVar("operation", default="other")
_domain: contextvars.ContextVar[str] = contextvars.ContextVar("domain", default="")
_principal: contextvars.ContextVar[str] = contextvars.ContextVar("principal", default="")


class attribute_to:
    """Mark everything inside as belonging to one operation."""

    def __init__(self, operation: str, *, domain: str = "", principal: str = "") -> None:
        self.operation, self.domain, self.principal = operation, domain, principal
        self._tokens: list[Any] = []

    def __enter__(self) -> "attribute_to":
        self._tokens = [_operation.set(self.operation)]
        if self.domain:
            self._tokens.append(_domain.set(self.domain))
        if self.principal:
            self._tokens.append(_principal.set(self.principal))
        return self

    def __exit__(self, *exc: Any) -> None:
        for var, token in zip([_operation, _domain, _principal], self._tokens):
            try:
                var.reset(token)
            except ValueError:
                pass

    async def __aenter__(self) -> "attribute_to":
        return self.__enter__()

    async def __aexit__(self, *exc: Any) -> None:
        self.__exit__(*exc)


def current_attribution() -> tuple[str, str, str]:
    return _operation.get(), _domain.get(), _principal.get()
