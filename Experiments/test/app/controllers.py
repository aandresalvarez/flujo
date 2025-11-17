from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from .comparators import ComparisonRecord, compare_with_reason
from .evidence import Badges, copy_badges


@dataclass
class OrdinalControllerConfig:
    tournament_k: int = 5
    max_frontier: int = 20
    min_support: int = 3


@dataclass
class OrdinalControllerResult:
    winner: Dict[str, Any]
    log: List[str] = field(default_factory=list)
    goal_met: bool = False


class OrdinalController:
    """Lightweight ordinal controller for experimental pipelines.

    Performs pairwise comparisons using the provided comparator and
    records a human-readable log for traceability.
    """

    def __init__(
        self,
        *,
        config: OrdinalControllerConfig | None = None,
    ) -> None:
        self._config = config or OrdinalControllerConfig()

    def run(self, candidates: Iterable[Dict[str, Any]]) -> OrdinalControllerResult:
        candidate_list = list(candidates)
        if not candidate_list:
            return OrdinalControllerResult(winner={}, log=[], goal_met=False)
        log: List[str] = []
        champion = candidate_list[0]
        champion_badges = copy_badges(champion.get("badges"))

        for contender in candidate_list[1:]:
            contender_badges = copy_badges(contender.get("badges"))
            record: ComparisonRecord = compare_with_reason(
                contender_badges, champion_badges
            )
            winner = contender if record.winner > 0 else champion
            log.append(
                self._format_log_entry(
                    contender_id=contender.get("id", "unknown"),
                    champion_id=champion.get("id", "unknown"),
                    winning_id=winner.get("id", "unknown"),
                    record=record,
                )
            )
            if record.winner > 0:
                champion = contender
                champion_badges = contender_badges

            if self._goal_met(champion_badges):
                return OrdinalControllerResult(
                    winner=_finalize_winner(champion),
                    log=log,
                    goal_met=True,
                )

        return OrdinalControllerResult(
            winner=_finalize_winner(champion),
            log=log,
            goal_met=self._goal_met(champion_badges),
        )

    def _goal_met(self, badges: Badges) -> bool:
        return bool(badges.constraints_ok and badges.support >= self._config.min_support)

    def _format_log_entry(
        self,
        *,
        contender_id: str,
        champion_id: str,
        winning_id: str,
        record: ComparisonRecord,
    ) -> str:
        return (
            f"duel contender={contender_id} vs champion={champion_id} "
            f"=> winner={winning_id} by {record.reason} ({record.details})"
        )


def _finalize_winner(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Return candidate dict with badges normalized to plain dict."""
    badges_raw = copy_badges(candidate.get("badges"))
    finalized = dict(candidate)
    finalized["badges"] = badges_raw.to_dict()
    return finalized
