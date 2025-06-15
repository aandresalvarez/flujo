"""Temperature schedule logic for flujo."""

from flujo.infra.settings import settings


def temp_for_round(round_i: int) -> float:
    """
    Returns the temperature for a given iteration round from the schedule
    defined in settings. The last value is used for any rounds beyond
    the schedule's length.
    """
    schedule = settings.t_schedule
    return schedule[min(round_i, len(schedule) - 1)]
