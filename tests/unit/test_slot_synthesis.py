import pytest

from flujo.helpers.slot_synthesis import synthesize_slots
from flujo.domain.models import PipelineContext, HumanInteraction


@pytest.mark.asyncio
async def test_synthesize_slots_extracts_fields_from_hitl_history():
    ctx = PipelineContext()
    ctx.hitl_history = [
        HumanInteraction(message_to_human="What metric?", human_response="count"),
        HumanInteraction(
            message_to_human="Which cohort/population?",
            human_response="male 20 to 30 years old",
        ),
        HumanInteraction(
            message_to_human="What is the time window?",
            human_response="between 2020 and 2025",
        ),
        HumanInteraction(
            message_to_human="Any groupings/dimensions?",
            human_response="by state and age",
        ),
        HumanInteraction(
            message_to_human="Any filters or exclusions?",
            human_response="no",
        ),
    ]

    update = await synthesize_slots(None, context=ctx)

    assert isinstance(update, dict)
    hitl_data = update.get("hitl_data") or {}
    slots = hitl_data.get("slots") or {}

    assert slots.get("metric") == "count"
    cohort = slots.get("cohort") or {}
    assert cohort.get("sex") == "male"
    assert cohort.get("age_min") == 20
    assert cohort.get("age_max") == 30
    tw = slots.get("time_window") or {}
    assert tw.get("start_year") == 2020
    assert tw.get("end_year") == 2025
    grouping = slots.get("grouping") or []
    # Order is not strictly guaranteed; assert subset
    assert set(grouping) & {"age", "state"}
    assert slots.get("filters") is None  # 'no' implies no filters

    # Also ensure filled/missing bookkeeping present
    assert set(hitl_data.get("slots_filled") or [])
    assert isinstance(hitl_data.get("slots_text_summary"), str)
