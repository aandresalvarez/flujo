from __future__ import annotations

import json
import pytest

from flujo.utils.template_vars import render_template
from flujo.utils.prompting import AdvancedPromptFormatter


def test_upper_and_lower_filters() -> None:
    assert render_template("{{ previous_step|upper }}", previous_step="aBc") == "ABC"
    assert render_template("{{ previous_step|lower }}", previous_step="aBc") == "abc"


def test_length_filter_on_list_and_string() -> None:
    assert render_template("{{ previous_step|length }}", previous_step=[1, 2, 3]) == "3"
    assert render_template("{{ previous_step|length }}", previous_step="hello") == "5"


def test_tojson_filter_serializes_values() -> None:
    data = {"a": 1, "b": [2, 3]}
    out = render_template("{{ previous_step|tojson }}", previous_step=data)
    # Should be valid JSON and equal to dumps(data)
    assert json.loads(out) == data


def test_join_filter_with_delimiter_and_non_str_values() -> None:
    items = [1, "b", 3]
    out = render_template("{{ previous_step|join(',') }}", previous_step=items)
    assert out == "1,b,3"


def test_filter_chain_applies_after_fallback_expression() -> None:
    # previous_step is missing; fallback literal should be upper-cased
    out = render_template("{{ previous_step or 'hi there' | upper }}")
    assert out == "HI THERE"


def test_unknown_filter_raises_value_error() -> None:
    fmt = AdvancedPromptFormatter("{{ previous_step|nope }}")
    with pytest.raises(ValueError):
        _ = fmt.format(previous_step="x")


def test_each_block_respects_filters() -> None:
    tpl = "{{#each context.items}}{{ this|upper }} {{/each}}"
    out = render_template(
        tpl,
        context=None,
        steps=None,
        previous_step=None,
    )
    # items is not provided; nothing rendered
    assert out.strip() == ""

    tpl2 = "{{#each context.items}}{{ this|lower }}{{/each}}"
    out2 = render_template(
        tpl2,
        context=None,
        steps=None,
        previous_step=None,
    )
    assert out2 == ""

    tpl3 = "{{#each context.items}}{{ this|upper }}-{{/each}}"
    out3 = render_template(tpl3, context={"items": ["a", "B"]})
    assert out3 == "A-B-"
