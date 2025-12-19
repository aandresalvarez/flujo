import pytest

from flujo.builtins.extras import render_jinja_template


@pytest.mark.asyncio
async def test_jinja_sandbox_access_restriction() -> None:
    """Dangerous Jinja templates should not execute or expose internals."""

    template = '{{ "".__class__.__base__.__subclasses__() }}'
    rendered = await render_jinja_template(template, variables={})

    # Sandbox should prevent execution; returning the template unchanged is acceptable.
    assert rendered.strip() == template
