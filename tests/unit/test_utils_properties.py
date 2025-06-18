from hypothesis import given, strategies as st
from flujo.utils.redact import summarize_and_redact_prompt
from flujo.infra.settings import Settings

secrets = ["sk-abc123DEF456", "gk_xyz987ZYX654", "ak_mno789MNO456"]


@st.composite
def text_with_secrets(draw):
    text = draw(st.text())
    secret = draw(st.sampled_from(secrets))
    return f"{text} here is a secret: {secret}"


@given(prompt=st.text(), max_length=st.integers(min_value=10, max_value=200))
def test_property_summary_length(prompt, max_length):
    """The output should never exceed max_length."""
    summary = summarize_and_redact_prompt(prompt, max_length=max_length)
    assert len(summary) <= max_length


@given(prompt=text_with_secrets())
def test_property_redaction_hides_secrets(prompt):
    """Known secret patterns should be redacted from the output."""
    mock_settings = Settings(
        openai_api_key="sk-abc123DEF456",
        google_api_key="gk_xyz987ZYX654",
        anthropic_api_key="ak_mno789MNO456",
    )
    from flujo.infra import settings as settings_module

    original_settings = settings_module.settings
    settings_module.settings = mock_settings
    try:
        summary = summarize_and_redact_prompt(prompt)
        for secret in secrets:
            assert secret not in summary
            assert secret[:8] not in summary
    finally:
        settings_module.settings = original_settings
