from hypothesis import given, strategies as st
from flujo.utils.redact import summarize_and_redact_prompt
from pydantic import SecretStr, BaseModel


# Simple test settings class that doesn't load from environment
class _TestSettings(BaseModel):
    openai_api_key: SecretStr
    google_api_key: SecretStr
    anthropic_api_key: SecretStr


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
    mock_settings = _TestSettings(
        openai_api_key=SecretStr("sk-abc123DEF456"),
        google_api_key=SecretStr("gk_xyz987ZYX654"),
        anthropic_api_key=SecretStr("ak_mno789MNO456"),
    )
    summary = summarize_and_redact_prompt(prompt, settings=mock_settings)
    for secret in secrets:
        assert secret not in summary
        assert secret[:8] not in summary
