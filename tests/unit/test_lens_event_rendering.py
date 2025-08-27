from rich.console import Console
from io import StringIO

from flujo.cli import lens_trace as lt


def test_lens_renders_agent_prompt_event():
    # Build a minimal fake trace with an agent.prompt event
    trace = {
        "name": "pipeline_run",
        "status": "completed",
        "start_time": 0.0,
        "end_time": 1.0,
        "attributes": {},
        "events": [],
        "children": [
            {
                "name": "clarify",
                "status": "completed",
                "start_time": 0.1,
                "end_time": 0.9,
                "attributes": {},
                "events": [
                    {
                        "name": "agent.prompt",
                        "attributes": {"rendered_history": "User: hello\nAssistant: hi there"},
                    }
                ],
                "children": [],
            }
        ],
    }

    tree = lt._render_trace_tree(trace)  # noqa: SLF001 (accessing module-private for test)
    # Render to text and assert the event preview appears
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    console.print(tree)
    output = buf.getvalue()
    assert "agent.prompt" in output
    assert "User: hello" in output
