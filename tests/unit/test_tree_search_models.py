from flujo.domain.models import PipelineContext, SearchNode, SearchState


def test_search_node_snapshot_excludes_tree_search_state():
    ctx = PipelineContext(initial_prompt="goal")
    ctx.tree_search_state = SearchState()
    node = SearchNode(node_id="n0", state_hash="hash")
    node.attach_context(ctx)
    snapshot = node.context_snapshot or {}
    assert "tree_search_state" not in snapshot


def test_search_state_priority_queue_ordering():
    state = SearchState()
    nodes = [
        SearchNode(node_id="n3", state_hash="h3", f_cost=1.0, g_cost=1.0, depth=1),
        SearchNode(node_id="n2", state_hash="h2", f_cost=1.0, g_cost=1.0, depth=2),
        SearchNode(node_id="n1", state_hash="h1", f_cost=1.0, g_cost=2.0, depth=1),
        SearchNode(node_id="n0", state_hash="h0", f_cost=2.0, g_cost=0.5, depth=0),
    ]
    state.nodes = {node.node_id: node for node in nodes}
    state.open_set = ["n0", "n2", "n1", "n3"]

    ordered = state.sorted_open_nodes()
    assert [node.node_id for node in ordered] == ["n3", "n2", "n1", "n0"]

    best = state.pop_best_open()
    assert best is not None
    assert best.node_id == "n3"
    assert "n3" not in state.open_set


def test_search_node_serialization_round_trip():
    ctx = PipelineContext(initial_prompt="goal", extra_field="value")
    ctx.tree_search_state = SearchState()
    node = SearchNode(node_id="n1", state_hash="hash", candidate={"k": "v"})
    node.attach_context(ctx)

    payload = node.model_dump()
    restored = SearchNode.model_validate(payload)
    rehydrated = restored.rehydrate_context(PipelineContext)

    assert rehydrated is not None
    assert rehydrated.initial_prompt == "goal"
    assert getattr(rehydrated, "extra_field") == "value"
    assert rehydrated.tree_search_state is None
