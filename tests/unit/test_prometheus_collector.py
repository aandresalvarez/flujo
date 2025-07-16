from flujo.telemetry.prometheus import PrometheusCollector


class DummyBackend:
    async def get_workflow_stats(self):
        return {
            "total_workflows": 3,
            "status_counts": {"completed": 2, "failed": 1},
            "average_execution_time_ms": 100,
        }


def test_prometheus_collector_yields_metrics():
    backend = DummyBackend()
    collector = PrometheusCollector(backend)
    metrics = list(collector.collect())
    names = {m.name for m in metrics}
    assert "flujo_runs_total" in names
    assert "flujo_runs_by_status" in names
    assert "flujo_avg_duration_ms" in names
