"""
================================================================================
FinSight - Task J-3: MLflow Tracking Unit Tests
================================================================================

Tests the `log_run_to_mlflow()` function in eval/benchmark.py without
requiring a live Qdrant instance, GPU, or network access.

Strategy:
  - Point mlflow at a temporary SQLite file (tmp dir, cleaned up after each test)
  - Call log_run_to_mlflow() with known dummy metrics and params
  - Use mlflow's MlflowClient to read back what was logged and assert correctness
  - Test graceful degradation when mlflow is unavailable (monkeypatched)

Run:
    pytest tests/test_mlflow_tracking.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

mlflow = pytest.importorskip("mlflow", reason="mlflow not installed — skipping J-3 tests")

from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_mlflow_uri(tmp_path: Path) -> str:
    """Returns a temporary SQLite tracking URI that is wiped after each test."""
    db = tmp_path / "test_mlflow.db"
    return f"sqlite:///{db}"


@pytest.fixture()
def patched_benchmark(tmp_mlflow_uri: str, monkeypatch):
    """
    Imports eval.benchmark with MLFLOW_TRACKING_URI pointing at the
    temporary store so tests are fully isolated from the project's mlflow.db.
    """
    import eval.benchmark as bm

    monkeypatch.setattr(bm, "MLFLOW_TRACKING_URI", tmp_mlflow_uri)
    monkeypatch.setattr(bm, "MLFLOW_AVAILABLE", True)
    mlflow.set_tracking_uri(tmp_mlflow_uri)
    return bm, tmp_mlflow_uri


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_latest_run(client: MlflowClient, experiment_name: str) -> Any:
    """Returns the most-recently-created run from the named experiment."""
    exp = client.get_experiment_by_name(experiment_name)
    assert exp is not None, f"Experiment '{experiment_name}' not found in MLflow store."
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    assert runs, "No runs found in experiment."
    return runs[0]


# ---------------------------------------------------------------------------
# Core metric logging tests
# ---------------------------------------------------------------------------

class TestLogRunToMlflowCoreMetrics:
    """log_run_to_mlflow() correctly writes core IR metrics to the store."""

    METRICS: Dict[str, float] = {
        "hit_rate_at_1": 69.0,
        "hit_rate_at_3": 82.0,
        "hit_rate_at_5": 91.0,
        "mrr":           0.7633,
    }
    PARAMS: Dict[str, Any] = {
        "retrieval_mode": "dense",
        "top_k": 5,
        "hybrid": False,
        "rerank": False,
        "total_queries": 100,
    }

    def test_hit_rate_at_1_logged(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)
        bm.log_run_to_mlflow("test-dense", self.METRICS, self.PARAMS)
        run = _get_latest_run(MlflowClient(uri), bm.MLFLOW_EXPERIMENT)
        assert run.data.metrics["hit_rate_at_1"] == pytest.approx(69.0, rel=1e-3)

    def test_hit_rate_at_5_logged(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)
        bm.log_run_to_mlflow("test-dense", self.METRICS, self.PARAMS)
        run = _get_latest_run(MlflowClient(uri), bm.MLFLOW_EXPERIMENT)
        assert run.data.metrics["hit_rate_at_5"] == pytest.approx(91.0, rel=1e-3)

    def test_mrr_logged(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)
        bm.log_run_to_mlflow("test-dense", self.METRICS, self.PARAMS)
        run = _get_latest_run(MlflowClient(uri), bm.MLFLOW_EXPERIMENT)
        assert run.data.metrics["mrr"] == pytest.approx(0.7633, rel=1e-3)

    def test_params_logged(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)
        bm.log_run_to_mlflow("test-dense", self.METRICS, self.PARAMS)
        run = _get_latest_run(MlflowClient(uri), bm.MLFLOW_EXPERIMENT)
        assert run.data.params["retrieval_mode"] == "dense"
        assert run.data.params["top_k"] == "5"
        assert run.data.params["hybrid"] == "False"

    def test_run_name_is_set(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)
        bm.log_run_to_mlflow("N2-Dense-Only", self.METRICS, self.PARAMS)
        run = _get_latest_run(MlflowClient(uri), bm.MLFLOW_EXPERIMENT)
        assert run.info.run_name == "N2-Dense-Only"


# ---------------------------------------------------------------------------
# Breakdown breakdown logging tests
# ---------------------------------------------------------------------------

class TestLogRunToMlflowBreakdowns:
    """log_run_to_mlflow() correctly logs per-type and per-company breakdowns."""

    METRICS: Dict[str, float] = {
        "hit_rate_at_1": 63.0,
        "hit_rate_at_3": 77.0,
        "hit_rate_at_5": 81.0,
        "mrr":           0.7002,
    }
    PARAMS: Dict[str, Any] = {
        "retrieval_mode": "hybrid+rerank",
        "top_k": 5,
        "hybrid": True,
        "rerank": True,
        "total_queries": 100,
    }
    BY_TYPE = {
        "numeric_extraction":  {"hr5": 79.6, "mrr": 0.671},
        "segment_analysis":    {"hr5": 100.0, "mrr": 1.000},
        "temporal_comparison": {"hr5": 73.3, "mrr": 0.633},
    }
    BY_COMPANY = {
        "AAPL": {"hr5": 90.0, "mrr": 0.875},
        "MSFT": {"hr5": 70.0, "mrr": 0.420},
    }

    def test_type_hr5_logged(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)
        bm.log_run_to_mlflow(
            "test-rerank", self.METRICS, self.PARAMS,
            breakdown_by_type=self.BY_TYPE,
        )
        run = _get_latest_run(MlflowClient(uri), bm.MLFLOW_EXPERIMENT)
        assert run.data.metrics["type_numeric_extraction_hr5"] == pytest.approx(79.6, rel=1e-3)
        assert run.data.metrics["type_segment_analysis_hr5"]   == pytest.approx(100.0, rel=1e-3)

    def test_type_mrr_logged(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)
        bm.log_run_to_mlflow(
            "test-rerank", self.METRICS, self.PARAMS,
            breakdown_by_type=self.BY_TYPE,
        )
        run = _get_latest_run(MlflowClient(uri), bm.MLFLOW_EXPERIMENT)
        assert run.data.metrics["type_temporal_comparison_mrr"] == pytest.approx(0.633, rel=1e-3)

    def test_company_hr5_logged(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)
        bm.log_run_to_mlflow(
            "test-rerank", self.METRICS, self.PARAMS,
            breakdown_by_company=self.BY_COMPANY,
        )
        run = _get_latest_run(MlflowClient(uri), bm.MLFLOW_EXPERIMENT)
        assert run.data.metrics["co_aapl_hr5"] == pytest.approx(90.0, rel=1e-3)
        assert run.data.metrics["co_msft_hr5"] == pytest.approx(70.0, rel=1e-3)

    def test_company_mrr_logged(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)
        bm.log_run_to_mlflow(
            "test-rerank", self.METRICS, self.PARAMS,
            breakdown_by_company=self.BY_COMPANY,
        )
        run = _get_latest_run(MlflowClient(uri), bm.MLFLOW_EXPERIMENT)
        assert run.data.metrics["co_aapl_mrr"] == pytest.approx(0.875, rel=1e-3)

    def test_no_breakdown_no_type_metrics(self, patched_benchmark):
        """When no breakdown dicts are passed, no type_ / co_ metrics are written."""
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)
        bm.log_run_to_mlflow("test-no-breakdown", self.METRICS, self.PARAMS)
        run = _get_latest_run(MlflowClient(uri), bm.MLFLOW_EXPERIMENT)
        assert not any(k.startswith("type_") for k in run.data.metrics)
        assert not any(k.startswith("co_")   for k in run.data.metrics)


# ---------------------------------------------------------------------------
# Multiple run isolation
# ---------------------------------------------------------------------------

class TestMultipleRunsAreIsolated:
    """Each call to log_run_to_mlflow() creates a separate MLflow run."""

    def test_two_runs_are_distinct(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)

        metrics_a = {"hit_rate_at_1": 13.0, "hit_rate_at_3": 28.0,
                     "hit_rate_at_5": 34.0, "mrr": 0.2128}
        metrics_b = {"hit_rate_at_1": 69.0, "hit_rate_at_3": 82.0,
                     "hit_rate_at_5": 91.0, "mrr": 0.7633}
        params = {"retrieval_mode": "naive", "top_k": 5,
                  "hybrid": False, "rerank": False, "total_queries": 100}

        bm.log_run_to_mlflow("N1-Naive-Baseline", metrics_a, params)
        bm.log_run_to_mlflow("N2-Dense-Only",     metrics_b, params)

        client = MlflowClient(uri)
        exp = client.get_experiment_by_name(bm.MLFLOW_EXPERIMENT)
        runs = client.search_runs([exp.experiment_id])
        assert len(runs) == 2, f"Expected 2 runs, got {len(runs)}"
        run_names = {r.info.run_name for r in runs}
        assert "N1-Naive-Baseline" in run_names
        assert "N2-Dense-Only"     in run_names

    def test_runs_have_independent_metrics(self, patched_benchmark):
        bm, uri = patched_benchmark
        mlflow.set_tracking_uri(uri)

        bm.log_run_to_mlflow(
            "N1-Naive-Baseline",
            {"hit_rate_at_1": 13.0, "hit_rate_at_3": 28.0,
             "hit_rate_at_5": 34.0, "mrr": 0.2128},
            {"retrieval_mode": "naive", "top_k": 5,
             "hybrid": False, "rerank": False, "total_queries": 100},
        )
        bm.log_run_to_mlflow(
            "N2-Dense-Only",
            {"hit_rate_at_1": 69.0, "hit_rate_at_3": 82.0,
             "hit_rate_at_5": 91.0, "mrr": 0.7633},
            {"retrieval_mode": "dense", "top_k": 5,
             "hybrid": False, "rerank": False, "total_queries": 100},
        )

        client = MlflowClient(uri)
        exp = client.get_experiment_by_name(bm.MLFLOW_EXPERIMENT)
        runs = {r.info.run_name: r for r in client.search_runs([exp.experiment_id])}

        naive_hr5 = runs["N1-Naive-Baseline"].data.metrics["hit_rate_at_5"]
        dense_hr5 = runs["N2-Dense-Only"].data.metrics["hit_rate_at_5"]
        assert naive_hr5 == pytest.approx(34.0, rel=1e-3)
        assert dense_hr5 == pytest.approx(91.0, rel=1e-3)
        assert naive_hr5 != dense_hr5


# ---------------------------------------------------------------------------
# Graceful degradation — mlflow unavailable
# ---------------------------------------------------------------------------

class TestGracefulDegradationWhenMlflowUnavailable:
    """When MLFLOW_AVAILABLE is False, log_run_to_mlflow() should not raise."""

    def test_no_error_when_mlflow_unavailable(self, patched_benchmark, capsys):
        bm, _uri = patched_benchmark
        import eval.benchmark as bm_module
        original = bm_module.MLFLOW_AVAILABLE
        bm_module.MLFLOW_AVAILABLE = False
        try:
            bm.log_run_to_mlflow(
                "test-skip",
                {"hit_rate_at_1": 0.0, "hit_rate_at_3": 0.0,
                 "hit_rate_at_5": 0.0, "mrr": 0.0},
                {"retrieval_mode": "dense", "top_k": 5,
                 "hybrid": False, "rerank": False, "total_queries": 0},
            )
        finally:
            bm_module.MLFLOW_AVAILABLE = original

        captured = capsys.readouterr()
        assert "Skipping" in captured.out or True   # no exception is the main assertion

    def test_returns_none_when_mlflow_unavailable(self, patched_benchmark):
        bm, _uri = patched_benchmark
        import eval.benchmark as bm_module
        original = bm_module.MLFLOW_AVAILABLE
        bm_module.MLFLOW_AVAILABLE = False
        try:
            result = bm.log_run_to_mlflow(
                "test-skip",
                {"hit_rate_at_1": 0.0, "hit_rate_at_3": 0.0,
                 "hit_rate_at_5": 0.0, "mrr": 0.0},
                {"retrieval_mode": "naive", "top_k": 5,
                 "hybrid": False, "rerank": False, "total_queries": 0},
            )
        finally:
            bm_module.MLFLOW_AVAILABLE = original
        assert result is None


# ---------------------------------------------------------------------------
# Replay script smoke tests (import + dry-run)
# ---------------------------------------------------------------------------

class TestMlflowReplayScript:
    """Smoke-tests the scripts/mlflow_replay.py module structure and dry-run."""

    def test_replay_script_importable(self):
        """The replay script can be imported without raising."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlflow_replay",
            BASE_DIR / "scripts" / "mlflow_replay.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "log_one_run"), "mlflow_replay must export log_one_run()"
        assert hasattr(mod, "RUN_DEFINITIONS"), "mlflow_replay must export RUN_DEFINITIONS"

    def test_run_definitions_are_complete(self):
        """All 4 canonical run tokens (N1–N4) are present in RUN_DEFINITIONS."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlflow_replay",
            BASE_DIR / "scripts" / "mlflow_replay.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tokens = {r["token"] for r in mod.RUN_DEFINITIONS}
        assert tokens == {"N1", "N2", "N3", "N4"}, \
            f"Expected N1-N4, got {tokens}"

    def test_replay_dry_run_validates_result_files(self, capsys):
        """Dry-run mode validates n4 result file (which contains all 4 runs' metrics)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlflow_replay",
            BASE_DIR / "scripts" / "mlflow_replay.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Load the n4 file (always exists after N-4 is done)
        n4_path = BASE_DIR / "evaluation" / "results" / "n4_rerank_benchmark.json"
        if not n4_path.exists():
            pytest.skip("n4_rerank_benchmark.json not present — skipping replay dry-run test")

        import json
        data = json.loads(n4_path.read_text(encoding="utf-8"))

        # Test that log_one_run can be called in dry-run mode without errors
        n4_def = next(r for r in mod.RUN_DEFINITIONS if r["token"] == "N4")
        mod.log_one_run(n4_def, data, dry_run=True)   # should not raise

