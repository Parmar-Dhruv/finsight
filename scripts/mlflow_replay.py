"""
================================================================================
FinSight - Task J-3: MLflow Replay Script
================================================================================

Reads the already-saved benchmark result JSON files from evaluation/results/
and re-logs them into MLflow so the experiment dashboard is populated without
needing to re-run the full benchmark (which requires Qdrant + GPU).

Each benchmark result file (n2, n3, n4) corresponds to one or more runs:
  - N1-Naive-Baseline    (embedded inside every result file as naive_metrics)
  - N2-Dense-Only        (from n2_retrieval_benchmark.json  → dense_metrics)
  - N3-Hybrid-Dense-BM25-RRF  (from n3_hybrid_benchmark.json → hybrid_metrics)
  - N4-Hybrid-BM25-Rerank     (from n4_rerank_benchmark.json → hybrid_rerank_metrics)

Usage:
    python scripts/mlflow_replay.py                 # logs all 4 runs
    python scripts/mlflow_replay.py --run N4        # logs only N4-Hybrid-BM25-Rerank
    python scripts/mlflow_replay.py --dry-run       # validate files without logging

Then launch the UI:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    # Open: http://localhost:5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT   = "FinSight-Retrieval"
RESULTS_DIR         = BASE_DIR / "evaluation" / "results"

# ---------------------------------------------------------------------------
# Result file manifest — ordered so N1 is always logged first
# ---------------------------------------------------------------------------

RESULT_FILES: dict[str, Path] = {
    "n2": RESULTS_DIR / "n2_retrieval_benchmark.json",
    "n3": RESULTS_DIR / "n3_hybrid_benchmark.json",
    "n4": RESULTS_DIR / "n4_rerank_benchmark.json",
}

# Map run-name token → (result_file_key, metrics_key, params)
RUN_DEFINITIONS: list[dict] = [
    {
        "token": "N1",
        "run_name": "N1-Naive-Baseline",
        "source_file": "n4",          # n4 file contains naive_metrics in its summary
        "metrics_key": "naive_metrics",
        "breakdown_type_key": None,   # n2/n3/n4 store only the main pipeline's breakdown
        "breakdown_company_key": None,
        "params": {
            "retrieval_mode": "naive",
            "top_k": 5,
            "hybrid": False,
            "rerank": False,
        },
    },
    {
        "token": "N2",
        "run_name": "N2-Dense-Only",
        "source_file": "n4",
        "metrics_key": "dense_metrics",
        "breakdown_type_key": None,
        "breakdown_company_key": None,
        "params": {
            "retrieval_mode": "dense",
            "top_k": 5,
            "hybrid": False,
            "rerank": False,
        },
    },
    {
        "token": "N3",
        "run_name": "N3-Hybrid-Dense-BM25-RRF",
        "source_file": "n4",
        "metrics_key": "hybrid_metrics",
        "breakdown_type_key": None,
        "breakdown_company_key": None,
        "params": {
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "hybrid": True,
            "rerank": False,
        },
    },
    {
        "token": "N4",
        "run_name": "N4-Hybrid-BM25-Rerank",
        "source_file": "n4",
        "metrics_key": "hybrid_rerank_metrics",
        "breakdown_type_key": "by_type",
        "breakdown_company_key": "by_company",
        "params": {
            "retrieval_mode": "hybrid+rerank",
            "top_k": 5,
            "hybrid": True,
            "rerank": True,
        },
    },
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_result_file(key: str) -> dict:
    """Loads and returns the parsed JSON for the given result-file key."""
    path = RESULT_FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}\n"
            "Run eval/benchmark.py first to generate it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Single-run logger
# ---------------------------------------------------------------------------

def log_one_run(
    run_def: dict,
    data: dict,
    *,
    dry_run: bool = False,
) -> None:
    """
    Logs one experiment run to MLflow from the pre-loaded result data dict.

    Parameters
    ----------
    run_def : dict
        Entry from RUN_DEFINITIONS describing which keys/params to use.
    data : dict
        Parsed JSON from the source result file.
    dry_run : bool
        If True, validate keys but do not write to MLflow.
    """
    run_name = run_def["run_name"]
    summary  = data.get("summary", data)   # n4 wraps everything in "summary"

    metrics = summary.get(run_def["metrics_key"], {})
    if not metrics:
        raise KeyError(
            f"metrics_key '{run_def['metrics_key']}' not found in result file "
            f"'{RESULT_FILES[run_def['source_file']].name}'."
        )

    total_queries = metrics.get("total_queries", summary.get("total_questions", 100))

    params = dict(run_def["params"])
    params["total_queries"] = total_queries

    # Optional breakdown dicts
    by_type    = data.get(run_def["breakdown_type_key"] or "", {}) or {}
    by_company = data.get(run_def["breakdown_company_key"] or "", {}) or {}

    if dry_run:
        logger.info("[DRY-RUN] Would log run '%s'  HR@5=%.1f%%  MRR=%.4f",
                    run_name, metrics.get("hit_rate_at_5", 0), metrics.get("mrr", 0))
        return

    with mlflow.start_run(run_name=run_name):
        # Core IR metrics
        mlflow.log_metric("hit_rate_at_1", metrics.get("hit_rate_at_1", 0.0))
        mlflow.log_metric("hit_rate_at_3", metrics.get("hit_rate_at_3", 0.0))
        mlflow.log_metric("hit_rate_at_5", metrics.get("hit_rate_at_5", 0.0))
        mlflow.log_metric("mrr",           metrics.get("mrr", 0.0))

        # Run configuration
        mlflow.log_params(params)

        # Per-question-type HR@5 & MRR (prefixed type_)
        for q_type, stat in by_type.items():
            safe = q_type.replace(" ", "_").replace("/", "_").lower()
            mlflow.log_metric(f"type_{safe}_hr5", stat.get("hr5", stat.get("hr1", 0.0)))
            mlflow.log_metric(f"type_{safe}_mrr",  stat.get("mrr", 0.0))

        # Per-company HR@5 & MRR (prefixed co_)
        for company, stat in by_company.items():
            safe = company.replace(" ", "_").lower()
            mlflow.log_metric(f"co_{safe}_hr5", stat.get("hr5", stat.get("hr1", 0.0)))
            mlflow.log_metric(f"co_{safe}_mrr",  stat.get("mrr", 0.0))

        # Log the source JSON file as an artifact for traceability
        artifact_path = RESULT_FILES.get(run_def["source_file"])
        if artifact_path and artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="benchmark_results")

    logger.info("[OK] Logged run '%s'  HR@5=%.1f%%  MRR=%.4f",
                run_name, metrics.get("hit_rate_at_5", 0), metrics.get("mrr", 0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay FinSight benchmark results into MLflow experiment tracking."
    )
    p.add_argument(
        "--run",
        choices=["N1", "N2", "N3", "N4", "all"],
        default="all",
        help="Which run(s) to log (default: all).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate result files without writing to MLflow.",
    )
    p.add_argument(
        "--tracking-uri",
        default=MLFLOW_TRACKING_URI,
        help=f"MLflow tracking URI (default: {MLFLOW_TRACKING_URI}).",
    )
    p.add_argument(
        "--experiment",
        default=MLFLOW_EXPERIMENT,
        help=f"MLflow experiment name (default: {MLFLOW_EXPERIMENT}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dry_run:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
        logger.info("MLflow tracking URI : %s", args.tracking_uri)
        logger.info("MLflow experiment   : %s", args.experiment)
    else:
        logger.info("[DRY-RUN] Validating result files — no MLflow writes will occur.")

    # Pre-load all required result files (avoid duplicate I/O)
    loaded: dict[str, dict] = {}
    for run_def in RUN_DEFINITIONS:
        key = run_def["source_file"]
        if key not in loaded:
            try:
                loaded[key] = load_result_file(key)
                logger.info("Loaded  : %s", RESULT_FILES[key])
            except FileNotFoundError as exc:
                logger.error(str(exc))
                sys.exit(1)

    # Filter runs
    target_token = args.run.upper()
    runs_to_log = (
        RUN_DEFINITIONS
        if target_token == "ALL"
        else [r for r in RUN_DEFINITIONS if r["token"] == target_token]
    )

    if not runs_to_log:
        logger.error("No matching run definition for token '%s'.", target_token)
        sys.exit(1)

    print()
    print("=" * 70)
    print(f"  FinSight J-3 — MLflow Replay  ({len(runs_to_log)} run(s))")
    print("=" * 70)

    for run_def in runs_to_log:
        data = loaded[run_def["source_file"]]
        log_one_run(run_def, data, dry_run=args.dry_run)

    print()
    if args.dry_run:
        print("Dry-run complete — all result files validated.")
    else:
        print("All runs logged successfully!")
        print()
        print("Launch the MLflow UI with:")
        print(f"  mlflow ui --backend-store-uri {args.tracking_uri}")
        print("  Then open: http://localhost:5000")
    print("=" * 70)


if __name__ == "__main__":
    main()

