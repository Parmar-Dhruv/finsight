"""
Quick MLflow test script — logs mock benchmark results to verify
the MLflow dashboard is working correctly before running the real benchmark.

Run:
    python scripts/test_mlflow.py
Then open: http://127.0.0.1:5000
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import mlflow

MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT   = "FinSight-Retrieval"

# ── Paste the real benchmark results from evaluation/results/ ──────────────────
MOCK_RUNS = [
    {
        "run_name": "N1-Naive-Baseline",
        "metrics": {
            "hit_rate_at_1": 13.0,
            "hit_rate_at_3": 28.0,
            "hit_rate_at_5": 34.0,
            "mrr": 0.2128,
        },
        "params": {
            "retrieval_mode": "naive",
            "top_k": 5,
            "hybrid": False,
            "rerank": False,
            "total_queries": 100,
        },
        "type_breakdown": {
            "numeric_extraction":  {"hr5": 18.4, "mrr": 0.131},
            "numeric_reasoning":   {"hr5": 14.3, "mrr": 0.103},
            "tabular":             {"hr5": 60.0, "mrr": 0.470},
            "qualitative":         {"hr5": 33.3, "mrr": 0.233},
            "segment_analysis":    {"hr5": 40.0, "mrr": 0.290},
            "temporal_comparison": {"hr5": 20.0, "mrr": 0.150},
        },
        "company_breakdown": {
            "AAPL": {"hr5": 35.0, "mrr": 0.230},
            "MSFT": {"hr5": 30.0, "mrr": 0.200},
            "AMZN": {"hr5": 35.0, "mrr": 0.220},
            "GOOGL": {"hr5": 35.0, "mrr": 0.210},
            "META": {"hr5": 35.0, "mrr": 0.200},
        },
    },
    {
        "run_name": "N2-Dense-Only",
        "metrics": {
            "hit_rate_at_1": 69.0,
            "hit_rate_at_3": 82.0,
            "hit_rate_at_5": 91.0,
            "mrr": 0.7633,
        },
        "params": {
            "retrieval_mode": "dense",
            "top_k": 5,
            "hybrid": False,
            "rerank": False,
            "total_queries": 100,
        },
        "type_breakdown": {
            "numeric_extraction":  {"hr5": 91.8, "mrr": 0.731},
            "numeric_reasoning":   {"hr5": 92.9, "mrr": 0.762},
            "tabular":             {"hr5": 100.0, "mrr": 0.950},
            "qualitative":         {"hr5": 88.9, "mrr": 0.800},
            "segment_analysis":    {"hr5": 100.0, "mrr": 0.950},
            "temporal_comparison": {"hr5": 80.0, "mrr": 0.683},
        },
        "company_breakdown": {
            "AAPL": {"hr5": 95.0, "mrr": 0.810},
            "MSFT": {"hr5": 90.0, "mrr": 0.780},
            "AMZN": {"hr5": 85.0, "mrr": 0.740},
            "GOOGL": {"hr5": 90.0, "mrr": 0.760},
            "META": {"hr5": 95.0, "mrr": 0.780},
        },
    },
    {
        "run_name": "N3-Hybrid-Dense-BM25-RRF",
        "metrics": {
            "hit_rate_at_1": 55.0,
            "hit_rate_at_3": 68.0,
            "hit_rate_at_5": 77.0,
            "mrr": 0.6277,
        },
        "params": {
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "hybrid": True,
            "rerank": False,
            "total_queries": 100,
        },
        "type_breakdown": {
            "numeric_extraction":  {"hr5": 75.5, "mrr": 0.590},
            "numeric_reasoning":   {"hr5": 78.6, "mrr": 0.610},
            "tabular":             {"hr5": 100.0, "mrr": 0.900},
            "qualitative":         {"hr5": 66.7, "mrr": 0.550},
            "segment_analysis":    {"hr5": 100.0, "mrr": 0.900},
            "temporal_comparison": {"hr5": 86.7, "mrr": 0.694},
        },
        "company_breakdown": {
            "AAPL": {"hr5": 80.0, "mrr": 0.660},
            "MSFT": {"hr5": 75.0, "mrr": 0.620},
            "AMZN": {"hr5": 75.0, "mrr": 0.610},
            "GOOGL": {"hr5": 75.0, "mrr": 0.620},
            "META": {"hr5": 80.0, "mrr": 0.640},
        },
    },
    {
        "run_name": "N4-Hybrid-BM25-Rerank",
        "metrics": {
            "hit_rate_at_1": 63.0,
            "hit_rate_at_3": 77.0,
            "hit_rate_at_5": 81.0,
            "mrr": 0.7002,
        },
        "params": {
            "retrieval_mode": "hybrid+rerank",
            "top_k": 5,
            "hybrid": True,
            "rerank": True,
            "total_queries": 100,
        },
        "type_breakdown": {
            "numeric_extraction":  {"hr5": 79.6, "mrr": 0.671},
            "numeric_reasoning":   {"hr5": 85.7, "mrr": 0.691},
            "tabular":             {"hr5": 100.0, "mrr": 0.950},
            "qualitative":         {"hr5": 72.2, "mrr": 0.611},
            "segment_analysis":    {"hr5": 100.0, "mrr": 1.000},
            "temporal_comparison": {"hr5": 80.0, "mrr": 0.650},
        },
        "company_breakdown": {
            "AAPL": {"hr5": 85.0, "mrr": 0.720},
            "MSFT": {"hr5": 80.0, "mrr": 0.700},
            "AMZN": {"hr5": 80.0, "mrr": 0.690},
            "GOOGL": {"hr5": 80.0, "mrr": 0.700},
            "META": {"hr5": 80.0, "mrr": 0.710},
        },
    },
]


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print(f"[MLflow] Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"[MLflow] Experiment   : {MLFLOW_EXPERIMENT}")
    print()

    for run in MOCK_RUNS:
        with mlflow.start_run(run_name=run["run_name"]):
            # Core metrics
            for k, v in run["metrics"].items():
                mlflow.log_metric(k, v)

            # Params
            mlflow.log_params(run["params"])

            # Per-type breakdown
            for q_type, stat in run["type_breakdown"].items():
                safe = q_type.replace(" ", "_").replace("/", "_").lower()
                mlflow.log_metric(f"type_{safe}_hr5", stat["hr5"])
                mlflow.log_metric(f"type_{safe}_mrr",  stat["mrr"])

            # Per-company breakdown
            for company, stat in run["company_breakdown"].items():
                safe = company.lower()
                mlflow.log_metric(f"co_{safe}_hr5", stat["hr5"])
                mlflow.log_metric(f"co_{safe}_mrr",  stat["mrr"])

            print(f"  [OK] Logged run: {run['run_name']}  "
                  f"HR@5={run['metrics']['hit_rate_at_5']:.1f}%  "
                  f"MRR={run['metrics']['mrr']:.4f}")

    print()
    print("=" * 60)
    print("All 4 runs logged successfully!")
    print(f"Open the dashboard: http://127.0.0.1:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
