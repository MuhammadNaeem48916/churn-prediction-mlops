"""
Metric gating script — run after training in CI.
Fails the pipeline if metrics don't meet minimum thresholds.
"""
import json
import sys

THRESHOLDS = {
    "roc_auc":   0.70,
    "recall":    0.60,   # catching churners matters most
    "f1_score":  0.55,
}

def check_metrics(metrics_path="metrics/scores.json"):
    with open(metrics_path) as f:
        metrics = json.load(f)

    print("=" * 50)
    print("  METRIC GATE CHECK")
    print("=" * 50)

    failed = []
    for metric_name, threshold in THRESHOLDS.items():
        value = metrics.get(metric_name)
        status = "PASS" if value >= threshold else "FAIL"
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {metric_name:12s}: {value:.4f}  (threshold: {threshold})  [{status}]")
        if status == "FAIL":
            failed.append(metric_name)

    print("=" * 50)

    if failed:
        print(f"\n❌ METRIC GATE FAILED — {', '.join(failed)} below threshold")
        print("Model will NOT be deployed.")
        sys.exit(1)
    else:
        print("\n✅ METRIC GATE PASSED — model meets all thresholds")
        sys.exit(0)


if __name__ == "__main__":
    check_metrics()
