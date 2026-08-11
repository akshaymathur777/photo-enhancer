"""Characterise run-to-run latency variance of the removal eval bench.

The regression gate compares one fresh-process bench run against a baseline
recorded by another fresh-process run, so that is exactly what this measures:
N sequential invocations of eval_removal_bench.py, no shared warm state.

Reports per-case spread for every metric the gate can trip on, plus the worst
run-to-run jump actually observed — the number a tolerance has to clear to stop
firing on noise.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "eval_removal_bench.py"

LATENCY_METRICS = ("latency_segment_s", "latency_inpaint_s", "latency_total_s")
QUALITY_METRICS = (
    "ghost_residual_pct",
    "plaque_score",
    "protect_drift",
    "edge_persist_pct",
    "perceptual_context_score",
    "perceptual_blur_ratio",
    "mean_abs_diff_mask",
)


def _spread(values: list[float]) -> dict:
    lo, hi = min(values), max(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    # Worst relative jump between any two runs, i.e. what a baseline diff sees.
    worst_rel = max(
        (b - a) / a for a in values for b in values if a > 0
    ) if all(v > 0 for v in values) else float("nan")
    return {
        "n": len(values),
        "values": [round(v, 3) for v in values],
        "min": round(lo, 3),
        "max": round(hi, 3),
        "mean": round(mean, 3),
        "sd": round(sd, 3),
        "cv_pct": round(sd / mean * 100, 2) if mean else None,
        "max_over_min_pct": round((hi - lo) / lo * 100, 2) if lo > 0 else None,
        "worst_pairwise_increase_pct": round(worst_rel * 100, 2),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="bench latency variance")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--mode", default="quick")
    parser.add_argument("--quality", default="best")
    parser.add_argument("--scratch", default=str(HERE / "_eval_variance"))
    parser.add_argument("--out", default=str(HERE / "_eval" / "latency_variance.json"))
    args = parser.parse_args()

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    for i in range(args.runs):
        out_dir = scratch / f"run_{i:02d}"
        cmd = [
            sys.executable,
            str(BENCH),
            "--mode",
            args.mode,
            "--quality",
            args.quality,
            "--out",
            str(out_dir),
        ]
        print(f"[run {i + 1}/{args.runs}] {args.mode}/{args.quality} -> {out_dir}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        summary_file = out_dir / "summary.json"
        if not summary_file.is_file():
            print(f"  FAILED (rc={proc.returncode})\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}", flush=True)
            continue
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        runs.append({"index": i, "rc": proc.returncode, "cases": summary["cases"]})
        for case in summary["cases"]:
            print(
                f"  {case['name']}: seg={case['latency_segment_s']}s "
                f"inpaint={case['latency_inpaint_s']}s total={case['latency_total_s']}s "
                f"ghost={case['ghost_residual_pct']} plaque={case['plaque_score']}",
                flush=True,
            )

    names = sorted({c["name"] for r in runs for c in r["cases"]})
    stats: dict = {}
    for name in names:
        series = [c for r in runs for c in r["cases"] if c["name"] == name]
        stats[name] = {}
        for metric in LATENCY_METRICS + QUALITY_METRICS:
            vals = [float(c[metric]) for c in series if isinstance(c.get(metric), (int, float))]
            if len(vals) >= 2:
                stats[name][metric] = _spread(vals)

    payload = {
        "config": {"runs": args.runs, "mode": args.mode, "quality": args.quality},
        "completed_runs": len(runs),
        "stats": stats,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'case.metric':<34}{'mean':>9}{'sd':>8}{'cv%':>7}{'min':>9}{'max':>9}{'worst jump%':>13}")
    for name, metrics in stats.items():
        for metric, s in metrics.items():
            print(
                f"{name + '.' + metric:<34}{s['mean']:>9.3f}{s['sd']:>8.3f}"
                f"{(s['cv_pct'] or 0):>7.2f}{s['min']:>9.3f}{s['max']:>9.3f}"
                f"{s['worst_pairwise_increase_pct']:>13.2f}",
                flush=True,
            )
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
