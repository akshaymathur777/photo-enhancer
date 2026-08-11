"""Coverage-aware summaries and named baselines for the removal eval bench.

Two jobs:

* ``build_summary`` — a summary may only read as a full-suite pass when every
  registered case actually ran and passed. A ``--only`` run is explicitly marked
  ``PARTIAL`` and its ``all_pass`` / ``full_suite_pass`` stay false, so a partial
  artifact can never be mistaken for a green suite.
* ``save_baseline`` / ``diff_against_baseline`` — record a run under a name, then
  report per-metric deltas on a later run and flag regressions past a tolerance.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

SCHEMA_VERSION = 2

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_PARTIAL = "PARTIAL"
STATUS_NO_CASES = "NO_CASES"

BASELINE_DIRNAME = "_baselines"
BASELINE_ARTIFACTS = ("before.jpg", "after.jpg", "mask.jpg", "scores.json")


@dataclass(frozen=True)
class MetricSpec:
    direction: str
    abs_tol: float = 0.0
    rel_tol: float = 0.0
    # False: report the delta but never fail the run (see LATENCY IS ADVISORY).
    gating: bool = True


LOWER = "lower_better"
HIGHER = "higher_better"
INFO = "info"

# A latency change of this factor or better is called out as a notable speedup,
# so a real optimisation stays visible even though latency cannot gate.
ADVISORY_SPEEDUP_FACTOR = 0.5

# LATENCY IS ADVISORY — calibrated, not guessed.
#
# scripts/latency_variance.py, 8 consecutive full-suite runs at --mode quick on
# an idle machine (GPU at 0%, 584 MiB used), fresh process per run:
#
#   metric                       cv%    worst run-to-run increase
#   coastal.latency_segment_s    8.2%   +35.1%
#   coastal.latency_inpaint_s   11.6%   +37.9%
#   coastal.latency_total_s     10.8%   +34.0%
#   library.latency_segment_s    8.4%   +26.5%
#   library.latency_inpaint_s    5.5%   +13.9%
#   library.latency_total_s      5.7%   +14.7%
#
#   every quality metric, both cases: sd = 0.000 (bit-identical, all 8 runs)
#
# So the old 25-35% latency tolerances sat *inside* the noise floor and fired on
# identical output. Worse, contention makes it unbounded: the same SDXL inpaint
# measured 46s idle and 824s (+1690%) with another GPU app resident. Latency here
# is a property of the machine, not of the code, while the quality metrics are
# exactly reproducible — so quality gates and latency only informs. The tolerance
# below (3x) is kept as the threshold for shouting about a latency change, not
# for failing on one.
METRIC_SPECS: dict[str, MetricSpec] = {
    "latency_segment_s": MetricSpec(LOWER, 5.0, 2.0, gating=False),
    "latency_inpaint_s": MetricSpec(LOWER, 5.0, 2.0, gating=False),
    "latency_total_s": MetricSpec(LOWER, 5.0, 2.0, gating=False),
    "ghost_residual_pct": MetricSpec(LOWER, 2.0, 0.15),
    "plaque_score": MetricSpec(LOWER, 0.05, 0.20),
    "protect_drift": MetricSpec(LOWER, 1.0, 0.25),
    "edge_persist_pct": MetricSpec(LOWER, 1.0, 0.25),
    "perceptual_context_score": MetricSpec(LOWER, 0.10, 0.15),
    "perceptual_blur_ratio": MetricSpec(HIGHER, 0.08, 0.12),
    "mean_abs_diff_mask": MetricSpec(INFO),
    "mask_coverage_pct": MetricSpec(INFO),
    "hf_energy_fill": MetricSpec(INFO),
    "hf_energy_ring": MetricSpec(INFO),
    "perceptual_fill_vs_context": MetricSpec(INFO),
    "perceptual_context_vs_context": MetricSpec(INFO),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_summary(
    *,
    registered: Iterable[str],
    requested: Iterable[str],
    cases: list[dict],
    skipped: dict[str, str],
    config: dict | None = None,
) -> dict:
    registered = list(registered)
    requested = list(requested)
    ran = [c["name"] for c in cases]
    missing = [n for n in registered if n not in ran]
    complete = not missing
    passed = [c["name"] for c in cases if bool(c.get("pass_case"))]
    ran_cases_pass = bool(cases) and len(passed) == len(cases)
    full_suite_pass = bool(complete and ran_cases_pass)

    if not cases:
        status = STATUS_NO_CASES
    elif not ran_cases_pass:
        status = STATUS_FAIL
    elif complete:
        status = STATUS_PASS
    else:
        status = STATUS_PARTIAL

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "status": status,
        "partial_run": not complete,
        "coverage": {
            "registered": registered,
            "requested": requested,
            "ran": ran,
            "missing": missing,
            "skipped": dict(skipped),
            "complete": complete,
        },
        "config": dict(config or {}),
        "cases": cases,
        "cases_passed": passed,
        "ran_cases_pass": ran_cases_pass,
        "full_suite_pass": full_suite_pass,
        "all_pass": full_suite_pass,
        "apple_claim_allowed": False,
    }


def exit_code_for(summary: dict, *, partial_allowed: bool) -> int:
    """0 only for a clean full suite, or an explicitly requested partial run that passed."""
    if summary["status"] == STATUS_PASS:
        return 0
    if summary["status"] == STATUS_PARTIAL and partial_allowed:
        return 0
    return 1


def baselines_dir(out: Path) -> Path:
    return out / BASELINE_DIRNAME


def baseline_path(out: Path, name: str) -> Path:
    if not name or any(ch in name for ch in '\\/:*?"<>|'):
        raise ValueError(f"invalid baseline name: {name!r}")
    return baselines_dir(out) / name


def list_baselines(out: Path) -> list[str]:
    root = baselines_dir(out)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def save_baseline(out: Path, name: str, summary: dict, *, overwrite: bool = False) -> Path:
    dest = baseline_path(out, name)
    if dest.exists():
        if not overwrite:
            raise FileExistsError(
                f"baseline '{name}' already exists at {dest}; pass --overwrite-baseline "
                f"or choose another name"
            )
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for case in summary.get("cases", []):
        case_src = out / case["name"]
        case_dst = dest / case["name"]
        case_dst.mkdir(parents=True, exist_ok=True)
        for artifact in BASELINE_ARTIFACTS:
            src = case_src / artifact
            if src.is_file():
                shutil.copy2(src, case_dst / artifact)
    (dest / "meta.json").write_text(
        json.dumps(
            {
                "name": name,
                "recorded_at": _now(),
                "status": summary.get("status"),
                "partial_run": bool(summary.get("partial_run")),
                "coverage": summary.get("coverage", {}),
                "config": summary.get("config", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dest


def load_baseline(out: Path, name: str) -> dict:
    dest = baseline_path(out, name)
    summary_file = dest / "summary.json"
    if not summary_file.is_file():
        known = list_baselines(out)
        raise FileNotFoundError(
            f"no baseline '{name}' under {baselines_dir(out)}"
            + (f" (known: {', '.join(known)})" if known else " (none recorded yet)")
        )
    return json.loads(summary_file.read_text(encoding="utf-8"))


def _tolerance(spec: MetricSpec, baseline: float, scale: float) -> float:
    return scale * max(spec.abs_tol, spec.rel_tol * abs(baseline))


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def diff_metrics(base_case: dict, cur_case: dict, *, scale: float = 1.0) -> dict:
    metrics: dict[str, dict] = {}
    for name, spec in METRIC_SPECS.items():
        b = _numeric(base_case.get(name))
        c = _numeric(cur_case.get(name))
        if b is None and c is None:
            continue
        entry: dict[str, object] = {
            "baseline": b,
            "current": c,
            "direction": spec.direction,
        }
        if b is None or c is None:
            lost = b is not None and c is None and spec.direction != INFO and spec.gating
            entry["regressed"] = bool(lost)
            entry["note"] = (
                "metric lost — recorded in baseline, missing now"
                if lost
                else "missing on one side"
            )
            metrics[name] = entry
            continue
        delta = c - b
        tol = _tolerance(spec, b, scale)
        exceeded = False
        if spec.direction == LOWER:
            exceeded = delta > tol
        elif spec.direction == HIGHER:
            exceeded = -delta > tol
        entry.update(
            {
                "delta": round(delta, 4),
                "pct": round(delta / abs(b) * 100.0, 2) if b else None,
                "tolerance": round(tol, 4),
                "exceeded_tolerance": bool(exceeded),
                "gating": bool(spec.gating),
                "regressed": bool(exceeded and spec.gating),
            }
        )
        if not spec.gating and spec.direction != INFO:
            entry["advisory"] = True
            factor = round(c / b, 3) if b > 0 and c > 0 else None
            entry["factor"] = factor
            if exceeded:
                entry["advisory_flag"] = "slower"
            elif b > 0 and c <= b * ADVISORY_SPEEDUP_FACTOR:
                entry["advisory_flag"] = "faster"
                entry["speedup"] = round(b / c, 2) if c > 0 else None
        metrics[name] = entry
    return metrics


def _advisory_shape(entry: dict) -> str:
    """Human-readable size of a latency change: '24.68x faster', '10.0x slower'."""
    flag = entry.get("advisory_flag")
    magnitude = entry.get("speedup") if flag == "faster" else entry.get("factor")
    return f"{magnitude}x {flag}" if magnitude else str(flag)


def diff_against_baseline(
    baseline: dict,
    current: dict,
    *,
    name: str = "",
    scale: float = 1.0,
    image_distance: Callable[[str], tuple[float | None, str | None]] | None = None,
) -> dict:
    base_cases = {c["name"]: c for c in baseline.get("cases", [])}
    cur_cases = {c["name"]: c for c in current.get("cases", [])}
    regressions: list[str] = []
    advisories: list[str] = []
    cases: dict[str, dict] = {}

    for case_name in sorted(set(base_cases) | set(cur_cases)):
        base_case = base_cases.get(case_name)
        cur_case = cur_cases.get(case_name)
        entry: dict[str, object] = {
            "in_baseline": base_case is not None,
            "in_current": cur_case is not None,
        }
        if base_case is None or cur_case is None:
            if cur_case is None:
                regressions.append(f"{case_name}: present in baseline but did not run")
            cases[case_name] = entry
            continue
        base_pass = bool(base_case.get("pass_case"))
        cur_pass = bool(cur_case.get("pass_case"))
        entry["pass_case_baseline"] = base_pass
        entry["pass_case_current"] = cur_pass
        metrics = diff_metrics(base_case, cur_case, scale=scale)
        entry["metrics"] = metrics
        if base_pass and not cur_pass:
            regressions.append(f"{case_name}.pass_case: True -> False")
        for metric, m in metrics.items():
            if m.get("advisory_flag"):
                advisories.append(
                    f"{case_name}.{metric}: {m['baseline']} -> {m['current']} "
                    f"({_advisory_shape(m)}) — advisory, does not fail the run"
                )
            if not m.get("regressed"):
                continue
            if m.get("delta") is None:
                regressions.append(f"{case_name}.{metric}: {m['note']}")
            else:
                regressions.append(
                    f"{case_name}.{metric}: {m['baseline']} -> {m['current']} "
                    f"(delta {m['delta']:+}, tol {m['tolerance']})"
                )
        if image_distance is not None:
            dist, reason = image_distance(case_name)
            entry["perceptual_vs_baseline"] = None if dist is None else round(dist, 4)
            entry["perceptual_vs_baseline_reason"] = reason
        cases[case_name] = entry

    coverage_regression = bool(set(base_cases) - set(cur_cases))
    return {
        "baseline": name,
        "baseline_recorded_at": baseline.get("generated_at"),
        "baseline_status": baseline.get("status"),
        "baseline_partial": bool(baseline.get("partial_run")),
        "current_status": current.get("status"),
        "current_partial": bool(current.get("partial_run")),
        "tolerance_scale": scale,
        "cases": cases,
        "coverage_regression": coverage_regression,
        "regressions": regressions,
        "has_regression": bool(regressions),
        "advisories": advisories,
        "has_advisory": bool(advisories),
    }


def format_diff(diff: dict) -> str:
    lines: list[str] = []
    for name, case in diff["cases"].items():
        if not case.get("in_current"):
            lines.append(f"  {name}: MISSING in current run (was in baseline)")
            continue
        if not case.get("in_baseline"):
            lines.append(f"  {name}: new case, no baseline to compare")
            continue
        lines.append(f"  {name}:")
        for metric, m in case.get("metrics", {}).items():
            flag = "REGRESSED" if m.get("regressed") else ""
            if m.get("advisory"):
                flag = "advisory (never fails)"
                if m.get("advisory_flag"):
                    flag = f"ADVISORY — {_advisory_shape(m)}"
            if m.get("baseline") is None or m.get("current") is None:
                lines.append(f"    {metric:<30} {m.get('note', 'n/a')} {flag}")
                continue
            pct = "" if m.get("pct") is None else f" ({m['pct']:+.1f}%)"
            lines.append(
                f"    {metric:<30} {m['baseline']:>10.4f} -> {m['current']:>10.4f}"
                f" {m['delta']:+.4f}{pct} {flag}"
            )
        dist = case.get("perceptual_vs_baseline")
        if dist is not None:
            lines.append(f"    {'perceptual_vs_baseline':<30} {dist:>10}")
    if diff["baseline_partial"]:
        lines.append("  NOTE: baseline itself was a PARTIAL run")
    return "\n".join(lines)
