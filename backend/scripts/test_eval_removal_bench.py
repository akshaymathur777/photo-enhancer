"""Fast unit checks for the removal eval harness helpers.

Covers the perceptual metric on synthetic fills, the coverage/partial-run rules in
summary.json, and baseline save/diff regression flagging. Runs in seconds and never
touches the real inpaint backends or backend/scripts/_eval artifacts.

    python scripts\\test_eval_removal_bench.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from removal_baseline import (  # noqa: E402
    build_summary,
    diff_against_baseline,
    exit_code_for,
    list_baselines,
    load_baseline,
    save_baseline,
)
from removal_metrics import (  # noqa: E402
    get_backend,
    perceptual_context_metrics,
    perceptual_image_distance,
)

SIDE = 420
HOLE = slice(150, 280)


def _textured_scene(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
    base = 120 + 45 * np.sin(xx / 7.0) * np.cos(yy / 9.0) + 25 * np.sin((xx + yy) / 4.0)
    arr = np.stack([base, base * 0.92 + 15, base * 0.8 + 30], axis=-1)
    arr += rng.normal(0, 12.0, size=arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _mask() -> np.ndarray:
    m = np.zeros((SIDE, SIDE), dtype=bool)
    m[HOLE, HOLE] = True
    return m


def _fill_flat(scene: np.ndarray, hard: np.ndarray) -> np.ndarray:
    out = scene.copy()
    out[hard] = scene[~hard].reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return out


def _fill_blurred(scene: np.ndarray, hard: np.ndarray) -> np.ndarray:
    out = scene.copy()
    blur = cv2.GaussianBlur(scene, (0, 0), 6.0)
    out[hard] = blur[hard]
    return out


def test_perceptual_metric() -> None:
    backend = get_backend("squeeze")
    scene = _textured_scene()
    hard = _mask()

    if not backend.available:
        res = perceptual_context_metrics(scene, hard)
        assert res.context_score is None, "score must be None without a backend"
        assert res.reason and "lpips" in res.reason.lower(), res.reason
        dist, reason = perceptual_image_distance(scene, scene)
        assert dist is None and reason, "image distance must degrade, not crash"
        print(f"SKIP perceptual values — backend unavailable: {res.reason}")
        return

    consistent = perceptual_context_metrics(scene, hard)
    flat = perceptual_context_metrics(_fill_flat(scene, hard), hard)
    blurred = perceptual_context_metrics(_fill_blurred(scene, hard), hard)

    for label, r in (("consistent", consistent), ("flat", flat), ("blurred", blurred)):
        assert r.context_score is not None, f"{label}: no score ({r.reason})"
        assert r.blur_ratio is not None, f"{label}: no blur ratio"
        print(
            f"  {label:<11} context_score={r.context_score:.3f} "
            f"blur_ratio={r.blur_ratio:.3f} patch={r.patch_px}px backend={r.backend}"
        )

    assert flat.context_score > consistent.context_score, (
        f"flat plaque must score worse than real texture "
        f"({flat.context_score:.3f} vs {consistent.context_score:.3f})"
    )
    assert blurred.context_score > consistent.context_score, (
        f"blurry fill must score worse than real texture "
        f"({blurred.context_score:.3f} vs {consistent.context_score:.3f})"
    )
    assert blurred.blur_ratio < consistent.blur_ratio, (
        f"blur_ratio must drop for a smoothed fill "
        f"({blurred.blur_ratio:.3f} vs {consistent.blur_ratio:.3f})"
    )
    assert flat.blur_ratio < consistent.blur_ratio, "blur_ratio must drop for a flat fill"

    thin = np.zeros((SIDE, SIDE), dtype=bool)
    thin[200:204, 100:300] = True
    res_thin = perceptual_context_metrics(scene, thin)
    assert res_thin.context_score is None and res_thin.reason, "thin mask must degrade cleanly"

    empty = perceptual_context_metrics(scene, np.zeros((SIDE, SIDE), dtype=bool))
    assert empty.context_score is None and empty.reason == "mask empty"

    same, reason = perceptual_image_distance(scene, scene)
    assert reason is None and same is not None and same < 1e-4, f"self distance {same}"
    other, _ = perceptual_image_distance(scene, _fill_flat(scene, hard))
    assert other is not None and other > same, "flat fill must be perceptually distant"
    print(f"  image distance: self={same:.5f} vs-flat={other:.5f}")
    print("OK perceptual metric")


def _case(name: str, *, passed: bool = True, **over) -> dict:
    case = {
        "name": name,
        "pass_case": passed,
        "latency_total_s": 100.0,
        "ghost_residual_pct": 10.0,
        "plaque_score": 0.20,
        "protect_drift": 0.0,
        "edge_persist_pct": 4.0,
        "perceptual_context_score": 1.30,
        "perceptual_blur_ratio": 0.90,
        "mean_abs_diff_mask": 50.0,
    }
    case.update(over)
    return case


def test_summary_coverage() -> None:
    registered = ["coastal", "library"]

    full = build_summary(
        registered=registered,
        requested=registered,
        cases=[_case("coastal"), _case("library")],
        skipped={},
    )
    assert full["status"] == "PASS"
    assert full["all_pass"] is True and full["full_suite_pass"] is True
    assert full["partial_run"] is False
    assert exit_code_for(full, partial_allowed=False) == 0

    partial = build_summary(
        registered=registered,
        requested=["coastal"],
        cases=[_case("coastal")],
        skipped={"library": "not requested (--only coastal)"},
        config={"only": "coastal"},
    )
    assert partial["status"] == "PARTIAL", partial["status"]
    assert partial["all_pass"] is False, "a --only run must never report all_pass"
    assert partial["full_suite_pass"] is False
    assert partial["partial_run"] is True
    assert partial["ran_cases_pass"] is True, "the case that ran still reports its own pass"
    assert partial["coverage"]["missing"] == ["library"]
    assert partial["coverage"]["skipped"]["library"]
    assert partial["apple_claim_allowed"] is False
    assert exit_code_for(partial, partial_allowed=True) == 0
    assert exit_code_for(partial, partial_allowed=False) == 1

    missing_asset = build_summary(
        registered=registered,
        requested=registered,
        cases=[_case("coastal")],
        skipped={"library": "asset missing: assets/foo.jpeg"},
    )
    assert missing_asset["status"] == "PARTIAL" and missing_asset["all_pass"] is False
    assert exit_code_for(missing_asset, partial_allowed=False) == 1, (
        "a full-suite run that lost a case must not exit 0"
    )

    failed = build_summary(
        registered=registered,
        requested=registered,
        cases=[_case("coastal"), _case("library", passed=False)],
        skipped={},
    )
    assert failed["status"] == "FAIL" and failed["all_pass"] is False
    assert exit_code_for(failed, partial_allowed=True) == 1

    none_ran = build_summary(
        registered=registered, requested=registered, cases=[], skipped={"coastal": "x"}
    )
    assert none_ran["status"] == "NO_CASES" and none_ran["all_pass"] is False
    assert exit_code_for(none_ran, partial_allowed=True) == 1

    for s in (full, partial, missing_asset, failed, none_ran):
        json.dumps(s)
        assert s["apple_claim_allowed"] is False
    print("OK summary coverage (partial run cannot read as a full pass)")


def test_baseline_diff() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="eval_baseline_"))
    try:
        for case in ("coastal", "library"):
            (tmp / case).mkdir(parents=True)
            (tmp / case / "after.jpg").write_bytes(b"fake")
            (tmp / case / "scores.json").write_text("{}", encoding="utf-8")

        base_summary = build_summary(
            registered=["coastal", "library"],
            requested=["coastal", "library"],
            cases=[_case("coastal"), _case("library")],
            skipped={},
        )
        dest = save_baseline(tmp, "v1", base_summary)
        assert (dest / "summary.json").is_file()
        assert (dest / "meta.json").is_file()
        assert (dest / "coastal" / "after.jpg").is_file(), "artifacts must be copied"
        assert list_baselines(tmp) == ["v1"]
        assert json.loads((dest / "meta.json").read_text(encoding="utf-8"))["partial_run"] is False

        try:
            save_baseline(tmp, "v1", base_summary)
        except FileExistsError:
            pass
        else:
            raise AssertionError("overwriting a baseline must require --overwrite-baseline")
        save_baseline(tmp, "v1", base_summary, overwrite=True)

        loaded = load_baseline(tmp, "v1")
        assert [c["name"] for c in loaded["cases"]] == ["coastal", "library"]

        clean = build_summary(
            registered=["coastal", "library"],
            requested=["coastal", "library"],
            cases=[_case("coastal", latency_total_s=101.5), _case("library")],
            skipped={},
        )
        diff = diff_against_baseline(loaded, clean, name="v1")
        assert diff["has_regression"] is False, diff["regressions"]
        assert diff["baseline"] == "v1"
        assert diff["cases"]["coastal"]["metrics"]["latency_total_s"]["delta"] == 1.5

        worse = build_summary(
            registered=["coastal", "library"],
            requested=["coastal", "library"],
            cases=[
                _case(
                    "coastal",
                    latency_total_s=180.0,
                    ghost_residual_pct=25.0,
                    perceptual_context_score=1.90,
                    perceptual_blur_ratio=0.40,
                ),
                _case("library", passed=False),
            ],
            skipped={},
        )
        diff = diff_against_baseline(loaded, worse, name="v1")
        assert diff["has_regression"] is True
        flagged = {
            m
            for case in diff["cases"].values()
            for m, entry in case.get("metrics", {}).items()
            if entry.get("regressed")
        }
        assert {"ghost_residual_pct", "perceptual_context_score",
                "perceptual_blur_ratio"} <= flagged, flagged
        assert "latency_total_s" not in flagged, (
            "latency is advisory — machine load moves it far more than code does"
        )
        assert any("pass_case: True -> False" in r for r in diff["regressions"])
        assert diff["cases"]["coastal"]["metrics"]["mean_abs_diff_mask"]["regressed"] is False, (
            "informational metrics must never flag a regression"
        )

        # Latency: reported, never fatal, and still loud when it is real.
        noisy = build_summary(
            registered=["coastal", "library"],
            requested=["coastal", "library"],
            cases=[_case("coastal", latency_total_s=138.0), _case("library")],
            skipped={},
        )
        diff = diff_against_baseline(loaded, noisy, name="v1")
        assert diff["has_regression"] is False, diff["regressions"]
        assert diff["has_advisory"] is False, "noise-scale latency drift must stay silent"

        slow = build_summary(
            registered=["coastal", "library"],
            requested=["coastal", "library"],
            cases=[_case("coastal", latency_total_s=1000.0), _case("library")],
            skipped={},
        )
        diff = diff_against_baseline(loaded, slow, name="v1")
        assert diff["has_regression"] is False, "a 10x slowdown must not fail the quality gate"
        assert diff["has_advisory"] is True
        entry = diff["cases"]["coastal"]["metrics"]["latency_total_s"]
        assert entry["advisory_flag"] == "slower" and entry["factor"] == 10.0, entry
        assert any("10.0x slower" in a for a in diff["advisories"]), diff["advisories"]

        fast = build_summary(
            registered=["coastal", "library"],
            requested=["coastal", "library"],
            cases=[_case("coastal", latency_total_s=100.0 / 45.0), _case("library")],
            skipped={},
        )
        diff = diff_against_baseline(loaded, fast, name="v1")
        assert diff["has_regression"] is False
        entry = diff["cases"]["coastal"]["metrics"]["latency_total_s"]
        assert entry["advisory_flag"] == "faster" and entry["speedup"] == 45.0, entry
        assert any("45.0x faster" in a for a in diff["advisories"]), diff["advisories"]

        loose = diff_against_baseline(loaded, worse, name="v1", scale=50.0)
        assert not any(
            entry.get("regressed")
            for case in loose["cases"].values()
            for entry in case.get("metrics", {}).values()
        ), "--regression-scale must widen tolerances"

        lost = build_summary(
            registered=["coastal", "library"],
            requested=["coastal", "library"],
            cases=[
                _case("coastal", perceptual_context_score=None, perceptual_blur_ratio=None),
                _case("library"),
            ],
            skipped={},
        )
        diff = diff_against_baseline(loaded, lost, name="v1")
        assert diff["has_regression"] is True, "losing a gated metric must be a regression"
        assert diff["cases"]["coastal"]["metrics"]["perceptual_context_score"]["regressed"] is True
        assert any("metric lost" in r for r in diff["regressions"])

        dropped = build_summary(
            registered=["coastal", "library"],
            requested=["coastal"],
            cases=[_case("coastal")],
            skipped={"library": "not requested (--only coastal)"},
        )
        diff = diff_against_baseline(loaded, dropped, name="v1")
        assert diff["coverage_regression"] is True
        assert diff["current_partial"] is True
        assert any("did not run" in r for r in diff["regressions"])

        try:
            load_baseline(tmp, "nope")
        except FileNotFoundError as exc:
            assert "v1" in str(exc)
        else:
            raise AssertionError("missing baseline must raise")
        print("OK baseline save/diff/regression flagging")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_summary_coverage()
    test_baseline_diff()
    test_perceptual_metric()
    print("\nall eval-harness unit checks passed")
