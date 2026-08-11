"""Local object-removal eval harness — coastal/library golden gates.

Writes before/after crops + JSON scores under backend/scripts/_eval/.
Does not claim Apple parity; prints PASS/FAIL against automated thresholds.

summary.json records coverage: a `--only` run is marked PARTIAL and can never
report full_suite_pass/all_pass. `--save-baseline NAME` records a run and
`--baseline NAME` diffs the current run against it, flagging regressions.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import configure_runtime_paths  # noqa: E402

configure_runtime_paths()

from services.object_removal import (  # noqa: E402
    auto_detect_others_mask,
    inpaint_objects,
    mask_overlay_preview,
    segment_from_points,
    _non_target_people_protect,
    _pil_to_rgb_array,
)

from removal_baseline import (  # noqa: E402
    baseline_path,
    build_summary,
    diff_against_baseline,
    exit_code_for,
    format_diff,
    list_baselines,
    load_baseline,
    save_baseline,
)
from removal_metrics import (  # noqa: E402
    PerceptualResult,
    perceptual_context_metrics,
    perceptual_image_distance,
)

OUT = Path(__file__).resolve().parent / "_eval"
OUT.mkdir(parents=True, exist_ok=True)

COASTAL = ROOT / "assets" / "WhatsApp Image 2026-07-16 at 9.31.13 PM.jpeg"
LIBRARY = ROOT / "assets" / "WhatsApp Image 2026-07-16 at 9.23.51 PM.jpeg"


@dataclass
class CaseResult:
    name: str
    mode: str
    quality: str
    w: int
    h: int
    mask_coverage_pct: float
    latency_segment_s: float
    latency_inpaint_s: float
    latency_total_s: float
    mean_abs_diff_mask: float
    ghost_residual_pct: float
    plaque_score: float
    hf_energy_fill: float
    hf_energy_ring: float
    protect_drift: float
    edge_persist_pct: float
    pass_ghost: bool
    pass_plaque: bool
    pass_protect: bool
    pass_case: bool
    perceptual_backend: str | None = None
    perceptual_reason: str | None = None
    perceptual_patch_px: int | None = None
    perceptual_fill_vs_context: float | None = None
    perceptual_context_vs_context: float | None = None
    perceptual_context_score: float | None = None
    perceptual_blur_ratio: float | None = None
    pass_perceptual: bool | None = None
    perceptual_gated: bool = False


@dataclass
class CaseSpec:
    name: str
    path: Path
    click: tuple[int, int]
    ghost_max: float
    plaque_max: float
    protect_max: float
    remove_others: bool = False
    lock: tuple[int, int] | None = None
    max_side_cap: int | None = None


# Full-res coastal click (1088,1200) on 2316x3088 — third person behind couple.
# Library: remove-others with lock on the foreground subject (not a tiny face click).
CASES: dict[str, CaseSpec] = {
    "coastal": CaseSpec(
        name="coastal",
        path=COASTAL,
        click=(1088, 1200),
        ghost_max=40.0,
        plaque_max=0.60,
        protect_max=14.0,
    ),
    "library": CaseSpec(
        name="library",
        path=LIBRARY,
        click=(520, 780),
        ghost_max=45.0,
        plaque_max=0.65,
        protect_max=14.0,
        remove_others=True,
        lock=(520, 780),
        max_side_cap=1280,
    ),
}


def _r(v: float | None, nd: int) -> float | None:
    return None if v is None else round(float(v), nd)


def _hf_energy(gray: np.ndarray, region: np.ndarray) -> float:
    if not region.any():
        return 0.0
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    return float(np.abs(gray[region] - blur[region]).mean())


def _plaque_score(orig: np.ndarray, result: np.ndarray, hard: np.ndarray) -> float:
    """Higher = more milky/flat plaque vs textured ring (bad)."""
    if not hard.any():
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    g_o = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g_r = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fill_l = float(g_r[hard].mean())
    ring_l = float(g_o[ring].mean()) if ring.any() else fill_l
    wash = max(0.0, fill_l - ring_l) / 40.0
    hf_f = _hf_energy(g_r, hard)
    hf_r = _hf_energy(g_o, ring) if ring.any() else hf_f
    flat = max(0.0, (hf_r - hf_f) / max(hf_r, 1.0))
    return float(np.clip(0.55 * wash + 0.45 * flat, 0.0, 2.0))


def _edge_persist(orig: np.ndarray, result: np.ndarray, hard: np.ndarray) -> float:
    if not hard.any():
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    band = (cv2.dilate(hard.astype(np.uint8), k) > 0)
    e0 = cv2.Canny(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), 50, 130) > 0
    e1 = cv2.Canny(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY), 50, 130) > 0
    persist = (e0 & e1 & band & hard).sum()
    return float(persist) / max(float(hard.sum()), 1.0) * 100.0


def _protect_drift(orig: np.ndarray, result: np.ndarray, hard: np.ndarray) -> float:
    """Mean abs diff on non-target people (should stay near 0)."""
    protect = _non_target_people_protect(orig, hard.astype(np.uint8) * 255)
    if not protect.any():
        return 0.0
    diff = np.abs(result.astype(np.float32) - orig.astype(np.float32)).mean(axis=2)
    return float(diff[protect].mean())


def run_case(
    name: str,
    path: Path,
    click: tuple[int, int],
    *,
    mode: str = "ai",
    quality: str = "best",
    max_side: int | None = 1280,
    ghost_max: float = 35.0,
    plaque_max: float = 0.55,
    protect_max: float = 12.0,
    remove_others: bool = False,
    lock: tuple[int, int] | None = None,
    perceptual_net: str | None = "squeeze",
    perceptual_max: float = 2.2,
    perceptual_blur_min: float = 0.55,
    gate_perceptual: bool = False,
    require_perceptual: bool = False,
) -> CaseResult:
    im = Image.open(path).convert("RGB")
    ow, oh = im.size
    scale = 1.0
    if max_side and max(ow, oh) > max_side:
        scale = max_side / float(max(ow, oh))
        im = im.resize((int(ow * scale), int(oh * scale)), Image.Resampling.LANCZOS)
    w, h = im.size
    cx = int(round(click[0] * scale))
    cy = int(round(click[1] * scale))
    lock_pt = None
    if lock is not None:
        lock_pt = (float(lock[0] * scale), float(lock[1] * scale))

    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    im.save(case_dir / "before.jpg", quality=92)

    t0 = time.time()
    if remove_others:
        # Library golden: keep locked subject, remove bystanders.
        lp = lock_pt if lock_pt is not None else (float(cx), float(cy))
        mask = auto_detect_others_mask(im, lock_point=lp)
    else:
        mask = segment_from_points(im, [[float(cx), float(cy)]], [1])
    t_seg = time.time() - t0
    hard = np.array(mask) > 127
    mask_overlay_preview(im, mask).save(case_dir / "mask.jpg", quality=92)

    t1 = time.time()
    result = inpaint_objects(
        im,
        mask,
        mode=mode,
        edge_softness=1.0,
        quality=quality,
        lock_point=lock_pt if remove_others else None,
    )  # type: ignore[arg-type]
    t_inp = time.time() - t1
    result.save(case_dir / "after.jpg", quality=92)

    orig = _pil_to_rgb_array(im)
    after = _pil_to_rgb_array(result)
    diff = np.abs(after.astype(np.float32) - orig.astype(np.float32)).mean(axis=2)
    ghost = float(((diff < 18) & hard).sum() / max(hard.sum(), 1) * 100) if hard.any() else 0.0
    plaque = _plaque_score(orig, after, hard)
    g = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g0 = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    ring = (cv2.dilate(hard.astype(np.uint8), k) > 0) & (~hard)
    hf_f = _hf_energy(g, hard)
    hf_r = _hf_energy(g0, ring)
    protect = _protect_drift(orig, after, hard)
    edge_p = _edge_persist(orig, after, hard)

    if perceptual_net:
        perc = perceptual_context_metrics(after, hard, net=perceptual_net)
    else:
        perc = PerceptualResult(reason="disabled via --no-perceptual")

    pass_ghost = bool(ghost <= ghost_max)
    pass_plaque = bool(plaque <= plaque_max)
    pass_protect = bool(protect <= protect_max)
    # Require meaningful change inside mask (not a no-op)
    mad = float(diff[hard].mean()) if hard.any() else 0.0
    pass_change = bool(mad >= 8.0)
    if perc.context_score is None:
        pass_perceptual = False if require_perceptual else None
    else:
        pass_perceptual = bool(
            perc.context_score <= perceptual_max
            and (perc.blur_ratio or 0.0) >= perceptual_blur_min
        )
    perceptual_gated = bool(gate_perceptual or require_perceptual)
    pass_case = bool(
        pass_ghost and pass_plaque and pass_protect and pass_change and bool(hard.any())
    )
    if perceptual_gated:
        pass_case = bool(pass_case and pass_perceptual)

    cr = CaseResult(
        name=name,
        mode=mode,
        quality=quality,
        w=int(w),
        h=int(h),
        mask_coverage_pct=float(hard.mean() * 100),
        latency_segment_s=round(float(t_seg), 3),
        latency_inpaint_s=round(float(t_inp), 3),
        latency_total_s=round(float(t_seg + t_inp), 3),
        mean_abs_diff_mask=round(mad, 3),
        ghost_residual_pct=round(ghost, 3),
        plaque_score=round(plaque, 4),
        hf_energy_fill=round(hf_f, 3),
        hf_energy_ring=round(hf_r, 3),
        protect_drift=round(protect, 3),
        edge_persist_pct=round(edge_p, 3),
        pass_ghost=pass_ghost,
        pass_plaque=pass_plaque,
        pass_protect=pass_protect,
        pass_case=pass_case,
        perceptual_backend=perc.backend,
        perceptual_reason=perc.reason,
        perceptual_patch_px=perc.patch_px,
        perceptual_fill_vs_context=_r(perc.fill_vs_context, 4),
        perceptual_context_vs_context=_r(perc.context_vs_context, 4),
        perceptual_context_score=_r(perc.context_score, 4),
        perceptual_blur_ratio=_r(perc.blur_ratio, 4),
        pass_perceptual=pass_perceptual,
        perceptual_gated=perceptual_gated,
    )
    (case_dir / "scores.json").write_text(json.dumps(asdict(cr), indent=2), encoding="utf-8")
    status = "PASS" if pass_case else "FAIL"
    if perc.context_score is None:
        perc_txt = f"perc=n/a ({perc.reason})"
    else:
        perc_txt = f"perc={perc.context_score:.2f} blur={perc.blur_ratio:.2f}"
    print(
        f"[{status}] {name} ghost={ghost:.1f}% plaque={plaque:.3f} protect={protect:.1f} "
        f"mad={mad:.1f} cov={hard.mean()*100:.2f}% {perc_txt} total={t_seg+t_inp:.1f}s",
        flush=True,
    )
    return cr


def _baseline_image_distance(baseline_name: str, net: str | None):
    def compare(case: str) -> tuple[float | None, str | None]:
        if not net:
            return None, "perceptual metric disabled"
        cur = OUT / case / "after.jpg"
        base = baseline_path(OUT, baseline_name) / case / "after.jpg"
        if not cur.is_file() or not base.is_file():
            return None, "after.jpg missing on one side"
        a = _pil_to_rgb_array(Image.open(cur).convert("RGB"))
        b = _pil_to_rgb_array(Image.open(base).convert("RGB"))
        return perceptual_image_distance(a, b, net=net)

    return compare


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Removal eval bench")
    parser.add_argument("--only", choices=[*CASES, "all"], default="all")
    parser.add_argument("--mode", choices=["ai", "quick"], default="ai")
    parser.add_argument("--quality", default="best")
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument(
        "--perceptual-net",
        default="squeeze",
        help="LPIPS backbone (squeeze|alex|vgg); squeeze has the smallest weights",
    )
    parser.add_argument(
        "--no-perceptual", action="store_true", help="skip the perceptual metric entirely"
    )
    parser.add_argument("--perceptual-max", type=float, default=2.2)
    parser.add_argument("--perceptual-blur-min", type=float, default=0.55)
    parser.add_argument(
        "--gate-perceptual",
        action="store_true",
        help="fold pass_perceptual into pass_case (thresholds are uncalibrated; off by default)",
    )
    parser.add_argument(
        "--require-perceptual",
        action="store_true",
        help="fail a case when the perceptual metric could not be computed",
    )
    parser.add_argument("--save-baseline", metavar="NAME")
    parser.add_argument("--overwrite-baseline", action="store_true")
    parser.add_argument("--baseline", metavar="NAME", help="diff this run against a saved baseline")
    parser.add_argument(
        "--regression-scale",
        type=float,
        default=1.0,
        help="multiplier on every per-metric regression tolerance",
    )
    parser.add_argument("--no-fail-on-regression", action="store_true")
    parser.add_argument("--list-baselines", action="store_true")
    parser.add_argument(
        "--out",
        metavar="DIR",
        help="write artifacts somewhere other than backend/scripts/_eval (scratch runs)",
    )
    args = parser.parse_args()

    if args.out:
        global OUT
        OUT = Path(args.out).expanduser().resolve()
        OUT.mkdir(parents=True, exist_ok=True)

    if args.list_baselines:
        known = list_baselines(OUT)
        print("\n".join(known) if known else "(no baselines recorded)", flush=True)
        return

    net = None if args.no_perceptual else args.perceptual_net
    requested = list(CASES) if args.only == "all" else [args.only]
    skipped: dict[str, str] = {}
    results: list[CaseResult] = []

    for name, spec in CASES.items():
        if name not in requested:
            skipped[name] = f"not requested (--only {args.only})"
            continue
        if not spec.path.is_file():
            skipped[name] = f"asset missing: {spec.path}"
            print(f"[SKIP] {name} — {skipped[name]}", flush=True)
            continue
        max_side = args.max_side
        if spec.max_side_cap is not None:
            max_side = min(max_side, spec.max_side_cap)
        results.append(
            run_case(
                spec.name,
                spec.path,
                spec.click,
                mode=args.mode,
                quality=args.quality,
                max_side=max_side,
                ghost_max=spec.ghost_max,
                plaque_max=spec.plaque_max,
                protect_max=spec.protect_max,
                remove_others=spec.remove_others,
                lock=spec.lock,
                perceptual_net=net,
                perceptual_max=args.perceptual_max,
                perceptual_blur_min=args.perceptual_blur_min,
                gate_perceptual=args.gate_perceptual,
                require_perceptual=args.require_perceptual,
            )
        )

    summary = build_summary(
        registered=list(CASES),
        requested=requested,
        cases=[asdict(r) for r in results],
        skipped=skipped,
        config={
            "only": args.only,
            "mode": args.mode,
            "quality": args.quality,
            "max_side": args.max_side,
            "perceptual_net": net,
            "gate_perceptual": bool(args.gate_perceptual),
            "require_perceptual": bool(args.require_perceptual),
        },
    )

    if args.save_baseline:
        try:
            dest = save_baseline(
                OUT, args.save_baseline, summary, overwrite=args.overwrite_baseline
            )
        except (FileExistsError, ValueError) as exc:
            print(f"baseline error: {exc}", flush=True)
            sys.exit(2)
        if summary["partial_run"]:
            print(
                f"WARNING: baseline '{args.save_baseline}' recorded from a PARTIAL run "
                f"(ran {summary['coverage']['ran']})",
                flush=True,
            )
        print(f"Baseline saved → {dest}", flush=True)

    diff = None
    if args.baseline:
        try:
            baseline = load_baseline(OUT, args.baseline)
        except (FileNotFoundError, ValueError) as exc:
            print(f"baseline error: {exc}", flush=True)
            sys.exit(2)
        diff = diff_against_baseline(
            baseline,
            summary,
            name=args.baseline,
            scale=args.regression_scale,
            image_distance=_baseline_image_distance(args.baseline, net),
        )
        summary["baseline_diff"] = diff
        (OUT / "baseline_diff.json").write_text(json.dumps(diff, indent=2), encoding="utf-8")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cov = summary["coverage"]
    print(
        f"\nSummary [{summary['status']}]: {len(summary['cases_passed'])}/{len(results)} PASS "
        f"of {len(cov['registered'])} registered "
        f"(ran={cov['ran'] or '-'}, missing={cov['missing'] or '-'}) "
        f"full_suite_pass={summary['full_suite_pass']} → {OUT}",
        flush=True,
    )
    for name, reason in skipped.items():
        print(f"  skipped {name}: {reason}", flush=True)
    if summary["partial_run"] and results:
        print(
            "PARTIAL RUN — this is not a full-suite pass; run without --only before "
            "trusting the gate.",
            flush=True,
        )

    if diff is not None:
        print(f"\nBaseline diff vs '{args.baseline}':", flush=True)
        print(format_diff(diff), flush=True)
        if diff["has_regression"]:
            print("\nRegressions:", flush=True)
            for line in diff["regressions"]:
                print(f"  {line}", flush=True)
        else:
            print("  no regressions beyond tolerance", flush=True)
        # Latency is advisory: run-to-run noise on an idle machine already reaches
        # ~38%, so it is reported loudly but never fails the run.
        if diff.get("has_advisory"):
            print("\nLatency advisories (do not fail the run):", flush=True)
            for line in diff["advisories"]:
                print(f"  {line}", flush=True)

    code = exit_code_for(summary, partial_allowed=args.only != "all")
    if diff is not None and diff["has_regression"] and not args.no_fail_on_regression:
        code = 1
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
