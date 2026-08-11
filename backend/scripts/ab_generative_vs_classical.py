"""A/B: generative erase (SDXL / PowerPaint) vs the hand-tuned classical pipeline.

Reuses eval_removal_bench's case definitions and run_case so the numbers are
directly comparable to existing eval artifacts. Two variants per case:

* classical  — mode=ai, quality=best with generative_engine_ready pinned False
               (isolates the generative decision; everything else identical)
* generative — mode=ai, quality=best with weights available

Diffusion is nondeterministic, so the generative side runs several independent
seed families. The backends already do their own best-of-N seed selection; this
script shifts every seed by a constant offset so each family is an independent
draw of whatever the pipeline would hand a user (no extra best-of on top).

Writes per-variant artifacts, ab_results.json, and the side-by-side sheets
backend/scripts/_eval/_ab/{coastal,library}_ab.jpg.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(HERE))

AB = HERE / "_eval" / "_ab"
VARIANTS = AB / "variants"
RUNS = AB / "runs"

EXPECT_GENERATIVE = {"sdxl", "powerpaint"}
EXPECT_CLASSICAL = {"water_exemplar", "classical", "lama_pm"}

# Seed shifting -------------------------------------------------------------
# The backends hardcode their seed tuples, so repeat runs are bit-identical.
# Wrapping torch.Generator lets us draw independent seed families without
# touching services/. diffusers never constructs a Generator itself (checked),
# so only the pipeline's own explicit generator is affected.
_SEED_OFFSET = 0


def _install_seed_shift() -> None:
    """Replace torch.Generator with a subclass that offsets manual_seed.

    Must stay a real class: diffusers evaluates `torch.Generator | None` type
    unions at runtime, which a plain factory function breaks.
    """
    import torch

    if getattr(torch, "_pe_seed_shift_installed", False):
        return

    class SeedShiftedGenerator(torch.Generator):  # type: ignore[misc]
        def manual_seed(self, seed):
            return super().manual_seed(int(seed) + _SEED_OFFSET)

    torch.Generator = SeedShiftedGenerator  # type: ignore[assignment]
    torch._pe_seed_shift_installed = True  # type: ignore[attr-defined]


# Engine census -------------------------------------------------------------
# Structured cases inpaint per connected component, so get_last_removal_meta()
# only describes the final component. Counting every low-level engine call
# proves no component silently fell back to something else.
_CENSUS: dict[str, dict] = {}


def _reset_census() -> None:
    _CENSUS.clear()


def _install_census() -> None:
    from services import inpaint_backends as ib
    from services import object_removal as orem

    def wrap(mod, attr, label):
        original = getattr(mod, attr)
        if getattr(original, "_pe_census", False):
            return

        def wrapper(*args, **kwargs):
            slot = _CENSUS.setdefault(label, {"calls": 0, "seconds": 0.0, "errors": []})
            slot["calls"] += 1
            t = time.time()
            try:
                return original(*args, **kwargs)
            except Exception as exc:
                slot["errors"].append(f"{type(exc).__name__}: {exc}"[:300])
                raise
            finally:
                slot["seconds"] = round(slot["seconds"] + (time.time() - t), 3)

        wrapper._pe_census = True  # type: ignore[attr-defined]
        setattr(mod, attr, wrapper)

    for attr, label in (
        ("inpaint_patch_sdxl", "sdxl"),
        ("inpaint_patch_powerpaint", "powerpaint"),
        ("inpaint_patch_sd", "sd15"),
        ("inpaint_patch_onnx_tiled", "lama_onnx_tiled"),
        ("inpaint_patch_torch", "lama_torch"),
        ("_get_sdxl_pipe", "load_sdxl_pipe"),
        ("_get_powerpaint_pipe", "load_powerpaint_pipe"),
    ):
        wrap(ib, attr, label)
    # object_removal imported these by name at import time.
    wrap(orem, "fill_for_scene", "classical_fill_for_scene")
    wrap(orem, "inpaint_patch_opencv", "classical_opencv")


def _disable_classical_overwrites() -> list[str]:
    """Neutralise the classical passes that hard-replace the hole after inpaint.

    On both scene families the pipeline treats classical synthesis as
    authoritative and overwrites the generative fill, so the shipped Best
    Quality output cannot show what the engine produced. Turning those passes
    into pass-throughs exposes the raw diffusion fill for visual judgement.
    Diagnostic only — nothing here changes the product code.
    """
    from services import object_removal as orem
    from services import scene_context as sctx

    disabled: list[str] = []

    def passthrough(mod, attr, index):
        original = getattr(mod, attr)

        def wrapper(*args, **kwargs):
            return args[index]

        setattr(mod, attr, wrapper)
        disabled.append(attr)
        return original

    def returns_none(mod, attr):
        setattr(mod, attr, lambda *a, **k: None)
        disabled.append(attr)

    # (module, attribute, index of the image argument to hand straight back)
    passthrough(orem, "fill_for_scene", 1)
    passthrough(orem, "_rescue_soft_generative_water", 1)
    passthrough(orem, "_hard_exemplar_overwrite", 1)
    passthrough(orem, "_ghost_mad_expand_refill", 1)
    passthrough(orem, "_inject_context_texture", 1)
    passthrough(sctx, "_match_fill_hf_to_ring", 1)
    returns_none(orem, "_soft_plaque_residual_mask")
    returns_none(orem, "_residual_ghost_mask")
    return disabled


def _font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _check() -> dict:
    import torch

    from services import object_removal as orem
    from services.inpaint_backends import _erase_engine_name, generative_removal_readiness
    from services.scene_context import classify_hole_scene

    import eval_removal_bench as bench

    info = {
        "cuda": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "erase_engine_env": _erase_engine_name(),
        "readiness": generative_removal_readiness(),
        "scenes": {},
    }
    for name, spec in bench.CASES.items():
        im = Image.open(spec.path).convert("RGB")
        ow, oh = im.size
        max_side = min(1280, spec.max_side_cap or 1280)
        scale = max_side / float(max(ow, oh)) if max(ow, oh) > max_side else 1.0
        im = im.resize((int(ow * scale), int(oh * scale)), Image.Resampling.LANCZOS)
        cx, cy = int(round(spec.click[0] * scale)), int(round(spec.click[1] * scale))
        if spec.remove_others:
            lp = (float(spec.lock[0] * scale), float(spec.lock[1] * scale))  # type: ignore[index]
            mask = orem.auto_detect_others_mask(im, lock_point=lp)
        else:
            mask = orem.segment_from_points(im, [[float(cx), float(cy)]], [1])
        hard = np.array(mask) > 127
        info["scenes"][name] = {
            "size": list(im.size),
            "scene": classify_hole_scene(orem._pil_to_rgb_array(im), hard),
            "mask_coverage_pct": round(float(hard.mean() * 100), 3),
        }
    return info


def run_variant(
    case: str,
    variant: str,
    *,
    generative: bool,
    seed_offset: int,
    mode: str = "ai",
    raw_engine: bool = False,
) -> dict:
    global _SEED_OFFSET

    from services import object_removal as orem

    import eval_removal_bench as bench

    spec = bench.CASES[case]
    out_dir = VARIANTS / f"{case}__{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_side = min(1280, spec.max_side_cap or 1280)
    saved_out = bench.OUT
    saved_ready = orem.generative_engine_ready
    _reset_census()
    _SEED_OFFSET = int(seed_offset)
    bench.OUT = out_dir
    if not generative:
        orem.generative_engine_ready = lambda scene=None: False  # type: ignore[assignment]
    disabled = _disable_classical_overwrites() if raw_engine else []

    t0 = time.time()
    try:
        cr = bench.run_case(
            spec.name,
            spec.path,
            spec.click,
            mode=mode,
            quality="best",
            max_side=max_side,
            ghost_max=spec.ghost_max,
            plaque_max=spec.plaque_max,
            protect_max=spec.protect_max,
            remove_others=spec.remove_others,
            lock=spec.lock,
        )
        meta = orem.get_last_removal_meta()
        error = None
    except Exception as exc:  # OOM or engine failure is a finding, not a crash
        cr, meta, error = None, orem.get_last_removal_meta(), f"{type(exc).__name__}: {exc}"
    finally:
        wall = time.time() - t0
        bench.OUT = saved_out
        orem.generative_engine_ready = saved_ready  # type: ignore[assignment]
        _SEED_OFFSET = 0

    engine = str(meta.get("engine"))
    expected = EXPECT_GENERATIVE if generative else EXPECT_CLASSICAL
    row: dict = {
        "case": case,
        "variant": variant,
        "mode": mode,
        "raw_engine": raw_engine,
        "classical_overwrites_disabled": disabled,
        "generative_requested": generative,
        "seed_offset": seed_offset,
        "engine": engine,
        "scene": meta.get("scene"),
        "engine_expected": sorted(expected),
        "engine_ok": engine in expected,
        "wall_s": round(wall, 3),
        "error": error,
        "out_dir": str(out_dir),
        "engine_census": json.loads(json.dumps(_CENSUS)),
    }
    census_gen = sum(_CENSUS.get(k, {}).get("calls", 0) for k in ("sdxl", "powerpaint", "sd15"))
    census_fallback = sum(
        _CENSUS.get(k, {}).get("calls", 0) for k in ("lama_onnx_tiled", "lama_torch")
    )
    row["census_generative_calls"] = census_gen
    row["census_classical_engine_calls"] = census_fallback
    if generative:
        # Any classical engine call or swallowed generative error means a fallback.
        gen_errors = [
            e for k in ("sdxl", "powerpaint", "sd15") for e in _CENSUS.get(k, {}).get("errors", [])
        ]
        row["silent_fallback"] = bool(census_fallback or gen_errors or census_gen == 0)
        row["generative_errors"] = gen_errors
        row["engine_ok"] = bool(row["engine_ok"] and not row["silent_fallback"])
    else:
        row["silent_fallback"] = bool(census_gen)
        row["engine_ok"] = bool(row["engine_ok"] and not census_gen)
    if cr is not None:
        row["scores"] = asdict(cr)
    print(
        f"  -> {case}/{variant}: engine={engine} scene={meta.get('scene')} "
        f"ok={row['engine_ok']} wall={wall:.1f}s"
        + (f" ERROR {error}" if error else ""),
        flush=True,
    )
    return row


# Comparison sheets ---------------------------------------------------------


def _after(row: dict) -> np.ndarray | None:
    p = Path(row["out_dir"]) / row["case"] / "after.jpg"
    if not p.is_file():
        return None
    return np.array(Image.open(p).convert("RGB"))


def _before(row: dict) -> np.ndarray:
    return np.array(Image.open(Path(row["out_dir"]) / row["case"] / "before.jpg").convert("RGB"))


def _fill_region(before: np.ndarray, afters: list[np.ndarray]) -> np.ndarray:
    """Union of pixels any variant changed — the fill region plus its seam."""
    acc = np.zeros(before.shape[:2], dtype=bool)
    for a in afters:
        d = np.abs(a.astype(np.float32) - before.astype(np.float32)).mean(axis=2)
        acc |= d > 8.0
    acc = cv2.morphologyEx(
        acc.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ) > 0
    return acc


def _bbox(region: np.ndarray, shape: tuple[int, int], pad_frac: float = 0.25):
    # Zoom on the largest changed blob, not the union of every bystander, so the
    # crop row is actually a close-up of a hole and its seam.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
    if n > 2:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region = lab == biggest
    ys, xs = np.where(region)
    if len(ys) == 0:
        h, w = shape
        return 0, 0, w, h
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    ph = int((y1 - y0 + 1) * pad_frac)
    pw = int((x1 - x0 + 1) * pad_frac)
    h, w = shape
    return max(0, x0 - pw), max(0, y0 - ph), min(w, x1 + pw + 1), min(h, y1 + ph + 1)


def _outline(img: np.ndarray, region: np.ndarray) -> np.ndarray:
    out = img.copy()
    edges = cv2.dilate(
        region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ).astype(bool) & (~region)
    out[edges] = (255, 60, 60)
    return out


def _wrap(text: str, font, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            trial = f"{cur} {word}".strip()
            if font.getlength(trial) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def _fit(img: np.ndarray, w: int, h: int) -> Image.Image:
    pil = Image.fromarray(img)
    pil.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), (18, 18, 20))
    canvas.paste(pil, ((w - pil.width) // 2, (h - pil.height) // 2))
    return canvas


def build_sheet(case: str, rows: list[dict], dest: Path) -> dict:
    usable = [r for r in rows if r["case"] == case and _after(r) is not None]
    if not usable:
        return {"case": case, "written": False, "reason": "no usable variant output"}
    def rank(r: dict) -> tuple:
        if r["variant"] == "quick":
            kind = 0
        elif not r["generative_requested"]:
            kind = 1
        elif not r.get("raw_engine"):
            kind = 2
        else:
            kind = 3
        return (kind, r["seed_offset"])

    # One panel per kind so the generative column is always visible even when it
    # is byte-identical to classical; extra seed families that add no new pixels
    # are folded into the label instead of another column.
    by_variant = {r["variant"]: r for r in usable}
    digests = {v: _after(r).tobytes() for v, r in by_variant.items() if _after(r) is not None}  # type: ignore[union-attr]
    keep = {v for v in ("quick", "gen_s0") if v in by_variant}
    keep |= {v for v in by_variant if v.startswith("genraw_")}
    if "classical" in by_variant and digests.get("classical") != digests.get("quick"):
        keep.add("classical")
    duplicates = {
        v: sorted(o for o, d in digests.items() if o != v and d == digests.get(v)) for v in keep
    }
    order = sorted([r for r in usable if r["variant"] in keep], key=rank)
    before = _before(order[0])
    afters = [_after(r) for r in order]
    region = _fill_region(before, [a for a in [_after(r) for r in usable] if a is not None])
    x0, y0, x1, y1 = _bbox(region, before.shape[:2])

    panels: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    panels.append(
        (
            "ORIGINAL (removal target outlined)",
            f"{before.shape[1]}x{before.shape[0]}",
            _outline(before, region),
            _outline(before, region)[y0:y1, x0:x1],
        )
    )
    for r, a in zip(order, afters):
        if a is None:
            continue
        sc = r.get("scores") or {}
        if r["variant"] == "quick":
            label = "CLASSICAL (mode=quick)"
        elif not r["generative_requested"]:
            label = "CLASSICAL (best tier, gen off)"
        elif r.get("raw_engine"):
            label = f"GEN RAW seed+{r['seed_offset']}"
        else:
            label = f"GEN as shipped seed+{r['seed_offset']}"
        sub = (
            f"{r['engine']} | {sc.get('latency_inpaint_s', '?')}s | "
            f"ghost {sc.get('ghost_residual_pct', '?')}% | "
            f"plaque {sc.get('plaque_score', '?')} | blur {sc.get('perceptual_blur_ratio', '?')}"
        )
        if r.get("raw_engine"):
            sub += "\n(classical post-fill disabled)"
        dupes = duplicates.get(r["variant"])
        if dupes:
            sub += f"\nidentical to: {', '.join(dupes)}"
        panels.append((label, sub, a, a[y0:y1, x0:x1]))

    cols = len(panels)
    cell_w = max(280, min(360, int(9600 / cols)))
    full_h = int(cell_w * before.shape[0] / before.shape[1])
    crop_ar = (y1 - y0) / max(x1 - x0, 1)
    crop_h = int(min(cell_w * 1.5, max(cell_w * 0.55, cell_w * crop_ar)))
    pad, head, lab = 10, 54, 88
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = head + lab + full_h + pad + 30 + crop_h + pad * 3

    sheet = Image.new("RGB", (sheet_w, sheet_h), (12, 12, 14))
    d = ImageDraw.Draw(sheet)
    f_title, f_lab, f_sub = _font(24), _font(19), _font(14)
    d.text(
        (pad, 12),
        f"{case}: hand-tuned classical vs generative inpainting  (mode=ai quality=best, max_side=1280)",
        font=f_title,
        fill=(240, 240, 245),
    )
    for i, (label, sub, full, crop) in enumerate(panels):
        x = pad + i * (cell_w + pad)
        for j, line in enumerate(_wrap(label, f_lab, cell_w)[:1]):
            d.text((x, head + j * 20), line, font=f_lab, fill=(255, 210, 120) if i else (150, 220, 255))
        for j, line in enumerate(_wrap(sub, f_sub, cell_w)[:4]):
            d.text((x, head + 22 + j * 15), line, font=f_sub, fill=(175, 175, 185))
        sheet.paste(_fit(full, cell_w, full_h), (x, head + lab))
        if i == 0:
            d.text(
                (x, head + lab + full_h + pad + 6),
                "ZOOM — fill region + seam",
                font=f_lab,
                fill=(150, 220, 255),
            )
        sheet.paste(_fit(crop, cell_w, crop_h), (x, head + lab + full_h + pad + 30))
    dest.parent.mkdir(parents=True, exist_ok=True)
    for q in (90, 84, 76, 68):
        sheet.save(dest, quality=q, optimize=True)
        if dest.stat().st_size <= 2_000_000:
            break
    return {
        "case": case,
        "written": True,
        "path": str(dest),
        "size_px": [sheet.width, sheet.height],
        "bytes": dest.stat().st_size,
        "zoom_bbox": [x0, y0, x1, y1],
        "panels": [p[0] for p in panels],
    }


def cross_compare(case: str, rows: list[dict]) -> dict:
    """How much do the variants actually differ, and where?"""
    usable = [r for r in rows if r["case"] == case and _after(r) is not None]
    if len(usable) < 2:
        return {}
    before = _before(usable[0])
    imgs = {r["variant"]: _after(r) for r in usable}
    region = _fill_region(before, [v for v in imgs.values() if v is not None])
    out: dict = {"fill_region_px": int(region.sum())}
    gens = {k: v for k, v in imgs.items() if k.startswith("gen_")}

    def mad(a, b, m):
        return round(
            float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)[m].mean()), 3
        )

    for ref_name in ("classical", "quick"):
        ref = imgs.get(ref_name)
        if ref is None:
            continue
        out[f"gen_vs_{ref_name}_mad_in_fill"] = {k: mad(v, ref, region) for k, v in gens.items()}
        out[f"gen_vs_{ref_name}_pct_pixels_changed_gt3"] = {
            k: round(
                float(
                    (
                        np.abs(v.astype(np.float32) - ref.astype(np.float32)).mean(axis=2)[region]
                        > 3
                    ).mean()
                    * 100
                ),
                2,
            )
            for k, v in gens.items()
        }
    if imgs.get("classical") is not None and imgs.get("quick") is not None:
        out["classical_vs_quick_mad_in_fill"] = mad(imgs["classical"], imgs["quick"], region)
    keys = sorted(gens)
    out["gen_seed_pairwise_mad_in_fill"] = {
        f"{a}|{b}": mad(gens[a], gens[b], region)
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    }
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="generative vs classical removal A/B")
    parser.add_argument("--check", action="store_true", help="environment / routing check only")
    parser.add_argument("--only", choices=["coastal", "library", "all"], default="all")
    parser.add_argument(
        "--seed-offsets",
        default="0,1000,2000",
        help="independent generative seed families (offset added to every backend seed)",
    )
    parser.add_argument(
        "--single",
        metavar="CASE:KIND:OFFSET",
        help="run one variant in this process (KIND=classical|generative); "
        "one process per variant keeps VRAM clean between engines",
    )
    parser.add_argument(
        "--sheets", action="store_true", help="merge _ab/runs/*.json into sheets + ab_results.json"
    )
    args = parser.parse_args()

    from paths import configure_runtime_paths

    configure_runtime_paths()
    _install_seed_shift()
    if not args.check:
        _install_census()

    AB.mkdir(parents=True, exist_ok=True)
    results_path = AB / "ab_results.json"

    if args.check:
        info = _check()
        print(json.dumps(info, indent=2), flush=True)
        (AB / "env_check.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return

    import eval_removal_bench as bench

    cases = list(bench.CASES) if args.only == "all" else [args.only]
    offsets = [int(s) for s in args.seed_offsets.split(",") if s.strip()]
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.single:
        case, kind, offset = args.single.split(":")
        generative = kind in ("generative", "genraw")
        variant = {
            "generative": f"gen_s{offset}",
            "genraw": f"genraw_s{offset}",
            "quick": "quick",
        }.get(kind, "classical")
        row = run_variant(
            case,
            variant,
            generative=generative,
            seed_offset=int(offset),
            mode="quick" if kind == "quick" else "ai",
            raw_engine=kind == "genraw",
        )
        (RUNS / f"{case}__{variant}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        return

    if args.sheets:
        rows = [
            json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*.json"))
        ]
        payload = {"runs": rows}
    else:
        rows = []
        for case in cases:
            print(f"[{case}] classical", flush=True)
            rows.append(run_variant(case, "classical", generative=False, seed_offset=0))
            for off in offsets:
                print(f"[{case}] generative seed+{off}", flush=True)
                rows.append(
                    run_variant(case, f"gen_s{off}", generative=True, seed_offset=off)
                )
            for r in rows:
                (RUNS / f"{r['case']}__{r['variant']}.json").write_text(
                    json.dumps(r, indent=2), encoding="utf-8"
                )
        payload = {"runs": rows}

    cases = sorted({r["case"] for r in rows}) if args.sheets else cases
    payload["compare"] = {c: cross_compare(c, rows) for c in cases}
    payload["sheets"] = [
        build_sheet(c, rows, AB / f"{c}_ab.jpg") for c in cases
    ]
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {results_path}", flush=True)
    for s in payload["sheets"]:
        print(f"  sheet {s}", flush=True)
    bad = [r for r in rows if not r["engine_ok"]]
    if bad:
        print("\nENGINE MISMATCH (comparison invalid for these):", flush=True)
        for r in bad:
            print(f"  {r['case']}/{r['variant']}: got {r['engine']} want {r['engine_expected']}", flush=True)


if __name__ == "__main__":
    main()
