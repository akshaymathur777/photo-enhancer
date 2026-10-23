"""Engine routing for object removal: scene x quality tier x CUDA x weights.

Asserts which engine _inpaint_once dispatches to. No real model runs here —
backends, exemplar fills and the CUDA / weight probes are stubbed, so this
covers the routing decision on machines with no GPU and no downloaded weights.

Run standalone:  python tests/test_removal_routing.py   (from backend/)
"""

from __future__ import annotations

import sys
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import inpaint_backends, object_removal, scene_context  # noqa: E402
from services.ai_modes import resolve_inpaint_mode, resolve_removal_quality  # noqa: E402


def _sample_image() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    rgb = np.zeros((200, 200, 3), np.uint8)
    rgb[:, :, 0] = 90
    rgb[:, :, 1] = 120
    rgb[:, :, 2] = 160
    rgb = np.clip(rgb.astype(np.int16) + rng.integers(-14, 14, rgb.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((200, 200), np.uint8)
    mask[70:130, 70:130] = 255
    return rgb, mask


class _StubBackend(inpaint_backends.InpaintBackend):
    def __init__(self, state: dict, name: str, generative: bool, fail: bool = False):
        self._state = state
        self._name = name
        self.is_generative = generative
        self.engine = name
        self._fail = fail

    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        name = self._name
        if name == "generative":
            # Mirror AiRemoveBackend's scene-routed preference order.
            prefers_ppt = inpaint_backends._erase_engine_prefers_powerpaint(rgb, mask > 127)
            name = "powerpaint" if prefers_ppt else "sdxl"
            self.engine = name
        self._state["calls"].append(name)
        if self._fail:
            raise inpaint_backends.InpaintError(f"stub {name} failure")
        self._state["filled"] = True
        return rgb.copy()


def route(
    scene: str,
    tier: str,
    *,
    cuda: bool,
    weights: bool,
    mode: str | None = None,
    generative_fails: bool = False,
) -> list[str]:
    """Return the engines _inpaint_once dispatched to, in order.

    fill_for_scene is recorded as `exemplar:*` only while no backend has produced
    a fill; afterwards it is the existing post-fill polish stage (`postfill:*`).
    """
    rgb, mask = _sample_image()
    state: dict = {"calls": [], "filled": False}
    calls: list[str] = state["calls"]
    if mode is None:
        mode = "quick" if tier == "fast" else "ai"

    def _fill_for_scene(original, inpainted, m, scene=None):
        stage = "postfill" if state["filled"] else "exemplar"
        calls.append(f"{stage}:{scene}")
        state["filled"] = True
        return inpainted

    with ExitStack() as stack:
        p = stack.enter_context
        p(mock.patch.object(object_removal, "classify_hole_scene", lambda *a, **k: scene))
        p(mock.patch.object(scene_context, "classify_hole_scene", lambda *a, **k: scene))
        p(mock.patch.object(
            object_removal, "_is_water_or_sky_ring", lambda *a, **k: scene in ("water", "sky")
        ))
        p(mock.patch.object(object_removal, "fill_for_scene", _fill_for_scene))
        p(mock.patch.object(object_removal, "_scrub_other_people", lambda rgb, m: rgb))
        p(mock.patch.object(
            object_removal, "_hybrid_context_fill", lambda original, lama, m, scene=None: lama
        ))
        p(mock.patch.object(
            scene_context, "_match_fill_hf_to_ring", lambda original, filled, *a, **k: filled
        ))
        p(mock.patch("torch.cuda.is_available", return_value=cuda))
        p(mock.patch("services.model_download._powerpaint_present", return_value=weights))
        p(mock.patch("services.model_download._sdxl_inpaint_present", return_value=weights))
        p(mock.patch.object(
            object_removal,
            "get_backend",
            lambda m: _StubBackend(
                state, "generative" if m == "ai" else "lama_onnx", m == "ai", generative_fails
            ),
        ))
        p(mock.patch.object(
            inpaint_backends,
            "QuickRemoveBackend",
            lambda: _StubBackend(state, "lama_onnx", False),
        ))
        inpaint_backends.set_removal_quality(tier)
        object_removal._inpaint_once(rgb, mask, mode=mode, edge_softness=1.0)
    return calls


def test_water_always_uses_exemplar_even_on_best():
    # Classical water matches SDXL+rescue quality without GPU thrash.
    assert route("water", "best", cuda=True, weights=True) == ["exemplar:water"]
    assert route("water", "best", cuda=False, weights=True) == ["exemplar:water"]
    assert route("water", "best", cuda=True, weights=False) == ["exemplar:water"]
    assert route("water", "balanced", cuda=True, weights=True) == ["exemplar:water"]
    assert route("water", "fast", cuda=True, weights=True) == ["exemplar:water"]


def test_structured_uses_powerpaint_only_on_best_with_cuda_and_weights():
    assert route("structured", "best", cuda=True, weights=True) == ["powerpaint"]
    assert route("structured", "best", cuda=False, weights=True) == ["lama_onnx"]
    assert route("structured", "best", cuda=True, weights=False) == ["lama_onnx"]
    assert route("structured", "balanced", cuda=True, weights=True) == ["lama_onnx"]
    assert route("structured", "fast", cuda=True, weights=True) == ["lama_onnx"]


def test_generic_keeps_existing_behaviour():
    for cuda in (False, True):
        for weights in (False, True):
            assert route("generic", "best", cuda=cuda, weights=weights) == ["powerpaint"]
            assert route("generic", "balanced", cuda=cuda, weights=weights) == ["powerpaint"]
            assert route("generic", "fast", cuda=cuda, weights=weights) == ["lama_onnx"]


def test_fast_tier_with_ai_remove_forced_stays_classical_on_water_and_structured():
    # The editor lets a Fast-preset user pick "AI Remove" explicitly.
    assert route("water", "fast", cuda=True, weights=True, mode="ai") == ["exemplar:water"]
    assert route("structured", "fast", cuda=True, weights=True, mode="ai") == ["lama_onnx"]
    assert route("generic", "fast", cuda=True, weights=True, mode="ai") == ["powerpaint"]


def test_generative_failure_falls_back_to_classical():
    # Water never enters SDXL; structured PowerPaint still falls back to LaMa.
    assert route(
        "water", "best", cuda=True, weights=True, generative_fails=True
    ) == ["exemplar:water"]
    assert route(
        "structured", "best", cuda=True, weights=True, generative_fails=True
    ) == ["powerpaint", "lama_onnx"]


def test_water_meta_engine_reports_exemplar_fallback():
    route("water", "balanced", cuda=True, weights=True)
    assert object_removal.get_last_removal_meta()["engine"] == "water_exemplar"


def test_water_best_short_circuits_to_exemplar_without_sdxl():
    """Best water must not load SDXL — classical exemplar is the product path."""
    rgb, mask = _sample_image()
    state: dict = {"sdxl_calls": 0}

    class GenBackend(inpaint_backends.InpaintBackend):
        def __init__(self):
            self.is_generative = True
            self.engine = "sdxl"

        def inpaint_crop(self, crop_rgb, crop_mask):
            state["sdxl_calls"] += 1
            return crop_rgb

    with ExitStack() as stack:
        p = stack.enter_context
        p(mock.patch.object(object_removal, "classify_hole_scene", lambda *a, **k: "water"))
        p(mock.patch.object(scene_context, "classify_hole_scene", lambda *a, **k: "water"))
        p(mock.patch.object(object_removal, "_is_water_or_sky_ring", lambda *a, **k: True))
        p(mock.patch.object(object_removal, "_scrub_other_people", lambda rgb, m: rgb))
        p(mock.patch.object(
            scene_context, "_match_fill_hf_to_ring", lambda original, filled, *a, **k: filled
        ))
        p(mock.patch("torch.cuda.is_available", return_value=True))
        p(mock.patch("services.model_download._powerpaint_present", return_value=True))
        p(mock.patch("services.model_download._sdxl_inpaint_present", return_value=True))
        p(mock.patch.object(object_removal, "get_backend", lambda m: GenBackend()))
        p(mock.patch.object(
            object_removal, "_non_target_people_protect", lambda *a, **k: np.zeros(rgb.shape[:2], bool)
        ))
        p(mock.patch.object(
            object_removal, "_occlusion_cutout_protect", lambda *a, **k: np.zeros(rgb.shape[:2], bool)
        ))
        p(mock.patch.object(
            object_removal,
            "refine_mask_hard",
            lambda mask, edge_softness=1.0, rgb=None: np.array(mask.convert("L")),
        ))
        inpaint_backends.set_removal_quality("best")
        from PIL import Image

        object_removal.inpaint_objects(
            Image.fromarray(rgb),
            Image.fromarray(mask),
            mode="ai",
            quality="best",
        )

    meta = object_removal.get_last_removal_meta()
    assert meta["engine"] == "water_exemplar"
    assert meta["postfill"] == "classical"
    assert state["sdxl_calls"] == 0


def test_quality_tiers_are_distinct():
    for raw, expected in (
        ("best_quality", "best"),
        ("best", "best"),
        ("balanced", "balanced"),
        ("fast", "fast"),
        (None, "balanced"),
    ):
        inpaint_backends.set_removal_quality(raw)
        assert inpaint_backends.get_removal_quality() == expected

    inpaint_backends.set_removal_quality("fast")
    assert inpaint_backends._tier_num_steps(28, 40, 48) == 28
    assert not inpaint_backends._tier_allows_multi_seed()
    inpaint_backends.set_removal_quality("balanced")
    assert inpaint_backends._tier_num_steps(28, 40, 48) == 40
    assert inpaint_backends._tier_allows_multi_seed()
    inpaint_backends.set_removal_quality("best")
    assert inpaint_backends._tier_num_steps(28, 40, 48) == 48


def test_generative_engine_ready_needs_cuda_and_scene_weights():
    def ready(scene, *, cuda, ppt, sdxl):
        with ExitStack() as stack:
            stack.enter_context(mock.patch("torch.cuda.is_available", return_value=cuda))
            stack.enter_context(
                mock.patch("services.model_download._powerpaint_present", return_value=ppt)
            )
            stack.enter_context(
                mock.patch("services.model_download._sdxl_inpaint_present", return_value=sdxl)
            )
            return inpaint_backends.generative_engine_ready(scene)

    assert ready("water", cuda=True, ppt=False, sdxl=True) is True
    assert ready("water", cuda=True, ppt=True, sdxl=False) is False
    assert ready("water", cuda=False, ppt=True, sdxl=True) is False
    assert ready("structured", cuda=True, ppt=True, sdxl=False) is True
    assert ready("structured", cuda=True, ppt=False, sdxl=True) is False
    assert ready(None, cuda=True, ppt=False, sdxl=True) is True


def test_mode_presets_map_to_three_tiers():
    assert [resolve_removal_quality(m) for m in ("best_quality", "balanced", "fast")] == [
        "best",
        "balanced",
        "fast",
    ]
    assert resolve_inpaint_mode("fast") == "quick"
    assert resolve_inpaint_mode("balanced") == "ai"
    assert resolve_inpaint_mode("best_quality") == "ai"


def test_convex_open_fill_mask_widens_head():
    mask = np.zeros((120, 80), np.uint8)
    mask[10:40, 32:48] = 1  # narrow head
    mask[40:100, 18:62] = 1  # wider torso
    hull = inpaint_backends._convex_open_fill_mask(mask.astype(bool))
    head_w = int(hull[20].sum())
    torso_w = int(hull[70].sum())
    assert head_w > 16, f"head still a silhouette ({head_w}px)"
    assert head_w >= torso_w * 0.45


def test_blobify_grows_head_into_hills():
    rgb = np.zeros((160, 120, 3), np.uint8)
    rgb[:70] = (70, 95, 55)  # hills
    rgb[70:] = (95, 145, 185)  # water
    hard = np.zeros((160, 120), np.uint8)
    hard[20:55, 50:70] = 255  # head on hills
    hard[55:130, 40:80] = 255  # body in water
    out = object_removal._blobify_water_hole(rgb, hard, protect=None, raw=hard)
    blob = out > 127
    # Head band widens horizontally into hills; must not climb a sky column above the head.
    assert int(blob[35].sum()) > 24
    assert int(blob.sum()) > int((hard > 127).sum())
    assert int(blob[:15].sum()) == 0, "blob climbed into sky above the head"


def test_water_seed_does_not_copy_same_column_hills():
    rgb = np.zeros((180, 160, 3), np.uint8)
    rgb[:70] = (80, 100, 70)
    rgb[70:] = (150, 165, 185)  # grey-bay water (matches _water_source_map)
    for x in range(160):
        rgb[:70, x, 0] = x  # unique hill column
    hard = np.zeros((180, 160), dtype=bool)
    hard[30:140, 70:90] = True
    with mock.patch(
        "services.object_removal._all_people_binary",
        lambda *a, **k: np.zeros(hard.shape, np.uint8),
    ):
        seeded = scene_context.seed_hole_for_scene(rgb, hard, "water")
    head_r = seeded[30:55, 70:90, 0].astype(np.int16)
    stripe = np.arange(70, 90, dtype=np.int16)[None, :]
    orig_match = float((head_r == stripe).mean())
    assert orig_match < 0.20, f"head seed still traces hill columns (match={orig_match:.2f})"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print("\nrouting matrix (scene, tier, cuda, weights) -> engines")
    for scene in ("water", "structured", "generic", "vegetation"):
        for tier in ("fast", "balanced", "best"):
            for cuda in (False, True):
                for weights in (False, True):
                    engines = route(scene, tier, cuda=cuda, weights=weights)
                    print(f"  {scene:11} {tier:8} cuda={int(cuda)} weights={int(weights)} -> {engines}")
    sys.exit(1 if failures else 0)
