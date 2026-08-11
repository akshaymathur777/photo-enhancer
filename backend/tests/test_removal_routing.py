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


def test_water_uses_sdxl_only_on_best_with_cuda_and_weights():
    assert route("water", "best", cuda=True, weights=True) == ["sdxl"]
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
    assert route(
        "water", "best", cuda=True, weights=True, generative_fails=True
    ) == ["sdxl", "exemplar:water"]
    assert route(
        "structured", "best", cuda=True, weights=True, generative_fails=True
    ) == ["powerpaint", "lama_onnx"]


def test_water_meta_engine_reports_exemplar_fallback():
    route("water", "balanced", cuda=True, weights=True)
    assert object_removal.get_last_removal_meta()["engine"] == "water_exemplar"


def test_generative_success_skips_hard_exemplar_overwrite():
    """Best + generative fill must keep hole pixels — no classical fill_for_scene /
    hard exemplar overwrite of the interior (the A/B regression).
    """
    rgb, mask = _sample_image()
    # Distinct generative paint so we can prove it survives.
    gen_fill = rgb.copy()
    gen_fill[mask > 127] = (240, 40, 40)

    state: dict = {"calls": [], "filled": False}

    class GenBackend(inpaint_backends.InpaintBackend):
        def __init__(self):
            self.is_generative = True
            self.engine = "sdxl"

        def inpaint_crop(self, crop_rgb, crop_mask):
            state["calls"].append("sdxl")
            out = crop_rgb.copy()
            hard = crop_mask > 127
            # Paint a unique color into the hole so overwrite is detectable.
            out[hard] = (240, 40, 40)
            return out

    def _fill_for_scene(original, inpainted, m, scene=None):
        state["calls"].append(f"postfill:{scene}")
        # Classical overwrite would replace generative paint from original.
        hard = m > 127
        out = inpainted.copy()
        out[hard] = original[hard]
        return out

    hard_calls: list[str] = []

    def _hard_overwrite(original, result, mask_u8, protect=None, scene="water"):
        hard_calls.append(scene)
        return _fill_for_scene(original, result, mask_u8, scene=scene)

    with ExitStack() as stack:
        p = stack.enter_context
        p(mock.patch.object(object_removal, "classify_hole_scene", lambda *a, **k: "water"))
        p(mock.patch.object(scene_context, "classify_hole_scene", lambda *a, **k: "water"))
        p(mock.patch.object(object_removal, "_is_water_or_sky_ring", lambda *a, **k: True))
        p(mock.patch.object(object_removal, "fill_for_scene", _fill_for_scene))
        p(mock.patch.object(object_removal, "_hard_exemplar_overwrite", _hard_overwrite))
        p(mock.patch.object(object_removal, "_scrub_other_people", lambda rgb, m: rgb))
        p(mock.patch.object(
            object_removal, "_hybrid_context_fill", lambda original, lama, m, scene=None: lama
        ))
        p(mock.patch.object(
            scene_context, "_match_fill_hf_to_ring", lambda original, filled, *a, **k: filled
        ))
        p(mock.patch("torch.cuda.is_available", return_value=True))
        p(mock.patch("services.model_download._powerpaint_present", return_value=True))
        p(mock.patch("services.model_download._sdxl_inpaint_present", return_value=True))
        p(mock.patch.object(object_removal, "get_backend", lambda m: GenBackend()))
        p(mock.patch.object(
            object_removal, "_soft_plaque_residual_mask", lambda *a, **k: None
        ))
        p(mock.patch.object(object_removal, "_residual_ghost_mask", lambda *a, **k: None))
        p(mock.patch.object(
            object_removal, "_ghost_mad_expand_refill", lambda *a, **k: a[1]
        ))
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

        out = object_removal.inpaint_objects(
            Image.fromarray(rgb),
            Image.fromarray(mask),
            mode="ai",
            quality="best",
        )

    meta = object_removal.get_last_removal_meta()
    assert meta["engine"] == "sdxl"
    assert meta["postfill"] == "seam_only"
    assert state["calls"] == ["sdxl"]
    assert hard_calls == []
    arr = np.array(out)
    hole = mask > 127
    # Generative red paint must survive — classical overwrite would restore ocean blue.
    assert int(arr[hole][:, 0].mean()) > 200
    assert int(np.abs(arr[hole].astype(np.float32) - rgb[hole].astype(np.float32)).mean()) > 30


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
