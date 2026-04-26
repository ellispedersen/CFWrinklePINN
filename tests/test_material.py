"""Tests for wp3_features.material_parser and wp3_features.material_mapping.

Covers:
- Parsing both canonical .afl files (skipped if files not present)
- Physics feature vector shape, dtype, and value behaviour
- num_plies and ply_thickness_mm override correctness
- Lazy caching in get_physics_material_card
- Fallback to hardcoded defaults when .afl is absent
- auto_detect_material heuristics
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from wp3_features.material_parser import (
    AFL_TWINTEX,
    AFL_UD,
    TWINTEX_GF_PP,
    UD_THERMOPLASTIC,
    MaterialProperties,
    FiberProperties,
    parse_material_file,
)
from wp3_features.material_mapping import (
    _DEFAULT_MATERIALS_DIR,
    auto_detect_material,
    get_physics_material_card,
    _PHYSICS_CARD_CACHE,
)
from wp3_features.material import MATERIAL_FEATURES, compute_material_card

_MAT_DIR = Path(_DEFAULT_MATERIALS_DIR)
_UD_AFL = _MAT_DIR / AFL_UD
_TWINTEX_AFL = _MAT_DIR / AFL_TWINTEX
_AFL_PRESENT = _UD_AFL.exists() and _TWINTEX_AFL.exists()


# ---------------------------------------------------------------------------
# Hardcoded defaults — always run (no file dependency)
# ---------------------------------------------------------------------------

class TestHardcodedDefaults:
    def test_ud_is_not_woven(self) -> None:
        assert not UD_THERMOPLASTIC.is_woven

    def test_twintex_is_woven(self) -> None:
        assert TWINTEX_GF_PP.is_woven

    def test_ud_has_one_fiber(self) -> None:
        assert len(UD_THERMOPLASTIC.fibers) == 1
        assert UD_THERMOPLASTIC.fibers[0].young_modulus == 25000.0

    def test_twintex_has_two_fibers(self) -> None:
        assert len(TWINTEX_GF_PP.fibers) == 2
        fibers = {f.orientation for f in TWINTEX_GF_PP.fibers}
        assert 0.0 in fibers and 90.0 in fibers


# ---------------------------------------------------------------------------
# Physics feature vector — always run (uses hardcoded defaults)
# ---------------------------------------------------------------------------

class TestPhysicsFeatures:
    def test_shape_and_dtype(self) -> None:
        vec = UD_THERMOPLASTIC.to_physics_features(num_plies=2, ply_thickness_mm=0.15)
        assert vec.shape == (8,)
        assert vec.dtype == np.float32

    def test_all_finite(self) -> None:
        for props in (UD_THERMOPLASTIC, TWINTEX_GF_PP):
            vec = props.to_physics_features(num_plies=2, ply_thickness_mm=0.20)
            assert np.isfinite(vec).all(), f"Non-finite in {props.name}: {vec}"

    def test_material_class_feature(self) -> None:
        ud_vec = UD_THERMOPLASTIC.to_physics_features(num_plies=2, ply_thickness_mm=0.15)
        tw_vec = TWINTEX_GF_PP.to_physics_features(num_plies=3, ply_thickness_mm=0.85)
        # Feature 7: 1.0 = UD, 0.0 = woven
        assert ud_vec[7] == pytest.approx(1.0)
        assert tw_vec[7] == pytest.approx(0.0)

    def test_num_plies_affects_features_5_and_6(self) -> None:
        v2 = UD_THERMOPLASTIC.to_physics_features(num_plies=2, ply_thickness_mm=0.15)
        v4 = UD_THERMOPLASTIC.to_physics_features(num_plies=4, ply_thickness_mm=0.15)
        # Feature 5 = num_plies / 4.0
        assert v2[5] == pytest.approx(0.5)
        assert v4[5] == pytest.approx(1.0)
        # Feature 6 = t * num_plies / 2.0 => should differ
        assert v4[6] > v2[6]

    def test_ply_thickness_override_affects_features_2_4_6(self) -> None:
        """Passing ply_thickness_mm must override self.ply_thickness."""
        t_thin = 0.10
        t_thick = 0.30
        v_thin = UD_THERMOPLASTIC.to_physics_features(num_plies=2, ply_thickness_mm=t_thin)
        v_thick = UD_THERMOPLASTIC.to_physics_features(num_plies=2, ply_thickness_mm=t_thick)
        # Feature 2 (thickness ratio) and 6 (total thickness) must differ
        assert v_thick[2] > v_thin[2], "thickness_ratio should increase with thicker ply"
        assert v_thick[6] > v_thin[6], "total_thickness should increase with thicker ply"
        # Feature 4 (stiffness ratio E_bend * t^2 / E_fibre) should differ by t^2
        ratio = v_thick[4] / v_thin[4]
        expected = (t_thick / t_thin) ** 2
        assert ratio == pytest.approx(expected, rel=1e-4)

    def test_afl_thickness_not_used_when_override_given(self) -> None:
        """The .afl ply_thickness must be ignored when ply_thickness_mm is passed."""
        afl_t = UD_THERMOPLASTIC.ply_thickness  # 0.15 mm
        override_t = afl_t * 2                   # 0.30 mm
        v_override = UD_THERMOPLASTIC.to_physics_features(
            num_plies=2, ply_thickness_mm=override_t
        )
        v_afl = UD_THERMOPLASTIC.to_physics_features(num_plies=2, ply_thickness_mm=afl_t)
        # If override is ignored, these would be equal
        assert not np.allclose(v_override, v_afl), (
            "Override ply_thickness_mm had no effect — .afl value used instead"
        )


# ---------------------------------------------------------------------------
# get_physics_material_card — caching and correctness
# ---------------------------------------------------------------------------

class TestGetPhysicsMaterialCard:
    def test_returns_correct_shape(self) -> None:
        card = get_physics_material_card("A", num_plies=2, ply_thickness_mm=0.15)
        assert card.shape == (len(MATERIAL_FEATURES),)
        assert card.dtype == np.float32

    def test_both_batches_differ(self) -> None:
        card_a = get_physics_material_card("A", num_plies=2, ply_thickness_mm=0.15)
        card_b = get_physics_material_card("B", num_plies=3, ply_thickness_mm=0.85)
        assert not np.allclose(card_a, card_b), "Batch A and B cards should differ"

    def test_caching_returns_same_object(self) -> None:
        """Second call with same args must return the identical cached array."""
        c1 = get_physics_material_card("A", num_plies=2, ply_thickness_mm=0.15)
        c2 = get_physics_material_card("A", num_plies=2, ply_thickness_mm=0.15)
        assert c1 is c2, "Expected cached object identity — new array allocated on each call"

    def test_different_num_plies_gives_different_card(self) -> None:
        c2 = get_physics_material_card("A", num_plies=2, ply_thickness_mm=0.15)
        c4 = get_physics_material_card("A", num_plies=4, ply_thickness_mm=0.15)
        assert not np.allclose(c2, c4)

    def test_different_thickness_gives_different_card(self) -> None:
        c_thin = get_physics_material_card("A", num_plies=2, ply_thickness_mm=0.10)
        c_thick = get_physics_material_card("A", num_plies=2, ply_thickness_mm=0.30)
        assert not np.allclose(c_thin, c_thick)

    def test_cache_key_separates_batch_and_plies(self) -> None:
        """Ensure (A,2,0.15) and (B,2,0.15) are separate cache entries."""
        ca = get_physics_material_card("A", num_plies=2, ply_thickness_mm=0.15)
        cb = get_physics_material_card("B", num_plies=2, ply_thickness_mm=0.15)
        assert ca is not cb

    @pytest.mark.parametrize(
        "batch,num_plies,ply_thickness_mm",
        [
            ("C", 2, 0.15),
            ("A", 0, 0.15),
            ("A", 2, 0.0),
        ],
    )
    def test_rejects_invalid_inputs(self, batch: str, num_plies: int, ply_thickness_mm: float) -> None:
        with pytest.raises(ValueError):
            get_physics_material_card(batch, num_plies=num_plies, ply_thickness_mm=ply_thickness_mm)


class TestComputeMaterialCard:
    def test_prefers_sim_metadata_over_registry(self) -> None:
        sim_attrs = {
            "batch": "A",
            "n_plies": 4,
            "ply_thickness_afi_mm": 0.18,
            "punch_stroke_loadset2_mm": 120.0,
            "solve_dt": [2.0, 2.0, 0.05],
            "blank_initial_temp_C": 315.0,
        }
        registry_row = {
            "batch": "A",
            "n_plies": 4,
            "ply_thickness_afi_mm": 0.30,
            "punch_stroke_loadset2_mm": 75.0,
            "solve_dt": [2.0, 2.0, 0.1],
            "blank_initial_temp_C": 290.0,
        }
        card = compute_material_card(sim_attrs, registry_row)
        assert card.shape == (len(MATERIAL_FEATURES),)
        assert card[0] == pytest.approx(4.0)
        assert card[1] == pytest.approx(0.18)
        assert card[2] == pytest.approx(2400.0)
        assert card[3] == pytest.approx(315.0)

    def test_uses_registry_when_sim_metadata_missing(self) -> None:
        sim_attrs = {"batch": "B", "n_plies": 3}
        registry_row = {
            "batch": "B",
            "n_plies": 3,
            "ply_thickness_afi_mm": 0.85,
            "punch_stroke_loadset2_mm": 90.0,
            "solve_dt": 0.03,
            "blank_initial_temp_C": 300.0,
        }
        card = compute_material_card(sim_attrs, registry_row)
        assert card[1] == pytest.approx(0.85)
        assert card[2] == pytest.approx(3000.0)
        assert card[3] == pytest.approx(300.0)
        assert card[4] == pytest.approx(1.0)

    def test_defaults_when_metadata_and_registry_missing(self) -> None:
        card = compute_material_card({"batch": "A", "n_plies": 2}, None)
        assert card[1] == pytest.approx(0.3)
        assert card[2] == pytest.approx(75.0 / 0.0333, rel=1e-6)
        assert card[3] == pytest.approx(300.0)

    @pytest.mark.parametrize(
        "sim_attrs,registry_row",
        [
            ({"batch": "A", "n_plies": 2}, {"batch": "A", "n_plies": 4}),
            ({"batch": "A", "n_plies": 2}, {"batch": "B", "n_plies": 2}),
            ({"batch": "A", "n_plies": 2, "solve_dt": 0.0}, None),
            ({"batch": "A", "n_plies": 0}, None),
        ],
    )
    def test_rejects_mismatched_or_invalid_inputs(
        self, sim_attrs: dict[str, object], registry_row: dict[str, object] | None
    ) -> None:
        with pytest.raises(ValueError):
            compute_material_card(sim_attrs, registry_row)


# ---------------------------------------------------------------------------
# auto_detect_material heuristics
# ---------------------------------------------------------------------------

class TestAutoDetectMaterial:
    @pytest.mark.parametrize("name,expected", [
        ("mold_set_001", "woven"),
        ("MOLD_SET_001", "woven"),
        ("twintex_run", "woven"),
        ("2x2_twill_sim", "woven"),
        ("geom_0_3_pair1", "ud"),
        ("UD_laminate", "ud"),
        ("unidirectional_tape", "ud"),
        ("tape_run_05", "ud"),
        ("unknown_sim_xyz", "unknown"),
    ])
    def test_detection(self, name: str, expected: str) -> None:
        assert auto_detect_material(name) == expected


# ---------------------------------------------------------------------------
# .afl file parsing — skipped if files not present
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _AFL_PRESENT, reason="config/materials/*.afl files not found")
class TestAflParsing:
    def test_ud_fiber_modulus(self) -> None:
        props = parse_material_file(_UD_AFL)
        assert props.fiber_modulus == pytest.approx(25000.0, rel=1e-3)

    def test_ud_not_woven(self) -> None:
        props = parse_material_file(_UD_AFL)
        assert not props.is_woven

    def test_ud_ply_thickness(self) -> None:
        props = parse_material_file(_UD_AFL)
        assert props.ply_thickness == pytest.approx(0.15, rel=1e-3)

    def test_ud_bending_modulus(self) -> None:
        props = parse_material_file(_UD_AFL)
        # OrthoElastic E1=450 in bending layer
        assert props.bending_modulus == pytest.approx(450.0, rel=1e-2)

    def test_twintex_fiber_moduli(self) -> None:
        props = parse_material_file(_TWINTEX_AFL)
        assert len(props.fibers) == 2
        assert all(f.young_modulus == pytest.approx(10000.0, rel=1e-3) for f in props.fibers)

    def test_twintex_is_woven(self) -> None:
        props = parse_material_file(_TWINTEX_AFL)
        assert props.is_woven

    def test_twintex_ply_thickness(self) -> None:
        props = parse_material_file(_TWINTEX_AFL)
        assert props.ply_thickness == pytest.approx(0.85, rel=1e-3)

    def test_twintex_mooney_rivlin(self) -> None:
        props = parse_material_file(_TWINTEX_AFL)
        assert props.mooney_rivlin_c01 == pytest.approx(0.020563, rel=1e-4)

    def test_parsed_matches_hardcoded_defaults(self) -> None:
        """Parsed UD props must be consistent with the UD_THERMOPLASTIC fallback."""
        parsed = parse_material_file(_UD_AFL)
        assert parsed.fiber_modulus == pytest.approx(UD_THERMOPLASTIC.fiber_modulus, rel=1e-3)
        assert parsed.ply_thickness == pytest.approx(UD_THERMOPLASTIC.ply_thickness, rel=1e-3)
        assert parsed.bending_modulus == pytest.approx(UD_THERMOPLASTIC.bending_modulus, rel=1e-2)
