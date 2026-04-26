"""AniForm .afl material file parser.

Parses AniForm Layup files to extract mechanical, thermal and geometric
properties. The two canonical material files live in config/materials/.

This is a reference/utility module. The active material card used during
training is computed by wp3_features/material.py:compute_material_card(),
which derives an 8-D process-parameter vector from simulation metadata.
This parser provides the underlying AniForm material properties and a
physics-based 8-D feature vector (to_physics_features) that can be used
for future WP iterations or ablation studies.

Usage:
    from wp3_features.material_parser import parse_material_file, MaterialProperties

    props = parse_material_file("config/materials/UD reinforced thermoplastic unitMPa 2025-04-14.afl")
    print(props.fiber_modulus)        # average fiber stiffness in MPa
    print(props.to_physics_features(num_plies=2))  # 8-D feature vector
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Canonical material file names (relative to config/materials/)
# ---------------------------------------------------------------------------
AFL_UD = "UD reinforced thermoplastic unitMPa 2025-04-14.afl"
AFL_TWINTEX = "Twintex GF-PP-unconsolidated-2x2twill-1485gsm fromLiterature RT unitMPA 2025-04-14.afl"


@dataclass
class FiberProperties:
    """Properties for a single fiber direction."""
    young_modulus: float  # MPa
    orientation: float    # degrees


@dataclass
class MaterialProperties:
    """Complete material properties extracted from an AniForm .afl file."""

    # Identification
    name: str = ""
    unit_system: str = "MPA"   # MPA or BIN

    # Fiber properties (multiple entries for woven)
    fibers: List[FiberProperties] = field(default_factory=list)

    # In-plane mechanical
    mooney_rivlin_c10: float = 0.0   # MPa
    mooney_rivlin_c01: float = 0.0   # MPa
    in_plane_viscosity: float = 0.0  # MPa·s
    density: float = 0.0             # tonne/mm³

    # Bending
    bending_modulus: float = 0.0     # MPa
    bending_viscosity: float = 0.0   # MPa·s

    # Geometry
    ply_thickness: float = 0.0       # mm

    # Interface / friction
    friction_coefficient: float = 0.0  # dimensionless
    penalty_stiffness: float = 0.0     # MPa/mm
    adhesion_tension: float = 0.0      # MPa

    # Thermal
    heat_capacity: float = 0.0             # J/(tonne·K)
    conductivity_in_plane: float = 0.0     # W/(mm·K)
    conductivity_through_thickness: float = 0.0  # W/(mm·K)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def fiber_modulus(self) -> float:
        """Average fiber modulus across all fiber directions."""
        if not self.fibers:
            return 0.0
        return float(np.mean([f.young_modulus for f in self.fibers]))

    @property
    def fiber_orientations(self) -> List[float]:
        return [f.orientation for f in self.fibers]

    @property
    def is_woven(self) -> bool:
        """True when the material has more than one fiber direction."""
        return len(self.fibers) > 1

    # ------------------------------------------------------------------
    # Feature vectors
    # ------------------------------------------------------------------

    def to_physics_features(self, num_plies: int = 2,
                            reference_thickness: float = 0.85,
                            ply_thickness_mm: Optional[float] = None) -> np.ndarray:
        """Physics-aware dimensionless 8-D feature vector.

        Args:
            num_plies: actual ply count from simulation metadata (NOT from the
                .afl file — always pass the real value from the sim).
            reference_thickness: normalisation constant (mm).
            ply_thickness_mm: actual per-ply thickness from simulation metadata.
                If None, falls back to the .afl value (self.ply_thickness).
                The .afl thickness and the simulated thickness can differ, so
                always pass the real value when available.

        Features:
            0  Fibre anisotropy ratio  log10(E1/E2 + 1)
            1  Bending anisotropy      log10(E_bend1/E_bend2 + 1)
            2  Thickness ratio         t / reference_thickness
            3  Viscosity ratio         log10(eta_bend / eta_inplane + 1)
            4  Stiffness ratio         E_bend * t^2 / E_fibre
            5  Num plies               num_plies / 4
            6  Total thickness         (t * num_plies) / 2 mm
            7  Material class          0 = woven, 1 = UD
        """
        if self.fibers:
            E1 = self.fibers[0].young_modulus
            E2 = self.fibers[1].young_modulus if len(self.fibers) > 1 else 1.0
        else:
            E1 = max(self.fiber_modulus, 1.0)
            E2 = 1.0
        E2 = max(E2, 1.0)
        anisotropy = np.log10(E1 / E2 + 1.0)

        bending_E1 = max(self.bending_modulus, 1e-6)
        bending_E2 = bending_E1 if self.is_woven else bending_E1 / 100.0
        bending_anisotropy = np.log10(bending_E1 / bending_E2 + 1.0)

        t = ply_thickness_mm if ply_thickness_mm is not None else (self.ply_thickness or 0.0)
        thickness_ratio = t / reference_thickness
        total_thickness = t * num_plies / 2.0

        eta_ip = max(self.in_plane_viscosity, 1e-6)
        eta_b = max(self.bending_viscosity, 1e-6)
        viscosity_ratio = np.log10(eta_b / eta_ip + 1.0)

        stiffness_ratio = bending_E1 * t ** 2 / max(E1, 1e-6)

        return np.array([
            anisotropy,
            bending_anisotropy,
            thickness_ratio,
            viscosity_ratio,
            stiffness_ratio,
            num_plies / 4.0,
            total_thickness,
            0.0 if self.is_woven else 1.0,
        ], dtype=np.float32)

    def to_feature_vector(self, num_plies: int = 2) -> np.ndarray:
        """Alias for to_physics_features (backward compatibility)."""
        return self.to_physics_features(num_plies=num_plies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit_system": self.unit_system,
            "fibers": [{"young_modulus": f.young_modulus, "orientation": f.orientation}
                       for f in self.fibers],
            "mooney_rivlin_c10": self.mooney_rivlin_c10,
            "mooney_rivlin_c01": self.mooney_rivlin_c01,
            "in_plane_viscosity": self.in_plane_viscosity,
            "density": self.density,
            "bending_modulus": self.bending_modulus,
            "bending_viscosity": self.bending_viscosity,
            "ply_thickness": self.ply_thickness,
            "friction_coefficient": self.friction_coefficient,
            "penalty_stiffness": self.penalty_stiffness,
            "adhesion_tension": self.adhesion_tension,
            "heat_capacity": self.heat_capacity,
            "conductivity_in_plane": self.conductivity_in_plane,
            "conductivity_through_thickness": self.conductivity_through_thickness,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaterialProperties":
        props = cls()
        props.name = data.get("name", "")
        props.unit_system = data.get("unit_system", "MPA")
        props.fibers = [FiberProperties(**f) for f in data.get("fibers", [])]
        for attr in (
            "mooney_rivlin_c10", "mooney_rivlin_c01", "in_plane_viscosity",
            "density", "bending_modulus", "bending_viscosity", "ply_thickness",
            "friction_coefficient", "penalty_stiffness", "adhesion_tension",
            "heat_capacity", "conductivity_in_plane", "conductivity_through_thickness",
        ):
            setattr(props, attr, data.get(attr, 0.0))
        return props


# ---------------------------------------------------------------------------
# Hardcoded defaults (used when .afl is unavailable)
# ---------------------------------------------------------------------------

TWINTEX_GF_PP = MaterialProperties(
    name="Twintex GF-PP-unconsolidated-2x2twill-1485gsm",
    unit_system="MPA",
    fibers=[
        FiberProperties(young_modulus=10000.0, orientation=0.0),
        FiberProperties(young_modulus=10000.0, orientation=90.0),
    ],
    mooney_rivlin_c10=0.0,
    mooney_rivlin_c01=0.020563,
    in_plane_viscosity=0.01,
    density=1.92e-9,
    bending_modulus=11.95,
    bending_viscosity=1.0,
    ply_thickness=0.85,
    friction_coefficient=0.22,
    penalty_stiffness=10.0,
    adhesion_tension=0.01,
    heat_capacity=1.1443e9,
    conductivity_in_plane=0.828,
    conductivity_through_thickness=0.3821,
)

UD_THERMOPLASTIC = MaterialProperties(
    name="UD reinforced thermoplastic",
    unit_system="MPA",
    fibers=[FiberProperties(young_modulus=25000.0, orientation=0.0)],
    in_plane_viscosity=0.07,
    density=2e-9,
    bending_modulus=450.0,
    bending_viscosity=37500.0,
    ply_thickness=0.15,
)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _extract_property(data: Any, property_type: str) -> Optional[dict]:
    """Return the first nested dict whose $type contains property_type."""
    if isinstance(data, dict):
        if property_type in data.get("$type", ""):
            return data
        for v in data.values():
            result = _extract_property(v, property_type)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = _extract_property(item, property_type)
            if result is not None:
                return result
    return None


def _extract_all_properties(data: Any, property_type: str) -> List[dict]:
    """Return all nested dicts whose $type contains property_type."""
    results: List[dict] = []
    if isinstance(data, dict):
        if property_type in data.get("$type", ""):
            results.append(data)
        for v in data.values():
            results.extend(_extract_all_properties(v, property_type))
    elif isinstance(data, list):
        for item in data:
            results.extend(_extract_all_properties(item, property_type))
    return results


def _sanitize_aniform_json(content: str) -> str:
    """Fix common AniForm JSON export quirks (missing commas)."""
    content = re.sub(r'([}\]])"MaterialType"', r'\1,"MaterialType"', content)
    content = re.sub(r'([}\]])"Kind"', r'\1,"Kind"', content)
    content = re.sub(r'([}\]])"Name"', r'\1,"Name"', content)
    return content


def _get_number(data: dict, *keys: str, default: float = 0.0) -> float:
    """Return first numeric field for keys, tolerating accidental trailing commas."""
    for key in keys:
        for candidate in (key, f"{key},"):
            if candidate in data:
                try:
                    return float(data[candidate])
                except (TypeError, ValueError):
                    return default
    return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_material_file(file_path: str | Path) -> MaterialProperties:
    """Parse an AniForm .afl material file and return a MaterialProperties."""
    file_path = Path(file_path)
    props = MaterialProperties()

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    if lines[0].startswith("Aniform"):
        content = "\n".join(lines[1:])

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        content = _sanitize_aniform_json(content)
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Warning: could not parse {file_path}: {e}")
            return props

    props.name = data.get("Name", file_path.stem)
    path_upper = str(file_path).upper()
    props.unit_system = "BIN" if "BIN" in path_upper else "MPA"

    # Fibers
    for fp in _extract_all_properties(data, "LinearElasticFiber"):
        for fiber in fp.get("Fibers", {}).get("$values", []):
            props.fibers.append(FiberProperties(
                young_modulus=fiber.get("Young", 0.0),
                orientation=fiber.get("Orientation", 0.0),
            ))

    # Mooney-Rivlin
    mr = _extract_property(data, "MooneyRivlin")
    if mr:
        props.mooney_rivlin_c10 = _get_number(mr, "C10")
        props.mooney_rivlin_c01 = _get_number(mr, "C01")

    # Viscosity — higher value assumed to be bending, lower is in-plane
    viscosity_props = _extract_all_properties(data, "Viscous")
    viscosity_props.extend(_extract_all_properties(data, "CrossViscosityFluid"))
    for vp in viscosity_props:
        eta = _get_number(vp, "Eta", "Eta0")
        if eta >= 0.5:
            props.bending_viscosity = eta
        else:
            props.in_plane_viscosity = eta

    # Density and bending modulus from IsoElastic / OrthoElastic entries
    for ie in _extract_all_properties(data, "IsoElastic"):
        d = _get_number(ie, "Density")
        if d > 0:
            props.density = d
        y = _get_number(ie, "Young")
        if y > 1.0:
            props.bending_modulus = y
    for oe in _extract_all_properties(data, "OrthoElastic"):
        e1 = _get_number(oe, "E1")
        if e1 > 1.0:
            props.bending_modulus = e1

    # Friction
    friction = _extract_property(data, "PenaltyCoulomb")
    if not friction:
        friction = _extract_property(data, "PenaltyPolymer")
    if friction:
        props.friction_coefficient = _get_number(friction, "Mu")
        props.penalty_stiffness = _get_number(friction, "PenaltyStiffness")

    # Adhesion
    adhesion = _extract_property(data, "Adhesion")
    if adhesion:
        props.adhesion_tension = adhesion.get("Tension", 0.0)

    # Thermal
    thermal = _extract_property(data, "OrthoThermal")
    if thermal:
        props.heat_capacity = _get_number(thermal, "Capacity")
        props.conductivity_in_plane = _get_number(thermal, "Conductivity1")
        props.conductivity_through_thickness = _get_number(thermal, "Conductivity3")

    # Ply thickness — first ContinuumLayer with Thickness > 0
    for layer in data.get("Layers", {}).get("$values", []):
        if "ContinuumLayer" in layer.get("$type", ""):
            t = layer.get("Thickness", 0.0)
            if t > 0:
                props.ply_thickness = float(t)
                break

    return props


def parse_material_directory(dir_path: str | Path) -> Dict[str, MaterialProperties]:
    """Parse all .afl files in dir_path; returns {stem: MaterialProperties}."""
    results: Dict[str, MaterialProperties] = {}
    for afl in Path(dir_path).rglob("*.afl"):
        try:
            results[afl.stem] = parse_material_file(afl)
        except Exception as e:
            print(f"Warning: could not parse {afl}: {e}")
    return results
