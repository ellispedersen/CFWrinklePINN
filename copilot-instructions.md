# Copilot Instructions — WP2 Execution Plan
## HDF5 Dataset Builder for CFWrinklePINN

**Generated:** 2026-03-29  
**Purpose:** Complete automation instructions for WP2 (HDF5 Dataset Schema) implementation  
**Status:** Ready to execute (WP1 gate closed 2026-03-29)  
**Target deliverable:** `data/cfwrinkle_dataset.h5` + `reports/WP2_gate.md`

---

## Executive Summary

Build a production-ready HDF5 dataset containing **66 usable simulation pairs** (21 Batch A UD + 45 Batch B Twintex) with fine and coarse mesh variants. Each simulation includes 13+ time-varying field arrays, mesh geometry, metadata, and wrinkle severity labels. The dataset supports stratified k-fold cross-validation for Physics-Informed Neural Network training in subsequent work packages.

**Key constraints:**
- Raw data in `CFWrinklePredict2/` is read-only — reference in place, never copy
- Follow group filtering rules strictly (ply vs bending groups per field type)
- Handle Batch A/B asymmetries (crystallinity, group numbering, node counts)
- Exclude aborted simulation `geom_0_1_pair2` (no AFR files present)

---

## Environment Setup

```powershell
cd "C:\Users\ellis\Documents\VS Code\CFWrinklePINN"
.\.venv\Scripts\Activate.ps1  # Virtual environment must be active

# Verify dependencies
python -c "import h5py, sklearn, yaml, numpy, pandas; print('All deps OK')"

# If missing, install:
pip install h5py>=3.10 scikit-learn>=1.4
```

**Context files to read before starting:**
- `config/pipeline_config.yaml` — paths, batch parameters, group definitions
- `config/field_registry.yaml` — all 13+ confirmed field definitions
- `reports/input_param_registry.json` — 65 simulation input parameters
- `reports/wrinkle_onset_registry.json` — 66 compound severity labels (0-1 range)
- `reports/force_stroke_summary.json` — 130 force-stroke time series records
- `CLAUDE.md` — AniForm reader APIs, critical data conventions
- `ANIFORM_REFERENCE.md` — group management rules (Section 1.6)
- `reports/WP2_spec.md` — complete technical specification (THIS IS THE AUTHORITATIVE SPEC)

---

## Implementation Sequence

### Phase 1: Schema Definition

#### Task 1.1: Create `wp2_build/__init__.py`
Create empty `__init__.py` to make `wp2_build/` a Python package.

#### Task 1.2: Create `wp2_build/schema.py`

Define HDF5 schema constants, field mappings, and data structures:

```python
"""
WP2 — HDF5 Dataset Schema for CFWrinklePINN
Defines hierarchy, field mappings, and data structures.
"""
from pathlib import Path
from dataclasses import dataclass

# ── HDF5 File Configuration ───────────────────────────────────────────────────
H5_PATH = Path("data/cfwrinkle_dataset.h5")
SCHEMA_VERSION = "1.0"

# ── Field Definitions ──────────────────────────────────────────────────────────
# Mapping: field_name → (afr_id, sub_id, shape_suffix, groups_key)
# groups_key: "ply" uses ply_groups (ScalarT/VectorT fields)
#             "bending" uses bending_groups (STensorPSST fields: strain, stress)

FIELD_DEFS = {
    # Kinematic fields (ply groups)
    "displacement":   (40,  1, (3,),  "ply"),      # VectorT [dx, dy, dz] mm
    "temperature":    (44,  1, (3,),  "ply"),      # VectorT [T1, T2, T3] degC
    
    # Scalar ply fields
    "thickness":      (105, 1, (),    "ply"),      # ScalarT mm
    "eq_shear_rate":  (106, 1, (),    "ply"),      # ScalarT 1/s
    
    # Fiber direction fields (ply groups)
    "fiber_dir_1":    (203, 1, (3,),  "ply"),      # VectorT unit vector
    "fiber_dir_2":    (203, 2, (3,),  "ply"),      # VectorT unit vector
    
    # Fiber deformation fields (ply groups)
    "shear_angle":    (204, 1, (),    "ply"),      # ScalarT deg (woven wrinkle indicator)
    "fiber_strain_1": (205, 1, (),    "ply"),      # ScalarT dimensionless
    "fiber_strain_2": (205, 2, (),    "ply"),      # ScalarT dimensionless
    "fiber_stress_1": (206, 1, (),    "ply"),      # ScalarT MPa ← PRIMARY TARGET
    "fiber_stress_2": (206, 2, (),    "ply"),      # ScalarT MPa
    
    # Tensor fields (bending groups — include sub-element layers)
    "gl_strain":      (102, 1, (3,),  "bending"),  # STensorPSST [E11, E22, E12]
    "stress":         (200, 1, (3,),  "bending"),  # STensorPSST [s11, s22, s12] MPa
    
    # Batch A only field
    "crystallinity":  (214, 1, (),    "ply"),      # ScalarT (Nakamura model)
}

# ── Simulation Record Structure ───────────────────────────────────────────────
@dataclass
class SimRecord:
    """Metadata for one simulation pair (fine + coarse mesh variants)."""
    sim_id: str                    # e.g., "geom_0_0_pair1" or "mold_set_000"
    batch: str                     # "A" | "B"
    fine_dir: Path                 # Path to fine mesh .Results directory
    coarse_dir: Path | None        # Path to coarse mesh (None if unavailable)
    ply_groups: list[int]          # Groups for ScalarT/VectorT fields
    bending_groups: list[int]      # Groups for STensorPSST fields (includes sub-elements)
    compound_severity: float       # [0, 1] from wrinkle_onset_registry.json
    is_wrinkled: bool              # True if compound_severity > threshold
    onset_increment: int           # Increment number of wrinkle onset (-1 if none)
    onset_stroke_frac: float       # Normalised stroke at onset (NaN if none)
    uz_range_mm: list[float]       # [min_uz, max_uz] from force_stroke_summary
    max_rfz_N: float               # Maximum reaction force Z component
    n_plies: int                   # 2 (Batch A) or 3 (Batch B)
    material: str                  # "UD_thermoplastic" | "Twintex_2x2_twill"
    ply_orientations_deg: list[float]  # Fiber angles per ply

# ── HDF5 Chunking and Compression ─────────────────────────────────────────────
CHUNK_TIME_DIM = 1
CHUNK_NODE_DIM = 4096
COMPRESSION = "lzf"

# ── Cross-Validation Configuration ────────────────────────────────────────────
CV_N_FOLDS = 5
CV_RANDOM_SEED = 42
```

**Critical implementation notes:**
- `fiber_stress_1` is the **PRIMARY TARGET** for wrinkle prediction (0→80% compressive during forming)
- Batch B has NO crystallinity field → write zero-size dataset with `available: False` attribute
- Group filtering is CRITICAL: use `ply_groups` for ScalarT/VectorT, `bending_groups` for STensorPSST
  - Batch A ply: [6, 10], bending: [6, 7, 10, 11]
  - Batch B ply: [4, 8, 12], bending: [4, 5, 8, 9, 12, 13]
- NEVER use Batch B contact groups [6, 7, 11, 15] for ply data — they overlap with Batch A numbering

---

### Phase 2: Dataset Builder Implementation

#### Task 2.1: Create `wp2_build/build_dataset.py` (Main Builder)

This is the core implementation. Structure:

```python
"""
WP2 — HDF5 Dataset Builder
Reads raw AniForm simulation data and writes cfwrinkle_dataset.h5

Usage:
    python -m wp2_build.build_dataset              # full build (66 sims)
    python -m wp2_build.build_dataset --dry-run    # validate only, no write
    python -m wp2_build.build_dataset --sim geom_0_0_pair1  # single sim
    python -m wp2_build.build_dataset --batch A    # Batch A only (21 sims)

Output: data/cfwrinkle_dataset.h5 (~500+ MB)
"""

import sys
from pathlib import Path
import argparse
import json
import yaml
import numpy as np
import h5py
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

# Add io/ to path for AniForm readers
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "io"))

from aniform_readers.ReadAFResult import ReadAFResult
from aniform_readers.ReadAFMesh import ReadAFMesh
from aniform_readers.ReadAFSFile import ReadAFSFile

from .schema import (
    H5_PATH, SCHEMA_VERSION, FIELD_DEFS, SimRecord,
    CHUNK_TIME_DIM, CHUNK_NODE_DIM, COMPRESSION,
    CV_N_FOLDS, CV_RANDOM_SEED
)
```

**Functions to implement (in order):**

##### 2.1.1 `load_configs() -> dict`
Load and merge all configuration files into a unified dictionary:
- `config/pipeline_config.yaml`
- `config/field_registry.yaml`
- `reports/input_param_registry.json`
- `reports/wrinkle_onset_registry.json`
- `reports/force_stroke_summary.json`

Return structure:
```python
{
    "pipeline": {...},  # YAML content
    "fields": {...},    # Field registry
    "input_params": {...},  # Simulation input parameters
    "wrinkle_registry": {...},  # Severity labels
    "force_stroke": {...}  # Time series data
}
```

##### 2.1.2 `build_sim_catalog(configs: dict) -> list[SimRecord]`
Create one `SimRecord` per usable simulation pair.

**Logic:**
1. Iterate through both batch roots (A and B) from `pipeline_config.yaml`
2. Find `.Results` directories matching `folder_glob` patterns
3. For each directory:
   - Check for required AFR files (at least displacement model_40_1.afr)
   - Skip `geom_0_1_pair2` (aborted, no AFR files)
   - Detect fine vs coarse by node count from displacement field
   - Extract metadata from `wrinkle_onset_registry.json` using sim_id key
   - Extract uz_range, max_rfz from `force_stroke_summary.json`
4. Return list of 66 `SimRecord` objects

**Edge cases:**
- `geom_0_6` has anomalous coarse mesh (86mm UZ, 0 force) — include but note
- If coarse variant not found, set `coarse_dir: None`

##### 2.1.3 `read_reference_mesh(run_dir: Path, groups: list[int]) -> tuple[np.ndarray, np.ndarray]`
Use `ReadAFMesh` to extract mesh geometry from `Part.section 1.1.msh`.

**Returns:**
- `nodes`: float32 array shape (n_nodes, 3) [x, y, z] in mm
- `elements`: int32 array shape (n_elements, 3) [node indices]

**Critical:** AniForm node indices are 1-based (Fortran convention). Convert to 0-based for Python/HDF5 by subtracting 1 from element connectivity.

##### 2.1.4 `read_field_all_increments(afr_dir: Path, afr_id: int, sub_id: int, groups: list[int]) -> tuple[np.ndarray, list[int]]`
Read all time increments of one field using `ReadAFResult`.

**Process:**
1. Call `ReadAFResult(filename, elemGrNrs=groups, IncsToExport=[], silent=True)`
2. `ResultsIncr[incr_nr][group_id]` → ndarray shape (n_nodes, n_components+1)
   - Column 0: node index (1-based)
   - Columns 1+: field components
3. For each increment:
   - Concatenate data across all groups
   - Sort by node index (col 0) to ensure consistent ordering
   - Remove duplicate nodes if any (keep first occurrence)
   - Strip node index column, keep only field values
4. Stack increments into shape `(n_incr, n_nodes, n_comp)` or `(n_incr, n_nodes)` for scalars

**Returns:**
- `data`: float32 ndarray
- `incr_numbers`: list of int (increment IDs from AniForm)

**Special handling:**
- For Batch B crystallinity: return empty ndarray shape `(n_incr, 0)` — field not present
- Check `isinstance(res_type, list)` — this means ReadAFResult found no data (error case)

##### 2.1.5 `read_increment_times(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
Parse `.afs` solution file using `ReadAFSFile` (returns pandas DataFrame).

**Returns:**
- `incr_numbers`: int32 array
- `times_s`: float32 array (simulation time in seconds)
- `stroke_frac`: float32 array [0, 1] (normalised stroke position)

**Stroke fraction calculation:**
1. Filter increments where `IncrementInfo['loadblock'] == 2` (forming phase)
2. `t_start = forming_increments['t_end'].min()`
3. `t_end = forming_increments['t_end'].max()`
4. `stroke_frac = (time - t_start) / (t_end - t_start)`
5. Clamp to [0, 1] if loadblock detection fails

**DataFrame access pattern:**
```python
times = IncrementInfo.loc[:, 't_end'].values.astype(float)  # NOT dict-style iteration
```

##### 2.1.6 `compute_derived(displacement: np.ndarray, fiber_stress_1: np.ndarray) -> dict`
Compute derived time series stored under `fine/derived/` and `coarse/derived/`.

**Returns dict with:**
- `comp_frac_f1`: float32 (n_incr,) — fraction of nodes with fiber_stress_1 < -0.05 MPa (compression indicator)
  ```python
  comp_frac_f1 = (fiber_stress_1 < -0.05).mean(axis=1)  # mean over spatial dim
  ```

- `dz_variance`: float32 (n_incr,) — variance of out-of-plane displacement (mm²)
  ```python
  dz = displacement[:, :, 2]  # Z component
  dz_variance = np.var(dz, axis=1)  # var over spatial dim
  ```

- `severity`: float32 (n_incr,) — compound severity per increment (same formula as `validation/wrinkle_detector.py`)
  - Load weights from `pipeline_config.yaml` under `dataset.compound_severity_weights`
  - Default: `fiber_stress: 0.50, dz_variance: 0.35, shear_angle: 0.15`
  - Normalise each component to [0, 1] using batch-specific scales, then weighted sum

##### 2.1.7 `write_sim_to_h5(h5: h5py.File, rec: SimRecord, field_data: dict)`
Write one simulation to open HDF5 file.

**Structure:**
```
simulations/{sim_id}/
  ├── attrs: batch, material, n_plies, ply_groups, fine_run_dir, ...
  ├── mesh/fine/nodes, mesh/fine/elements
  ├── mesh/coarse/nodes, mesh/coarse/elements (if coarse_dir not None)
  ├── fine/increments, fine/times_s, fine/stroke_frac
  ├── fine/displacement, fine/temperature, fine/thickness, ...
  ├── fine/derived/comp_frac_f1, fine/derived/dz_variance, fine/derived/severity
  └── coarse/... (same structure if available)
```

**Chunking strategy:**
For time-indexed arrays shape `(n_incr, n_nodes, n_comp)`:
```python
chunks = (CHUNK_TIME_DIM, min(n_nodes, CHUNK_NODE_DIM), n_comp) if n_comp > 1 else \
         (CHUNK_TIME_DIM, min(n_nodes, CHUNK_NODE_DIM))
ds = grp.create_dataset(name, data=arr, dtype='float32', chunks=chunks, compression=COMPRESSION)
```

**Batch B crystallinity handling:**
```python
if batch == "B" and field_name == "crystallinity":
    # Write zero-size dataset
    grp.create_dataset("crystallinity", shape=(n_incr, 0), dtype='float32')
    grp["crystallinity"].attrs["available"] = False
```

**Attributes:**
Store all `SimRecord` fields as HDF5 attributes. Convert numpy integers to Python `int` before writing (HDF5 doesn't support np.uint32 in attrs).

##### 2.1.8 `write_metadata(h5: h5py.File, configs: dict)`
Write `metadata/` group with JSON-serialized configs and build info.

```python
meta = h5.create_group("metadata")
meta.create_dataset("field_registry", data=json.dumps(configs["fields"]))
meta.create_dataset("pipeline_config", data=json.dumps(configs["pipeline"]))
meta.attrs["schema_version"] = SCHEMA_VERSION
meta.attrs["build_date"] = datetime.now().isoformat()
meta.attrs["n_pairs"] = 66
meta.attrs["n_fields"] = len(FIELD_DEFS)
# Optional: git hash via subprocess.run(['git', 'rev-parse', 'HEAD'])
```

##### 2.1.9 `write_splits(h5: h5py.File, sim_ids: list[str], labels: dict)`
Generate stratified k-fold cross-validation splits.

**Stratification strategy:**
Create composite label combining:
1. Batch (A vs B)
2. Severity quartile (bin `compound_severity` into 4 groups)

```python
from sklearn.model_selection import StratifiedKFold

# Build stratification labels
strat_labels = []
for sim_id in sim_ids:
    batch = labels[sim_id]["batch"]
    severity = labels[sim_id]["compound_severity"]
    quartile = int(severity * 4)  # 0, 1, 2, 3
    strat_labels.append(f"{batch}_Q{quartile}")

# Generate splits (train/val/test per fold)
skf = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=CV_RANDOM_SEED)
splits_grp = h5.create_group("splits")
splits_grp.attrs["strategy"] = "stratified_batch_severity_quartile"
splits_grp.attrs["seed"] = CV_RANDOM_SEED
splits_grp.attrs["n_folds"] = CV_N_FOLDS

for k, (train_idx, test_idx) in enumerate(skf.split(sim_ids, strat_labels)):
    fold_grp = splits_grp.create_group(f"fold_{k}")
    # Use variable-length string dtype
    str_dtype = h5py.string_dtype(encoding='utf-8')
    train_pool_ids = np.array([sim_ids[i] for i in train_idx], dtype=object)
    train_pool_labels = np.array([strat_labels[i] for i in train_idx], dtype=object)
    train_ids, val_ids = train_test_split(
        train_pool_ids,
        test_size=0.2,
        random_state=CV_RANDOM_SEED + k,
        stratify=train_pool_labels,
    )
    fold_grp.create_dataset("train", data=train_ids, dtype=str_dtype)
    fold_grp.create_dataset("val", data=val_ids, dtype=str_dtype)
    fold_grp.create_dataset("test", data=[sim_ids[i] for i in test_idx], dtype=str_dtype)
```

##### 2.1.10 `main()`
CLI entry point with argument parsing.

```python
def main():
    parser = argparse.ArgumentParser(description="Build CFWrinklePINN HDF5 dataset")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    parser.add_argument("--sim", type=str, help="Process single sim_id only")
    parser.add_argument("--batch", choices=["A", "B"], help="Process one batch only")
    args = parser.parse_args()
    
    print("=== WP2 Dataset Builder ===")
    print(f"Target: {H5_PATH}")
    
    # Load configs
    configs = load_configs()
    print("✓ Loaded configs")
    
    # Build catalog
    catalog = build_sim_catalog(configs)
    print(f"✓ Built catalog: {len(catalog)} simulations")
    
    # Filter by CLI args
    if args.sim:
        catalog = [rec for rec in catalog if rec.sim_id == args.sim]
    if args.batch:
        catalog = [rec for rec in catalog if rec.batch == args.batch]
    
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        for rec in catalog:
            # Validate AFR files exist
            for field_name, (afr_id, sub_id, _, _) in FIELD_DEFS.items():
                afr_path = rec.fine_dir / "results" / f"model_{afr_id}_{sub_id}.afr"
                status = "✓" if afr_path.exists() else "✗ MISSING"
                print(f"  {status} {rec.sim_id}: {afr_path.name}")
        print("\nDry run complete. No HDF5 file written.")
        return
    
    # Full build mode
    H5_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(H5_PATH, "w") as h5:
        write_metadata(h5, configs)
        print("✓ Wrote metadata")
        
        for i, rec in enumerate(catalog, 1):
            print(f"\n[{i}/{len(catalog)}] Processing {rec.sim_id}...")
            
            # Read all fields for this sim
            field_data = {}
            for field_name, (afr_id, sub_id, shape, groups_key) in FIELD_DEFS.items():
                groups = rec.bending_groups if groups_key == "bending" else rec.ply_groups
                
                # Skip crystallinity for Batch B
                if field_name == "crystallinity" and rec.batch == "B":
                    field_data[field_name] = None
                    continue
                
                afr_path = rec.fine_dir / "results" / f"model_{afr_id}_{sub_id}.afr"
                data, incrs = read_field_all_increments(afr_path, afr_id, sub_id, groups)
                field_data[field_name] = data
            
            # Read mesh
            mesh_path = rec.fine_dir / "Part.section 1.1.msh"
            nodes, elements = read_reference_mesh(mesh_path, rec.ply_groups)
            field_data["mesh_nodes"] = nodes
            field_data["mesh_elements"] = elements
            
            # Read increment times
            afs_path = list(rec.fine_dir.glob("*.afs"))[0]
            incr_nums, times, stroke = read_increment_times(afs_path)
            field_data["increments"] = incr_nums
            field_data["times_s"] = times
            field_data["stroke_frac"] = stroke
            
            # Compute derived
            derived = compute_derived(field_data["displacement"], field_data["fiber_stress_1"])
            field_data["derived"] = derived
            
            # Write to HDF5
            write_sim_to_h5(h5, rec, field_data)
            print(f"  ✓ Wrote {rec.sim_id} ({rec.batch})")
        
        # Generate CV splits
        sim_ids = [rec.sim_id for rec in catalog]
        labels = {rec.sim_id: {"batch": rec.batch, "compound_severity": rec.compound_severity} for rec in catalog}
        write_splits(h5, sim_ids, labels)
        print("\n✓ Generated CV splits")
    
    print(f"\n=== BUILD COMPLETE ===")
    print(f"Output: {H5_PATH}")
    print(f"Size: {H5_PATH.stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
```

---

### Phase 3: Validation Script

#### Task 3.1: Create `wp2_build/validate_dataset.py`

This script verifies the built dataset meets all requirements.

```python
"""
WP2 Validation — Verify HDF5 dataset integrity

Usage:
    python -m wp2_build.validate_dataset
    python -m wp2_build.validate_dataset --verbose

Exit code 0 = all checks pass, non-zero = failures detected
"""

import sys
from pathlib import Path
import h5py
import numpy as np
import json
import argparse
from typing import List, Tuple

from .schema import H5_PATH, FIELD_DEFS

# Expected dataset structure
REQUIRED_FINE_FIELDS = [
    "displacement", "temperature", "thickness", "eq_shear_rate",
    "fiber_dir_1", "fiber_dir_2", "shear_angle",
    "fiber_strain_1", "fiber_strain_2", "fiber_stress_1", "fiber_stress_2",
    "gl_strain", "stress", "increments", "times_s", "stroke_frac"
]

REQUIRED_DERIVED = ["comp_frac_f1", "dz_variance", "severity"]
BATCH_A_ONLY = ["crystallinity"]
EXPECTED_N_SIMS = 66
CV_N_FOLDS = 5

class ValidationError(Exception):
    pass

def check_sim_count(h5: h5py.File, verbose: bool) -> bool:
    """Verify 66 simulation groups present."""
    sim_ids = list(h5["simulations"].keys())
    n_sims = len(sim_ids)
    
    if n_sims != EXPECTED_N_SIMS:
        print(f"✗ FAIL: Expected {EXPECTED_N_SIMS} sims, found {n_sims}")
        return False
    
    if verbose:
        print(f"✓ Simulation count: {n_sims}")
    return True

def check_required_fields(h5: h5py.File, verbose: bool) -> bool:
    """Verify all required fields present per sim."""
    all_ok = True
    
    for sim_id in h5["simulations"].keys():
        sim_grp = h5[f"simulations/{sim_id}"]
        batch = sim_grp.attrs["batch"]
        
        # Check fine mesh fields
        for field in REQUIRED_FINE_FIELDS:
            if field not in sim_grp["fine"]:
                print(f"✗ FAIL: {sim_id} missing fine/{field}")
                all_ok = False
        
        # Check derived fields
        for field in REQUIRED_DERIVED:
            if field not in sim_grp["fine/derived"]:
                print(f"✗ FAIL: {sim_id} missing fine/derived/{field}")
                all_ok = False
        
        # Batch A crystallinity
        if batch == "A":
            if "crystallinity" not in sim_grp["fine"]:
                print(f"✗ FAIL: {sim_id} (Batch A) missing crystallinity")
                all_ok = False
        elif batch == "B":
            # Batch B should have zero-size crystallinity with available=False
            if "crystallinity" in sim_grp["fine"]:
                cryst = sim_grp["fine/crystallinity"]
                if cryst.shape[1] != 0:
                    print(f"✗ FAIL: {sim_id} (Batch B) crystallinity should have shape[1]==0")
                    all_ok = False
                if cryst.attrs.get("available", True):
                    print(f"✗ FAIL: {sim_id} (Batch B) crystallinity missing available=False attr")
                    all_ok = False
    
    if all_ok and verbose:
        print(f"✓ All required fields present")
    return all_ok

def check_no_nan_inf(h5: h5py.File, verbose: bool) -> bool:
    """Check critical fields for NaN/Inf."""
    critical_fields = ["displacement", "fiber_stress_1", "stress"]
    all_ok = True
    
    for sim_id in h5["simulations"].keys():
        sim_grp = h5[f"simulations/{sim_id}"]
        
        for field in critical_fields:
            data = sim_grp[f"fine/{field}"][:]
            if np.any(np.isnan(data)):
                print(f"✗ FAIL: {sim_id} fine/{field} contains NaN")
                all_ok = False
            if np.any(np.isinf(data)):
                print(f"✗ FAIL: {sim_id} fine/{field} contains Inf")
                all_ok = False
    
    if all_ok and verbose:
        print(f"✓ No NaN/Inf in critical fields")
    return all_ok

def check_severity_labels(h5: h5py.File, verbose: bool) -> bool:
    """Verify compound_severity attrs match wrinkle_onset_registry.json."""
    registry_path = Path("reports/wrinkle_onset_registry.json")
    with open(registry_path) as f:
        registry = json.load(f)
    
    all_ok = True
    for sim_id in h5["simulations"].keys():
        sim_grp = h5[f"simulations/{sim_id}"]
        h5_severity = sim_grp.attrs["compound_severity"]
        
        if sim_id not in registry:
            print(f"✗ FAIL: {sim_id} not found in wrinkle_onset_registry.json")
            all_ok = False
            continue
        
        registry_severity = registry[sim_id]["max_compound_severity"]
        if abs(h5_severity - registry_severity) > 1e-6:
            print(f"✗ FAIL: {sim_id} severity mismatch: h5={h5_severity:.3f}, registry={registry_severity:.3f}")
            all_ok = False
    
    if all_ok and verbose:
        print(f"✓ Severity labels match registry")
    return all_ok

def check_splits_integrity(h5: h5py.File, verbose: bool) -> bool:
    """Verify CV splits have no overlap and cover all sims."""
    all_sim_ids = set(h5["simulations"].keys())
    all_ok = True
    
    for k in range(CV_N_FOLDS):
        fold_grp = h5[f"splits/fold_{k}"]
        train_ids = set(fold_grp["train"].asstr()[:])
        val_ids = set(fold_grp["val"].asstr()[:])
        test_ids = set(fold_grp["test"].asstr()[:])
        
        # Check no overlap between train/val/test
        overlap_tv = train_ids & val_ids
        overlap_tt = train_ids & test_ids
        overlap_vt = val_ids & test_ids
        overlap = overlap_tv | overlap_tt | overlap_vt
        if overlap:
            print(f"✗ FAIL: fold_{k} has split overlap: {overlap}")
            all_ok = False
        
        # Check union matches all sims
        union = train_ids | val_ids | test_ids
        if union != all_sim_ids:
            missing = all_sim_ids - union
            extra = union - all_sim_ids
            if missing:
                print(f"✗ FAIL: fold_{k} missing sims: {missing}")
            if extra:
                print(f"✗ FAIL: fold_{k} has extra sims: {extra}")
            all_ok = False
    
    if all_ok and verbose:
        print(f"✓ CV splits integrity OK")
    return all_ok

def check_node_count_spot(h5: h5py.File, verbose: bool) -> bool:
    """Spot-check node counts against ReadAFResult for 3 A + 3 B sims."""
    all_ok = True
    
    for batch in ("A", "B"):
        batch_sims = [k for k in h5["simulations"].keys() if h5[f"simulations/{k}"].attrs["batch"] == batch]
        sample = batch_sims[:3]  # deterministic sample of 3
        
        for sim_id in sample:
            sim_grp = h5[f"simulations/{sim_id}"]
            afr_path = Path(sim_grp.attrs["fine_run_dir"]) / "results" / "model_40_1.afr"
            ply_groups = list(map(int, sim_grp.attrs["ply_groups"]))
            data, _ = read_field_all_increments(afr_path, 40, 1, ply_groups)
            ref_nodes = int(data.shape[1])
            h5_nodes = int(sim_grp.attrs["n_nodes_fine"])
            if h5_nodes != ref_nodes:
                print(f"✗ FAIL: {sim_id} node mismatch: h5={h5_nodes}, afr={ref_nodes}")
                all_ok = False
    
    if all_ok and verbose:
        print(f"✓ Node count spot-check OK")
    return all_ok

def check_increment_count_spot(h5: h5py.File, verbose: bool) -> bool:
    """Spot-check increment counts against .afs for 3 A + 3 B sims."""
    all_ok = True
    for batch in ("A", "B"):
        batch_sims = [k for k in h5["simulations"].keys() if h5[f"simulations/{k}"].attrs["batch"] == batch]
        sample = batch_sims[:3]
        for sim_id in sample:
            sim_grp = h5[f"simulations/{sim_id}"]
            afs_candidates = list(Path(sim_grp.attrs["fine_run_dir"]).glob("*.afs"))
            if not afs_candidates:
                print(f"✗ FAIL: {sim_id} no .afs file found")
                all_ok = False
                continue
            afs_incr, _, _ = read_increment_times(afs_candidates[0])
            h5_incr = sim_grp["fine/increments"][:]
            if len(h5_incr) != len(afs_incr):
                print(f"✗ FAIL: {sim_id} increment mismatch: h5={len(h5_incr)}, afs={len(afs_incr)}")
                all_ok = False
    if all_ok and verbose:
        print("✓ Increment count spot-check OK")
    return all_ok

def main():
    parser = argparse.ArgumentParser(description="Validate CFWrinklePINN HDF5 dataset")
    parser.add_argument("--verbose", action="store_true", help="Print detailed pass messages")
    args = parser.parse_args()
    
    if not H5_PATH.exists():
        print(f"✗ FAIL: Dataset not found at {H5_PATH}")
        sys.exit(1)
    
    print("=== WP2 Dataset Validation ===")
    print(f"Dataset: {H5_PATH}")
    print(f"Size: {H5_PATH.stat().st_size / 1e6:.1f} MB\n")
    
    with h5py.File(H5_PATH, "r") as h5:
        checks = [
            ("Simulation count", check_sim_count),
            ("Required fields", check_required_fields),
            ("NaN/Inf check", check_no_nan_inf),
            ("Increment count spot-check", check_increment_count_spot),
            ("Severity labels", check_severity_labels),
            ("CV splits integrity", check_splits_integrity),
            ("Node count spot-check", check_node_count_spot),
        ]
        
        results = []
        for name, func in checks:
            print(f"Running: {name}...")
            ok = func(h5, args.verbose)
            results.append((name, ok))
            if not ok:
                print()
        
        print("\n=== VALIDATION SUMMARY ===")
        all_passed = all(ok for _, ok in results)
        for name, ok in results:
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"{status}: {name}")
        
        if all_passed:
            print("\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
            sys.exit(0)
        else:
            print("\n✗✗✗ VALIDATION FAILED ✗✗✗")
            sys.exit(1)

if __name__ == "__main__":
    main()
```

---

### Phase 4: Configuration Updates

#### Task 4.1: Update `config/pipeline_config.yaml`

Add dataset section with build parameters:

```yaml
# ── HDF5 Dataset Configuration (WP2) ──────────────────────────────────────────
dataset:
  hdf5_path: data/cfwrinkle_dataset.h5
  schema_version: "1.0"
  chunk_time_dim: 1
  chunk_node_dim: 4096
  compression: lzf
  
  # Cross-validation
  cv_n_folds: 5
  cv_random_seed: 42
  
  # Compound severity calculation (weights sum to 1.0)
  compound_severity_weights:
    fiber_stress: 0.50      # Compression fraction weight
    dz_variance: 0.35       # Out-of-plane variance weight
    shear_angle: 0.15       # Shear angle weight
  
  # Component-specific parameters
  comp_frac_deadband_MPa: -0.05         # Threshold for compression detection
  comp_frac_severity_scale: 0.50        # Max compression fraction for normalization
  
  dz_var_severity_scale:                # Batch-specific variance scales (mm²)
    A: 10.0
    B: 0.40
  
  shear_angle_severity_scale:           # Batch-specific shear angle scales (deg)
    A: 90.0
    B: 45.0
```

---

### Phase 5: Gate Checklist

#### Task 5.1: Create `reports/WP2_gate.md`

```markdown
# WP2 Gate Checklist
## HDF5 Dataset Build — CFWrinklePINN

**Status:** ⏳ In progress  
**Date:** 2026-03-29  
**Blocker for:** WP3 (Feature Extraction)

**Do not begin WP3 until every item below is checked.**

---

## Step 2.1 — Schema Design

- [ ] `wp2_build/__init__.py` created (empty package marker)
- [ ] `wp2_build/schema.py` defines:
  - [ ] `H5_PATH`, `SCHEMA_VERSION`, `FIELD_DEFS`
  - [ ] `SimRecord` dataclass with all required fields
  - [ ] All 13 confirmed feature fields included (displacement through crystallinity)
  - [ ] crystallinity marked with note "Batch A only"
  - [ ] Chunking and compression constants defined

---

## Step 2.2 — Dataset Build Script

- [ ] `wp2_build/build_dataset.py` implemented with functions:
  - [ ] `load_configs()` — loads all 5 config files
  - [ ] `build_sim_catalog()` — creates 66 SimRecord objects
  - [ ] `read_reference_mesh()` — extracts nodes/elements from .msh
  - [ ] `read_field_all_increments()` — reads AFR time series
  - [ ] `read_increment_times()` — parses .afs for times/stroke
  - [ ] `compute_derived()` — calculates comp_frac_f1, dz_variance, severity
  - [ ] `write_sim_to_h5()` — writes one sim to HDF5
  - [ ] `write_metadata()` — writes metadata group
  - [ ] `write_splits()` — generates stratified CV folds
  - [ ] `main()` — CLI with --dry-run, --sim, --batch args

- [ ] **Dry-run validation:**
  ```bash
  python -m wp2_build.build_dataset --dry-run
  ```
  - [ ] Completes with 0 errors
  - [ ] Reports 66 simulations (21 Batch A + 45 Batch B)
  - [ ] All required AFR files found (except geom_0_1_pair2 skipped)

- [ ] **Full build execution:**
  ```bash
  python -m wp2_build.build_dataset
  ```
  - [ ] Completes without exceptions
  - [ ] `data/cfwrinkle_dataset.h5` written
  - [ ] File size > 500 MB (expected ~500-800 MB)
  - [ ] 66 simulation groups present in `simulations/`
  - [ ] `splits/` group present with 5 folds (fold_0 through fold_4)

---

## Step 2.3 — Validation

- [ ] `wp2_build/validate_dataset.py` implemented
- [ ] **Validation run:**
  ```bash
  python -m wp2_build.validate_dataset --verbose
  ```
  - [ ] ✓ Simulation count: 66
  - [ ] ✓ All required fields present for all sims
  - [ ] ✓ Zero NaN/Inf in displacement, fiber_stress_1, stress
  - [ ] ✓ Node count spot-check passes (3 A + 3 B sims)
  - [ ] ✓ compound_severity attrs match `wrinkle_onset_registry.json`
  - [ ] ✓ CV splits integrity check passes (no train/val/test overlap)
  - [ ] ✓ Batch B crystallinity has shape[1]==0 and available=False

---

## Step 2.4 — Configuration Updates

- [ ] `config/pipeline_config.yaml` updated with `dataset:` section
- [ ] Dataset parameters match schema.py constants

---

## Step 2.5 — Manual Verification

Open Python REPL and verify:

```python
import h5py

h5 = h5py.File("data/cfwrinkle_dataset.h5", "r")

# Check sim count
sim_ids = list(h5["simulations"].keys())
print(f"Sim count: {len(sim_ids)}")  # Should be 66

# Check metadata
print(f"Schema version: {h5['metadata'].attrs['schema_version']}")  # "1.0"
print(f"Build date: {h5['metadata'].attrs['build_date']}")

# Spot-check specific sims
mold_004 = h5["simulations/mold_set_004"]
print(f"mold_set_004 severity: {mold_004.attrs['compound_severity']:.3f}")  # ~0.167 (lowest)

geom_08 = h5["simulations/geom_0_8"]
print(f"geom_0_8 severity: {geom_08.attrs['compound_severity']:.3f}")  # ~0.473 (anomalous A)

# Check Batch B crystallinity
mold_000 = h5["simulations/mold_set_000"]
cryst = mold_000["fine/crystallinity"]
print(f"Batch B crystallinity shape: {cryst.shape}")  # (n_incr, 0)
print(f"Batch B crystallinity available: {cryst.attrs['available']}")  # False

# Check fiber_stress_1 final increment (should show compression)
fs1 = mold_000["fine/fiber_stress_1"][-1, :]  # Last increment
print(f"fiber_stress_1 range: [{fs1.min():.2f}, {fs1.max():.2f}] MPa")
print(f"Nodes in compression (<-0.05 MPa): {(fs1 < -0.05).sum()}")

h5.close()
```

**Manual checks:**
- [ ] `mold_set_004` has `compound_severity` ≈ 0.167 (lowest severity, borderline clean)
- [ ] `geom_0_8` has `compound_severity` ≈ 0.473 (anomalous Batch A case)
- [ ] Batch B `crystallinity` dataset has shape `(n_incr, 0)` and `attrs["available"] == False`
- [ ] At least 1 Batch A and 1 Batch B `fiber_stress_1` final-increment array shows compression (negative values)
- [ ] Plotted at least one `fine/displacement` array (z-component) to visually confirm wrinkle shape

---

## Step 2.6 — Known Issues Handled

- [ ] `geom_0_1_pair2` (aborted run) correctly skipped — not in dataset
- [ ] `geom_0_6` anomalous coarse mesh (86mm UZ, 0 force) included with note
- [ ] Batch B missing crystallinity handled — zero-size dataset with `available=False`
- [ ] Node index conversion (1-based → 0-based) applied to mesh elements
- [ ] Multi-group node concatenation uses sorted, deduplicated node indices
- [ ] STensorPSST fields (gl_strain, stress) use `bending_groups`, not `ply_groups`

---

## Sign-Off

- [ ] All above items checked
- [ ] `data/cfwrinkle_dataset.h5` size and content verified
- [ ] No blockers remaining
- [ ] **WP3 (Feature Extraction) may begin**

**Signed:** _________________________  
**Date:** _________________________
```

---

## Execution Checklist for Copilot

Use this sequence to execute WP2:

### Stage 1: Setup and Schema
1. ✅ Verify environment: `.venv` activated, dependencies installed
2. ✅ Read all context files listed in "Context files to read"
3. ✅ Create `wp2_build/__init__.py` (empty)
4. ✅ Create `wp2_build/schema.py` with all constants and `SimRecord` dataclass

### Stage 2: Builder Implementation
5. ✅ Create `wp2_build/build_dataset.py` skeleton with imports
6. ✅ Implement `load_configs()`
7. ✅ Implement `build_sim_catalog()` — test with `--dry-run`
8. ✅ Implement `read_reference_mesh()`
9. ✅ Implement `read_field_all_increments()`
10. ✅ Implement `read_increment_times()`
11. ✅ Implement `compute_derived()`
12. ✅ Implement `write_sim_to_h5()`
13. ✅ Implement `write_metadata()`
14. ✅ Implement `write_splits()`
15. ✅ Implement `main()` with CLI argument parsing

### Stage 3: Validation
16. ✅ Create `wp2_build/validate_dataset.py` with all checks
17. ✅ Test validation script structure (before dataset exists)

### Stage 4: Execution
18. ✅ Run `python -m wp2_build.build_dataset --dry-run`
   - Fix any AFR file discovery issues
   - Verify 66 sims cataloged
19. ✅ Run `python -m wp2_build.build_dataset --sim geom_0_0_pair1`
   - Test single-sim write
   - Verify HDF5 structure with h5py
20. ✅ Run `python -m wp2_build.build_dataset --batch A`
   - Test Batch A (21 sims)
   - Check crystallinity field present
21. ✅ Run `python -m wp2_build.build_dataset`
   - Full build (66 sims)
   - Monitor for errors, memory usage

### Stage 5: Validation and Gate
22. ✅ Run `python -m wp2_build.validate_dataset --verbose`
   - Address any failures
   - Re-run builder if needed
23. ✅ Update `config/pipeline_config.yaml` with dataset section
24. ✅ Create `reports/WP2_gate.md`
25. ✅ Perform manual verification checks from gate checklist
26. ✅ Mark all gate items complete

---

## Critical Edge Cases and Gotchas

### AniForm Reader Quirks
1. **Node indices are 1-based:** Always subtract 1 when storing element connectivity
2. **ReadAFSFile returns DataFrame:** Use `.loc[row, col].values`, NOT dict-style access
3. **Empty results return list for res_type:** Check `isinstance(res_type, list)` before using
4. **Numpy uint32 in attrs:** Convert to Python `int()` before HDF5 attribute write
5. **Multi-group concatenation:** Sort by node index column 0, deduplicate if overlaps exist

### Batch Asymmetries
1. **Crystallinity:** Batch A has field, Batch B does not → zero-size dataset with `available=False`
2. **Group numbering:** Batch B contact groups [6, 7, 11, 15] OVERLAP Batch A ply groups → NEVER mix
3. **Node counts:** Batch A fine ~75k, Batch B fine ~40k → adjust chunking if needed
4. **Shear angle scale:** Batch B approaches 45° (locking angle), Batch A approaches 90°

### Data Quality
1. **geom_0_1_pair2:** Aborted run, no AFR files → skip in catalog
2. **geom_0_6 coarse:** Anomalous (86mm UZ, 0 force) → include but note in attrs
3. **Loadblock detection:** If AFS has no loadblock 2, fall back to global time normalisation

### HDF5 Best Practices
1. **Chunking:** Use `(1, min(n_nodes, 4096), n_comp)` for time-series data
2. **Compression:** `lzf` is fast and sufficient (avoid gzip/blosc overhead)
3. **String datasets:** Use `h5py.string_dtype(encoding='utf-8')` for variable-length strings
4. **Attribute types:** HDF5 attrs don't support numpy scalar types → cast to Python types

---

## Dependencies and Imports

Required packages (already in `requirements.txt`):
- `h5py>=3.10`
- `scikit-learn>=1.4`
- `numpy>=1.24`
- `pandas>=2.0`
- `pyyaml>=6.0`

AniForm readers (in `io/aniform_readers/`):
- `ReadAFResult.py`
- `ReadAFMesh.py`
- `ReadAFSFile.py`

---

## Success Criteria

WP2 is **complete** when:
1. ✅ `data/cfwrinkle_dataset.h5` exists and is 500-800 MB
2. ✅ Contains 66 simulation groups (21 A + 45 B)
3. ✅ All validation checks pass with exit code 0
4. ✅ Manual verification confirms:
   - Severity labels match registry
   - Batch B crystallinity handled correctly
   - Fiber stress shows compression in wrinkled sims
   - CV splits are stratified and non-overlapping
5. ✅ `reports/WP2_gate.md` all items checked
6. ✅ No blocker issues preventing WP3 start

---

## Post-WP2 Actions

After gate sign-off:
1. Commit `wp2_build/` module code to git
2. Commit `config/pipeline_config.yaml` updates
3. Commit `reports/WP2_gate.md`
4. Add `data/cfwrinkle_dataset.h5` to `.gitignore` (already present)
5. Document dataset location and schema in `README.md`
6. Proceed to **WP3: Feature Extraction Pipeline**

---

## Notes for Future Work Packages

**WP3 will use:**
- `h5py.File("data/cfwrinkle_dataset.h5", "r")` to load simulations
- `splits/fold_{k}/train`, `splits/fold_{k}/val`, and `splits/fold_{k}/test` for CV
- `simulations/{sim_id}/fine/fiber_stress_1` as primary target
- `simulations/{sim_id}/attrs["compound_severity"]` for stratification

**Dataset assumptions WP3+ must honour:**
- Increments are time-ordered but NOT uniformly spaced
- Node ordering is consistent across fields within a sim, but varies between sims
- Coarse mesh may be absent (check `coarse_dir is not None` in attrs or presence of `mesh/coarse/`)
- Batch B has no crystallinity → skip or use dummy values in feature extractors

---

**END OF COPILOT INSTRUCTIONS**
