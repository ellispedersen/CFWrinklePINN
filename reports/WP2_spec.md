# WP2 — HDF5 Dataset Schema
## Specification Checklist for Automated Execution

**Status:** Ready to begin (WP1 gate closed 2026-03-29)
**Executor:** Codex / GitHub Copilot
**Output:** `data/cfwrinkle_dataset.h5` (gitignored) + `reports/WP2_gate.md`
**Prerequisite reads:**
  - `config/pipeline_config.yaml` — batch paths, group IDs
  - `config/field_registry.yaml` — all confirmed field definitions
  - `reports/input_param_registry.json` — 65 simulation input params
  - `reports/wrinkle_onset_registry.json` — 66 compound severity labels
  - `reports/force_stroke_summary.json` — 130 force-stroke records
  - `CLAUDE.md` — AniForm reader APIs and data conventions
  - `ANIFORM_REFERENCE.md` — group management rules (Section 1.6)

---

## Context and Constraints

**Dataset:** 66 usable simulation pairs (21 Batch A UD + 45 Batch B Twintex).
Each pair has a coarse mesh run and a fine mesh run. Raw data lives in
`CFWrinklePredict2/` (read-only, referenced in-place, never copied).

**AniForm file types per run:**
- `model_40_1.afr` — Displacement VectorT [dx, dy, dz]
- `model_102_1.afr` — GL strain STensorPSST [E11, E22, E12]
- `model_105_1.afr` — Thickness ScalarT
- `model_106_1.afr` — Eq shear rate ScalarT
- `model_200_1.afr` — Stress STensorPSST [s11, s22, s12] (bending groups)
- `model_203_1.afr` + `model_203_2.afr` — Fiber direction 1/2 VectorT
- `model_204_1.afr` — Shear angle f1_f2 ScalarT
- `model_205_1.afr` + `model_205_2.afr` — Fiber strain 1/2 ScalarT
- `model_206_1.afr` — Fiber stress 1 ScalarT (**PRIMARY TARGET**)
- `model_206_2.afr` — Fiber stress 2 ScalarT
- `model_44_1.afr` — Temperature VectorT [T1, T2, T3]
- `model_214_1.afr` — Nakamura crystallinity ScalarT (**Batch A only**)

**Excluded fields (contact artifacts):** 302, 304, 401

**Group filtering rules (critical — see ANIFORM_REFERENCE.md §1.6):**
- ScalarT/VectorT fields: use `ply_groups` only
  - Batch A: [6, 10]   Batch B: [4, 8, 12]
- STensorPSST fields (stress model_200, strain model_102): use `bending_groups`
  - Batch A: [6, 7, 10, 11]   Batch B: [4, 5, 8, 9, 12, 13]
- NEVER use Batch B contact groups [6, 7, 11, 15] for ply data — they overlap Batch A ply groups

**Reader API:**
```python
from aniform_readers.ReadAFResult import ReadAFResult
(ResultsIncr, Groups, Increments, res_type, indices_included, res_id) = ReadAFResult(
    filename, elemGrNrs=[], IncsToExport=[], silent=True
)
# ResultsIncr[incr_nr][group_id] -> ndarray shape (n_nodes, n_components+1), col 0 = node index
```

**Target label:** `compound_severity` float [0, 1] from `wrinkle_onset_registry.json`
(field `max_compound_severity`). Also store binary `is_wrinkled` bool.

---

## Step 2.1 — Design HDF5 Schema

### [ ] 2.1.1 Create `wp2_build/schema.py`

Define the HDF5 group/dataset hierarchy as constants. Schema:

```
cfwrinkle_dataset.h5
├── metadata/
│   ├── field_registry          [JSON string, attrs: version]
│   ├── pipeline_config         [JSON string]
│   └── build_info              [attrs: date, git_hash, n_pairs, n_fields]
│
├── simulations/
│   └── {sim_id}/               (e.g., "geom_0_0_pair1", "mold_set_000")
│       ├── attrs:
│       │   batch               str  "A" | "B"
│       │   material            str  "UD_thermoplastic" | "Twintex_2x2_twill"
│       │   n_plies             int
│       │   ply_groups          int[N]
│       │   fine_run_dir        str
│       │   n_nodes_fine        int
│       │   n_nodes_coarse      int
│       │   uz_range_mm         float[2]
│       │   max_rfz_N           float
│       │   ply_orientations_deg float[N]
│       │   compound_severity   float  [0, 1]
│       │   is_wrinkled         bool
│       │   onset_increment     int | -1
│       │   onset_stroke_frac   float | NaN
│       │
│       ├── mesh/
│       │   ├── fine/
│       │   │   ├── nodes       float32 (n_nodes, 3) [x, y, z in mm]
│       │   │   └── elements    int32   (n_elements, 3) [node indices, 0-based]
│       │   └── coarse/
│       │       ├── nodes       float32 (n_nodes_c, 3)
│       │       └── elements    int32   (n_elements_c, 3)
│       │
│       ├── fine/
│       │   ├── increments      int32   (n_incr,)   increment numbers
│       │   ├── times_s         float32 (n_incr,)   simulation time in seconds
│       │   ├── stroke_frac     float32 (n_incr,)   normalised stroke [0, 1]
│       │   │
│       │   ├── displacement    float32 (n_incr, n_nodes, 3)  [dx, dy, dz] mm
│       │   ├── temperature     float32 (n_incr, n_nodes, 3)  [T1, T2, T3] degC
│       │   ├── thickness       float32 (n_incr, n_nodes)     mm
│       │   ├── eq_shear_rate   float32 (n_incr, n_nodes)     1/s
│       │   ├── fiber_dir_1     float32 (n_incr, n_nodes, 3)  unit vector
│       │   ├── fiber_dir_2     float32 (n_incr, n_nodes, 3)  unit vector
│       │   ├── shear_angle     float32 (n_incr, n_nodes)     deg
│       │   ├── fiber_strain_1  float32 (n_incr, n_nodes)     dimensionless
│       │   ├── fiber_strain_2  float32 (n_incr, n_nodes)     dimensionless
│       │   ├── fiber_stress_1  float32 (n_incr, n_nodes)     MPa  ← PRIMARY TARGET
│       │   ├── fiber_stress_2  float32 (n_incr, n_nodes)     MPa
│       │   ├── gl_strain       float32 (n_incr, n_nodes_b, 3) [E11, E22, E12] bending groups
│       │   ├── stress          float32 (n_incr, n_nodes_b, 3) [s11, s22, s12] bending groups
│       │   ├── crystallinity   float32 (n_incr, n_nodes)     Batch A only (absent for B)
│       │   │
│       │   └── derived/
│       │       ├── comp_frac_f1  float32 (n_incr,)  fraction nodes with fiber_stress_1 < -0.05 MPa
│       │       ├── dz_variance   float32 (n_incr,)  mm^2
│       │       └── severity      float32 (n_incr,)  compound severity per increment
│       │
│       └── coarse/
│           └── (same structure as fine/, for coarse mesh fields)
│
└── splits/
    ├── attrs: strategy, seed, n_folds
    └── fold_{k}/
        ├── train  str[N]  sim_ids in training set
        ├── val    str[N]  sim_ids in validation set
        └── test   str[N]  sim_ids in test set
```

Constants to define in `schema.py`:
```python
H5_PATH = Path("data/cfwrinkle_dataset.h5")
SCHEMA_VERSION = "1.0"

# Field name → (afr_id, sub_id, array_shape_suffix, groups_key)
# groups_key: "ply" | "bending"
FIELD_DEFS = {
    "displacement":   (40,  1, (3,),  "ply"),
    "temperature":    (44,  1, (3,),  "ply"),
    "thickness":      (105, 1, (),    "ply"),
    "eq_shear_rate":  (106, 1, (),    "ply"),
    "fiber_dir_1":    (203, 1, (3,),  "ply"),
    "fiber_dir_2":    (203, 2, (3,),  "ply"),
    "shear_angle":    (204, 1, (),    "ply"),
    "fiber_strain_1": (205, 1, (),    "ply"),
    "fiber_strain_2": (205, 2, (),    "ply"),
    "fiber_stress_1": (206, 1, (),    "ply"),   # primary target
    "fiber_stress_2": (206, 2, (),    "ply"),
    "gl_strain":      (102, 1, (3,),  "bending"),
    "stress":         (200, 1, (3,),  "bending"),
    # Batch A only:
    "crystallinity":  (214, 1, (),    "ply"),
}
```

---

## Step 2.2 — Implement Builder Script

### [ ] 2.2.1 Create `wp2_build/__init__.py` (empty)

### [ ] 2.2.2 Create `wp2_build/build_dataset.py`

Main builder script. Structure:

```python
"""
WP2 — Build HDF5 dataset from raw AniForm simulation data.

Usage:
    python -m wp2_build.build_dataset              # full build
    python -m wp2_build.build_dataset --dry-run    # validate structure only
    python -m wp2_build.build_dataset --sim geom_0_0_pair1  # single sim
    python -m wp2_build.build_dataset --batch B    # one batch

Output: data/cfwrinkle_dataset.h5
"""
```

**Functions to implement:**

#### `load_configs() -> dict`
Load and merge pipeline_config.yaml, field_registry.yaml, wrinkle_onset_registry.json,
force_stroke_summary.json, input_param_registry.json. Return unified sim catalog.

#### `build_sim_catalog(configs) -> list[SimRecord]`
Create one `SimRecord` per usable pair with fields:
```python
@dataclass
class SimRecord:
    sim_id: str
    batch: str              # "A" | "B"
    fine_dir: Path
    coarse_dir: Path | None
    ply_groups: list[int]
    bending_groups: list[int]
    compound_severity: float
    is_wrinkled: bool
    onset_increment: int     # -1 if no onset
    onset_stroke_frac: float # NaN if no onset
    uz_range_mm: list[float]
    n_plies: int
    material: str
    ply_orientations_deg: list[float]
```
Exclude `geom_0_1_pair2` (aborted, no AFR files).

#### `read_reference_mesh(run_dir, groups) -> tuple[np.ndarray, np.ndarray]`
Use `ReadAFMesh` to get nodes (n, 3) and elements (m, 3).
Node indices in elements must be converted to 0-based.

#### `read_field_all_increments(afr_dir, afr_id, sub_id, groups) -> tuple[np.ndarray, list[int]]`
Read all increments of one field. Returns:
- `data`: float32 ndarray, shape `(n_incr, n_nodes, n_comp)` for vector/tensor,
  or `(n_incr, n_nodes)` for scalar
- `incr_numbers`: list of int

Handle the multi-group case: concatenate node data across groups per increment.
Node ordering must be consistent (sort by node index col 0 after concatenation).

For fields absent in a batch (e.g., crystallinity in Batch B): return empty ndarray
with n_incr rows but zero node dimension — write as zero-size dataset with attribute
`available: False`.

#### `read_increment_times(run_dir) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
Use `ReadAFSFile` to get `(incr_numbers, times_s, stroke_frac)`.
Stroke fraction = (time - t_forming_start) / (t_forming_end - t_forming_start)
for forming loadblock (loadblock == 2). Fall back to global normalisation if no loadblock 2.

#### `compute_derived(displacement, fiber_stress_1) -> dict`
Compute derived arrays stored under `derived/`:
- `comp_frac_f1(t)` = (fiber_stress_1[t] < -0.05).mean(axis=-1)  shape (n_incr,)
- `dz_variance(t)` = np.var(displacement[t, :, 2], axis=-1)       shape (n_incr,)
- `severity(t)` — compound compound_severity per increment (same formula as wrinkle_detector.py)

#### `write_sim_to_h5(h5: h5py.File, rec: SimRecord, field_data: dict)`
Write one simulation to the open HDF5 file. Create group `simulations/{sim_id}/`.
Set all attrs. Write mesh, fine/ fields, coarse/ fields, derived/ fields.
Use chunking: `chunks=(1, min(n_nodes, 4096), n_comp)` for time-indexed arrays.
Use `compression="lzf"` for all datasets.

#### `write_metadata(h5: h5py.File, configs: dict)`
Write `metadata/` group. Include schema version, build date, git hash if available.

#### `write_splits(h5: h5py.File, sim_ids: list[str], labels: dict)`
Stratified k-fold CV splits (k=5). Stratify by:
- batch (A vs B)
- severity quartile (low/mid-low/mid-high/high based on compound_severity)
Use `sklearn.model_selection.StratifiedKFold`. Random seed = 42.
Store as variable-length string arrays in `splits/fold_{k}/`.

#### `main()`
Parse args `--dry-run`, `--sim`, `--batch`.
In dry-run mode: discover all AFR files, validate field presence, print summary — no HDF5 write.
In full mode: build catalog → open h5 → write metadata → iterate sims → write splits → report.

---

## Step 2.3 — Implement Validation Script

### [ ] 2.3.1 Create `wp2_build/validate_dataset.py`

```python
"""
WP2 Validation — Verify the built HDF5 dataset.

Usage:
    python -m wp2_build.validate_dataset
    python -m wp2_build.validate_dataset --verbose

Checks:
  - All 66 sim_ids present
  - All required datasets exist per sim
  - No NaN/Inf in any array (except allowed: onset_stroke_frac for no-onset sims)
  - Node count matches ReadAFResult for 3 randomly sampled sims per batch
  - Increment count matches .afs file for 3 randomly sampled sims per batch
  - compound_severity attrs match wrinkle_onset_registry.json exactly
  - Splits: no overlap between train/val/test within a fold, union = all sims
  - Batch B crystallinity dataset has attribute available=False
"""
```

**Checks to implement:**

```python
REQUIRED_FINE_FIELDS = [
    "displacement", "temperature", "thickness", "eq_shear_rate",
    "fiber_dir_1", "fiber_dir_2", "shear_angle",
    "fiber_strain_1", "fiber_strain_2", "fiber_stress_1", "fiber_stress_2",
    "gl_strain", "stress", "increments", "times_s", "stroke_frac",
    "derived/comp_frac_f1", "derived/dz_variance", "derived/severity",
]
BATCH_A_ONLY_FIELDS = ["crystallinity"]
```

For each check: print PASS / FAIL with details. Exit code 0 only if all pass.

---

## Step 2.4 — Write WP2 Gate Checklist

### [ ] 2.4.1 Create `reports/WP2_gate.md`

```markdown
# WP2 Gate Checklist
## HDF5 Dataset Build

**Do not begin WP3 until every item is checked.**

## Step 2.1 — Schema Design
[ ] schema.py defines H5_PATH, SCHEMA_VERSION, FIELD_DEFS
[ ] All 13 confirmed feature fields included (excludes 302, 304, 401)
[ ] crystallinity marked Batch A only in schema

## Step 2.2 — Dataset Build
[ ] build_dataset.py --dry-run completes with 0 errors for all 66 sims
[ ] build_dataset.py full run completes without exceptions
[ ] data/cfwrinkle_dataset.h5 written (check file size > 500 MB)
[ ] 66 sim groups present in simulations/
[ ] splits/ group present with 5 folds

## Step 2.3 — Validation
[ ] validate_dataset.py: all REQUIRED_FINE_FIELDS present for all 66 sims
[ ] validate_dataset.py: zero NaN/Inf in displacement, fiber_stress_1, stress
[ ] validate_dataset.py: node count spot-check passes (3 A + 3 B sims)
[ ] validate_dataset.py: increment count spot-check passes
[ ] validate_dataset.py: compound_severity attrs match registry JSON
[ ] validate_dataset.py: split integrity check passes (no train/val/test overlap)

## Step 2.4 — Manual Checks
[ ] h5py.File('data/cfwrinkle_dataset.h5')['simulations'].keys() shows all 66
[ ] mold_set_004 compound_severity attr == 0.167 (lowest severity, borderline clean)
[ ] geom_0_8 compound_severity attr == 0.473 (anomalous Batch A case)
[ ] Batch B crystallinity dataset has shape[1] == 0 and attr available=False
[ ] At least 1 Batch A and 1 Batch B fiber_stress_1 final-increment array plotted

## Sign-off
[ ] WP2_gate.md reviewed
[ ] data/cfwrinkle_dataset.h5 size and content verified
[ ] WP3 (feature extraction) may begin
```

---

## Step 2.5 — Package Structure

### [ ] 2.5.1 Ensure `wp2_build/` follows existing pattern

```
wp2_build/
  __init__.py       (empty)
  schema.py         (HDF5 schema constants + SimRecord dataclass)
  build_dataset.py  (main builder)
  validate_dataset.py (validation checks)
```

Add to `requirements.txt` if not present: `h5py>=3.10`, `scikit-learn>=1.4`

### [ ] 2.5.2 Update `config/pipeline_config.yaml`

Add section:
```yaml
dataset:
  hdf5_path: data/cfwrinkle_dataset.h5
  schema_version: "1.0"
  chunk_time_dim: 1
  chunk_node_dim: 4096
  compression: lzf
  cv_n_folds: 5
  cv_random_seed: 42
  compound_severity_weights:
    fiber_stress: 0.50
    dz_variance: 0.35
    shear_angle: 0.15
  comp_frac_deadband_MPa: -0.05
  comp_frac_severity_scale: 0.50
  dz_var_severity_scale:
    A: 10.0
    B: 0.40
  shear_angle_severity_scale:
    A: 90.0
    B: 45.0
```

---

## Execution Order

```bash
# 1. Schema and builder
python -m wp2_build.build_dataset --dry-run       # validate without writing
python -m wp2_build.build_dataset                 # write data/cfwrinkle_dataset.h5

# 2. Validate
python -m wp2_build.validate_dataset

# 3. Gate check
# Manually verify WP2_gate.md items, then begin WP3
```

---

## Known Edge Cases for Builder

| Issue | Handling |
|---|---|
| geom_0_1_pair2 (aborted) | Skip — no AFR files, mark in metadata |
| geom_0_6 anomalous coarse (86mm UZ, 0 force) | Include coarse mesh but note anomaly in attrs |
| Batch B missing crystallinity | Write zero-size dataset, attr available=False |
| Multi-group nodes overlap | Sort by node index col after concat; dedup if needed |
| STensorPSST bending groups have more nodes than ply groups | Use bending_groups for gl_strain + stress, ply_groups for rest |
| Node indices in ReadAFResult are 1-based (Fortran) | Subtract 1 after reading for 0-based storage |
| Stroke fraction outside [0,1] if AFS loadblock detection fails | Clamp to [0,1], set attr afs_loadblock_used=False |
| Batch B shear angle approaching 45 deg (locking) | Expected; not an error |

---

## Notes for Codex / Copilot

- This repository uses `io/aniform_readers/` for AniForm binary parsing — do not reimpleent these.
- All paths must use `pathlib.Path` and come from `pipeline_config.yaml`, not hardcoded.
- `ReadAFResult` returns `res_type` as a list when no data found — check `isinstance(res_type, list)` before using it.
- Numpy integer types from ReadAFResult are `np.uint32` — cast to `int` before HDF5 attribute storage.
- HDF5 variable-length strings: use `h5py.special_dtype(vlen=str)` or `h5py.string_dtype()` depending on h5py version.
- The `io/` directory is on `sys.path` via `sys.path.insert(0, str(_ROOT / "io"))` in all scripts.
- `ReadAFSFile` returns a pandas DataFrame, not a dict — access via `.loc['row_name'].values`.
