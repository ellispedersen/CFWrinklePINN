# AniForm Data Reference
## Extraction Utilities, Binary Formats, and Field Mapping

**Purpose:** WP1 archaeology reference for both batches. Authoritative source for pipeline (WP2+) design.
**Last updated:** 2026-03-29 — corrected field mapping from ReadAFResult(silent=False) binary headers.

---

## 1. Dataset Structure

### 1.1 Total Dataset

| Batch | Label | Material | Geometry | Pairs | Ply groups | Bending groups | Contact groups | Plies |
|---|---|---|---|---|---|---|---|---|
| Batch A | UD | UD thermoplastic (AS4/PEEK-like) | flat_smooth_0 punch + flat die | **21** | 6, 10 | 6, 7, 10, 11 | 8, 9, 13 | 2 |
| Batch B | Twintex | Twintex 2×2 twill GF-PP | composite_plate_NNN | **45** | 4, 8, 12 | 4, 5, 8, 9, 12, 13 | 6, 7, 11, 15 | 3 |
| **Total** | — | — | — | **66–67** | — | — | — | — |

One "pair" = one coarse run + one fine run of the same geometry.

A planned **3rd batch** (40 geometries × 2 thickness variants) was attempted but **all runs aborted** — both zip archives (`Updated 40x2 sim set.zip`, `SIM Batch Final Aniform.zip`) contain only warmup `.tmp` files with zero `.afr` outputs. Not usable.

---

### 1.2 Batch A — Directory Layout

```
CFWrinklePredict2/data/aniform_raw/
├── UD reinforced thermoplastic unitMPa 2025-04-14.afl  ← material library
├── geom 0_0.afp                                        ← project file (binary, not parseable)
├── geom 0_0.Results/        ← 21 simulation pairs total: geom 0_0 has 2 pairs
│   ├── 2025-08-31 13_12_15/     ← FINE run (Part.section 1.1.msh ≈ 9312 KB)
│   │   ├── model.afi            ← ASCII input file
│   │   ├── model.afm            ← binary mesh (reference config, NOT deformed)  [NOT in results/]
│   │   ├── meshes/
│   │   │   ├── Part.section 1.1.msh   ← ply 0 deformed mesh (size → fine/coarse ID)
│   │   │   ├── Part.section 1.2.msh   ← ply 1 deformed mesh
│   │   │   └── flat_smooth_0_*.msh    ← tool meshes (ignore)
│   │   └── results/
│   │       ├── model.afs              ← solution file (increment timing, convergence)
│   │       ├── model_40_1.afr         ← Displacement [dx,dy,dz] (VectorT)
│   │       ├── model_42_1.afr         ← Rotation (tool DOF, near-zero on plies)
│   │       ├── model_44_1.afr         ← Temperature [T1,T2,T3] (VectorT)
│   │       ├── model_48_1.afr         ← Temperature 1 (top surface, ScalarT)
│   │       ├── model_49_1.afr         ← Temperature 2 (bottom surface, ScalarT)
│   │       ├── model_102_1.afr        ← Green-Lagrange strain [E11,E22,E12] (STensorPSST)
│   │       ├── model_105_1.afr        ← Thickness (ScalarT, mm)
│   │       ├── model_106_1.afr        ← Eq shear rate (ScalarT, 1/s)
│   │       ├── model_200_1.afr        ← Stress [s11,s22,s12] (STensorPSST, MPa)
│   │       ├── model_203_1.afr        ← Fiber direction 1 [fd_x,fd_y,fd_z] (VectorT)
│   │       ├── model_203_2.afr        ← Fiber direction 2 (fiber family 2)
│   │       ├── model_204_1.afr        ← Shear angle f1_f2 (ScalarT, deg)
│   │       ├── model_205_1.afr        ← Fiber strain 1 (ScalarT)
│   │       ├── model_205_2.afr        ← Fiber strain 2 (fiber family 2)
│   │       ├── model_206_1.afr        ← Fiber stress 1 (ScalarT, MPa) ★ PRIMARY WRINKLE PRECURSOR
│   │       ├── model_206_2.afr        ← Fiber stress 2 (fiber family 2)
│   │       ├── model_214_1.afr        ← Rel. crystallinity Nakamura (Batch A only)
│   │       ├── model_302_1.afr        ← Penetration depth (CONTACT field, ScalarT)
│   │       ├── model_304_1.afr        ← Slip path length (CONTACT field, ScalarT)
│   │       └── model_401_1.afr        ← Traction (CONTACT field, all zeros)
│   └── 2025-08-31 14_58_22/     ← COARSE run (Part.section 1.1.msh ≈ 456 KB)
│       └── ...same structure...
├── geom 0_1.Results/        ← 1 pair (3 runs: 1 fine + 1 coarse + 1 aborted = use the 2 valid)
├── geom 0_2.Results/ ... geom 0_19.Results/   ← 1 pair each
```

**geom 0_0 has 4 runs = 2 independent coarse+fine pairs — use both:**

| Pair | Fine run | Coarse run | Status |
|---|---|---|---|
| pair 1 | 2025-08-31 13_12_15 (9.5 MB) | 2025-08-31 14_58_22 (464 KB) | use |
| pair 2 | 2025-09-30 15_19_36 (9.5 MB) | 2025-09-30 15_20_12 (391 KB) | use |

**geom 0_1 has 3 runs** — one has 0 AFR files (aborted); identify and skip it.
**geom 0_6** — one run has anomalous 1 KB mesh; use the valid pair.

---

### 1.3 Batch B — Directory Layout

```
CFWrinklePredict2/data/simulation_batch_271125/
├── mold set 000.Results/
│   ├── 2025-11-27 14_36_06/     ← FINE run (Part.section 1.1.msh ≈ 4984 KB)
│   │   ├── model.afi
│   │   ├── meshes/
│   │   │   ├── Part.section 1.1.msh   ← ply 0
│   │   │   ├── Part.section 1.2.msh   ← ply 1
│   │   │   ├── Part.section 1.3.msh   ← ply 2 (Twintex has 3 plies)
│   │   │   └── composite_plate_000_*.msh
│   │   └── results/                   ← AFRs here (same as Batch A — results/ subdir)
│   │       ├── model.afs
│   │       ├── model_40_1.afr
│   │       ├── ...                    ← 19 AFR files (no model_214_1.afr)
│   │       └── model_401_1.afr
│   └── 2025-11-27 15_02_11/     ← COARSE run (≈ 1252 KB)
├── mold set 001.Results/
│   └── ...
├── mold set 044.Results/        ← 45 mold sets total (000–044)
└── mold set NNN.afp             ← project files (binary, mixed in with .Results dirs)
```

**Key difference from Batch A:** `glob("mold set *")` also matches `.afp` files — use `glob("mold set *.Results")` to get only result directories.

---

### 1.4 Coarse vs Fine Identification

**Definitive method:** Compare `meshes/Part.section 1.1.msh` file size across runs in the same `.Results` directory.

| Batch | Fine mesh size | Coarse mesh size |
|---|---|---|
| A (UD) | ~9310 KB | ~382–456 KB |
| B (Twintex) | ~4984 KB | ~1252 KB |

Do NOT use the `.msh` files for node coordinates — they are deformed configurations at the final timestep. Use `results/model.afm` via `ReadAFMesh` for reference (t=0) coordinates.

Note: `model.afm` is in the run root directory alongside `model.afi`, NOT in `results/`.

**Node count method (more robust):** Read displacement field (model_40_1.afr) at any increment, count nodes per group:
```python
data = ResultsIncr[incr_nr][group_id]  # shape (N_nodes, 4)
n_nodes = data.shape[0]
# Batch A: fine ~75,356 vs coarse ~3,721
# Batch B: fine ~40,401 vs coarse ~1,400
```

---

### 1.6 Group Management — Mesh and AFR Files Contain Non-Ply Components

**CRITICAL:** Both `model.afm` (mesh) and `model_*.afr` (result fields) contain data for ALL element groups — tool surfaces, contact elements, and bending sub-elements, not just laminate plies.

When `ReadAFResult` or `ReadAFMesh` is called with `elemGrNrs=[]` (all groups), the returned data includes:
- **Tool groups** (1–5 in Batch A, 1–3 in Batch B): die, punch, frame meshes and DOFs
- **Ply shell groups** (main kinematic data — the groups you want):
  - Batch A: [6, 10]
  - Batch B: [4, 8, 12]
- **Bending sub-elements** (share nodes with ply shell, present in STensorPSST fields):
  - Batch A: [7, 11] (paired with ply groups 6, 10)
  - Batch B: [5, 9, 13] (paired with ply groups 4, 8, 12)
- **Contact groups** (penetration/slip/traction data — exclude from features):
  - Batch A: [8, 9, 13]
  - Batch B: [6, 7, 11, 15]

**Filtering rules for pipeline code:**
1. For scalar and vector ply fields (displacement, temperature, thickness, fiber strain/stress, etc.): filter to `ply_groups` — returns one result per ply
2. For STensorPSST fields (GL strain field 102, stress field 200): these appear in both main ply groups AND bending sub-element groups. Filter to `ply_groups` for the main shell result, or `bending_groups` if you need both membrane and bending contributions
3. For contact fields (302 penetration, 304 slip, 401 traction): these only exist in contact groups. Exclude from ply features entirely
4. For mesh (`ReadAFMesh`): always filter to `ply_groups` to get ply node coordinates and element connectivity. Tool meshes have different node counts and are not spatially registered with ply data

**Batch B trap:** Note that Batch B contact groups [6, 7, 11, 15] overlap with Batch A ply groups [6, 10]. Code that hardcodes group 6 as "ply 0" will silently read contact data when processing Batch B. Always use the batch-specific `ply_groups` from `pipeline_config.yaml`.

---

### 1.5 Batch Differences Summary

| Property | Batch A | Batch B |
|---|---|---|
| Root dir | `aniform_raw/` | `simulation_batch_271125/` |
| Folder pattern | `geom 0_N.Results/` | `mold set NNN.Results/` |
| Geometry | flat punch + flat die | varying mold geometry |
| Material | UD thermoplastic | Twintex 2×2 twill GF-PP |
| Plies | 2 | 3 |
| Ply element groups | 6, 10 | **4, 8, 12** |
| Bending sub-element groups | 6, 7, 10, 11 | 4, 5, 8, 9, 12, 13 |
| Contact/tool groups | 8, 9, 13 | 6, 7, 11, 15 |
| Fine mesh nodes/ply | **~75,356** | **~40,401** |
| Coarse mesh nodes/ply | **~3,721** | ~1,400 |
| Fine/coarse ratio | ~20:1 | ~29:1 |
| AFR file count | 20 | 19 (no model_214_1.afr) |
| Stroke direction | +Z (75 mm) | −Z (~10–12 mm depth) |
| AFR location | `<run>/results/` | `<run>/results/` (same) |
| Ply thickness (model_105) | 0.300 mm at t=0, 0.312 mm final | verify |
| Crystallisation (model_214) | yes (Nakamura, [0,1]) | **not present** |

---

## 2. Extraction Utilities

All five readers are in `io/aniform_readers/`. Import from project root:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "io"))
from aniform_readers.ReadAFResult import ReadAFResult
from aniform_readers.ReadAFSFile  import ReadAFSFile
from aniform_readers.ReadAFMesh   import ReadAFMesh
from aniform_readers.ReadMSHFile  import ReadMSHFile, find_ply_mesh_files
```

`ReadAFProject` is a stub with speculative binary parsing — do not use.
`ReadMSHFile` is a best-effort text/binary reader — use only for coarse/fine identification via file size.

---

### 2.1 ReadAFResult — Primary field reader

```python
ResultsIncr, Groups, Increments, res_type, indices_included, res_id = ReadAFResult(
    filename,          # str: absolute path to model_XX_Y.afr
    elemGrNrs=[],      # list[int]: [] = all groups; [6, 10] = ply groups only
    IncsToExport=[],   # list[int]: [] = all increments; [5, 10] = specific increments
    silent=True        # False prints file header and per-group info
)
```

**Return values:**

| Name | Type | Description |
|---|---|---|
| `ResultsIncr` | `dict[int, dict[int, np.ndarray]]` | `ResultsIncr[incr_nr][group_id]` → data array |
| `Groups` | `list[int]` | All group IDs present (may have duplicates across increments) |
| `Increments` | `list[int]` | All increment numbers present (may have duplicates) |
| `res_type` | `int` | Result type code (see table below) |
| `indices_included` | `bool` | If True, column 0 of data is the node index |
| `res_id` | `int` | Result ID from file header (matches X in `model_X_Y.afr`) |

**Data array shape:** `ResultsIncr[incr_nr][group_id]` → `np.ndarray` of shape `(N_nodes, n_floats + 1)`
- Column 0: node index (uint32, only meaningful if `indices_included=True`)
- Columns 1 to `n_floats`: field values (float32)

**Result type codes (res_type → n_floats, component names):**

| Code | Name | n_floats | Component order |
|---|---|---|---|
| 10 | ScalarT | 1 | s |
| 20 | VectorT | 3 | vx, vy, vz |
| 30 | TensorUNIT | 1 | txx |
| 31 | STensorPSST | 3 | txx, tyy, txy |
| 32 | TensorPSST | 4 | txx, tyy, txy, tyx |
| 33 | STensorPSNT | 4 | txx, tyy, tzz, txy |
| 34 | TensorPSNT | 5 | txx, tyy, tzz, txy, tyx |
| 35 | STensorT | 6 | txx, tyy, tzz, txy, tyz, txz |
| 36 | TensorT | 9 | txx, tyy, tzz, txy, tyz, txz, tyx, tzy, tzx |
| 40 | EulerT | 3 | rx, ry, rz |

**Common access pattern:**
```python
ResultsIncr, Groups, Increments, res_type, indices_included, res_id = ReadAFResult(
    str(afr_path), elemGrNrs=[6, 10], IncsToExport=[], silent=True
)
unique_incrs = sorted(set(Increments))
unique_groups = sorted(set(Groups))

# Get data at last increment, group 6
last_incr = unique_incrs[-1]
data = ResultsIncr[last_incr][6]   # shape: (N_nodes, n_floats+1)
node_ids = data[:, 0].astype(int)  # column 0 = node indices
values   = data[:, 1:]             # columns 1: = field values
```

**Element groups (Batch A):**

| Group | Content | Include? |
|---|---|---|
| 1–5 | Tool surfaces (die, punch sections, shuttle frame) | **No** |
| 6 | Ply 0 (0°) — Tri3LDT membrane | **Yes** |
| 7 | Ply 0 bending sub-element (shares nodes with 6) | STensorPSST fields only |
| 8, 9 | Contact elements for ply 0 | **No** (fields 302, 304, 401 live here) |
| 10 | Ply 1 (90°) — Tri3LDT membrane | **Yes** |
| 11 | Ply 1 bending sub-element | STensorPSST fields only |
| 13 | Contact elements for ply 1 | **No** |

**Element groups (Batch B):**

| Group | Content | Include? |
|---|---|---|
| 1–3 | Tool surfaces | **No** |
| 4 | Ply 0 — Tri3LDT membrane | **Yes** |
| 5 | Ply 0 bending sub-element | STensorPSST fields only |
| 6, 7 | Contact elements for ply 0 | **No** (fields 302, 304, 401 live here) |
| 8 | Ply 1 — Tri3LDT membrane | **Yes** |
| 9 | Ply 1 bending sub-element | STensorPSST fields only |
| 11 | Contact elements for ply 1 | **No** |
| 12 | Ply 2 — Tri3LDT membrane | **Yes** |
| 13 | Ply 2 bending sub-element | STensorPSST fields only |
| 15 | Contact elements for ply 2 | **No** |

---

### 2.2 ReadAFSFile — Increment timing and convergence

```python
import pandas as pd
Name, Description, IncrementInfo, Version, Core, SimulationInfo, Timing = ReadAFSFile(
    str(afs_path),   # str: path to model.afs
    silent=True
)
```

**Return values:**

| Name | Type | Description |
|---|---|---|
| `Name` | `str` | Simulation name (from .afi file) |
| `Description` | `str` | Simulation description |
| `Version` | `int` | AFS file format version (2 or 3) |
| `IncrementInfo` | `pd.DataFrame` | Per-increment metadata — **columns = increment numbers, index = field names** |
| `Core` | `pd.DataFrame` | Solver version and build info |
| `SimulationInfo` | `pd.DataFrame` | Start datetime, .afi MD5 checksum |
| `Timing` | `pd.DataFrame` | Wall-clock timing per increment |

**IMPORTANT — IncrementInfo is a DataFrame, not a list of dicts:**

```python
# CORRECT — access by row label:
times      = IncrementInfo.loc['t_end'].values.astype(float)
loadblocks = IncrementInfo.loc['current_loadblock'].values.astype(int)
converged  = IncrementInfo.loc['has_converged'].values.astype(bool)
incr_nrs   = IncrementInfo.loc['incr_nr'].values.astype(int)

# CORRECT — iterate over increments:
for col in IncrementInfo.columns:
    row = IncrementInfo[col]
    t      = float(row['t_end'])
    lb     = int(row['current_loadblock'])
    conv   = bool(row['has_converged'])

# WRONG — this iterates over column names (integers), not rows:
# for info in IncrementInfo: ...  ← DO NOT DO THIS
```

**IncrementInfo field names (DataFrame index):**

| Field | Type | Description |
|---|---|---|
| `incr_nr` | uint32 | Increment number |
| `t_end` | float64 | Simulation time at end of increment (seconds) |
| `nr_iterations` | uint32 | Solver iterations used |
| `norm_unbalance` | float64 | Convergence norm (unbalanced forces) |
| `norm_displacement` | float64 | Convergence norm (displacement) |
| `has_converged` | bool | False → non-converged; skip these increments |
| `current_loadblock` | uint32 | Load step number (1 = heat, 2 = form, 3 = cool for Batch A) |
| `nr_of_loadblocks` | uint32 | Total number of load steps |
| `progress_current_loadblock` | float64 | Fraction of current load step complete [0–1] |
| `next_increment_size` | float64 | Δt for next increment (adaptive) |
| `nr_incr_to_go` | uint32 | Remaining increments in current load step |
| `restartfile_written` | bool | Restart file dumped at this increment |
| `resultsfile_written` | bool | AFR result files written at this increment |

---

### 2.3 ReadAFMesh — Binary mesh file (reference configuration)

```python
Nodes, Elements, elemGrNrsAll = ReadAFMesh(
    str(afm_path),              # str: path to model.afm (in run root, NOT results/)
    elemGrNrsToExport=[6, 10],  # list[int]: [] = all groups
    silent=True
)
```

**Return values:**

| Name | Type | Description |
|---|---|---|
| `Nodes` | `dict[int, np.ndarray]` | `Nodes[group_id]` → `(N_nodes, 4)` — col0=external node ID, cols 1-3=XYZ (float32) |
| `Elements` | `dict[int, np.ndarray]` | `Elements[group_id]` → `(N_elems, nodes_per_elem+1)` — col0=external elem ID, cols 1:=node indices |
| `elemGrNrsAll` | `list[int]` | All element group IDs in the file |

**Access pattern:**
```python
Nodes, Elements, _ = ReadAFMesh(str(afm_path), elemGrNrsToExport=[6, 10])

node_ids_g6 = Nodes[6][:, 0].astype(int)    # external node numbers
xyz_g6      = Nodes[6][:, 1:4]              # reference coordinates (t=0), mm
elem_ids_g6 = Elements[6][:, 0].astype(int) # external element numbers
connectivity = Elements[6][:, 1:]           # node index columns (NOT external node IDs)
```

**Element format codes (elem_format field):**

| Code | Type | Nodes |
|---|---|---|
| 1 | Triangle 3D | 3 |
| 2 | Tet | 4 |
| 6 | Quad 3D | 4 |
| Others | see source | varies |

Ply elements are always format 1 (Tri3 triangles with 3 nodes each).

**CRITICAL:** `model.afm` is in the run's root directory (alongside `model.afi`), NOT in `results/`. Path: `<run_dir>/model.afm` — not `<run_dir>/results/model.afm`.

---

### 2.4 ReadMSHFile — Per-ply deformed mesh (limited use)

`ReadMSHFile` is a best-effort heuristic reader, not an authoritative binary parser for AniForm's `.msh` format. Use only for coarse/fine identification via file size.

```python
# Only reliable use: file size for coarse/fine identification
import os
msh_path = run_dir / "meshes" / "Part.section 1.1.msh"
size_kb = os.path.getsize(msh_path) / 1024
is_fine = size_kb > 2000   # Batch A: fine ≈ 9310 KB, Batch B: fine ≈ 4984 KB
```

`find_ply_mesh_files(run_dir)` → `dict[int, Path]` mapping ply index (0-based) to `.msh` path. Useful for enumerating available ply meshes.

---

### 2.5 ReadAFProject — Do not use

`ReadAFProject` is a speculative stub based on guessed binary structure. The `.afp` files are not needed for data extraction (all training data comes from `.afr`, `.afs`, `.afm`, `.afi`, and `model.track.txt`).

---

## 3. AFR Field Mapping (CONFIRMED via ReadAFResult silent=False headers)

All field names below were confirmed by reading the binary header "Name:" string from each AFR file.

### 3.1 Ply fields (features and targets)

| AFR file | Header Name | res_type | n_floats | Components | Units | Groups A | Groups B | Feature? |
|---|---|---|---|---|---|---|---|---|
| model_40_1.afr | Displacement | VectorT (20) | 3 | dx, dy, dz | mm | 6, 10 | 4, 8, 12 | yes |
| model_42_1.afr | Rotation | VectorT (20) | 3 | rx, ry, rz | rad | 6, 10 | 4, 8, 12 | **no** (tool DOF) |
| model_44_1.afr | Temperature | VectorT (20) | 3 | T1, T2, T3 | °C | 6, 10 | 4, 8, 12 | yes |
| model_48_1.afr | Temperature 1 | ScalarT (10) | 1 | T_top | °C | 6, 10 | 4, 8, 12 | pending |
| model_49_1.afr | Temperature 2 | ScalarT (10) | 1 | T_bot | °C | 6, 10 | 4, 8, 12 | pending |
| model_102_1.afr | Green-Lagrange strain | STensorPSST (31) | 3 | E11, E22, E12 | — | 6,7,10,11 | 4,5,8,9,12,13 | yes |
| model_105_1.afr | Thickness | ScalarT (10) | 1 | t | mm | 6, 10 | 4, 8, 12 | yes |
| model_106_1.afr | Eq shear rate | ScalarT (10) | 1 | gamma_dot | 1/s | 6, 10 | 4, 8, 12 | yes |
| model_200_1.afr | Stress | STensorPSST (31) | 3 | s11, s22, s12 | MPa | 6,7,10,11 | 4,5,8,9,12,13 | yes |
| model_203_1.afr | Fiber direction 1 | VectorT (20) | 3 | fd1_x, fd1_y, fd1_z | — | 6, 10 | 4, 8, 12 | yes |
| model_203_2.afr | Fiber direction 2 | VectorT (20) | 3 | fd2_x, fd2_y, fd2_z | — | 6, 10 | 4, 8, 12 | yes |
| model_204_1.afr | Shear angle f1_f2 | ScalarT (10) | 1 | gamma_f1f2 | deg | 6, 10 | 4, 8, 12 | **key** (woven) |
| model_205_1.afr | Fiber strain 1 | ScalarT (10) | 1 | eps_f1 | — | 6, 10 | 4, 8, 12 | yes |
| model_205_2.afr | Fiber strain 2 | ScalarT (10) | 1 | eps_f2 | — | 6, 10 | 4, 8, 12 | yes |
| model_206_1.afr | **Fiber stress 1** | ScalarT (10) | 1 | sigma_f1 | MPa | 6, 10 | 4, 8, 12 | **PRIMARY target** |
| model_206_2.afr | Fiber stress 2 | ScalarT (10) | 1 | sigma_f2 | MPa | 6, 10 | 4, 8, 12 | yes |
| model_214_1.afr | Rel. crystallinity (Nakamura) | ScalarT (10) | 1 | alpha | — | 6, 10 | **A only** | pending |

### 3.2 Contact/tool fields (EXCLUDED from features)

| AFR file | Header Name | res_type | Groups A | Groups B | Notes |
|---|---|---|---|---|---|
| model_302_1.afr | Penetration depth | ScalarT (10) | 8, 9, 13 | 6, 7, 11, 15 | Contact interface |
| model_304_1.afr | Slip path length | ScalarT (10) | 8, 9, 13 | 6, 7, 11, 15 | 0 → 3.07mm at contact |
| model_401_1.afr | Traction | VectorT (20) | 8, 9, 13 | 6, 7, 11, 15 | All zeros — exclude |

### 3.3 Confirmed value ranges (Batch A fine mesh, geom_0_0)

| Field | Value at t=0 | Value at t_final | Notes |
|---|---|---|---|
| Thickness (105) | 0.300 mm uniform | mean 0.312 mm | Nominal 0.3mm from .afi confirmed |
| GL strain (102) | ~0 | E11≈0, E12≈0.00075 | Near-zero elastic (dominant viscous) |
| Stress (200) | ~0 | s11=-0.203, s22=-0.176, s12=-0.001 MPa | Very small viscous stress |
| Fiber dir 1 (203_1) | [1.000, 0.000, 0.000] | [0.948, 0.001, 0.004] | Unit vector evolves with deformation |
| Fiber stress 1 (206_1) | 0% compressive | **80% compressive** | Primary wrinkle precursor |
| Slip path (304) | 0 | max 3.07 mm | Contact nodes only |
| Crystallinity (214) | ~0 | rises on cooling | Batch A only, [0,1] range |

### 3.4 Sub-result interpretation

Sub_id corresponds to **fiber family**, not integration point or coordinate frame:
- `203_1` / `203_2` = Fiber direction vectors for family 1 and family 2
- `205_1` / `205_2` = Fiber strains for family 1 and family 2
- `206_1` / `206_2` = Fiber stresses for family 1 and family 2

### 3.5 Stress coordinate frame (model_200_1.afr) — PARTIALLY RESOLVED

Field 200 is confirmed "Stress" [s11, s22, s12]. The s11/s22 ratio at the final forming increment for group 6 (0° ply) is approximately 1.15:1 — suggesting global or near-isotropic viscous stress. The small magnitudes (~0.2 MPa) are consistent with viscous forming stresses in a thermoplastic composite above T_g.

**Note:** `model_302_1.afr` is NOT the stress tensor — it is "Penetration depth", a contact field. All stress resolution should target `model_200_1.afr`.

---

## 4. ASCII File Formats

### 4.1 Input File (.afi)

Text format (PAM-STAMP lineage). Key sections for extraction:

```
*title "flat_smooth_0"       ← geometry name

*elementgroup N              ← one block per element group
  type Tri3LDT
  Part.section 1.1           ← identifies ply groups (vs tool groups)
  localcs 1                  ← local coordinate system (1=0°, 2=90°)
  thickness 0.3              ← nominal ply thickness in .afi (may differ from .afl)
  ...

*loadset 1                   ← forming load step
  UZ 70 tool 3               ← punch Z-displacement (mm)
  T 180 tool 1               ← die temperature (°C)
  T 290 tool 3               ← punch temperature (°C)
  T0 320 group 6 10          ← blank initial temperature (°C)

*solve                       ← one block per solve step
  increments 10
  timeincrement 2.0

*post                        ← result output settings
  interval 50                ← write results every 50 increments
  doftrigger UZ > 0.55       ← also write on trigger condition
```

**Batch A specifics:**
- 2 loadsets (heat: UZ=70mm, form: UZ=75mm) + 1 cool step
- Tool geometry: `flat_smooth_0_die` + `flat_smooth_0_punch`

**Batch B specifics:**
- 1 loadset (forming only, ~10–12 mm depth, Z negative direction)
- Tool geometry: `composite_plate_NNN_die` + `composite_plate_NNN_punch`

### 4.2 Force-Stroke File (model.track.txt)

Location: `<run>/results/model.track.txt` (check also `<run>/model.track.txt` as fallback).

Format: ASCII, one row per tracked tool per increment, 8 columns:
```
UX   RFX   UY   RFY   UZ   RFZ   T   RQ
```
- `UZ` = punch displacement (mm); direction depends on batch (Batch A: +Z, Batch B: −Z)
- `RFZ` = punch reaction force (N) = **forming force**
- Multiple tools may be tracked → multiple rows per increment (deinterleave by striding)

**Extraction code (single-tool case):**
```python
import numpy as np
text = track_path.read_text(encoding="utf-8", errors="replace")
rows = []
header_found = False
for line in text.splitlines():
    s = line.strip()
    if not s or s.startswith('%'):
        continue
    if s.lower().startswith('increment'):
        header_found = True
        continue
    if not header_found:
        continue
    tokens = s.split()
    if len(tokens) >= 8:
        try:
            rows.append([float(t) for t in tokens[:8]])
        except ValueError:
            pass

arr = np.array(rows)
uz  = arr[:, 4]   # punch displacement
rfz = arr[:, 5]   # forming force
```

For multi-tool files, the punch is the tool with the largest UZ range. Deinterleave by trying n_tools ∈ {1,2,3,4,5} and picking the stride with the highest UZ range.

### 4.3 Material Library (.afl)

JSON-like text format. For Batch A (UD thermoplastic):

| Property | Value | Units |
|---|---|---|
| E_fiber (fiber Young modulus) | 25,000 | MPa |
| E1 bending (along fiber) | 450 | MPa |
| E2 bending (transverse) | 4.5 | MPa |
| Nu12 | 0.01 | — |
| G12 | 0.45 | MPa |
| η₀ in-plane viscosity | 70 | Pa·s |
| η₀ bending viscosity | 37,500 | Pa·s |
| Ply thickness | **0.15** | mm |
| Crystallisation | Nakamura (exponent=3, Tinf=320°C) | — |

**Thickness discrepancy:** `.afi` says `thickness 0.3`, `.afl` says `Thickness: 0.15`. The `.afl` value (0.15 mm) is the layer thickness; the `.afi` value (0.3 mm) is likely the element thickness including both plies. Use `model_102_1.afr` at t=0 as the ground-truth per-element thickness.

---

## 5. Data Access Patterns for Pipeline (WP2+)

### 5.1 Enumerate all simulation pairs

```python
from pathlib import Path

BATCH_A_ROOT = Path(r"...\data\aniform_raw")
BATCH_B_ROOT = Path(r"...\data\simulation_batch_271125")

def find_coarse_fine_pair(results_dir: Path):
    """Return (coarse_dir, fine_dir) by ply mesh size. Both None if not found."""
    runs = [d for d in results_dir.iterdir()
            if d.is_dir() and (d / "meshes").exists()]
    def size(r):
        p = r / "meshes" / "Part.section 1.1.msh"
        return p.stat().st_size if p.exists() else 0
    valid = sorted([(r, size(r)) for r in runs if size(r) > 0], key=lambda x: x[1])
    if len(valid) < 2:
        return None, None
    return valid[0][0], valid[-1][0]

# Batch A
for d in sorted(BATCH_A_ROOT.glob("geom 0_*.Results")):
    coarse, fine = find_coarse_fine_pair(d)
    # geom 0_0 has 4 runs — this returns only 1 pair (smallest and largest)
    # handle geom 0_0 separately to get both pairs

# Batch B
for d in sorted(BATCH_B_ROOT.glob("mold set *.Results")):  # .Results suffix avoids .afp files
    coarse, fine = find_coarse_fine_pair(d)
```

### 5.2 Read all ply fields at final increment

```python
from aniform_readers.ReadAFResult import ReadAFResult

PLY_GROUPS = {"A": [6, 10], "B": [4, 8, 12]}

def load_field_at_final_increment(afr_path, ply_groups):
    """Returns dict: group_id → (N_nodes, n_floats) float32 array."""
    ResultsIncr, Groups, Increments, res_type, indices_included, res_id = ReadAFResult(
        str(afr_path), elemGrNrs=ply_groups, IncsToExport=[], silent=True
    )
    last_incr = max(set(Increments))
    return {
        grp: ResultsIncr[last_incr][grp][:, 1:]   # drop node-index column
        for grp in set(Groups) if grp in ResultsIncr[last_incr]
    }
```

### 5.3 Read reference mesh coordinates

```python
from aniform_readers.ReadAFMesh import ReadAFMesh

def load_reference_mesh(run_dir, ply_groups):
    """Returns dict: group_id → (N_nodes, 3) XYZ float32."""
    afm_path = run_dir / "model.afm"   # NOT in results/ subdir
    Nodes, Elements, _ = ReadAFMesh(str(afm_path), elemGrNrsToExport=ply_groups)
    return {grp: Nodes[grp][:, 1:4] for grp in Nodes}  # drop external node ID column
```

### 5.4 Read increment time series from AFS

```python
from aniform_readers.ReadAFSFile import ReadAFSFile
import numpy as np

def load_increment_times(run_dir):
    """Returns (incr_nrs, times, loadblocks, converged) arrays."""
    afs_path = run_dir / "results" / "model.afs"
    Name, _, IncrementInfo, Version, _, _, _ = ReadAFSFile(str(afs_path))
    return (
        IncrementInfo.loc['incr_nr'].values.astype(int),
        IncrementInfo.loc['t_end'].values.astype(float),
        IncrementInfo.loc['current_loadblock'].values.astype(int),
        IncrementInfo.loc['has_converged'].values.astype(bool),
    )
```

### 5.5 Spatial correspondence — coarse to fine (no shared nodes)

Coarse and fine meshes share no node indices. Correspondence must be built spatially:

```python
from scipy.spatial import cKDTree

def build_coarse_to_fine_map(coarse_xyz, fine_xyz, k=1):
    """
    For each coarse node, find the nearest fine node.
    Returns indices into fine_xyz, shape (N_coarse,).
    """
    tree = cKDTree(fine_xyz)
    dist, idx = tree.query(coarse_xyz, k=k)
    return idx  # shape (N_coarse,) if k=1

# Use with reference coordinates (t=0) from ReadAFMesh, group 6:
coarse_xyz = load_reference_mesh(coarse_run_dir, [6])[6]
fine_xyz   = load_reference_mesh(fine_run_dir,   [6])[6]
coarse_to_fine = build_coarse_to_fine_map(coarse_xyz, fine_xyz)
```

---

## 6. Resolved Questions (WP1 Archaeology)

All previously open questions have been resolved via ReadAFResult(silent=False) header inspection:

| # | Question | Resolution | Date |
|---|---|---|---|
| 1 | Stress field identity | model_302 = Penetration depth (contact). model_200 = Stress (Cauchy) | 2026-03-28 |
| 2 | model_214_1.afr meaning | "Rel. crystallinity (Nakamura)" — Batch A only | 2026-03-28 |
| 3 | Thickness field | model_105 = Thickness (0.300mm at t=0). model_102 = GL strain (NOT thickness) | 2026-03-28 |
| 4 | Sub-results 203_2, 205_2, 206_2 | Sub_id = fiber family number (1 or 2), not frame/measure variant | 2026-03-28 |
| 5 | model_44_1.afr identity | "Temperature" (NOT acceleration) | 2026-03-28 |
| 6 | model_48/49 identity | "Temperature 1" / "Temperature 2" (surface temps, NOT contact pressure/status) | 2026-03-28 |
| 7 | model_401 identity | "Traction" (all zeros, NOT Nakamura crystallinity) | 2026-03-28 |

**Remaining to run:** Wrinkle classification for all 66 pairs (wrinkle_detector.py).

---

## 7. Pipeline Constraints (WP2+ Design)

| Constraint | Source | Implication |
|---|---|---|
| Only coarse mesh at inference | Project design | No fine topology features; no refinement ratio |
| No shared nodes between coarse and fine | Data structure | All correspondence via spatial KDTree, not index lookup |
| 66–67 pairs total (21 Batch A + 45 Batch B) | Data archaeology | ~5-fold CV → 13–14 val pairs per fold; stratify by batch |
| Batch A: 2 plies, Batch B: 3 plies | .afi files | Feature extraction must handle variable ply count or process per-ply |
| Ply groups per batch differ | Groups A=[6,10] B=[4,8,12] | Pipeline must receive `ply_groups` as a parameter |
| AFR files in `<run>/results/` for both batches | Confirmed by ls | Use `run_dir / "results" / "model_XX_Y.afr"` always |
| `model.afm` in run root, not results/ | Confirmed by ls | Use `run_dir / "model.afm"` not `run_dir / "results" / "model.afm"` |
| Batch B has no model_214_1.afr | File inventory | Field 214 cannot be a shared feature (exclude or mask) |
| Coarse/fine ratio differs by batch | Mesh sizes | Batch A ratio ≈ 20:1 nodes; Batch B ratio ≈ 29:1 nodes |
| model_102 = GL strain, model_105 = thickness | Header confirmation | Previous code comments about "col3 = Δt" referred to model_102 which is strain, not thickness |
| IncrementInfo from ReadAFSFile is a pandas DataFrame | Source code | Access via `.loc['field_name'].values`, not dict-style iteration |
| Batch B Twintex → shear locking expected | Material physics | model_204_1.afr (shear angle f1_f2) is a primary wrinkle precursor for woven; less relevant for UD |
| Nakamura crystallinity (214) Batch A only | Header confirmation | Not present in Batch B; exclude from cross-batch features or mask |
| Contact fields (302, 304, 401) in different groups per batch | Header confirmation | A: groups [8,9,13]; B: groups [6,7,11,15]. Exclude from ply features |
| Mesh files contain tool + ply groups | Data structure | ReadAFResult/ReadAFMesh return ALL groups if elemGrNrs=[]. Always filter to ply_groups to exclude tool surfaces |
