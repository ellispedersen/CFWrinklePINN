# WP2 — Data Structures & Schema
## HDF5 Layout, Field Registry Contract, Project Structure

**Depends on:** WP1 gate passed, field_registry.yaml locked
**Feeds into:** WP3 (writers), WP4 (readers), WP5 (readers), WP6/7 (dataloader)

---

## Objective

Define and stub every data structure before any data is written. WP3 writes into this schema; WP4 reads from it. Changes to the schema after WP3 begins require a migration — avoid by getting this right first.

---

## Inputs Received from WP1

| Artifact | Used For |
|---|---|
| `config/field_registry.yaml` | Defines N_fields and field index → name mapping for HDF5 tensor shape |
| `reports/input_param_registry.json` | Informs pair/metadata group contents |
| `reports/wrinkle_onset_registry.json` | Informs events/ and targets/ group contents |

---

## Outputs Produced by This WP

| Artifact | Location | Consumer |
|---|---|---|
| Empty `dataset.h5` with full group structure | `data/dataset.h5` | WP3 fills it |
| `hdf5_writer.py` | `io/hdf5_writer.py` | WP3 |
| `hdf5_reader.py` | `io/hdf5_reader.py` | WP4, WP6, WP7 |
| `pipeline_config.yaml` | `config/pipeline_config.yaml` | All WPs |
| Project skeleton (empty module files) | `project/` | All WPs |
| WP2 validation checklist | `reports/WP2_gate.md` | Gate review |

---

## 2.1 Project File Layout

Create all files and directories now — even if empty. This is the canonical structure; do not add modules outside it without updating INDEX.md.

```
project/
├── orchestrator.py              # top-level runner — narrative only, no logic
├── config/
│   ├── pipeline_config.yaml     # all paths, parameters, versions — no hardcoded values in modules
│   └── field_registry.yaml      # locked after WP1; version-bumped if changed
├── io/
│   ├── aniform_input_reader.py  # WP1 deliverable — reads input files
│   ├── aniform_output_reader.py # WP3 — wraps Aniform Python tool
│   ├── hdf5_writer.py           # WP2 deliverable — all write operations
│   └── hdf5_reader.py           # WP2 deliverable — all read operations for training
├── mesh/
│   ├── correspondence.py        # WP3 — coarse-to-fine spatial mapping
│   └── interpolation.py         # WP3 — barycentric, spatial queries
├── features/
│   ├── material_features.py     # WP4 — CLT, material card extraction
│   ├── kinematic_features.py    # WP4 — F, E, strain rates from nodal coords
│   ├── stress_features.py       # WP4 — principal stresses, invariants, wrinkling proximity
│   └── temporal_features.py     # WP4 — resampling, dt-normalisation, event detection
├── targets/
│   └── wrinkle_targets.py       # WP4 — fine mesh aggregation, wrinkle severity
├── training/
│   └── stratification.py        # WP5 — fold construction, composition report
├── model/                       # WP6 — do not create until WP5 gate passes
│   ├── gnn.py
│   ├── temporal.py
│   └── loss.py
├── validation/
│   ├── field_survey.py          # WP1 deliverable
│   ├── wrinkle_detector.py      # WP1 deliverable
│   └── sanity_checks.py         # WP3 — runs after each pair ingestion
└── tests/
    ├── test_correspondence.py
    ├── test_features.py
    ├── test_targets.py
    └── test_hdf5_schema.py      # validates schema against this document
```

```bash
# Create full project skeleton
mkdir -p project/{config,io,mesh,features,targets,training,model,validation,tests,reports,data}
touch project/orchestrator.py
touch project/io/{aniform_input_reader.py,aniform_output_reader.py,hdf5_writer.py,hdf5_reader.py}
touch project/mesh/{correspondence.py,interpolation.py}
touch project/features/{material_features.py,kinematic_features.py,stress_features.py,temporal_features.py}
touch project/targets/wrinkle_targets.py
touch project/training/stratification.py
touch project/validation/{field_survey.py,wrinkle_detector.py,sanity_checks.py}
touch project/tests/{test_correspondence.py,test_features.py,test_targets.py,test_hdf5_schema.py}
```

---

## 2.2 pipeline_config.yaml

```yaml
# config/pipeline_config.yaml
version: "1.0"

paths:
  data_root: "/path/to/aniform/simulations"   # root of raw sim files
  dataset_h5: "data/dataset.h5"
  reports_dir: "reports/"
  field_registry: "config/field_registry.yaml"
  input_registry: "reports/input_param_registry.json"
  wrinkle_registry: "reports/wrinkle_onset_registry.json"
  fold_assignments: "config/fold_assignments.yaml"

ingestion:
  n_resample_points: 256          # uniform stroke-fraction points after resampling
  correspondence_radius_factor: 1.5   # coarse element radius multiplier for fine mesh query
  min_fine_per_coarse: 1          # flag pairs where any coarse element maps to <N fine

features:
  feature_version: "v1.0"         # bump when feature engineering changes
  use_fields: []                  # populated from field_registry after WP1; field names only
  compute_kinematic: true         # always true — computable from nodal coords regardless
  compute_stress: true            # requires stress fields confirmed in field_registry
  compute_material: true

targets:
  wrinkle_severity_metric: "oop_displacement_variance"   # or thickness_variance
  min_wrinkle_severity_threshold: 0.05    # below this → clean label

training:
  n_folds: 5
  random_seed: 42
```

---

## 2.3 HDF5 Schema

This is the authoritative schema. `test_hdf5_schema.py` validates the actual file against it.

```
data/dataset.h5
│
├── metadata/
│   ├── simulation_index         # (80,) structured array: sim_id, pair_id, material, geometry_type
│   ├── material_cards/
│   │   ├── UD/                  # datasets: E1, E2, G12, nu12, rho — scalars from material card
│   │   └── twill_2x2/           # same fields
│   └── field_registry           # string dump of field_registry.yaml content at parse time
│
└── pairs/
    └── pair_000/                # one group per pair, named by pair_id
        │
        ├── metadata             # HDF5 attributes (not datasets):
        │                        #   geometry_type, material, blank_dim_x_mm, blank_dim_y_mm,
        │                        #   n_plies, ply_thickness_actual_mm, blank_holder_force_N,
        │                        #   forming_speed_mm_s, friction_coefficient,
        │                        #   wrinkle_outcome (0|1), feature_version
        │
        ├── mesh/
        │   ├── coarse/
        │   │   ├── nodes              # (N_c_nodes, 3) float32 — reference coords at t=0
        │   │   ├── elements           # (N_c_elem, n_per_elem) int32 — connectivity
        │   │   ├── edge_index         # (2, N_edges) int32 — PyG format, built from elements
        │   │   ├── edge_attr          # (N_edges, 4) float32 — [dx, dy, dz, dist] in ref config
        │   │   └── coarse_to_fine_map # variable-length: N_c_elem groups, each listing fine elem indices
        │   └── fine/
        │       ├── nodes              # (N_f_nodes, 3) float32
        │       └── elements           # (N_f_elem, n_per_elem) int32
        │
        ├── coarse/
        │   ├── raw_timesteps/         # IMMUTABLE after write
        │   │   ├── times              # (N_raw,) float32 — actual solver time values in seconds
        │   │   ├── fields             # (N_raw, N_c_nodes, N_fields) float32
        │   │   │                      # field index matches field_registry.yaml exactly
        │   │   └── rates              # (N_raw, N_c_nodes, N_fields) float32 — dt-normalised
        │   │                          # finite differences; NaN at t=0 (no prior timestep)
        │   └── resampled/
        │       ├── stroke_fractions   # (256,) float32 — uniform 0→1
        │       ├── fields             # (256, N_c_nodes, N_fields) float32
        │       └── rates              # (256, N_c_nodes, N_fields) float32
        │
        ├── events/                    # scalar event times — all as stroke fractions (0→1)
        │   ├── contact_initiation     # scalar float32
        │   ├── max_forming_force      # scalar float32
        │   └── wrinkle_onset          # scalar float32 — from fine mesh; NaN if wrinkle_outcome=0
        │
        ├── features/                  # computed by WP4; written separately from raw fields
        │   └── coarse/
        │       └── resampled/
        │           └── physics/       # (256, N_c_nodes, N_physics_features) float32
        │                              # feature names stored as dataset attribute 'feature_names'
        │
        └── fine/
            └── targets/               # precomputed by WP4; dataloader reads these directly
                ├── thickness_variance     # (N_c_elem,) float32 — variance over fine cluster
                ├── max_oop_displacement   # (N_c_elem,) float32 — max |z| over fine cluster at final step
                └── wrinkle_severity       # (N_c_elem,) float32 — composite severity per coarse element
```

### Provenance Attributes

Every dataset must carry these HDF5 attributes at write time:

```python
# io/hdf5_writer.py — apply to every ds.create_dataset call
ds.attrs['status']           = 'confirmed'      # confirmed | inferred | unknown
ds.attrs['physical_meaning'] = 'string'         # human-readable
ds.attrs['feature_version']  = 'v1.0'           # from pipeline_config
ds.attrs['source']           = 'aniform_direct' # aniform_direct | derived | back_calculated
ds.attrs['written_at']       = timestamp_iso    # when this dataset was written
```

### coarse_to_fine_map Storage

Because each coarse element maps to a variable number of fine elements, store as a group of datasets:

```python
# In pairs/pair_000/mesh/coarse/coarse_to_fine_map/
#   elem_0000  →  int32 array of fine element indices
#   elem_0001  →  int32 array of fine element indices
#   ...
# Also store:
#   refinement_ratios  →  (N_c_elem,) float32 — len(fine_indices)/1 per coarse elem
#   boundary_flags     →  (N_c_elem,) bool — True if coarse elem is at blank boundary
```

---

## 2.4 hdf5_writer.py Interface

```python
# io/hdf5_writer.py

def create_empty_dataset(path: str, config: dict) -> None:
    """Create dataset.h5 with full group structure but no pair data."""

def write_pair_metadata(h5: h5py.File, pair_id: str, meta: dict) -> None:
    """Write pair/metadata attributes from input_param_registry entry."""

def write_mesh(h5: h5py.File, pair_id: str,
               coarse_mesh: dict, fine_mesh: dict,
               mapping: dict) -> None:
    """Write mesh/coarse and mesh/fine groups including edge_index, edge_attr, mapping."""

def write_raw_timesteps(h5: h5py.File, pair_id: str,
                        times: np.ndarray,
                        fields: np.ndarray,
                        rates: np.ndarray) -> None:
    """Write immutable raw solver output. Refuses to overwrite existing data."""

def write_resampled(h5: h5py.File, pair_id: str,
                    stroke_fractions: np.ndarray,
                    fields: np.ndarray,
                    rates: np.ndarray) -> None:
    """Write 256-point resampled version."""

def write_events(h5: h5py.File, pair_id: str, events: dict) -> None:
    """Write scalar event times as stroke fractions."""

def write_features(h5: h5py.File, pair_id: str,
                   physics_features: np.ndarray,
                   feature_names: list[str],
                   version: str) -> None:
    """Write computed physics feature tensor. Includes feature_names as attribute."""

def write_targets(h5: h5py.File, pair_id: str, targets: dict) -> None:
    """Write precomputed targets. Refuses to overwrite without explicit overwrite=True."""
```

---

## 2.5 hdf5_reader.py Interface

```python
# io/hdf5_reader.py

def load_pair_for_training(h5_path: str, pair_id: str,
                           feature_version: str) -> dict:
    """
    Returns everything the dataloader needs for one pair.
    Reads only from resampled/ and features/ and targets/ — never from raw_timesteps.
    Raises if feature_version in file does not match requested version.
    """
    return {
        'pair_id': str,
        'material': str,
        'geometry_type': str,
        'wrinkle_outcome': int,           # 0 | 1
        'edge_index': torch.Tensor,       # (2, N_edges) long
        'edge_attr': torch.Tensor,        # (N_edges, 4) float
        'node_features': torch.Tensor,    # (256, N_nodes, N_feat) float
        'material_card': torch.Tensor,    # (N_mat_params,) float
        'events': dict,                   # scalar stroke fractions
        'targets': dict,                  # precomputed per-element targets
    }

def load_field_registry(h5_path: str) -> list[dict]:
    """Read field_registry snapshot from metadata/field_registry."""

def list_pairs(h5_path: str) -> list[str]:
    """Return sorted list of pair_ids present in the file."""

def check_schema_version(h5_path: str, expected_version: str) -> bool:
    """Return True if schema version attribute matches expected."""
```

---

## WP2 Validation Gate

```
Schema
  [ ] dataset.h5 created with full group structure (no pair data yet)
  [ ] test_hdf5_schema.py written and passes against empty dataset.h5
  [ ] All interface functions in hdf5_writer.py and hdf5_reader.py stubbed with docstrings
  [ ] pipeline_config.yaml complete with all paths and parameters

Project Structure
  [ ] All module files created (may be empty stubs)
  [ ] No hardcoded paths in any module file
  [ ] config/ directory contains field_registry.yaml (locked from WP1) and pipeline_config.yaml

Handoff Check
  [ ] WP3 lead has confirmed hdf5_writer.py interface matches their extraction output shapes
  [ ] WP4 lead has confirmed hdf5_reader.py interface matches their feature computation inputs
```

---

## Handoff to WP3

WP3 receives from WP2:

| Artifact | Path | Notes |
|---|---|---|
| Empty `dataset.h5` | `data/dataset.h5` | WP3 fills it via hdf5_writer.py |
| `hdf5_writer.py` (stubbed) | `io/hdf5_writer.py` | WP3 implements and uses |
| `pipeline_config.yaml` | `config/pipeline_config.yaml` | WP3 reads all paths from here |
| `field_registry.yaml` | `config/field_registry.yaml` | WP3 uses field indices for array slicing |
| Project skeleton | `project/` | WP3 implements io/, mesh/, validation/ |
