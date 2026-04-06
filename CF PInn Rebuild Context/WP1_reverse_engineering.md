# WP1 — Reverse Engineering
## Aniform Dataset Archaeology

**Status:** FIRST. Nothing else starts until this is done.
**Feeds into:** WP2 (field_registry.yaml), WP3 (aniform_input_reader.py), WP4 (feature decisions)

---

## Objective

Recover the complete physical meaning of every input parameter and every output field across all 80 simulations before a single line of pipeline code is written. The field_registry.yaml produced here is the contract that all downstream work packages depend on.

---

## Outputs Produced by This WP

| Artifact | Location | Consumer |
|---|---|---|
| `field_registry.yaml` | `config/field_registry.yaml` | WP2, WP3, WP4 |
| `field_survey_report.json` | `reports/field_survey_<pair_id>.json` | WP1 internal review |
| `input_param_registry.json` | `reports/input_param_registry.json` | WP3 (input reader), WP5 (stratification labels) |
| Thickness values per sim | manual → written to WP2 schema | WP4 (CLT features) |
| Force-stroke curves | manual → written to WP2 schema | WP4 (event features) |
| WP1 validation checklist | `reports/WP1_gate.md` | Gate review before WP2 |

---

## Step 1.1 — Input File Archaeology

**Goal:** Recover all simulation setup parameters from Aniform input files for all 80 simulations.

### Target Quantities

| Quantity | Expected Location | Verification Method |
|---|---|---|
| Forming feature geometry (depth, radius, type) | Geometry/tool section | Visual: 50mm dome should look 50mm |
| Ply layup (n_layers, stacking sequence, orientations) | Material/layup section | Cross-ref with material card |
| Layer thickness | Material section | May differ from nominal — see §1.3 |
| Process params (blank holder force, forming speed, friction) | Process/BHF section | Order-of-magnitude check |
| Blank geometry (dimensions, initial flat config) | Blank section | Compare to mesh node extents |

### Approach

1. Open one input file in a text editor — identify keyword structure (PAM-STAMP lineage uses `$` or `*` prefixed section headers)
2. Map every section header to a physical quantity; list any unrecognised sections
3. Build `io/aniform_input_reader.py` that extracts all recoverable parameters into a dict per simulation
4. Run against all 80 input files; flag any that fail to parse or return null for mandatory fields
5. **Visual verification is mandatory** — render geometry parameters against the actual mesh for at least one sim per geometry type

### Deliverable

```python
# io/aniform_input_reader.py
# Returns per simulation:
{
    'sim_id': str,
    'geometry_type': str,           # punch | dome | rib | mixed
    'feature_depth_mm': float,
    'feature_radius_mm': float,
    'n_plies': int,
    'stacking_sequence': list[float],   # ply angles in degrees
    'ply_thickness_mm': float,          # as specified in input, may be modified
    'blank_holder_force_N': float,
    'forming_speed_mm_s': float,
    'friction_coefficient': float,
    'blank_dims_mm': tuple[float, float],
    'material': str,                    # UD | twill_2x2
    'parse_status': str,                # complete | partial | failed
    'parse_notes': str,
}
```

Write all 80 records to `reports/input_param_registry.json`.

### Commands

```bash
# Inspect one input file manually
head -200 <path_to_aniform_input_file>

# Run parser on a single sim
python -m io.aniform_input_reader --sim pair_000 --config config/pipeline_config.yaml

# Run on all 80, write registry
python -m io.aniform_input_reader --all --output reports/input_param_registry.json

# Check for null mandatory fields
python -c "
import json
reg = json.load(open('reports/input_param_registry.json'))
for r in reg:
    nulls = [k for k,v in r.items() if v is None and k != 'parse_notes']
    if nulls: print(r['sim_id'], ':', nulls)
"
```

---

## Step 1.2 — Output Field Identification

**Goal:** Identify the physical meaning of every output field in the Aniform Python tool output across all 80 simulations.

### Expected Documented Fields

| Field | Expected Behaviour | Coordinate Frame |
|---|---|---|
| Nodal displacements (x, y, z) | Monotonically increasing magnitude | Global |
| In-plane stress σ₁₁, σ₂₂, σ₁₂ | Mixed sign; compressive precedes wrinkling | Fiber or global — must resolve |
| Thickness | Monotonically decreasing in formed regions | Scalar |
| Fiber angle | Evolves from initial; bounded by locking angle for 2×2 (~54°) | Local fiber frame |
| Contact pressure | Spiky; localised to tool contact zone | Normal to surface |

### Stress Coordinate Frame Resolution

This is critical for feature engineering. Disambiguate by:
- UD material: σ₁₁ in fiber direction should dominate; σ₂₂ much smaller
- 2×2 twill: σ₁₂ shear component should be significant near locking angle
- Compare signs and magnitudes against the forming direction — compressive σ in the forming plane should precede wrinkle onset

### Fingerprinting Approach for Unknown Fields

For every field not in the documented list, compute this fingerprint across all 80 simulations:

```python
# validation/field_survey.py
def fingerprint_unknown_variable(field_data, sim_metadata):
    """
    field_data: (N_timesteps, N_nodes) array for one field, one simulation
    sim_metadata: dict from input_param_registry for this sim
    """
    return {
        # Temporal behaviour
        'is_monotonic_increasing':   bool(is_monotonic(field_data)),
        'is_monotonic_decreasing':   bool(is_monotonic(-field_data)),
        'initial_value_mean':        float(field_data[0].mean()),
        'final_value_mean':          float(field_data[-1].mean()),
        'mean_rate_of_change':       float(temporal_derivative(field_data).mean()),
        'has_temporal_spike':        bool(has_spike(field_data)),
        'spike_timing_stroke_frac':  float(spike_timing(field_data)),  # when spike occurs
        # Spatial behaviour
        'is_spatially_uniform':      bool(field_data.std(axis=1).mean() < 0.01 * field_data.mean()),
        'is_localized_at_contact':   float(contact_correlation(field_data, sim_metadata)),
        'is_localized_at_wrinkle':   float(wrinkle_correlation(field_data, sim_metadata)),
        # Cross-simulation sensitivity
        'ud_vs_twill_difference':    float(material_contrast(field_data, sim_metadata)),
        'geometry_sensitivity':      float(geometry_contrast(field_data, sim_metadata)),
        # Dimensional analysis
        'order_of_magnitude':        float(np.log10(np.abs(field_data).mean() + 1e-12)),
        'always_positive':           bool((field_data >= 0).all()),
        'always_negative':           bool((field_data <= 0).all()),
    }
```

### Priority Identification Order

1. **High value, probably directly readable:** thickness, fiber angle, nodal displacements
2. **High value, identify via fingerprint:** stress components (resolve coordinate frame), contact pressure
3. **High value, computable from nodal coords regardless:** deformation gradient F, Green-Lagrange strains E, principal strains, strain rates — these do not depend on field identification success
4. **Medium value:** internal energy density, reaction forces
5. **Lower value:** solver convergence flags, damage variables (likely absent for elastic forming)

### Commands

```bash
# Run field survey on a single known-wrinkle pair first
python -m validation.field_survey \
    --pair pair_000 \
    --config config/pipeline_config.yaml \
    --output reports/field_survey_pair_000.json

# Visualise fingerprints for all unknown fields in one pair
python -m validation.field_survey \
    --report reports/field_survey_pair_000.json \
    --plot

# Run across all pairs and aggregate
python -m validation.field_survey \
    --all-pairs \
    --unknown-only \
    --config config/pipeline_config.yaml \
    --output reports/field_registry_candidates.json

# Print candidate field identifications sorted by confidence
python -m validation.field_survey \
    --rank-candidates \
    --report reports/field_registry_candidates.json
```

### field_registry.yaml Format

Update this file as identifications are made. Every field must have an entry before WP2 begins.

```yaml
# config/field_registry.yaml
# DO NOT reorder entries — index correspondence with Aniform output is fixed
# Update status and physical_meaning as archaeology progresses
# Never delete an entry; set status: confirmed or status: unknown

fields:
  - index: 0
    name: displacement_x
    status: confirmed           # confirmed | inferred | unknown | pending
    physical_meaning: nodal displacement in global X direction
    units: mm
    coordinate_frame: global
    use_as_feature: false       # displacements encoded via F; raw displacements not directly used
    use_as_target: false
    notes: ""

  - index: 1
    name: displacement_y
    status: confirmed
    physical_meaning: nodal displacement in global Y direction
    units: mm
    coordinate_frame: global
    use_as_feature: false
    use_as_target: false
    notes: ""

  - index: 2
    name: displacement_z
    status: confirmed
    physical_meaning: nodal displacement in global Z (out-of-plane) direction
    units: mm
    coordinate_frame: global
    use_as_feature: false
    use_as_target: false
    notes: "out-of-plane variance used in wrinkle severity target"

  # --- populate remaining fields below as identified ---

  - index: 47
    name: unknown_047
    status: unknown
    physical_meaning: null
    units: null
    coordinate_frame: null
    use_as_feature: pending
    use_as_target: false
    notes: "spike at wrinkle onset, spatially localised — candidate: shear locking proximity"
```

---

## Step 1.3 — Layer Thickness Back-Calculation

Aniform sometimes modifies nominal ply thickness for numerical stability. The value in the input file may not match what the simulation actually used.

### Three-Method Hierarchy (attempt in order)

**Method A — Direct output field**
If thickness appears as a documented output field, read it at t=0 and average across all elements. Should match n_plies × ply_thickness_mm from input file or a simple multiple thereof.

**Method B — CLT stiffness back-calculation**
If bending stiffness D or membrane stiffness A tensors are accessible:
- For UD symmetric layup: D₁₁ = E₁ × t³ / (12 × (1-ν₁₂ν₂₁)), solve for t
- For 2×2 twill: use in-plane stiffness A₁₁ = E₁ × t × n_plies

**Method C — Cross-simulation clustering**
Plot recovered thickness values across all simulations for each material. Discrete clusters indicate modification levels (e.g. nominal × 0.5, nominal × 1.0). Flag simulations that do not cluster cleanly.

### Deliverable

Write per-simulation thickness to `input_param_registry.json` with:
```json
{
    "sim_id": "pair_000",
    "ply_thickness_actual_mm": 0.25,
    "ply_thickness_method": "method_A",
    "thickness_consistent_with_material_group": true
}
```

---

## Step 1.4 — Forming Force Reconstruction

**Goal:** Force vs. stroke fraction curve for event detection (max force timing, rate of change).

### Approach

1. Check if Aniform writes a summary force output directly — inspect the Python tool's available outputs
2. If not: identify nodal reaction forces at tool/blank contact nodes; integrate spatially to get total forming force
3. Plot force vs. stroke fraction for at least one pair per geometry type
4. Validate profile shape: initial rise at contact → quasi-static rise → plateau → possible drop at wrinkle onset

### Use in Pipeline

- `events/max_forming_force` — stroke fraction at peak force (scalar conditioning input to model)
- Force rate of change — temporal feature indicating forming phase transitions

### Command

```bash
# Check what summary outputs the Aniform tool exposes
python -c "
from <aniform_tool_module> import load_simulation
sim = load_simulation('<path_to_coarse_sim>')
print(dir(sim))           # list available attributes/methods
print(sim.available_fields())
"
```

---

## Step 1.5 — Wrinkle Onset Identification from Fine Mesh

**Goal:** For each of the 40 pairs, identify the timestep and spatial location of wrinkle onset in the fine mesh. This is the primary training target.

### Detection Approach

Wrinkling in the fine mesh manifests as:
- Spatial variance spike in out-of-plane displacement (z) within a coarse element cluster
- Local thickness variance increasing above background
- In UD: compressive principal stress crossing zero followed by out-of-plane displacement growth
- In 2×2 twill: shear angle approaching locking angle (~54°) followed by abrupt fiber reorientation

### Per-Pair Deliverable

```json
{
    "pair_id": "pair_000",
    "wrinkle_onset_timestep": 142,
    "wrinkle_onset_stroke_fraction": 0.61,
    "wrinkle_location_node_ids": [1847, 1848, 1849, ...],
    "wrinkle_severity_max": 0.43,
    "wrinkle_outcome": "wrinkled",
    "detection_method": "oop_displacement_variance",
    "detection_confidence": "high"
}
```

Write all 40 records to `reports/wrinkle_onset_registry.json`. This is consumed by WP4 (targets) and WP5 (stratification labels).

```bash
# Run wrinkle detection on fine mesh for one pair
python -m validation.wrinkle_detector \
    --pair pair_000 \
    --config config/pipeline_config.yaml \
    --output reports/wrinkle_onset_pair_000.json

# Run all pairs
python -m validation.wrinkle_detector \
    --all-pairs \
    --output reports/wrinkle_onset_registry.json

# Review detection confidence distribution
python -c "
import json
reg = json.load(open('reports/wrinkle_onset_registry.json'))
for r in reg:
    print(r['pair_id'], r['wrinkle_outcome'], r['detection_confidence'])
"
```

---

## WP1 Validation Gate

**Do not proceed to WP2 until every item below is checked.**

Write results to `reports/WP1_gate.md`.

```
Input File Archaeology
  [ ] input_param_registry.json written for all 80 simulations
  [ ] Zero simulations with parse_status: failed
  [ ] geometry_type populated for all 80
  [ ] material populated for all 80
  [ ] Visual geometry verification done for ≥1 sim per geometry type

Output Field Identification
  [ ] field_registry.yaml has an entry for every field index in Aniform output
  [ ] Zero fields with status: pending remaining
  [ ] Stress tensor coordinate frame resolved (confirmed fiber or global, documented in registry)
  [ ] All fields used as features have status: confirmed or status: inferred with rationale

Thickness
  [ ] ply_thickness_actual_mm written for all 80 simulations
  [ ] back-calculation method documented per simulation
  [ ] Values consistent within material groups (no unexplained outliers)

Forming Force
  [ ] Force-stroke curve obtainable for all 40 pairs (directly or via integration)
  [ ] Profile shape validated against expected (rise → plateau) for ≥1 per geometry type

Wrinkle Onset
  [ ] wrinkle_onset_registry.json written for all 40 pairs
  [ ] wrinkle_outcome (wrinkled | clean) assigned for all 40
  [ ] detection_confidence: low flagged and reviewed manually for any pair

Handoff Check
  [ ] field_registry.yaml reviewed and committed — no further edits without version bump
  [ ] input_param_registry.json reviewed and committed
  [ ] wrinkle_onset_registry.json reviewed and committed
  [ ] WP2 lead has read WP1_gate.md and signed off
```

---

## Handoff to WP2

WP2 receives from WP1:

| Artifact | Path | Notes |
|---|---|---|
| `field_registry.yaml` | `config/field_registry.yaml` | Locked — defines HDF5 field index mapping |
| `input_param_registry.json` | `reports/input_param_registry.json` | Source of geometry_type, material, process params |
| `wrinkle_onset_registry.json` | `reports/wrinkle_onset_registry.json` | Source of wrinkle_outcome and onset timing |
| `aniform_input_reader.py` | `io/aniform_input_reader.py` | WP3 uses this directly |
| `field_survey.py` | `validation/field_survey.py` | WP3 sanity checks reference fingerprint logic |
