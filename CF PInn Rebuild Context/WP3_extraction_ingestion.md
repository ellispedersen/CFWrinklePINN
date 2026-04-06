# WP3 — Extraction & Ingestion
## Aniform Reader, Orchestrator, Mesh Correspondence, Temporal Resampling, Sanity Checks

**Depends on:** WP2 gate passed, `dataset.h5` schema exists, `hdf5_writer.py` interfaces defined
**Feeds into:** WP4 (reads from `dataset.h5`), WP5 (reads pair metadata)

---

## Objective

Extract all 40 coarse/fine pairs from the raw Aniform output, build spatial mesh correspondence, resample temporal sequences, and write into `dataset.h5` via `hdf5_writer.py`. Every pair must pass sanity checks before being committed.

---

## Inputs Received from WP2

| Artifact | Used For |
|---|---|
| `data/dataset.h5` (empty, structured) | Write destination |
| `io/hdf5_writer.py` (interfaces) | All write operations |
| `config/pipeline_config.yaml` | All paths and ingestion parameters |
| `config/field_registry.yaml` | Field index → name mapping for array slicing |
| `reports/input_param_registry.json` | Pair metadata (geometry_type, material, process params) |
| `reports/wrinkle_onset_registry.json` | events/wrinkle_onset values per pair |

---

## Outputs Produced by This WP

| Artifact | Location | Consumer |
|---|---|---|
| Populated `dataset.h5` (raw + resampled, no features yet) | `data/dataset.h5` | WP4 |
| `pipeline.log` | `reports/pipeline.log` | Review |
| Per-pair sanity reports | `reports/sanity_<pair_id>.json` | Gate review |
| WP3 validation checklist | `reports/WP3_gate.md` | Gate review |

---

## 3.1 aniform_output_reader.py

Wraps the Aniform Python tool. Returns structured objects; all Aniform-specific API calls are contained here — nothing outside this module touches the Aniform tool directly.

```python
# io/aniform_output_reader.py

from dataclasses import dataclass
import numpy as np

@dataclass
class MeshData:
    nodes: np.ndarray        # (N_nodes, 3) float32 — reference coords
    elements: np.ndarray     # (N_elements, n_per_elem) int32

@dataclass
class RawSimData:
    mesh: MeshData
    times: np.ndarray        # (N_timesteps,) float64 — solver time values
    fields: np.ndarray       # (N_timesteps, N_nodes, N_fields) float32
                             # field index matches field_registry.yaml exactly
    n_fields: int

def extract(sim_path: str, field_registry: list[dict]) -> RawSimData:
    """
    Load one Aniform simulation using the Aniform Python tool.
    Returns RawSimData. field_registry used to order output fields
    to match registry index — do not rely on Aniform's default ordering.

    Raises:
        AniformLoadError if sim_path is not a valid simulation
        FieldMismatchError if actual field count != len(field_registry)
    """
    raise NotImplementedError

def compute_rates(times: np.ndarray, fields: np.ndarray) -> np.ndarray:
    """
    Compute dt-normalised finite difference rates.
    rates[0] = NaN (no prior timestep).
    rates[i] = (fields[i] - fields[i-1]) / (times[i] - times[i-1])
    Returns: (N_timesteps, N_nodes, N_fields) float32
    """
    dt = np.diff(times, prepend=np.nan)[:, None, None]  # (N_t, 1, 1)
    diff = np.diff(fields, axis=0, prepend=np.nan)
    rates = diff / dt
    rates[0] = np.nan
    return rates.astype(np.float32)
```

---

## 3.2 Mesh Correspondence

Builds the coarse-to-fine spatial mapping. Stored once per pair at t=0; never recomputed.

```python
# mesh/correspondence.py

from scipy.spatial import KDTree
import numpy as np

def build_coarse_to_fine_map(coarse_mesh: MeshData,
                              fine_mesh: MeshData,
                              radius_factor: float = 1.5) -> dict:
    """
    For each coarse element, find all fine elements whose centroids
    fall within radius_factor × coarse element characteristic radius.

    Returns:
        mapping: dict with keys:
            'fine_indices':      list of np.ndarray — fine elem indices per coarse elem
            'refinement_ratios': np.ndarray (N_c_elem,) — n_fine / 1 per coarse elem
            'boundary_flags':    np.ndarray (N_c_elem,) bool — True if at blank boundary
            'empty_flags':       np.ndarray (N_c_elem,) bool — True if 0 fine elems found

    Raises:
        CorrespondenceError if >5% of coarse elements map to 0 fine elements
    """
    fine_centroids   = _element_centroids(fine_mesh)
    coarse_centroids = _element_centroids(coarse_mesh)
    tree = KDTree(fine_centroids)

    fine_indices      = []
    refinement_ratios = np.zeros(len(coarse_centroids), dtype=np.float32)
    empty_flags       = np.zeros(len(coarse_centroids), dtype=bool)

    for i, centroid in enumerate(coarse_centroids):
        radius = _element_characteristic_radius(coarse_mesh, i)
        idx    = tree.query_ball_point(centroid, radius * radius_factor)
        fine_indices.append(np.array(idx, dtype=np.int32))
        refinement_ratios[i] = len(idx)
        empty_flags[i]       = len(idx) == 0

    empty_fraction = empty_flags.mean()
    if empty_fraction > 0.05:
        raise CorrespondenceError(
            f"{empty_fraction:.1%} of coarse elements map to 0 fine elements"
        )

    boundary_flags = _detect_boundary_elements(coarse_mesh)

    return {
        'fine_indices':      fine_indices,
        'refinement_ratios': refinement_ratios,
        'boundary_flags':    boundary_flags,
        'empty_flags':       empty_flags,
    }

def build_edge_index(mesh: MeshData) -> tuple[np.ndarray, np.ndarray]:
    """
    Build edge_index (2, N_edges) and edge_attr (N_edges, 4) from element connectivity.
    edge_attr columns: [dx, dy, dz, euclidean_dist] in reference configuration.
    Edges are undirected — both (i→j) and (j→i) included.
    """
    raise NotImplementedError
```

**Edge case handling:**
- Coarse elements at blank boundary → flagged in `boundary_flags`; include in training but flag in sanity report
- Empty coarse elements → flagged in `empty_flags`; exclude from targets computation; >5% raises error
- The `refinement_ratios` dataset is **diagnostic only** — stored in HDF5, never passed to the model (fine mesh unavailable at inference)

---

## 3.3 Temporal Resampling

```python
# features/temporal_features.py

import numpy as np
from scipy.interpolate import interp1d

def times_to_stroke_fractions(times: np.ndarray) -> np.ndarray:
    """Normalise raw solver times to [0, 1] stroke fraction."""
    t_min, t_max = times[0], times[-1]
    return (times - t_min) / (t_max - t_min)

def resample_to_stroke_fractions(times: np.ndarray,
                                  rates: np.ndarray,
                                  n_points: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample rate features (not raw fields) to uniform stroke-fraction grid.

    Interpolation is applied to rates because:
    - Rates are already dt-normalised — linear interpolation is physically reasonable
    - Raw field values should NOT be interpolated — this can generate non-physical states

    Args:
        times:    (N_raw,) float — solver time values
        rates:    (N_raw, N_nodes, N_fields) float — dt-normalised rates; NaN at t=0
        n_points: target number of uniform points (default 256)

    Returns:
        stroke_fractions: (n_points,) float — uniform 0→1
        rates_resampled:  (n_points, N_nodes, N_fields) float
    """
    stroke_fractions_raw = times_to_stroke_fractions(times)
    stroke_fractions_out = np.linspace(0.0, 1.0, n_points, dtype=np.float32)

    # Fill NaN at t=0 with zero before interpolating
    rates_filled = rates.copy()
    rates_filled[0] = 0.0

    N_nodes, N_fields = rates.shape[1], rates.shape[2]
    rates_resampled = np.zeros((n_points, N_nodes, N_fields), dtype=np.float32)

    for n in range(N_nodes):
        for f in range(N_fields):
            interp = interp1d(stroke_fractions_raw, rates_filled[:, n, f],
                              kind='linear', bounds_error=False, fill_value='extrapolate')
            rates_resampled[:, n, f] = interp(stroke_fractions_out)

    return stroke_fractions_out, rates_resampled

def detect_contact_initiation(times: np.ndarray,
                               contact_pressure_field: np.ndarray) -> float:
    """
    Return stroke fraction at which contact pressure first exceeds threshold.
    Returns NaN if contact pressure field not identified.
    """
    raise NotImplementedError

def detect_max_forming_force(times: np.ndarray,
                              force_curve: np.ndarray) -> float:
    """Return stroke fraction at peak forming force."""
    stroke_fracs = times_to_stroke_fractions(times)
    return float(stroke_fracs[np.argmax(force_curve)])
```

**Critical constraint:** Resample `rates`, not `fields`. Raw field values at non-solver timesteps are not physically defined — interpolating them can produce stress states that violate equilibrium. Rates between known points are a physically reasonable interpolation target.

---

## 3.4 Orchestrator

Reads as a narrative of what the pipeline does. No logic here — all logic in modules.

```python
# orchestrator.py

import logging
from pathlib import Path
import yaml, json
from io import aniform_input_reader, aniform_output_reader, hdf5_writer
from mesh import correspondence
from features import temporal_features
from targets import wrinkle_targets
from validation import sanity_checks

logging.basicConfig(level=logging.INFO, filename='reports/pipeline.log')
logger = logging.getLogger(__name__)

def run_pipeline(config_path: str, pair_filter: str = None) -> None:
    config          = yaml.safe_load(open(config_path))
    field_registry  = yaml.safe_load(open(config['paths']['field_registry']))['fields']
    input_registry  = json.load(open(config['paths']['input_registry']))
    wrinkle_registry = json.load(open(config['paths']['wrinkle_registry']))

    pairs = _discover_pairs(config, input_registry, pair_filter)

    with hdf5_writer.open_dataset(config['paths']['dataset_h5']) as h5:
        for pair in pairs:
            logger.info(f"=== Processing {pair['pair_id']} | {pair['geometry_type']} | {pair['material']} ===")
            try:
                # Stage 1: Extract raw data from Aniform
                coarse_raw = aniform_output_reader.extract(pair['coarse_path'], field_registry)
                fine_raw   = aniform_output_reader.extract(pair['fine_path'],   field_registry)

                # Stage 2: Build mesh correspondence — spatial, not topological
                mapping = correspondence.build_coarse_to_fine_map(
                    coarse_raw.mesh, fine_raw.mesh,
                    radius_factor=config['ingestion']['correspondence_radius_factor']
                )
                edge_index, edge_attr = correspondence.build_edge_index(coarse_raw.mesh)

                # Stage 3: Compute dt-normalised rates on raw timesteps
                rates_raw = aniform_output_reader.compute_rates(coarse_raw.times, coarse_raw.fields)

                # Stage 4: Resample rates to 256 uniform stroke-fraction points
                stroke_fractions, rates_resampled = temporal_features.resample_to_stroke_fractions(
                    coarse_raw.times, rates_raw,
                    n_points=config['ingestion']['n_resample_points']
                )

                # Stage 5: Detect scalar events
                events = _build_events(coarse_raw, pair, wrinkle_registry, config)

                # Stage 6: Write everything to HDF5
                hdf5_writer.write_pair_metadata(h5, pair['pair_id'], pair)
                hdf5_writer.write_mesh(h5, pair['pair_id'], coarse_raw.mesh, fine_raw.mesh, mapping, edge_index, edge_attr)
                hdf5_writer.write_raw_timesteps(h5, pair['pair_id'], coarse_raw.times, coarse_raw.fields, rates_raw)
                hdf5_writer.write_resampled(h5, pair['pair_id'], stroke_fractions, coarse_raw.fields, rates_resampled)
                hdf5_writer.write_events(h5, pair['pair_id'], events)

                # Stage 7: Sanity check — catch problems before they enter training data
                report = sanity_checks.validate_pair(h5, pair['pair_id'], config)
                if report['status'] == 'failed':
                    logger.error(f"{pair['pair_id']} FAILED sanity checks: {report['failures']}")
                else:
                    logger.info(f"{pair['pair_id']} passed sanity checks")

            except Exception as e:
                logger.error(f"{pair['pair_id']} ERROR: {e}", exc_info=True)
                continue   # --continue-on-error behaviour; remove to halt on first failure
```

---

## 3.5 Sanity Checks

Runs after each pair is written. Catches problems at parse time, not training time.

```python
# validation/sanity_checks.py

def validate_pair(h5: h5py.File, pair_id: str, config: dict) -> dict:
    """
    Returns {'status': 'passed'|'failed', 'warnings': [...], 'failures': [...]}
    """
    failures = []
    warnings = []
    grp = h5[f'pairs/{pair_id}']

    # 1. Conservation — total blank area approximately conserved
    nodes_ref = grp['mesh/coarse/nodes'][:]
    nodes_cur = grp['coarse/raw_timesteps/fields'][-1, :, :3]   # displacement fields
    _check_area_conservation(nodes_ref, nodes_cur, failures, warnings)

    # 2. Thickness monotonicity — should decrease in formed regions
    if 'thickness' in _get_field_names(h5):
        thickness_idx = _field_index(h5, 'thickness')
        thickness = grp['coarse/raw_timesteps/fields'][:, :, thickness_idx]
        _check_thickness_monotonic(thickness, warnings)

    # 3. Correspondence quality — mean mapping distance below threshold
    ratios = grp['mesh/coarse/coarse_to_fine_map/refinement_ratios'][:]
    if ratios.min() == 0:
        failures.append(f"coarse elements with 0 fine elements: {(ratios == 0).sum()}")

    # 4. Target sanity — wrinkle severity near zero at t=0
    severity = grp['fine/targets/wrinkle_severity'][:]
    if severity.max() > 0.1:
        warnings.append(f"wrinkle_severity non-zero at t=0: max={severity.max():.3f}")

    # 5. Events — wrinkle_onset NaN only if wrinkle_outcome=0
    outcome = grp.attrs['wrinkle_outcome']
    onset   = grp['events/wrinkle_onset'][()]
    if outcome == 1 and np.isnan(onset):
        failures.append("wrinkle_outcome=1 but wrinkle_onset is NaN")
    if outcome == 0 and not np.isnan(onset):
        warnings.append("wrinkle_outcome=0 but wrinkle_onset is set")

    # 6. Provenance — every dataset has required attributes
    required_attrs = {'status', 'physical_meaning', 'feature_version', 'source'}
    h5[f'pairs/{pair_id}'].visititems(
        lambda name, obj: _check_provenance(name, obj, required_attrs, warnings)
    )

    status = 'failed' if failures else 'passed'
    return {'status': status, 'failures': failures, 'warnings': warnings}
```

---

## Commands

```bash
# Dry-run single pair (extracts, builds correspondence, no HDF5 write)
python orchestrator.py --pair pair_000 --dry-run --config config/pipeline_config.yaml

# Ingest single pair (write to HDF5)
python orchestrator.py --pair pair_000 --config config/pipeline_config.yaml

# Ingest all 40 pairs, continue on error, write log
python orchestrator.py --all --continue-on-error \
    --config config/pipeline_config.yaml \
    --log reports/pipeline.log

# Run sanity checks on all ingested pairs
python -m validation.sanity_checks \
    --all --dataset data/dataset.h5 \
    --config config/pipeline_config.yaml \
    --report reports/WP3_sanity_summary.json

# Inspect HDF5 structure post-ingestion
python -c "
import h5py
with h5py.File('data/dataset.h5', 'r') as f:
    f.visit(print)
"

# Check how many pairs are ingested
python -c "
import h5py
with h5py.File('data/dataset.h5', 'r') as f:
    pairs = list(f['pairs'].keys())
    print(f'{len(pairs)} pairs ingested:', pairs)
"

# Check correspondence quality across all pairs
python -c "
import h5py, numpy as np
with h5py.File('data/dataset.h5', 'r') as f:
    for pid in f['pairs']:
        ratios = f[f'pairs/{pid}/mesh/coarse/coarse_to_fine_map/refinement_ratios'][:]
        print(pid, 'min_ratio:', ratios.min(), 'empty_elems:', (ratios==0).sum())
"

# Run unit tests
python -m pytest tests/test_correspondence.py tests/test_hdf5_schema.py -v
```

---

## WP3 Validation Gate

```
Extraction
  [ ] aniform_output_reader.extract() runs without error for all 80 sims (40 coarse + 40 fine)
  [ ] Field count in extracted data matches len(field_registry) for all sims
  [ ] times array is strictly increasing for all sims

Correspondence
  [ ] build_coarse_to_fine_map completes for all 40 pairs
  [ ] No pair has >5% empty coarse elements
  [ ] edge_index and edge_attr written for all pairs

Ingestion
  [ ] All 40 pairs present in dataset.h5/pairs/
  [ ] raw_timesteps/fields marked IMMUTABLE (no overwrite guard test)
  [ ] resampled/ written for all pairs with shape (256, N_nodes, N_fields)
  [ ] events/ written for all pairs; wrinkle_onset consistent with wrinkle_outcome

Sanity Checks
  [ ] Zero pairs with status: failed in WP3_sanity_summary.json
  [ ] All warnings reviewed and documented
  [ ] pipeline.log contains no unhandled exceptions

Handoff Check
  [ ] WP4 lead has confirmed dataset.h5 is readable via hdf5_reader.load_pair_for_training()
  [ ] feature_version in pipeline_config.yaml set to v1.0 before WP4 begins
```

---

## Handoff to WP4

WP4 receives from WP3:

| Artifact | Path | Notes |
|---|---|---|
| Populated `dataset.h5` | `data/dataset.h5` | Raw + resampled; no features yet |
| `hdf5_reader.py` | `io/hdf5_reader.py` | WP4 reads via this interface |
| `field_registry.yaml` | `config/field_registry.yaml` | Defines which fields WP4 can use |
| `wrinkle_onset_registry.json` | `reports/wrinkle_onset_registry.json` | WP4 uses for target computation |
| `pipeline_config.yaml` | `config/pipeline_config.yaml` | feature_version set before WP4 starts |
