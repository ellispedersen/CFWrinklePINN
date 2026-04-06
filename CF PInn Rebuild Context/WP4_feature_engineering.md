# WP4 — Feature Engineering & Target Computation
## Physics Features, Kinematic Features, Material Features, Wrinkle Targets

**Depends on:** WP3 gate passed, `dataset.h5` populated with raw and resampled data
**Feeds into:** WP5 (reads wrinkle_outcome for stratification), WP6/7 (dataloader reads features and targets)

---

## Objective

Compute all physics-based node features and per-element wrinkle severity targets from the ingested data. Write results back into `dataset.h5` under `features/` and `fine/targets/`. The dataloader in WP7 reads only from these groups — no on-the-fly computation at training time.

---

## Inputs Received from WP3

| Artifact | Path | Used For |
|---|---|---|
| Populated `dataset.h5` | `data/dataset.h5` | Read raw fields, mesh, resampled rates |
| `field_registry.yaml` | `config/field_registry.yaml` | Know which field indices are stress, fiber angle, etc. |
| `wrinkle_onset_registry.json` | `reports/wrinkle_onset_registry.json` | Target node locations per pair |
| `pipeline_config.yaml` | `config/pipeline_config.yaml` | feature_version, use_fields list |

---

## Outputs Produced by This WP

| Artifact | Location | Consumer |
|---|---|---|
| Physics feature tensor per pair | `pairs/<id>/features/coarse/resampled/physics/` | WP7 dataloader |
| Wrinkle severity targets per pair | `pairs/<id>/fine/targets/` | WP7 dataloader |
| `reports/WP4_feature_report.json` | feature value range checks per field | Gate review |
| `reports/WP4_gate.md` | Validation checklist | Gate review |

---

## 4.1 Feature Engineering Principles

### What the Model Needs vs. What the Solver Writes

The model needs proximity-to-instability indicators, not raw field values. Raw displacements and raw stresses are not directly useful — their derivatives and invariants are.

| Raw Solver Output | Derived Feature(s) |
|---|---|
| Nodal displacements (x, y, z) | Deformation gradient F, Green-Lagrange strain E |
| In-plane stress components | Principal stresses σ₁, σ₂; compressive stress magnitude; Huber wrinkling criterion |
| Thickness | Thickness reduction ratio t/t₀; thinning rate |
| Fiber angle | Fiber angle deviation from initial; proximity to locking angle |
| Any rate field (from WP3) | Already dt-normalised — use directly as rate features |

### Inference Constraint Check

Before adding any feature, ask: **is this computable from the coarse mesh alone at inference time?**

| Feature | Available at Inference? | Notes |
|---|---|---|
| Deformation gradient F | ✓ | From coarse nodal coords |
| Green-Lagrange strains | ✓ | From F |
| Principal stresses | ✓ | From coarse stress fields |
| Fiber angle proximity to locking | ✓ | From coarse fiber angle field |
| Refinement ratio (fine/coarse element count) | ✗ | Fine mesh unavailable — do not use |
| Fine mesh thickness | ✗ | Fine mesh unavailable — do not use |

---

## 4.2 Kinematic Features

Computable from nodal coordinates alone — independent of which stress/scalar fields were successfully identified in WP1. These are always available.

```python
# features/kinematic_features.py
import numpy as np
import torch

def deformation_gradient_per_element(ref_coords: np.ndarray,
                                      cur_coords: np.ndarray,
                                      connectivity: np.ndarray) -> np.ndarray:
    """
    Compute F = dx/dX per element using shape function gradients.

    Args:
        ref_coords:   (N_nodes, 3) — reference config (t=0 flat blank)
        cur_coords:   (N_nodes, 3) — current config at one timestep
        connectivity: (N_elements, n_per_elem) — element node indices

    Returns:
        F: (N_elements, 3, 3) — deformation gradient per element
    """
    raise NotImplementedError

def green_lagrange_strain(F: torch.Tensor) -> torch.Tensor:
    """
    E = 0.5 * (F^T @ F - I)
    Args:
        F: (..., 3, 3)
    Returns:
        E: (..., 3, 3)
    """
    C = F.transpose(-1, -2) @ F
    I = torch.eye(3, device=F.device, dtype=F.dtype)
    return 0.5 * (C - I)

def principal_strains(E: torch.Tensor) -> torch.Tensor:
    """
    Return principal strains (eigenvalues of E), sorted descending.
    Returns: (..., 3)
    """
    eigenvalues = torch.linalg.eigvalsh(E)  # eigvalsh for symmetric tensors
    return eigenvalues.flip(-1)             # descending order

def thickness_reduction_ratio(thickness_field: np.ndarray,
                               t0_thickness: float) -> np.ndarray:
    """
    t/t₀ — fraction of original thickness remaining.
    Values below ~0.85 in forming simulations warrant attention.
    """
    return thickness_field / t0_thickness
```

---

## 4.3 Stress Features

Requires stress fields confirmed in `field_registry.yaml`. Check `status: confirmed` before computing — log a warning and skip if not confirmed.

```python
# features/stress_features.py
import numpy as np
import torch

def principal_stresses(stress_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute principal stresses from stress components.
    Args:
        stress_tensor: (..., 3) — [sigma_11, sigma_22, sigma_12] in fiber or global frame
    Returns:
        (sigma_1, sigma_2): (..., 2) — max and min principal stresses
    """
    s11, s22, s12 = stress_tensor[..., 0], stress_tensor[..., 1], stress_tensor[..., 2]
    mean  = (s11 + s22) / 2.0
    delta = torch.sqrt(((s11 - s22) / 2.0) ** 2 + s12 ** 2)
    sigma_1 = mean + delta
    sigma_2 = mean - delta
    return torch.stack([sigma_1, sigma_2], dim=-1)

def compressive_stress_magnitude(sigma_1: torch.Tensor,
                                  sigma_2: torch.Tensor) -> torch.Tensor:
    """
    Return magnitude of most compressive principal stress.
    Zero if both principal stresses are tensile.
    """
    return torch.maximum(-sigma_2, torch.zeros_like(sigma_2))

def huber_wrinkling_criterion(sigma_1: torch.Tensor,
                               sigma_2: torch.Tensor,
                               E1: float, E2: float,
                               t: float, a: float) -> torch.Tensor:
    """
    Huber wrinkling criterion for composite shells.
    Returns criterion value — positive values indicate wrinkling tendency.
    sigma_2 + (E2/E1) * (t/a)^2 * pi^2 / 12 < 0  → wrinkle risk

    Args:
        sigma_1, sigma_2: principal stresses (...,)
        E1, E2: Young's moduli in fiber and transverse directions
        t: ply thickness
        a: characteristic element size (wavelength proxy)
    """
    critical = (E2 / E1) * (t / a) ** 2 * (np.pi ** 2 / 12.0)
    return -(sigma_2 + critical)   # positive → wrinkle risk
```

---

## 4.4 Material Features

```python
# features/material_features.py

def extract_material_card(material: str, h5_path: str) -> np.ndarray:
    """
    Read material card parameters from dataset.h5/metadata/material_cards/<material>.
    Returns flat vector: [E1, E2, G12, nu12, rho, ply_thickness]
    Normalised by typical forming values before returning.
    """
    raise NotImplementedError

def compute_clt_stiffness(material_card: np.ndarray,
                           layup: list[float],
                           ply_thickness: float) -> dict:
    """
    Classical Laminate Theory — compute A, B, D matrices.
    Returns dict with A11, A22, A12, D11, D22, membrane_to_bending_ratio.
    membrane_to_bending_ratio is a key wrinkling susceptibility indicator.
    """
    raise NotImplementedError

def fiber_angle_proximity_to_locking(fiber_angle_field: np.ndarray,
                                      locking_angle: float) -> np.ndarray:
    """
    Proximity of current fiber angle to shear locking angle.
    Returns (N_nodes,) in [0, 1] — 1.0 means at locking angle.
    locking_angle is ~54° for 2×2 twill; not applicable for UD (set to inf).
    """
    return np.clip(np.abs(fiber_angle_field) / locking_angle, 0.0, 1.0)
```

---

## 4.5 Target Computation (Wrinkle Severity)

Computed from the **fine mesh** and aggregated to coarse element clusters via the correspondence map built in WP3.

```python
# targets/wrinkle_targets.py

def compute_wrinkle_severity(h5: h5py.File,
                              pair_id: str,
                              config: dict) -> dict:
    """
    Computes per-coarse-element wrinkle severity from fine mesh data.
    Reads fine mesh fields and mapping from h5; writes results back to h5.

    Returns dict of arrays (N_c_elem,):
        thickness_variance:    variance of fine thickness within each coarse cluster
        max_oop_displacement:  max |z-displacement| within cluster at final timestep
        wrinkle_severity:      composite: alpha * thickness_variance + beta * max_oop
    """
    mapping = _load_mapping(h5, pair_id)
    fine_fields = h5[f'pairs/{pair_id}/fine/raw_timesteps/fields'][-1]  # final timestep only

    thickness_idx  = _field_index(h5, 'thickness')
    z_disp_idx     = _field_index(h5, 'displacement_z')

    thickness_variance   = np.zeros(len(mapping['fine_indices']), dtype=np.float32)
    max_oop_displacement = np.zeros(len(mapping['fine_indices']), dtype=np.float32)

    for i, fine_idx in enumerate(mapping['fine_indices']):
        if len(fine_idx) == 0:
            thickness_variance[i]   = np.nan   # empty element — excluded from loss
            max_oop_displacement[i] = np.nan
            continue
        fine_thickness = fine_fields[fine_idx, thickness_idx]
        fine_z_disp    = fine_fields[fine_idx, z_disp_idx]
        thickness_variance[i]   = float(np.var(fine_thickness))
        max_oop_displacement[i] = float(np.abs(fine_z_disp).max())

    alpha, beta = 0.5, 0.5   # from config — tune in WP6
    wrinkle_severity = alpha * thickness_variance + beta * max_oop_displacement

    return {
        'thickness_variance':    thickness_variance,
        'max_oop_displacement':  max_oop_displacement,
        'wrinkle_severity':      wrinkle_severity,
    }
```

**Note:** NaN entries (empty coarse elements) must be masked in the loss function in WP6 — they are not errors, they are boundary elements with no fine-mesh coverage.

---

## 4.6 Feature Assembly and Write

```python
# features pipeline — called from a WP4-specific runner, not orchestrator

def compute_and_write_features(h5_path: str, pair_id: str, config: dict) -> None:
    """
    Reads from dataset.h5, computes all physics features, writes back.
    Safe to re-run — increments feature_version check before overwriting.
    """
    with h5py.File(h5_path, 'a') as h5:
        pair = hdf5_reader.load_pair_for_training(h5_path, pair_id,
                                                   feature_version=None)  # load any version
        field_registry = yaml.safe_load(open(config['paths']['field_registry']))['fields']

        features_list  = []
        feature_names  = []

        # Kinematic features — always available
        for t_idx in range(pair['node_features'].shape[0]):   # 256 timesteps
            cur_coords = _get_coords_at_timestep(h5, pair_id, t_idx)
            F = kinematic_features.deformation_gradient_per_element(
                ref_coords=h5[f'pairs/{pair_id}/mesh/coarse/nodes'][:],
                cur_coords=cur_coords,
                connectivity=h5[f'pairs/{pair_id}/mesh/coarse/elements'][:]
            )
            E = kinematic_features.green_lagrange_strain(torch.from_numpy(F))
            eps_principal = kinematic_features.principal_strains(E)
            # ... accumulate into features array

        # Stress features — only if confirmed in field_registry
        confirmed_stress = [f for f in field_registry
                            if 'stress' in f['physical_meaning'] and f['status'] == 'confirmed']
        if confirmed_stress:
            # extract and compute stress-derived features
            pass
        else:
            logger.warning(f"{pair_id}: stress fields not confirmed — stress features skipped")

        # Material features — always available
        mat_card = material_features.extract_material_card(pair['material'], h5_path)
        # ... CLT stiffness, fiber angle proximity

        physics_tensor = np.stack(features_list, axis=-1)   # (256, N_nodes, N_feat)

        hdf5_writer.write_features(h5, pair_id, physics_tensor, feature_names,
                                    version=config['features']['feature_version'])

        # Targets — computed from fine mesh
        targets = wrinkle_targets.compute_wrinkle_severity(h5, pair_id, config)
        hdf5_writer.write_targets(h5, pair_id, targets)
```

---

## Commands

```bash
# Run feature engineering on one pair
python -m features.run_features \
    --pair pair_000 \
    --dataset data/dataset.h5 \
    --config config/pipeline_config.yaml

# Run on all pairs
python -m features.run_features \
    --all \
    --dataset data/dataset.h5 \
    --config config/pipeline_config.yaml \
    --log reports/feature_pipeline.log

# Check feature value ranges across all pairs
python -c "
import h5py, numpy as np
with h5py.File('data/dataset.h5', 'r') as f:
    for pid in f['pairs']:
        feat = f[f'pairs/{pid}/features/coarse/resampled/physics']
        if feat.shape[0] > 0:
            print(pid, 'feat_shape:', feat.shape,
                  'min:', np.nanmin(feat[:]), 'max:', np.nanmax(feat[:]))
"

# Check target distribution (wrinkle severity)
python -c "
import h5py, numpy as np
with h5py.File('data/dataset.h5', 'r') as f:
    severities = []
    for pid in f['pairs']:
        sev = f[f'pairs/{pid}/fine/targets/wrinkle_severity'][:]
        severities.append(np.nanmax(sev))
    print('max severity per pair:', sorted(severities))
    print('wrinkled (>0.05):', sum(s > 0.05 for s in severities))
"

# Verify NaN masking — check empty element fraction
python -c "
import h5py, numpy as np
with h5py.File('data/dataset.h5', 'r') as f:
    for pid in f['pairs']:
        sev = f[f'pairs/{pid}/fine/targets/wrinkle_severity'][:]
        nan_frac = np.isnan(sev).mean()
        if nan_frac > 0.1:
            print(pid, f'WARNING: {nan_frac:.1%} NaN targets (empty elements)')
"

# Run tests
python -m pytest tests/test_features.py tests/test_targets.py -v
```

---

## WP4 Validation Gate

```
Feature Computation
  [ ] physics feature tensor written for all 40 pairs
  [ ] feature_version attribute on all feature datasets matches pipeline_config.yaml
  [ ] feature_names attribute present and length matches last dimension of tensor
  [ ] No pair with all-NaN feature tensor
  [ ] Stress features skipped gracefully (warning logged) if not confirmed in field_registry

Target Computation
  [ ] wrinkle_severity, thickness_variance, max_oop_displacement written for all 40 pairs
  [ ] NaN fraction per pair < 10% (boundary element tolerance)
  [ ] wrinkle_severity > threshold for all pairs with wrinkle_outcome=1
  [ ] wrinkle_severity ≈ 0 for all pairs with wrinkle_outcome=0

Inference Constraint
  [ ] No feature uses fine mesh data (confirmed by code review)
  [ ] No feature uses refinement_ratio as input

Handoff Check
  [ ] hdf5_reader.load_pair_for_training() returns correct tensor shapes for all 40 pairs
  [ ] WP5 lead has confirmed wrinkle_outcome is set correctly for stratification
  [ ] feature_version locked — any change requires new version string and re-run of WP4
```

---

## Handoff to WP5

WP5 receives from WP4:

| Artifact | Path | Notes |
|---|---|---|
| Fully populated `dataset.h5` | `data/dataset.h5` | Features and targets written |
| `input_param_registry.json` | `reports/input_param_registry.json` | Source of geometry_type, material for stratification |
| `wrinkle_onset_registry.json` | `reports/wrinkle_onset_registry.json` | Source of wrinkle_outcome |
| `hdf5_reader.py` | `io/hdf5_reader.py` | WP5 uses list_pairs() and pair metadata |
