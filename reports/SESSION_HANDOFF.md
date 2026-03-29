# Session Handoff — 2026-03-28
## What was discovered / what needs updating

### Status
WP1 field extraction complete. All 20 AFR field identities confirmed via `ReadAFResult(silent=False)`.
**`config/field_registry.yaml`, `config/pipeline_config.yaml`, and `ANIFORM_REFERENCE.md` are all still wrong — they must be rewritten before running any pipeline scripts.**

---

## Definitive Field Identity Map (both batches)

| AFR ID | Sub | Name (from silent=False) | res_type | Groups (Batch A) | Groups (Batch B) | Use |
|--------|-----|--------------------------|----------|------------------|------------------|-----|
| 40 | 1 | Displacement | VectorT (3f) | 6, 10 | 4, 8, 12 | feature — dz wrinkle precursor |
| 42 | 1 | Rotation | VectorT (3f) | 6, 10 | 4, 8, 12 | exclude — rigid tool DOF, near-zero on ply |
| 44 | 1 | Temperature | VectorT (3f)? | 6, 10 | 4, 8, 12 | feature — thermal history drives crystallinity |
| 48 | 1 | Temperature 1 | ScalarT (1f) | 6, 10 | 4, 8, 12 | feature — top surface temp |
| 49 | 1 | Temperature 2 | ScalarT (1f) | 6, 10 | 4, 8, 12 | feature — bottom surface temp |
| 102 | 1 | Green-Lagrange strain | STensorPSST (3f) | 6,7,10,11 | 4,5,8,9,12,13 | feature — [E11, E22, E12] |
| 105 | 1 | Thickness | ScalarT (1f) | 6, 10 | 4, 8, 12 | feature — thinning indicator |
| 106 | 1 | Eq shear rate | ScalarT (1f) | 6, 10 | 4, 8, 12 | feature — viscous flow rate |
| 200 | 1 | Stress | STensorPSST (3f) | 6,7,10,11 | 4,5,8,9,12,13 | feature — [s11, s22, s12] Cauchy stress |
| 203 | 1 | Fiber direction 1 | VectorT (3f) | 6, 10 | 4, 8, 12 | feature — unit vec, evolves with deformation |
| 203 | 2 | Fiber direction 2 | VectorT (3f) | 6, 10 | 4, 8, 12 | feature — unit vec fiber family 2 |
| 204 | 1 | Shear angle f1_f2 | ScalarT (1f) | 6, 10 | 4, 8, 12 | **KEY feature** — primary woven wrinkle indicator |
| 205 | 1 | Fiber strain 1 | ScalarT (1f) | 6, 10 | 4, 8, 12 | feature — stretch ratio fiber family 1 |
| 205 | 2 | Fiber strain 2 | ScalarT (1f) | 6, 10 | 4, 8, 12 | feature — stretch ratio fiber family 2 |
| 206 | 1 | Fiber stress 1 | ScalarT (1f) | 6, 10 | 4, 8, 12 | **PRIMARY wrinkle precursor** — goes 0→80% compressive |
| 206 | 2 | Fiber stress 2 | ScalarT (1f) | 6, 10 | 4, 8, 12 | feature — fiber stress family 2 |
| 214 | 1 | Rel. crystallinity (Nakamura) | ScalarT (1f) | 6, 10 | **Batch A only** | feature — [0,1], thermal state |
| 302 | 1 | Penetration depth | ScalarT (1f) | 8,9,13 | 6,7,11,15 | exclude — contact tool field, not ply data |
| 304 | 1 | Slip path length | ScalarT (1f) | 8,9,13 | 6,7,11,15 | exclude — contact tool field |
| 401 | 1 | Traction | VectorT (3f) | 8,9,13 | 6,7,11,15 | exclude — all zeros, contact artifact |

**Key corrections from previous assumptions:**
- `model_200` = **Stress** (Cauchy [s11,s22,s12]) — was assumed to be curvature tensor
- `model_206_1/2` = **Fiber stress** — was assumed to be total strains; IS the primary wrinkle precursor
- `model_105` = **Thickness** — was assumed to be fiber angle; field 102 is GL strain (not thickness)
- `model_44` = **Temperature** — was assumed to be acceleration
- `model_48/49` = **Temperature 1/2** — were assumed to be contact pressure/status
- `model_302` = **Penetration depth** — was assumed to be stress tensor; is a CONTACT field
- `model_401` = **Traction** — all zeros; is NOT Nakamura crystallinity
- `model_214` = **Nakamura crystallinity** — Batch A only; was the unknown field

---

## Correct Group IDs (CRITICAL — current pipeline_config.yaml is WRONG)

```
Batch A:
  ply shell (main):     [6, 10]           — kinematic + scalar fields
  ply bending sub-elem: [6, 7, 10, 11]    — STensorPSST fields (102, 200)
  contact groups:       [8, 9, 13]        — fields 302, 304, 401

Batch B:
  ply shell (main):     [4, 8, 12]        — NOT [6, 10, 14] as in config
  ply bending sub-elem: [4, 5, 8, 9, 12, 13]
  contact groups:       [6, 7, 11, 15]
```

---

## Confirmed Value Ranges (from live extraction)

| Field | Value at t=0 | Value at t_final | Notes |
|-------|-------------|-----------------|-------|
| Thickness (105) | 0.300mm uniform | mean 0.312mm | Thinning occurs, nominal 0.3mm confirmed |
| GL strain (102) | ~0 | E11≈0, E12≈0.00075 | Near-zero — dominant viscous not elastic |
| Stress (200) | ~0 | s11=-0.203, s22=-0.176 MPa | Very small — viscous forming stress |
| Fiber dir 1 (203_1) | [1.000, 0.000, 0.000] | [0.948, 0.001, 0.004] | Pure unit vector, tracks deformation |
| Fiber stress 1 (206_1) | 0% compressive | 80% compressive | Key wrinkle indicator |
| Slip path (304) | 0 | max 3.07mm | Contact nodes only |
| Crystallinity (214) | ~0 | rises on cooling | Batch A only |

---

## Coarse vs Fine Identification

```python
# Node count from displacement field (most reliable):
data = ResultsIncr[incr_nr][group_id]  # shape (N_nodes, 4)
n_nodes = data.shape[0]

# Thresholds:
Batch A: fine > 30_000 nodes, coarse < 5_000 nodes (actual: 75,356 vs 3,721)
Batch B: fine > 20_000 nodes (actual: 40,401)

# File size fallback (Part.section 1.1.msh):
Batch A: fine > 5_000 KB, coarse < 500 KB
Batch B: fine > 2_000 KB
```

---

## Files That Must Be Rewritten Tomorrow

1. **`config/field_registry.yaml`** — complete rewrite; almost all entries have wrong names, groups, meanings
2. **`config/pipeline_config.yaml`** — fix `batches.B.ply_groups: [6,10,14]` → `[4,8,12]`; add bending groups and contact groups per batch
3. **`ANIFORM_REFERENCE.md`** — rewrite field table with correct names and value ranges

## Scripts That Need Fixes (after config rewrite)

4. **`validation/field_survey.py`** — stress frame resolution targets field 302 (wrong); should target field 200 in groups [6,7]
5. **`validation/wrinkle_detector.py`** — uses `ply_group_for_detection: 6`; correct for Batch A but must use group 4 for Batch B
6. **`wp1_survey/probe_afr_fields.py`** — ply groups filter was wrong; recheck after pipeline_config update

## Scripts Still to Run (WP1 gate unchecked items)

7. `python -m wp1_survey.parse_all_afi` → `reports/input_param_registry.json`
8. `python -m wp1_survey.read_track_force` → `reports/force_stroke_summary.json`
9. `python -m validation.field_survey` → `reports/field_survey_detail.json`
10. `python -m validation.wrinkle_detector` → `reports/wrinkle_onset_registry.json`

---

## Resume Order Tomorrow

1. Rewrite `config/pipeline_config.yaml` (Batch B groups)
2. Rewrite `config/field_registry.yaml` (all correct names)
3. Rewrite `ANIFORM_REFERENCE.md` (full field table)
4. Fix `validation/field_survey.py` (stress frame target)
5. Fix `validation/wrinkle_detector.py` (Batch B group)
6. Run the 4 remaining WP1 scripts
7. Manually review outputs, check all WP1_gate.md boxes
