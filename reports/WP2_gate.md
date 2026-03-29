# WP2 Gate Checklist
## HDF5 Dataset Build

**Do not begin WP3 until every item is checked.**

## Step 2.1 — Schema Design
[x] schema.py defines H5_PATH, SCHEMA_VERSION, FIELD_DEFS
[x] All 13 confirmed feature fields included (excludes 302, 304, 401)
[x] crystallinity marked Batch A only in schema

## Step 2.2 — Dataset Build
[x] build_dataset.py --dry-run completes with 0 errors for all 66 sims
[x] build_dataset.py full run completes without exceptions
[x] data/cfwrinkle_dataset.h5 written (check file size > 500 MB)
[x] 66 sim groups present in simulations/
[x] splits/ group present with 5 folds

## Step 2.3 — Validation
[x] validate_dataset.py: all REQUIRED_FINE_FIELDS present for all 66 sims
[x] validate_dataset.py: zero NaN/Inf in displacement, fiber_stress_1, stress
[x] validate_dataset.py: node count spot-check passes (3 A + 3 B sims)
[x] validate_dataset.py: increment count spot-check passes
[x] validate_dataset.py: compound_severity attrs match registry JSON
[x] validate_dataset.py: split integrity check passes (no train/val/test overlap)

## Step 2.4 — Manual Checks
[x] h5py.File('data/cfwrinkle_dataset.h5')['simulations'].keys() shows all 66
[x] mold_set_004 compound_severity attr == 0.167 (lowest severity, borderline clean)
[x] geom_0_8 compound_severity attr == 0.473 (anomalous Batch A case)
[x] Batch B crystallinity dataset has shape[1] == 0 and attr available=False
[x] At least 1 Batch A and 1 Batch B fiber_stress_1 final-increment array plotted

## Sign-off
[x] WP2_gate.md reviewed
[x] data/cfwrinkle_dataset.h5 size and content verified
[x] WP3 (feature extraction) may begin

