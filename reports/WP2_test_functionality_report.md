# WP2 Test and Functionality Report

## Summary

WP2 is implemented, validated, and test-covered for both functional behavior and WP1↔WP2 integration.

- Dataset build pipeline implemented under `wp2_build/`
- Dataset validation pipeline implemented under `wp2_build/validate_dataset.py`
- Gate checklist completed in `reports/WP2_gate.md`
- Test stack added under `tests/`

## Functional Deliverables

- `wp2_build/schema.py`
  - Defines `H5_PATH`, `SCHEMA_VERSION`, `FIELD_DEFS`, `SimRecord`, chunking and CV constants.
- `wp2_build/build_dataset.py`
  - Loads WP1 registries and pipeline config.
  - Builds 66-simulation catalog (excluding aborted `geom_0_1_pair2`).
  - Reads AniForm mesh/result/solution files.
  - Writes HDF5 dataset with metadata, fine/coarse groups, derived fields, and CV splits.
- `wp2_build/validate_dataset.py`
  - Verifies dataset integrity:
    - simulation count
    - required fields
    - NaN/Inf checks
    - node/increment spot checks
    - severity alignment to WP1 registry
    - split integrity
    - Batch B crystallinity handling

## Data and Validation Results

- Build dry-run (`--dry-run`): **passed** for all 66 sims (0 errors).
- Full build: **passed** and produced:
  - `data/cfwrinkle_dataset.h5`
  - size: `68,295,180,279` bytes
- Validator (`python -m wp2_build.validate_dataset --verbose`): **all checks passed**.

Manual spot checks:
- `mold_set_004` severity: `0.1671279` (expected ~0.167)
- `geom_0_8` severity: `0.4732135` (expected ~0.473)
- Batch B crystallinity: shape `(44, 0)`, `available=False`
- Final increment `fiber_stress_1` compression present in Batch A and Batch B samples

## Test Stack

### Unit tests

File: `tests/test_wp2_unit.py`

Coverage:
- `_severity_from_components` behavior (clamping and weighting)
- `compute_derived` array shapes and expected metric values
- `write_splits` invariants:
  - train/val/test disjointness per fold
  - union coverage equals full simulation set

### WP1↔WP2 integration tests

File: `tests/test_wp1_wp2_integration.py`

Coverage:
- HDF5 artifact presence and minimum size check
- Sim-ID alignment with `reports/wrinkle_onset_registry.json` (excluding aborted case)
- Severity anchor checks for known reference simulations
- Full validator check-suite execution against built dataset

## Latest Test Execution

Command:

```powershell
python -m pytest tests\test_wp2_unit.py tests\test_wp1_wp2_integration.py -q
```

Result:
- **7 passed**, **0 warnings**

## Additional Stability Fix Applied

To remove warning noise in test runs:
- Updated `io/aniform_readers/ReadAFSFile.py`
  - replaced deprecated `datetime.datetime.utcfromtimestamp(...)`
  - with timezone-aware `datetime.datetime.fromtimestamp(..., datetime.UTC)`

## Conclusion

WP2 functionality is complete and validated, and the WP2 test stack is green. The codebase is ready for WP3 feature extraction kickoff.

