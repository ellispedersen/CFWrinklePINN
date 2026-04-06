# WP1-WP3 Verification and Test Plan

## Scope
- Verify deliverables and data integrity for completed work packages:
  - WP1: survey/registry artifacts
  - WP2: dataset build + validation + integration
  - WP3: feature build + validation, including corrupted-run skip handling

## Acceptance Criteria

### WP1
- Required survey outputs exist:
  - `reports/afr_field_survey.json`
  - `reports/field_survey_detail.json`
  - `reports/force_stroke_summary.json`
  - `reports/input_param_registry.json`
  - `reports/wrinkle_onset_registry.json`
- Registry consistency:
  - `wrinkle_onset_registry.json` has 66 sims (including aborted id)
  - `input_param_registry.json` has expected parsed records

### WP2
- `python -m wp2_build.validate_dataset --verbose` passes all checks
- Pytests pass:
  - `tests/test_wp2_unit.py`
  - `tests/test_wp1_wp2_integration.py`
- HDF5 integrity:
  - Coarse/fine mesh elements are non-empty for representative sims
  - Increment alignment is consistent with field data

### WP3
- `python -m wp3_features.build_features` completes (with skip-safe behavior)
- `python -m wp3_features.validate_features --verbose` passes
- Pytest passes:
  - `tests/test_wp3_unit.py`
- Corrupted-run handling is explicitly verified:
  - Metadata includes `n_input_simulations`, `n_simulations`, `n_skipped`
  - `skipped_sim_ids` and `skipped_details_json` are present
  - Pre-fallback mapping quality is tracked (`pre_fallback_coverage`)

## Execution Steps
1. Run WP1 artifact sanity script (file presence + registry counts).
2. Run WP2 validator and WP2-related tests.
3. Run WP3 full build and validator.
4. Read WP3 metadata and confirm skip accounting fields.
5. Run consolidated pytest suite for WP1/WP2/WP3 tests.
6. Publish execution report with pass/fail and skip statistics.

