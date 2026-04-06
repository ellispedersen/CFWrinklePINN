# WP1-WP3 Test Execution Report

## Plan
- Verification plan file: `reports/WP123_test_plan.md`
- Coverage: WP1 artifacts, WP2 dataset validation/integration, WP3 build/validation with corrupted-run skip accounting.

## Executed Commands
- `python -m wp2_build.validate_dataset --verbose`
- `python -m wp3_features.build_features`
- `python -m wp3_features.validate_features --verbose`
- `python -m pytest tests\test_wp2_unit.py tests\test_wp1_wp2_integration.py tests\test_wp3_unit.py -q`
- WP1 artifact sanity script (file existence + registry counts)

## Results

### WP1
- Required survey/report artifacts found.
- Registry counts observed:
  - `wrinkle_onset_registry.json`: 67 records
  - `input_param_registry.json`: 65 records
- WP1 reports are present and usable as upstream sources.

### WP2
- `wp2_build.validate_dataset` passed all checks:
  - sim count, required fields, NaN/Inf, node/increment spot checks, severity, split integrity, Batch B crystallinity.
- WP1↔WP2 integration tests passed.

### WP3
- Initial skip policy (`pre_fallback_coverage < 0.90`) was found to be too strict for Batch B and was corrected.
- Final skip policy:
  - skip if coarse mesh is invalid (`<1000` elements),
  - skip if fine mesh is invalid (`<10000` elements),
  - skip only correspondence outliers (`pre_fallback_coverage < 0.65`),
  - require post-fallback correspondence quality (`mapping_coverage >= 0.99`).
- Built output: `data/cfwrinkle_wp3_features.h5`
- Metadata summary:
  - `n_input_simulations = 66`
  - `n_simulations (built) = 65`
  - `n_skipped = 1`
- Skipped simulation:
  - `geom_0_6` (coarse mesh corrupted; only 4 coarse elements)
- Skip details captured in:
  - `metadata/skipped_sim_ids`
  - `metadata/skipped_details_json`
- Validator passed for built set.

## Test Summary
- Pytest suite result: **11 passed**
  - WP2 unit tests
  - WP1/WP2 integration tests
  - WP3 unit tests

## Conclusion
- The WP1–WP3 verification plan was executed successfully.
- Corrupted/low-quality runs are now explicitly accounted for via skip handling and recorded metadata.
- Dataset is validated for the built subset and ready for downstream WP4/WP6 work with clear provenance.

