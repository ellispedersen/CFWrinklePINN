# WP3 Gate Checklist

## Build and Validation

- [x] WP2 rebuild completed before WP3 feature extraction
- [x] `wp3_features` package created with graph, temporal, correspondence, physics, targets, material, build, and validate modules
- [x] `python -m wp3_features.build_features` completes and writes `data/cfwrinkle_wp3_features.h5`
- [x] `python -m wp3_features.validate_features` passes
- [x] `pytest tests/test_wp3_unit.py -v` passes

## Data Quality Checks

- [x] Graph edge index/attributes written for all simulations
- [x] Resampled stroke grid fixed to 256 points in `[0, 1]`
- [x] Per-timestep wrinkle targets generated from fine-to-coarse mapping
- [x] Material cards normalised and statistics stored in metadata
- [x] Corrupted simulation runs are skipped safely and recorded in WP3 metadata (`skipped_sim_ids`, `skipped_details_json`)
- [x] Skip criteria calibrated to avoid false positives (final full build: built=65, skipped=1 known-corrupt sim)

## Sign-off

- [x] WP3 feature dataset built and validated
- [x] Ready to proceed to WP4/WP6 model and training stages

