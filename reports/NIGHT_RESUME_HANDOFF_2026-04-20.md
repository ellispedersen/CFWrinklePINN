# Night Resume Handoff (2026-04-20)

## Status
- Track B Level 3 passed and produced gate artifact (`reports/wp7_gate_level3_cross_scale.json`).
- Track B Level 4 run is intentionally paused for daytime.

## Saved recovery state
- `fold_0`: complete at epoch 50 with `latest.pt`, `best.pt`, `history.json`.
- `fold_1`: partial at epoch 21 with `latest.pt`, `best.pt`, `history.json`.
- Fine-input normalization persisted (`fine_input/normalized=1.0`).

## Resume command
```bash
AUTO_RESUME=1 bash run_cross_scale_level4.sh
```

## Keep fixed constraints
- `MAX_TIMESTEPS >= 96`
- fine-feature normalization ON

## Metric interpretation note
- `detection_rate` is simulation-level recall and saturates to `1.0` on all-positive folds.
- Use wrinkle-extent diagnostics for quality tracking:
  - coarse: `mean_wrinkled_frac_mae`
  - fine: `fine/wrinkled_match_rate`, `fine/wrinkled_frac_mae`
