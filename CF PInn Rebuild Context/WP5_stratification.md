# WP5 — Stratification & Cross-Validation Setup
## Label Scheme, Fold Construction, Pre-Flight Composition Report

**Depends on:** WP4 gate passed, all 40 pairs have features and targets written
**Feeds into:** WP6 (uses fold_assignments.yaml to understand training set size), WP7 (training loop reads folds)

---

## Objective

Assign stratification labels to all 40 pairs, construct 5 balanced folds, and produce a composition report that must be reviewed before any training run begins. This WP has no model code — it is purely data organisation.

---

## Inputs Received from WP4

| Artifact | Used For |
|---|---|
| `data/dataset.h5` | Read pair metadata (material, geometry_type, wrinkle_outcome) |
| `reports/input_param_registry.json` | geometry_type, forming_severity for stratification |
| `reports/wrinkle_onset_registry.json` | wrinkle_outcome labels |

---

## Outputs Produced by This WP

| Artifact | Location | Consumer |
|---|---|---|
| `fold_assignments.yaml` | `config/fold_assignments.yaml` | WP7 training loop |
| `fold_composition_report.txt` | `reports/fold_composition_report.txt` | Manual review gate |
| `WP5_gate.md` | `reports/WP5_gate.md` | Gate review |

---

## 5.1 Stratification Label Scheme

Each pair receives a 3-tuple label: `(material, geometry_complexity, wrinkle_outcome)`

| Dimension | Values |
|---|---|
| material | `UD` \| `twill_2x2` |
| geometry_complexity | `simple` (single feature type) \| `mixed` (2+ feature types) |
| wrinkle_outcome | `clean` \| `wrinkled` |

This gives a maximum of 8 cells. With 40 pairs some cells will have fewer than 5 entries — flag any with fewer than 5 and handle manually.

```python
# training/stratification.py

def assign_labels(pair_ids: list[str],
                  input_registry: list[dict],
                  wrinkle_registry: list[dict]) -> list[dict]:
    """
    Returns list of dicts with keys:
        pair_id, material, geometry_complexity, wrinkle_outcome, stratum_key
    stratum_key is the 3-tuple string used for fold balancing.
    """
    registry_by_id  = {r['sim_id']: r for r in input_registry}
    wrinkle_by_id   = {r['pair_id']: r for r in wrinkle_registry}

    labels = []
    for pid in pair_ids:
        meta    = registry_by_id[pid]
        wrinkle = wrinkle_by_id[pid]

        geometry_complexity = (
            'mixed' if meta['geometry_type'] == 'mixed' else 'simple'
        )
        wrinkle_outcome = wrinkle['wrinkle_outcome']   # 'wrinkled' | 'clean'
        material        = meta['material']             # 'UD' | 'twill_2x2'

        labels.append({
            'pair_id':             pid,
            'material':            material,
            'geometry_complexity': geometry_complexity,
            'wrinkle_outcome':     wrinkle_outcome,
            'stratum_key':         f"{material}__{geometry_complexity}__{wrinkle_outcome}",
        })
    return labels
```

---

## 5.2 Fold Construction

5-fold stratified cross-validation. Stratification is at the simulation level — entire simulations are held out, never individual timesteps.

```python
def build_stratified_folds(labels: list[dict],
                            n_folds: int = 5,
                            seed: int = 42) -> list[dict]:
    """
    Returns list of n_folds dicts:
        {'fold': int, 'train': list[str], 'val': list[str]}

    Strategy:
    - Group pairs by stratum_key
    - Distribute each stratum across folds as evenly as possible
    - Use seed for reproducibility
    - Flag strata with < n_folds pairs (can't guarantee 1 per fold)

    Mixed-geometry cases treated as their own stratum regardless of
    material or wrinkle_outcome, since geometry type is the dominant
    source of variation for these cases.
    """
    import random
    random.seed(seed)

    strata = {}
    for label in labels:
        key = label['stratum_key']
        strata.setdefault(key, []).append(label['pair_id'])

    # Flag thin strata
    for key, members in strata.items():
        if len(members) < n_folds:
            print(f"WARNING: stratum '{key}' has only {len(members)} pairs "
                  f"(<{n_folds}) — cannot guarantee 1 per fold")

    folds = [{'fold': i, 'train': [], 'val': []} for i in range(n_folds)]

    for key, members in strata.items():
        random.shuffle(members)
        for i, pid in enumerate(members):
            folds[i % n_folds]['val'].append(pid)

    # Train is everything not in val for that fold
    all_ids = [label['pair_id'] for label in labels]
    for fold in folds:
        fold['train'] = [pid for pid in all_ids if pid not in fold['val']]

    return folds
```

---

## 5.3 Pre-Flight Composition Report

**This report must be printed and reviewed before any training run.** It catches degenerate folds before wasted compute.

```python
def print_composition_report(folds: list[dict],
                              labels: list[dict],
                              output_path: str) -> None:
    """
    Print and write per-fold composition:
    - Material distribution in val set
    - Geometry type distribution in val set
    - Wrinkle/clean ratio in val set
    - Flags: folds with no wrinkled cases, no clean cases, single material only
    """
    label_by_id = {l['pair_id']: l for l in labels}
    lines = []

    for fold in folds:
        val_labels = [label_by_id[pid] for pid in fold['val']]
        lines.append(f"\n=== Fold {fold['fold']} | val={len(fold['val'])} train={len(fold['train'])} ===")

        mat_counts    = _count_attr(val_labels, 'material')
        geom_counts   = _count_attr(val_labels, 'geometry_complexity')
        outcome_counts = _count_attr(val_labels, 'wrinkle_outcome')

        lines.append(f"  Material:    {mat_counts}")
        lines.append(f"  Geometry:    {geom_counts}")
        lines.append(f"  Outcome:     {outcome_counts}")

        # Flags
        if 'wrinkled' not in outcome_counts:
            lines.append("  !! FLAG: No wrinkled cases in validation set")
        if 'clean' not in outcome_counts:
            lines.append("  !! FLAG: No clean cases in validation set")
        if len(mat_counts) == 1:
            lines.append(f"  !! FLAG: Single material in validation set ({list(mat_counts.keys())[0]})")

    report = '\n'.join(lines)
    print(report)
    with open(output_path, 'w') as f:
        f.write(report)
```

---

## fold_assignments.yaml Format

```yaml
# config/fold_assignments.yaml
# Generated by training/stratification.py
# Do not edit manually — re-run stratification.py if pair labels change

version: "1.0"
n_folds: 5
seed: 42
generated_at: "2025-XX-XX"

folds:
  - fold: 0
    train: [pair_001, pair_002, pair_003, ...]   # 32 pairs
    val:   [pair_000, pair_012, ...]             # 8 pairs

  - fold: 1
    train: [...]
    val:   [...]

  # ... folds 2, 3, 4

labels:
  - pair_id: pair_000
    material: UD
    geometry_complexity: simple
    wrinkle_outcome: wrinkled
    stratum_key: "UD__simple__wrinkled"
  # ... all 40 pairs
```

---

## Commands

```bash
# Build fold assignments and print composition report
python -m training.stratification \
    --dataset data/dataset.h5 \
    --input-registry reports/input_param_registry.json \
    --wrinkle-registry reports/wrinkle_onset_registry.json \
    --n-folds 5 \
    --seed 42 \
    --output config/fold_assignments.yaml \
    --report reports/fold_composition_report.txt

# Print composition report from existing assignments (before any training run)
python -m training.stratification \
    --report-only \
    --assignments config/fold_assignments.yaml \
    --output reports/fold_composition_report.txt

# Check cell sizes in stratification
python -c "
import yaml
assignments = yaml.safe_load(open('config/fold_assignments.yaml'))
from collections import Counter
strata = Counter(l['stratum_key'] for l in assignments['labels'])
for key, count in sorted(strata.items()):
    flag = '!! <5' if count < 5 else ''
    print(f'{count:3d}  {key}  {flag}')
"
```

---

## WP5 Validation Gate

```
Stratification
  [ ] All 40 pairs have a stratum_key assigned
  [ ] No pair missing material, geometry_complexity, or wrinkle_outcome
  [ ] Strata with <5 pairs flagged and documented in WP5_gate.md

Fold Construction
  [ ] fold_assignments.yaml written with n_folds=5
  [ ] Each fold has exactly ~8 pairs in val and ~32 in train (allow ±1)
  [ ] No pair appears in both train and val within any fold
  [ ] Seed and version recorded in fold_assignments.yaml

Composition Report
  [ ] fold_composition_report.txt written
  [ ] Report reviewed manually — all !! FLAG lines understood and accepted
  [ ] At least 1 wrinkled case in val for every fold (hard requirement)

Handoff Check
  [ ] WP6 author has read fold_composition_report.txt
  [ ] Training set effective size (~32 pairs) confirmed as input to architecture decisions in WP6
```

---

## Handoff to WP6

WP6 receives from WP5:

| Artifact | Path | Notes |
|---|---|---|
| `fold_assignments.yaml` | `config/fold_assignments.yaml` | WP7 training loop reads this |
| `fold_composition_report.txt` | `reports/fold_composition_report.txt` | WP6 must read before architecture decisions |
| `data/dataset.h5` (complete) | `data/dataset.h5` | WP6 dataloader reads features and targets |

**Note for WP6:** Effective training set size is ~32 pairs per fold. Every capacity decision — hidden_dim, n_message_passing_steps, n_attention_heads, dropout — must be justified against a dataset of this size.
