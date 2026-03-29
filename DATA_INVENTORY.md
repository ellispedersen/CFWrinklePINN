# Data Inventory
## Complete Dataset Survey — All Simulation Batches

---

## 1. Data Locations

Three directories hold simulation data. **sim batch 271125 is duplicated** between CFWrinklePredict2 and CFAIMesh — same timestamps, same file sizes, same content.

| Location | Contents | Canonical? |
|---|---|---|
| `CFWrinklePredict2/data/aniform_raw/` | 20 × geom 0_x (UD thermoplastic) | **yes** |
| `CFWrinklePredict2/data/simulation_batch_271125/` | 45 × mold set (Twintex 2×2 twill) | **yes** |
| `CFAIMesh/Aniform/` | Mirror of aniform_raw (same timestamps) | duplicate |
| `CFAIMesh/sim batch 271125/` | Mirror of simulation_batch_271125 | duplicate |
| `CFAIMesh/sim batch 261125/` | 1 × mold set 000 pilot only | **exclude** |
| `CFAIMesh/laminate setup.Results/` | 16+ development runs (no paired fine/coarse) | **exclude** |
| `CFAIMesh/testing setup.Results/` | 24+ development runs | **exclude** |

**Use only the canonical locations.** Do not read from CFAIMesh duplicates.

---

## 2. Batch A — UD Thermoplastic (geom 0_x)

**Material:** UD reinforced thermoplastic (FibreLD: E_fiber=25000 MPa, Orientation=0°)
**Material file:** `CFWrinklePredict2/data/aniform_raw/UD reinforced thermoplastic unitMPa 2025-04-14.afl`
**Tool geometry:** `flat_smooth_0` (die + punch, 3 sections)
**Plies:** 2 (Layer 0 @ 0°, Layer 1 @ 90°)
**Element groups:** Group 6 (ply 0, Tri3LDT), Group 10 (ply 1, Tri3LDT)
**Punch stroke:** 75 mm total (loadset 1: UZ=70mm, loadset 2: UZ=75mm in Z+)
**Thickness in .afi:** 0.3 mm — **do not use; derive from model_102_1.afr at t=0**
**AFR count per run:** 20 files
**Results directory:** inside `<run>/results/`
**Mesh directory:** `<run>/meshes/`

### Pair Inventory

| Sim ID | Fine run | Coarse run | Fine KB | Coarse KB | AFR | Status | Notes |
|---|---|---|---|---|---|---|---|
| geom_0_0 (pair 1) | 2025-08-31 13_12_15 | 2025-08-31 14_58_22 | 9310 | 454 | 20/20 | **use** | 4 runs total = 2 complete pairs. Pair 1 (1hr earlier) is independent variation, not a duplicate. |
| geom_0_0 (pair 2) | 2025-08-31 15_19_36 | 2025-08-31 15_20_12 | 9310 | 382 | 20/20 | **use** | Pair 2 was used by CFAIMesh Model Training project. Both valid — 2 separate simulation runs of same geom. |
| geom_0_1 | 2025-08-31 15_25_28 | 2025-08-31 15_26_01 | 9310 | 382 | 20/20 | **use** | 3rd run (17_24_01): 0 AFR, no .afs — FAILED, skip |
| geom_0_2 | 2025-08-31 15_28_04 | 2025-08-31 15_28_28 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_3 | 2025-08-31 15_29_28 | 2025-08-31 15_29_49 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_4 | 2025-08-31 15_31_29 | 2025-08-31 15_31_53 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_5 | 2025-08-31 15_33_37 | 2025-08-31 15_34_13 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_6 | 2025-08-31 15_36_27 | 2025-08-31 15_36_47 | 9310 | 0.4 | 20/20 | **⚠ verify** | "coarse" ply mesh is 436 bytes (not a real mesh file); AFR files present and non-zero. Likely: coarse mesh was not exported to .msh but results still computed. Run ReadAFMesh on model.afm to confirm node count. |
| geom_0_7 | 2025-08-31 15_37_57 | 2025-08-31 15_38_15 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_8 | 2025-08-31 15_39_31 | 2025-08-31 15_39_49 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_9 | 2025-08-31 15_41_00 | 2025-08-31 15_41_31 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_10 | 2025-08-31 15_42_42 | 2025-08-31 15_43_05 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_11 | 2025-08-31 15_44_04 | 2025-08-31 15_44_22 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_12 | 2025-08-31 15_46_28 | 2025-08-31 15_46_42 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_13 | 2025-08-31 15_47_39 | 2025-08-31 15_47_58 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_14 | 2025-08-31 15_49_15 | 2025-08-31 15_49_33 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_15 | 2025-08-31 15_50_34 | 2025-08-31 15_50_52 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_16 | 2025-08-31 15_52_02 | 2025-08-31 15_52_15 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_17 | 2025-08-31 15_53_37 | 2025-08-31 15_53_54 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_18 | 2025-08-31 15_55_14 | 2025-08-31 15_55_28 | 9310 | 382 | 20/20 | **use** | clean |
| geom_0_19 | 2025-08-31 15_56_28 | 2025-08-31 15_56_43 | 9310 | 382 | 20/20 | **use** | clean |

**Batch A summary:** geom_0_0 has 2 complete coarse+fine pairs (21 total Batch A pairs). geom_0_1 third run excluded (0 AFRs, never started). geom_0_6 coarse mesh pending verification. All others clean.

---

## 3. Batch B — Twintex 2×2 Twill (mold sets 000–039 + 100–119)

**Material:** Twintex GF-PP unconsolidated 2×2 twill 1485 gsm
**Material file:** `CFWrinklePredict2/data/simulation_batch_271125/Twintex GF-PP-unconsolidated-2x2twill-1485gsm fromLiterature RT unitMPA 2025-04-14.afl`
**Tool geometry:** `composite_plate_NNN` (upper_closed + lower_closed — closed die forming)
**Plies:** 3 (Layers 0°, 90°, 0° — or similar via localcs 1, 2, 3)
**Element groups:** Groups 6, 10, 14 (approximate — verify from AFI)
**Thickness in .afi:** 0.85 mm — **do not use; derive from model_102_1.afr at t=0**
**AFR count per run:** 19 files (no model_214_1.afr — that file is UD-only)
**Coarse mesh:** 1252 KB per ply, **Fine mesh:** 4984 KB per ply
**Results directory:** `<run>/results/` (same structure as Batch A — confirmed from directory listing)
**Mesh directory:** `<run>/meshes/`

### Two Forming Depths Tested

The 000–039 and 100–119 series use the **same geometry tool shapes** but **different punch depths**:
- **0-series:** tool 2 (upper die) UZ ≈ −12.048 mm (deeper)
- **100-series:** tool 2 (upper die) UZ ≈ −10.058 mm (shallower)
- tool 1 (lower die) UZ = −6.2461 mm for both

These are **independent experiments** — each mold set folder is its own coarse/fine pair. They are NOT paired with each other. The 100-series geometry tag maps: ms=100 → composite_plate_000, ms=104 → composite_plate_004, ms=106 → composite_plate_006, etc.

### Mold Set Inventory — 0-Series (28 sets, depth ≈ −12mm)

| Mold Set | Geometry | Depth t2 (mm) | Coarse KB | Fine KB | AFR | Status |
|---|---|---|---|---|---|---|
| 000 | composite_plate_000 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 001 | composite_plate_001 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 002 | composite_plate_002 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 004 | composite_plate_004 | −5.994 | 1252 | 4984 | 19/19 | **use** |
| 006 | composite_plate_006 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 008 | composite_plate_008 | (see note) | 1252 | 4984 | 19/19 | **use** |
| 010 | composite_plate_010 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 011 | composite_plate_011 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 012 | composite_plate_012 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 013 | composite_plate_013 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 014 | composite_plate_014 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 015 | composite_plate_015 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 016 | composite_plate_016 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 017 | composite_plate_017 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 019 | composite_plate_019 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 020 | composite_plate_020 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 021 | composite_plate_021 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 022 | composite_plate_022 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 023 | composite_plate_023 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 024 | composite_plate_024 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 025 | composite_plate_025 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 028 | composite_plate_028 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 029 | composite_plate_029 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 030 | composite_plate_030 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 031 | composite_plate_031 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 034 | composite_plate_034 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 037 | composite_plate_037 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 039 | composite_plate_039 | −12.048 | 1252 | 4984 | 19/19 | **use** |

Note: "force ctrl" = UZ line present but no magnitude (tool motion prescribed via RFZ force tracking, not displacement). These are force-controlled forming steps — valid, just different BC type.

### Mold Set Inventory — 100-Series (17 sets, depth ≈ −10mm)

| Mold Set | Geometry | Depth t2 (mm) | Coarse KB | Fine KB | AFR | Status |
|---|---|---|---|---|---|---|
| 100 | composite_plate_000 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 102 | composite_plate_002 | −10.058 | 1252 | 4984 | 19/19 | **use** |
| 103 | composite_plate_003 | −10.058 | 1252 | 4984 | 19/19 | **use** |
| 104 | composite_plate_004 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 105 | composite_plate_005 | −10.058 | 1252 | 4984 | 19/19 | **use** |
| 106 | composite_plate_006 | −10.058 | 1252 | 4984 | 19/19 | **use** |
| 107 | composite_plate_007 | −10.058 | 1252 | 4984 | 19/19 | **use** |
| 108 | composite_plate_008 | −10.058 | 1252 | 4984 | 19/19 | **use** |
| 111 | composite_plate_011 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 112 | composite_plate_012 | −10.058 | 1252 | 4984 | 19/19 | **use** |
| 113 | composite_plate_013 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 114 | composite_plate_014 | −10.058 | 1252 | 4984 | 19/19 | **use** |
| 115 | composite_plate_015 | −10.058 | 1252 | 4984 | 19/19 | **use** |
| 116 | composite_plate_016 | −10.058 | 1252 | 4984 | 19/19 | **use** |
| 117 | composite_plate_017 | −12.048 | 1252 | 4984 | 19/19 | **use** |
| 118 | composite_plate_018 | (force ctrl) | 1252 | 4984 | 19/19 | **use** |
| 119 | composite_plate_019 | −10.058 | 1252 | 4984 | 19/19 | **use** |

**Batch B summary:** 45 pairs, all complete (19/19 AFR, matching coarse/fine sizes), no failures detected.

### Missing Mold Set Numbers (Excluded / Never Run)

These geometry IDs have no .Results folder — they were excluded from the batch:
- **0-series missing:** 003, 005, 007, 009, 018, 026, 027, 032, 033, 035, 036, 038
- **100-series missing:** 101, 109, 110

These gaps are intentional — likely geometries that failed pre-screening, were duplicates, or were excluded for quality reasons.

---

## 4. Key Structural Differences Between Batches

| Property | Batch A (geom 0_x, UD) | Batch B (mold sets, Twintex) |
|---|---|---|
| Material | UD thermoplastic | 2×2 twill GF-PP |
| Plies | 2 | 3 |
| Tool type | Open punch-die (flat_smooth) | Closed die (composite_plate) |
| Forming direction | Z+ (punch up) | Z− (die closing down) |
| Tool 2 depth | N/A (die fixed) | −10.058 to −12.048 mm |
| Tool 1 depth (blank holder/lower) | −6.2461 mm | −6.2461 mm (same) |
| AFR file count | 20 | 19 (no model_214_1.afr) |
| Results location | `<run>/results/` | `<run>/results/` (same structure) |
| Ply mesh coarse size | ~382–453 KB | 1252 KB |
| Ply mesh fine size | ~9310 KB | 4984 KB |
| Locking angle relevant | No (UD — buckling, not locking) | Yes (~54° for GF-PP twill) |
| Shear angle field (106) | May be degenerate (near zero) | Key feature |

---

## 5. Thickness — Derive from Simulation Data

**Do not use thickness values from .afi or .afl files.** The .afi `thickness` keyword and .afl `Thickness` property may not match what the solver actually used.

**Correct approach:** Read `model_102_1.afr` (res_id=102, ScalarT) at increment 0 for each run. Average across all ply nodes to get the actual simulated ply thickness. This is what the solver computed with.

Expected values (not authoritative — verify from data):
- Batch A (UD): .afi says 0.3 mm, .afl says 0.15 mm — actual value TBD from model_102
- Batch B (Twintex): .afi says 0.85 mm — actual value TBD from model_102

---

## 6. Confirmed Failed / Excluded Runs

| Sim | Run | Reason | Action |
|---|---|---|---|
| ~~geom_0_0 first pair~~ | ~~13_12_15 + 14_58_22~~ | ~~Previously assumed duplicate~~ | **Reinstated** — both pairs are independent simulation runs; use both |
| geom_0_1 | 2025-08-31 17_24_01 | 0 AFR files, no .afs, no model.out — simulation never started | **Skip** — pair is still valid using earlier 2 runs |
| geom_0_6 | 2025-08-31 15_36_47 (coarse) | Ply .msh file is 436 bytes (not a real mesh export) but .afr files present; model.afm may still contain valid mesh | **Verify** via ReadAFMesh; if model.afm readable, use this run |
| sim batch 261125 / mold set 000 | all | Pilot test — only 1 mold set, run before main batch | **Exclude entirely** |
| CFAIMesh/laminate setup + testing setup | all | Development setup runs, no consistent coarse/fine pairing | **Exclude entirely** |

---

## 7. Final Dataset Count (Confirmed Usable)

| Batch | Pairs | Status |
|---|---|---|
| Batch A (UD, geom 0_x) | 20 confirmed + 1 extra (geom_0_0 pair 1) + 1 pending (geom_0_6) | 21–22 |
| Batch B (Twintex 0-series) | 28 | 28 |
| Batch B (Twintex 100-series) | 17 | 17 |
| **Total** | **66–67** | |

This is substantially larger than the 20 previously assumed. The WP plan's estimate of 40 pairs was wrong — we have ~66 usable pairs across two materials and two geometry families.

---

## 8b. Aborted 3rd Batch (No Usable Data)

A 3rd batch was planned and partially executed but all runs were aborted before producing AFR results:

- **`CFAIMesh/SIM Batch Final Aniform.zip`** (666 MB compressed): 82 .Results dirs, simulations of `NNN .45mm / .55mm / .85mm` — ply thickness variations on composite_plate geometries. All results directories contain only `.tmp` warmup files — never reached forming increment.
- **`CFAIMesh/Updated 40x2 sim set.zip`** (523 MB compressed): 80 .Results dirs, 40 composite_plate geometries × 2 thickness variants (`.5mm` + `1mm`). Same issue — warmup `.tmp` files only, no `.afr`.

**These archives contain zero usable simulation output.** Do not attempt to parse. The intended 3rd batch (40 geometry × 2 thickness = 80 pairs with ply thickness as a physics parameter) would have been valuable but was never completed.

---

## 8. Revised WP5 Stratification Labels

With 65 pairs, 5-fold CV gives ~13 val sims per fold — much more robust than the 4 initially expected.

Stratification dimensions:
- `material`: `UD` | `twill_2x2`
- `geometry_family`: `flat_smooth` | `composite_plate`
- `forming_depth`: `shallow` (≈10mm) | `deep` (≈12mm) | `ud_75mm` (UD flat punch)
- `wrinkle_outcome`: `clean` | `wrinkled` (to be determined from fine mesh)

The `composite_plate_NNN` geometry ID is the key geometry distinguisher within Batch B — same ID at two depths gives a natural experiment pair. Keep both depths of the same geometry in the same fold if possible.
