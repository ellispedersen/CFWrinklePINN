"""
WP1.5 — Wrinkle Onset Detection (Multi-Parameter)

Multi-parameter approach matching AniForm / CAE industry criteria using
confirmed field IDs from WP1 archaeology:

  1. Fiber stress 1 compressive fraction (model_206_1) — PRIMARY
     Physics: compressive fiber stress → Euler buckling → wrinkling.
     Field confirmed via ReadAFResult silent=False: "Fiber stress 1", ScalarT.

  2. Out-of-plane displacement variance (model_40_1) — SECONDARY
     Batch-specific normalisation (A: scale 10 mm², B: scale 0.4 mm²).

  3. Shear angle f1_f2 (model_204_1) — TERTIARY (Batch B woven only)
     Approaching locking angle (~45° for Twintex 2×2 twill) precedes wrinkling.

Compound severity score = 0.5 * S_fiber + 0.35 * S_var + 0.15 * S_shear
  where each S_x = clamp(metric / scale, 0, 1)

Onset: first increment where compound_severity > SEVERITY_ONSET_THRESHOLD

Classification:
  max_severity >= 0.50  → wrinkled (high confidence)
  max_severity >= 0.25  → wrinkled (medium confidence)
  max_severity >= 0.10  → wrinkled (low confidence / borderline)
  max_severity <  0.10  → clean

Usage:
    python -m validation.wrinkle_detector
    python -m validation.wrinkle_detector --pair geom_0_0 --batch A
    python -m validation.wrinkle_detector --batch B

Writes: reports/wrinkle_onset_registry.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "io"))
from aniform_readers.ReadAFResult import ReadAFResult
from aniform_readers.ReadAFSFile  import ReadAFSFile


# ── Config ────────────────────────────────────────────────────────────────────

BATCH_A_ROOT = Path(r"C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\aniform_raw")
BATCH_B_ROOT = Path(r"C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\simulation_batch_271125")
REPORT_OUT   = _ROOT / "reports" / "wrinkle_onset_registry.json"

PLY_GROUPS = {"A": [6, 10], "B": [4, 8, 12]}

# ── AFR field IDs (confirmed via ReadAFResult silent=False headers) ───────────
DISPLACEMENT_AFR_ID  = 40    # "Displacement", VectorT [dx, dy, dz]
FIBER_STRESS_AFR_ID  = 206   # "Fiber stress 1", ScalarT
SHEAR_ANGLE_AFR_ID   = 204   # "Shear angle f1_f2", ScalarT

# ── Fiber stress parameters ───────────────────────────────────────────────────
COMP_DEADBAND_MPA       = -0.05   # sigma_f1 below this is "compressive"
COMP_FRAC_SEVERITY_SCALE = 0.50   # comp_frac / scale → clamped to [0,1]
                                   # (expect max ~0.8; onset meaningful at ~0.25+)

# ── dz variance parameters (batch-specific) ───────────────────────────────────
DZ_COMPONENT_IDX       = 2        # column in [dx, dy, dz] (0-indexed, after node-idx col)
DZ_VAR_SEVERITY_SCALE  = {"A": 10.0, "B": 0.40}   # mm² — severity = 1.0 at this value
LOCAL_DEV_THRESHOLD    = 0.1      # mm — |dz - mean_dz| > this counts as locally deviated

# ── Shear angle parameters (Batch B woven only) ───────────────────────────────
SHEAR_ANGLE_SEVERITY_SCALE = {"A": 90.0, "B": 45.0}  # deg — UD: no locking; Twintex ~45-54°

# ── Compound severity weights ─────────────────────────────────────────────────
W_FIBER  = 0.50
W_VAR    = 0.35
W_SHEAR  = 0.15

# ── Classification thresholds ─────────────────────────────────────────────────
SEVERITY_ONSET_THRESHOLD = 0.15   # compound severity → onset detected
SEVERITY_HIGH_CONF       = 0.50
SEVERITY_MED_CONF        = 0.25
SEVERITY_LOW_CONF        = 0.10   # below → clean


# ── File helpers ──────────────────────────────────────────────────────────────

def find_coarse_and_fine(results_dir: Path):
    """Return (coarse_dir, fine_dir) by ply mesh size."""
    runs = [d for d in results_dir.iterdir()
            if d.is_dir() and (d / "meshes").exists()]
    def size(r):
        p = r / "meshes" / "Part.section 1.1.msh"
        return p.stat().st_size if p.exists() else 0
    valid = sorted([(r, size(r)) for r in runs if size(r) > 0], key=lambda x: x[1])
    if len(valid) < 2:
        return None, None
    return valid[0][0], valid[-1][0]


def find_afr_dir(run_dir: Path) -> Path:
    results = run_dir / "results"
    if results.is_dir() and any(results.glob("model_*.afr")):
        return results
    if any(run_dir.glob("model_*.afr")):
        return run_dir
    raise FileNotFoundError(f"No model_*.afr under {run_dir}")


def load_afs_times(run_dir: Path) -> dict:
    """Load increment timing from model.afs."""
    afs_path = run_dir / "results" / "model.afs"
    if not afs_path.exists():
        afs_path = run_dir / "model.afs"
    if not afs_path.exists():
        return {}
    try:
        _, _, IncrInfo, _, _, _, _ = ReadAFSFile(str(afs_path), silent=True)
        incr_nrs   = IncrInfo.loc['incr_nr'].values.astype(int)
        times      = IncrInfo.loc['t_end'].values.astype(float)
        loadblocks = IncrInfo.loc['current_loadblock'].values.astype(int)
        converged  = IncrInfo.loc['has_converged'].values.astype(bool)
        return {
            int(nr): {"time": float(t), "loadblock": int(lb), "converged": bool(c)}
            for nr, t, lb, c in zip(incr_nrs, times, loadblocks, converged)
        }
    except Exception:
        return {}


# ── Profile computation functions ─────────────────────────────────────────────

def compute_dz_variance_profile(afr_dir: Path, ply_groups: list) -> dict:
    """
    Read displacement (model_40_1.afr) and compute dz spatial variance per increment.
    Uses only the first ply group to avoid double-counting nodes.
    """
    disp_path = afr_dir / f"model_{DISPLACEMENT_AFR_ID}_1.afr"
    if not disp_path.exists():
        return {"error": f"model_{DISPLACEMENT_AFR_ID}_1.afr not found"}

    primary_group = ply_groups[0]
    ResultsIncr, _, Increments, _, _, _ = ReadAFResult(
        str(disp_path), elemGrNrs=[primary_group], IncsToExport=[], silent=True
    )

    unique_incrs = sorted(set(Increments))
    incr_numbers         = []
    variance_profile     = []
    affected_frac_profile = []

    for incr in unique_incrs:
        if primary_group not in ResultsIncr.get(incr, {}):
            continue
        data = ResultsIncr[incr][primary_group]   # (N, 4): [idx, dx, dy, dz]
        dz   = data[:, DZ_COMPONENT_IDX + 1]

        spatial_var = float(np.var(dz))
        mean_dz     = float(np.mean(dz))
        local_dev   = np.abs(dz - mean_dz)
        affected    = float((local_dev > LOCAL_DEV_THRESHOLD).sum() / max(len(dz), 1))

        incr_numbers.append(int(incr))
        variance_profile.append(spatial_var)
        affected_frac_profile.append(affected)

    return {
        "incr_numbers":       incr_numbers,
        "dz_variance":        variance_profile,
        "affected_fraction":  affected_frac_profile,
        "n_increments":       len(incr_numbers),
    }


def compute_fiber_stress_profile(afr_dir: Path, ply_groups: list) -> dict:
    """
    Read fiber stress 1 (model_206_1.afr) and compute per-increment compressive fraction.

    "Fiber stress 1" is ScalarT — shape (N, 2) per group: [node_idx, sigma_f1_MPa].
    Compressive fraction = fraction of ply nodes with sigma_f1 < COMP_DEADBAND_MPA.
    Uses all ply groups to cover both plies (groups may have non-overlapping node sets).
    """
    fstress_path = afr_dir / f"model_{FIBER_STRESS_AFR_ID}_1.afr"
    if not fstress_path.exists():
        return {"error": f"model_{FIBER_STRESS_AFR_ID}_1.afr not found"}

    ResultsIncr, _, Increments, _, _, _ = ReadAFResult(
        str(fstress_path), elemGrNrs=ply_groups, IncsToExport=[], silent=True
    )

    unique_incrs = sorted(set(Increments))
    incr_numbers     = []
    comp_frac_profile = []
    mean_stress_profile = []
    min_stress_profile  = []

    for incr in unique_incrs:
        all_stress = []
        for grp in ply_groups:
            if grp not in ResultsIncr.get(incr, {}):
                continue
            data = ResultsIncr[incr][grp]
            if data.ndim == 1:
                # Fallback: flat array — treat as single component after index col
                continue
            # ScalarT: col 0 = node index, col 1 = value
            n_cols = data.shape[1]
            val_col = 1 if n_cols >= 2 else 0
            all_stress.append(data[:, val_col])

        if not all_stress:
            continue

        stress = np.concatenate(all_stress)
        comp_frac = float((stress < COMP_DEADBAND_MPA).sum() / max(len(stress), 1))
        comp_frac_profile.append(comp_frac)
        mean_stress_profile.append(float(stress.mean()))
        min_stress_profile.append(float(stress.min()))
        incr_numbers.append(int(incr))

    return {
        "incr_numbers":     incr_numbers,
        "comp_frac":        comp_frac_profile,
        "mean_stress_MPa":  mean_stress_profile,
        "min_stress_MPa":   min_stress_profile,
        "n_increments":     len(incr_numbers),
    }


def compute_shear_angle_profile(afr_dir: Path, ply_groups: list) -> dict:
    """
    Read shear angle f1_f2 (model_204_1.afr) for Batch B woven wrinkle detection.

    "Shear angle f1_f2" is ScalarT (degrees). Maximum absolute shear angle across
    all ply nodes per increment indicates inter-fiber shear approaching locking.
    """
    sa_path = afr_dir / f"model_{SHEAR_ANGLE_AFR_ID}_1.afr"
    if not sa_path.exists():
        return {"error": f"model_{SHEAR_ANGLE_AFR_ID}_1.afr not found"}

    ResultsIncr, _, Increments, _, _, _ = ReadAFResult(
        str(sa_path), elemGrNrs=ply_groups, IncsToExport=[], silent=True
    )

    unique_incrs = sorted(set(Increments))
    incr_numbers      = []
    max_angle_profile = []
    mean_angle_profile = []

    for incr in unique_incrs:
        all_angles = []
        for grp in ply_groups:
            if grp not in ResultsIncr.get(incr, {}):
                continue
            data = ResultsIncr[incr][grp]
            if data.ndim < 2 or data.shape[1] < 2:
                continue
            all_angles.append(np.abs(data[:, 1]))   # absolute shear angle

        if not all_angles:
            continue

        angles = np.concatenate(all_angles)
        max_angle_profile.append(float(angles.max()))
        mean_angle_profile.append(float(angles.mean()))
        incr_numbers.append(int(incr))

    return {
        "incr_numbers":      incr_numbers,
        "max_shear_angle_deg":  max_angle_profile,
        "mean_shear_angle_deg": mean_angle_profile,
        "n_increments":      len(incr_numbers),
    }


# ── Compound severity ─────────────────────────────────────────────────────────

def compute_compound_severity(
    incr: int,
    dz_profile: dict, dz_scale: float,
    fs_profile: dict,
    sa_profile: dict, sa_scale: float,
) -> float:
    """
    Compute compound wrinkle severity [0,1] at a given increment number.
    Returns 0.0 if the increment is not present in a profile.
    """
    # Find index of this increment in each profile
    def _get_at_incr(profile: dict, key: str) -> float | None:
        try:
            idx = profile["incr_numbers"].index(incr)
            return profile[key][idx]
        except (ValueError, KeyError, IndexError):
            return None

    # Fiber stress component (primary)
    cf = _get_at_incr(fs_profile, "comp_frac")
    s_fiber = min(cf / COMP_FRAC_SEVERITY_SCALE, 1.0) if cf is not None else 0.0

    # dz variance component (secondary)
    dv = _get_at_incr(dz_profile, "dz_variance")
    s_var = min(dv / max(dz_scale, 1e-9), 1.0) if dv is not None else 0.0

    # Shear angle component (tertiary)
    ma = _get_at_incr(sa_profile, "max_shear_angle_deg")
    s_shear = min(ma / max(sa_scale, 1.0), 1.0) if ma is not None else 0.0

    return W_FIBER * s_fiber + W_VAR * s_var + W_SHEAR * s_shear


# ── Onset detection ────────────────────────────────────────────────────────────

def detect_onset(
    dz_profile: dict,
    fs_profile: dict,
    sa_profile: dict,
    afs_times: dict,
    batch: str,
    sim_id: str,
) -> dict:
    """
    Detect wrinkle onset using compound multi-parameter severity score.

    Returns the full onset record with both legacy dz metrics and new compound score.
    """
    dz_scale = DZ_VAR_SEVERITY_SCALE[batch]
    sa_scale = SHEAR_ANGLE_SEVERITY_SCALE[batch]

    # Collect all available increment numbers (union across profiles)
    all_incrs = sorted(set(
        dz_profile.get("incr_numbers", []) +
        fs_profile.get("incr_numbers", []) +
        sa_profile.get("incr_numbers", [])
    ))

    if not all_incrs:
        return {
            "sim_id":           sim_id,
            "wrinkle_outcome":  "unknown",
            "detection_confidence": "low",
            "error": "no increments in any profile",
        }

    # Build compound severity profile
    severity_profile = [
        compute_compound_severity(incr, dz_profile, dz_scale, fs_profile, sa_profile, sa_scale)
        for incr in all_incrs
    ]

    max_severity    = float(max(severity_profile))
    max_severity_idx = int(np.argmax(severity_profile))
    max_incr        = all_incrs[max_severity_idx]

    # Find onset: first increment exceeding threshold
    onset_incr = None
    onset_idx  = None
    for i, (incr, sev) in enumerate(zip(all_incrs, severity_profile)):
        if sev > SEVERITY_ONSET_THRESHOLD:
            onset_incr = incr
            onset_idx  = i
            break

    # Stroke fraction at onset
    forming_times = {nr: info["time"] for nr, info in afs_times.items()
                     if info.get("loadblock", 1) == 2}
    if not forming_times:
        forming_times = {nr: info["time"] for nr, info in afs_times.items()}
    max_forming_time  = max(forming_times.values()) if forming_times else None
    onset_time        = afs_times.get(onset_incr, {}).get("time") if onset_incr else None
    onset_stroke_frac = (
        float(onset_time / max_forming_time)
        if (onset_time is not None and max_forming_time and max_forming_time > 0)
        else None
    )

    # Classification
    if max_severity >= SEVERITY_HIGH_CONF:
        wrinkle_outcome = "wrinkled"
        confidence = "high"
    elif max_severity >= SEVERITY_MED_CONF:
        wrinkle_outcome = "wrinkled"
        confidence = "medium"
    elif max_severity >= SEVERITY_LOW_CONF:
        wrinkle_outcome = "wrinkled"
        confidence = "low"
    else:
        wrinkle_outcome = "clean"
        confidence = "high" if max_severity < 0.04 else "medium"

    # Summary values at final and onset increments
    final_incr  = all_incrs[-1]
    final_cf    = None
    onset_cf    = None
    try:
        final_cf_idx = fs_profile["incr_numbers"].index(final_incr)
        final_cf = float(fs_profile["comp_frac"][final_cf_idx])
    except (ValueError, KeyError):
        pass
    if onset_idx is not None:
        try:
            onset_cf_idx = fs_profile["incr_numbers"].index(onset_incr)
            onset_cf = float(fs_profile["comp_frac"][onset_cf_idx])
        except (ValueError, KeyError):
            pass

    max_dz_var = max(dz_profile.get("dz_variance", [0.0]), default=0.0)

    # Severity profile (sampled) for debugging
    n = len(severity_profile)
    if n > 10:
        sev_sample = severity_profile[:5] + ["..."] + severity_profile[-5:]
    else:
        sev_sample = severity_profile[:]

    return {
        "sim_id":                    sim_id,
        "wrinkle_outcome":           wrinkle_outcome,
        "detection_confidence":      confidence,
        "detection_method":          "compound_multiparameter",
        "max_compound_severity":     max_severity,
        "max_severity_at_increment": int(max_incr),
        "onset_increment":           int(onset_incr) if onset_incr else None,
        "onset_stroke_fraction":     onset_stroke_frac,
        "onset_compound_severity":   float(severity_profile[onset_idx]) if onset_idx is not None else None,
        "final_comp_frac":           final_cf,
        "onset_comp_frac":           onset_cf,
        "max_dz_variance_mm2":       max_dz_var,
        "n_increments_total":        len(all_incrs),
        "severity_threshold":        SEVERITY_ONSET_THRESHOLD,
        "severity_profile_sample":   sev_sample,
        "fs_profile_available":      "error" not in fs_profile,
        "sa_profile_available":      "error" not in sa_profile,
    }


# ── Batch processors ──────────────────────────────────────────────────────────

def process_pair_with_runs(pair_id: str, fine_dir: Path, _coarse_dir, batch: str, records: list) -> None:
    """Process a pair where fine_dir is already resolved."""
    ply_groups = PLY_GROUPS[batch]
    try:
        afr_dir   = find_afr_dir(fine_dir)
        afs_times = load_afs_times(fine_dir)

        dz_profile = compute_dz_variance_profile(afr_dir, ply_groups)
        fs_profile = compute_fiber_stress_profile(afr_dir, ply_groups)
        sa_profile = compute_shear_angle_profile(afr_dir, ply_groups)

        record = detect_onset(dz_profile, fs_profile, sa_profile, afs_times, batch, pair_id)
        record["batch"]    = batch
        record["fine_run"] = fine_dir.name
        records.append(record)

        outcome  = record["wrinkle_outcome"]
        conf     = record["detection_confidence"]
        sev      = record["max_compound_severity"]
        final_cf = record.get("final_comp_frac")
        max_dz   = record["max_dz_variance_mm2"]
        cf_str   = f"cf={final_cf:.2f}" if final_cf is not None else "cf=?"
        print(f"  {pair_id:30s} [{batch}] {outcome:8s} ({conf:6s}) | "
              f"sev={sev:.3f} {cf_str} dz={max_dz:.3f}mm2")

    except Exception as e:
        records.append({
            "sim_id": pair_id, "batch": batch,
            "wrinkle_outcome": "unknown", "detection_confidence": "low",
            "error": str(e),
        })
        print(f"  {pair_id}: ERROR — {e}")


def process_pair(sim_id: str, results_dir: Path, batch: str, records: list) -> None:
    """Process one coarse+fine pair."""
    coarse_dir, fine_dir = find_coarse_and_fine(results_dir)
    if fine_dir is None:
        records.append({
            "sim_id": sim_id, "batch": batch,
            "wrinkle_outcome": "unknown", "detection_confidence": "low",
            "error": "could not find fine run",
        })
        print(f"  {sim_id}: SKIP — no fine run found")
        return
    process_pair_with_runs(sim_id, fine_dir, coarse_dir, batch, records)


def process_batch_a(records: list) -> None:
    print(f"\n--- Batch A (UD thermoplastic) ---")
    for results_dir in sorted(BATCH_A_ROOT.glob("geom 0_*.Results")):
        sim_id = results_dir.stem.replace(".Results", "").replace(" ", "_")
        all_runs = [d for d in results_dir.iterdir()
                    if d.is_dir() and (d / "meshes").exists()]
        def ply_size(r):
            p = r / "meshes" / "Part.section 1.1.msh"
            return p.stat().st_size if p.exists() else 0
        valid = sorted([(r, ply_size(r)) for r in all_runs if ply_size(r) > 0], key=lambda x: x[1])
        fine_runs   = [r for r, s in valid if s > 5_000_000]
        coarse_runs = [r for r, s in valid if s < 2_000_000 and s > 0]

        if len(fine_runs) == 1:
            process_pair(sim_id, results_dir, "A", records)
        elif len(fine_runs) > 1:
            for i, fine_run in enumerate(fine_runs):
                pair_id = f"{sim_id}_pair{i+1}"
                process_pair_with_runs(pair_id, fine_run,
                                       coarse_runs[i] if i < len(coarse_runs) else None,
                                       "A", records)
        else:
            records.append({
                "sim_id": sim_id, "batch": "A",
                "wrinkle_outcome": "unknown", "detection_confidence": "low",
                "error": "no fine run found",
            })
            print(f"  {sim_id}: SKIP — no fine run")


def process_batch_b(records: list) -> None:
    print(f"\n--- Batch B (Twintex 2×2 twill) ---")
    for results_dir in sorted(BATCH_B_ROOT.glob("mold set *.Results")):
        sim_id = results_dir.stem.replace(".Results", "").replace(" ", "_")
        process_pair(sim_id, results_dir, "B", records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WP1.5 Multi-Parameter Wrinkle Onset Detector")
    parser.add_argument("--batch", choices=["A", "B"], default=None)
    parser.add_argument("--pair", type=str, default=None,
                        help="Single pair by sim_id (e.g. geom_0_0 or mold_set_001)")
    args = parser.parse_args()

    REPORT_OUT.parent.mkdir(exist_ok=True)
    records = []

    print("=" * 90)
    print("WP1.5 — Wrinkle Onset Detection (Multi-Parameter Compound Severity)")
    print("=" * 90)
    print(f"  Primary:   Fiber stress 1 compressive fraction (model_{FIBER_STRESS_AFR_ID}_1.afr)")
    print(f"             weight={W_FIBER:.0%}, comp_frac scale={COMP_FRAC_SEVERITY_SCALE:.2f}")
    print(f"  Secondary: dz variance (model_{DISPLACEMENT_AFR_ID}_1.afr)")
    print(f"             weight={W_VAR:.0%}, scale A={DZ_VAR_SEVERITY_SCALE['A']:.1f}mm², "
          f"B={DZ_VAR_SEVERITY_SCALE['B']:.2f}mm²")
    print(f"  Tertiary:  Shear angle f1_f2 (model_{SHEAR_ANGLE_AFR_ID}_1.afr)")
    print(f"             weight={W_SHEAR:.0%}, scale A={SHEAR_ANGLE_SEVERITY_SCALE['A']:.0f}°, "
          f"B={SHEAR_ANGLE_SEVERITY_SCALE['B']:.0f}°")
    print(f"  Onset threshold: severity > {SEVERITY_ONSET_THRESHOLD:.2f}")
    print(f"  Classification: >={SEVERITY_HIGH_CONF} high / >={SEVERITY_MED_CONF} med / "
          f">={SEVERITY_LOW_CONF} low / <{SEVERITY_LOW_CONF} clean")

    if args.pair:
        batch = args.batch or "A"
        root  = BATCH_A_ROOT if batch == "A" else BATCH_B_ROOT
        glob_pattern = "geom 0_*.Results" if batch == "A" else "mold set *.Results"
        target_id = args.pair.replace("_", " ")
        matches = [d for d in root.glob(glob_pattern)
                   if target_id in d.name or args.pair in d.name.replace(" ", "_")]
        if not matches:
            print(f"ERROR: no match for --pair {args.pair} in Batch {batch}")
            return
        for m in matches:
            sim_id = m.stem.replace(".Results", "").replace(" ", "_")
            process_pair(sim_id, m, batch, records)
    else:
        if args.batch in (None, "A"):
            process_batch_a(records)
        if args.batch in (None, "B"):
            process_batch_b(records)

    wrinkled = [r for r in records if r.get("wrinkle_outcome") == "wrinkled"]
    clean    = [r for r in records if r.get("wrinkle_outcome") == "clean"]
    unknown  = [r for r in records if r.get("wrinkle_outcome") == "unknown"]
    high_c   = [r for r in wrinkled if r.get("detection_confidence") == "high"]
    med_c    = [r for r in wrinkled if r.get("detection_confidence") == "medium"]
    low_c    = [r for r in wrinkled if r.get("detection_confidence") == "low"]

    print(f"\n{'=' * 90}")
    print(f"Summary: {len(records)} pairs")
    print(f"  Wrinkled: {len(wrinkled)} "
          f"({len(high_c)} high / {len(med_c)} med / {len(low_c)} low confidence)")
    print(f"  Clean:    {len(clean)}")
    print(f"  Unknown:  {len(unknown)}")

    if clean:
        print("\nClean simulations:")
        for r in clean:
            sev = r.get("max_compound_severity", 0.0)
            print(f"  {r['sim_id']} [{r['batch']}]  max_severity={sev:.4f}")

    with open(REPORT_OUT, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nDone. Written to {REPORT_OUT}")


if __name__ == "__main__":
    main()
