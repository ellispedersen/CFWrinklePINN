"""
WP1.2 — Field Survey and Stress Frame Resolution

Resolves three open questions from ANIFORM_REFERENCE.md §6:
  1. Stress coordinate frame for model_302_1.afr (fiber vs global)
  2. Physical meaning of model_214_1.afr (Batch A only)
  3. Sub-result meanings for 203_2, 205_2, 206_2

Reads one fine-mesh run from each batch, fingerprints every field at the
final forming increment, and emits a structured JSON report + console summary.

Usage:
    python -m validation.field_survey
    python -m validation.field_survey --resolve-stress-frame    (quick mode)
    python -m validation.field_survey --batch A                 (one batch only)

Writes: reports/field_survey_detail.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Allow running as module from project root
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "io"))
from aniform_readers.ReadAFResult import ReadAFResult
from aniform_readers.ReadAFSFile  import ReadAFSFile


# ── Config ────────────────────────────────────────────────────────────────────

BATCH_A_ROOT = Path(r"C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\aniform_raw")
BATCH_B_ROOT = Path(r"C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\simulation_batch_271125")
REPORT_OUT   = _ROOT / "reports" / "field_survey_detail.json"

PLY_GROUPS = {"A": [6, 10], "B": [4, 8, 12]}

# E_fiber for Batch A UD material (MPa) — used for stress frame resolution
E_FIBER_MPA = 25_000.0

# Sub-result pairs: sub_id = fiber family number (1 vs 2)
# 203: Fiber direction (VectorT), 205: Fiber strain (ScalarT), 206: Fiber stress (ScalarT)
SUB_RESULT_PAIRS = [
    (203, 1, 203, 2, "fiber_direction_1 vs fiber_direction_2"),
    (205, 1, 205, 2, "fiber_strain_1 vs fiber_strain_2"),
    (206, 1, 206, 2, "fiber_stress_1 vs fiber_stress_2"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_best_run(search_dir: Path) -> Path:
    runs = [d for d in search_dir.iterdir()
            if d.is_dir() and (d / "meshes").exists()]
    if not runs:
        raise FileNotFoundError(f"No run dirs in {search_dir}")
    def size(r):
        p = r / "meshes" / "Part.section 1.1.msh"
        return p.stat().st_size if p.exists() else 0
    return max(runs, key=size)


def find_afr_dir(run_dir: Path) -> Path:
    results = run_dir / "results"
    if results.is_dir() and any(results.glob("model_*.afr")):
        return results
    if any(run_dir.glob("model_*.afr")):
        return run_dir
    raise FileNotFoundError(f"No model_*.afr under {run_dir}")


def load_final_increment(afr_path: Path, ply_groups: list) -> dict:
    """Read all data at the final increment for the given ply groups."""
    ResultsIncr, Groups, Increments, res_type, indices_included, res_id = ReadAFResult(
        str(afr_path), elemGrNrs=ply_groups, IncsToExport=[], silent=True
    )
    if not Increments:
        return {}
    last_incr = max(set(Increments))
    data = {}
    for grp in set(Groups):
        if grp in ResultsIncr.get(last_incr, {}):
            arr = ResultsIncr[last_incr][grp]
            data[grp] = arr[:, 1:]   # drop node-index col
    return {"res_type": int(res_type), "res_id": int(res_id),
            "last_incr": int(last_incr), "data": data}


def fingerprint(values: np.ndarray, label: str = "") -> dict:
    """Statistical fingerprint of a field component array."""
    flat = values.flatten().astype(float)
    nonzero = flat[flat != 0]
    fp = {
        "label":         label,
        "shape":         list(values.shape),
        "n_components":  int(values.shape[1]) if values.ndim == 2 else 1,
        "mean":          float(np.nanmean(flat)),
        "std":           float(np.nanstd(flat)),
        "min":           float(np.nanmin(flat)),
        "max":           float(np.nanmax(flat)),
        "always_positive": bool((flat >= 0).all()),
        "always_negative": bool((flat <= 0).all()),
        "all_zero":      bool(np.allclose(flat, 0)),
        "pct_nonzero":   float(100.0 * len(nonzero) / max(len(flat), 1)),
        "order_of_mag":  float(np.log10(np.abs(flat).mean() + 1e-12)),
    }
    return fp


# ── Stress frame resolution ───────────────────────────────────────────────────

def resolve_stress_frame(afr_dir: Path, ply_groups: list) -> dict:
    """
    Determine whether model_200_1.afr (Stress) is in fiber or global frame.

    Method: At the final forming increment, for the 0° ply (group 6 Batch A / group 4 Batch B):
    - Fiber frame: s11 (along fiber) >> s22 (transverse). Expected ratio > 5
      because E_fiber = 25,000 MPa vs E_transverse ≈ 4.5 MPa in bending.
    - Global frame: loads distributed; s11 ≈ s22 in magnitude.

    Note: model_302_1.afr is "Penetration depth" (contact field), NOT stress.
    The actual stress tensor is model_200_1.afr (confirmed via silent=False header).

    Returns a dict with the resolved frame and supporting statistics.
    """
    stress_path = afr_dir / "model_200_1.afr"
    if not stress_path.exists():
        return {"frame": "unknown", "error": "model_200_1.afr not found"}

    # Stress (field 200) uses bending sub-element groups: A=[6,7,10,11], B=[4,5,8,9,12,13]
    # But the main ply groups (6/4) are included, so ply_groups works here
    result = load_final_increment(stress_path, ply_groups)
    if not result.get("data"):
        return {"frame": "unknown", "error": "no data read"}

    grp = ply_groups[0]   # group 6 (Batch A) or group 4 (Batch B) = first ply
    if grp not in result["data"]:
        return {"frame": "unknown", "error": f"group {grp} not in data"}

    s = result["data"][grp]   # (N, 3): [s11, s22, s12]
    s11 = s[:, 0]
    s22 = s[:, 1]
    s12 = s[:, 2]

    abs_s11_mean = float(np.abs(s11).mean())
    abs_s22_mean = float(np.abs(s22).mean())
    abs_s12_mean = float(np.abs(s12).mean())
    ratio = abs_s11_mean / max(abs_s22_mean, 1e-12)

    # Frame verdict
    if ratio > 5.0:
        frame = "fiber"
        confidence = "high" if ratio > 10.0 else "medium"
    elif ratio < 2.0:
        frame = "global"
        confidence = "medium"
    else:
        frame = "ambiguous"
        confidence = "low"

    pct_compressive_s11 = float(100.0 * (s11 < 0).sum() / max(len(s11), 1))

    verdict = (
        f"s11/s22 ratio = {ratio:.1f} → frame={frame} ({confidence} confidence). "
        f"{pct_compressive_s11:.1f}% of nodes have compressive s11 (wrinkle precursor indicator)."
    )
    print(f"\n  [STRESS FRAME] {verdict}")

    return {
        "frame":                frame,
        "confidence":           confidence,
        "abs_s11_mean_MPa":     abs_s11_mean,
        "abs_s22_mean_MPa":     abs_s22_mean,
        "abs_s12_mean_MPa":     abs_s12_mean,
        "s11_s22_ratio":        ratio,
        "pct_compressive_s11":  pct_compressive_s11,
        "verdict":              verdict,
        "resolution_rule":      "ratio > 5 → fiber; ratio < 2 → global; else ambiguous",
        "action_required":      frame in ("ambiguous", "unknown"),
    }


# ── Unknown field identification ──────────────────────────────────────────────

def identify_unknown_214(afr_dir: Path, ply_groups: list) -> dict:
    """
    Read model_214_1.afr with silent=False to get the header name string,
    then fingerprint the data for pattern analysis.
    """
    afr_214 = afr_dir / "model_214_1.afr"
    if not afr_214.exists():
        return {"status": "not_present", "note": "Batch B does not have model_214_1.afr"}

    print(f"\n  [FIELD 214] Reading header (silent=False for header only):")

    try:
        # Read once with silent=False to capture header name (first few lines)
        # Then re-read with silent=True for full data
        import io, contextlib
        header_buf = io.StringIO()
        with contextlib.redirect_stdout(header_buf):
            ReadAFResult(str(afr_214), elemGrNrs=ply_groups, IncsToExport=[0], silent=False)
        header_text = header_buf.getvalue()
        # Extract the Name line
        for line in header_text.splitlines():
            if "Name:" in line:
                print(f"  {line.strip()}")
                break

        ResultsIncr, Groups, Increments, res_type, indices_included, res_id = ReadAFResult(
            str(afr_214), elemGrNrs=ply_groups, IncsToExport=[], silent=True
        )

        last_incr = max(set(Increments)) if Increments else None
        first_incr = min(set(Increments)) if Increments else None

        fp_first, fp_last = {}, {}
        if first_incr and ply_groups[0] in ResultsIncr.get(first_incr, {}):
            vals = ResultsIncr[first_incr][ply_groups[0]][:, 1:]
            fp_first = fingerprint(vals, "first_incr")
        if last_incr and ply_groups[0] in ResultsIncr.get(last_incr, {}):
            vals = ResultsIncr[last_incr][ply_groups[0]][:, 1:]
            fp_last = fingerprint(vals, "last_incr")

        # Temporal profile: check if mean changes significantly
        means_over_time = []
        for incr in sorted(set(Increments)):
            if ply_groups[0] in ResultsIncr.get(incr, {}):
                v = ResultsIncr[incr][ply_groups[0]][:, 1:]
                means_over_time.append(float(np.abs(v).mean()))

        is_monotonic = (np.diff(means_over_time) >= 0).all() if len(means_over_time) > 1 else None
        initial_mean  = means_over_time[0]  if means_over_time else None
        final_mean    = means_over_time[-1] if means_over_time else None

        print(f"  [FIELD 214] res_type={res_type}, n_increments={len(set(Increments))}, groups={sorted(set(Groups))}")
        if fp_last:
            print(f"  [FIELD 214] Final incr stats: mean={fp_last['mean']:.4g}, std={fp_last['std']:.4g}, "
                  f"always_positive={fp_last['always_positive']}, pct_nonzero={fp_last['pct_nonzero']:.1f}%")

        return {
            "status":         "fingerprinted",
            "res_type":       int(res_type),
            "n_increments":   len(set(Increments)),
            "groups":         sorted(set(Groups)),
            "fingerprint_first": fp_first,
            "fingerprint_last":  fp_last,
            "temporal_mean_profile": means_over_time,
            "is_monotonic_increasing": bool(is_monotonic) if is_monotonic is not None else None,
            "initial_mean":   initial_mean,
            "final_mean":     final_mean,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Sub-result frame comparison ───────────────────────────────────────────────

def compare_sub_results(afr_dir: Path, ply_groups: list,
                        res_id: int, sub1: int, sub2: int, label: str) -> dict:
    """
    Compare sub1 vs sub2 for a given result ID.
    Sub_id corresponds to fiber family number (1 or 2).
    Handles both VectorT (3 components, e.g. 203) and ScalarT (1 component, e.g. 205/206).
    """
    p1 = afr_dir / f"model_{res_id}_{sub1}.afr"
    p2 = afr_dir / f"model_{res_id}_{sub2}.afr"

    if not p1.exists() or not p2.exists():
        return {"label": label, "status": "file_missing",
                "note": f"{p1.name}={p1.exists()}, {p2.name}={p2.exists()}"}

    r1 = load_final_increment(p1, ply_groups)
    r2 = load_final_increment(p2, ply_groups)

    grp = ply_groups[0]
    if grp not in r1.get("data", {}) or grp not in r2.get("data", {}):
        return {"label": label, "status": "no_data"}

    d1 = r1["data"][grp]
    d2 = r2["data"][grp]

    n_comp = min(d1.shape[1], d2.shape[1])
    comp_names = ["c1", "c2", "c3"][:n_comp]

    results = {"label": label, "status": "compared",
               "n_components_sub1": int(d1.shape[1]),
               "n_components_sub2": int(d2.shape[1]),
               "interpretation": "sub_id = fiber family number (1 vs 2)"}

    for i, comp in enumerate(comp_names):
        v1, v2 = d1[:, i], d2[:, i]
        scale_ratio = float(np.abs(v2).mean() / max(np.abs(v1).mean(), 1e-12))
        correlation = float(np.corrcoef(v1, v2)[0, 1]) if len(v1) > 1 else 0.0
        results[comp] = {
            "sub1_mean": float(v1.mean()),
            "sub1_std":  float(v1.std()),
            "sub2_mean": float(v2.mean()),
            "sub2_std":  float(v2.std()),
            "scale_ratio": scale_ratio,
            "correlation": correlation,
        }

    return results


# ── Main survey ───────────────────────────────────────────────────────────────

def survey_batch(batch_label: str, search_dir: Path) -> dict:
    print(f"\n{'=' * 80}")
    print(f"Field Survey — Batch {batch_label}: {search_dir.name}")
    print(f"{'=' * 80}")

    run_dir = find_best_run(search_dir)
    afr_dir = find_afr_dir(run_dir)
    ply_groups = PLY_GROUPS[batch_label]

    print(f"  Run: {run_dir.name}")
    print(f"  AFR dir: {afr_dir}")
    print(f"  Ply groups: {ply_groups}")

    results = {
        "batch":      batch_label,
        "sim_dir":    str(search_dir),
        "run_dir":    str(run_dir),
        "afr_dir":    str(afr_dir),
        "ply_groups": ply_groups,
    }

    # 1. Stress frame resolution (Batch A only — UD material has unambiguous fiber direction)
    if batch_label == "A":
        print("\n--- Stress frame resolution (model_200_1.afr) ---")
        results["stress_frame"] = resolve_stress_frame(afr_dir, ply_groups)

    # 2. Unknown field 214 (Batch A only)
    if batch_label == "A":
        print("\n--- Unknown field identification (model_214_1.afr) ---")
        results["field_214"] = identify_unknown_214(afr_dir, ply_groups)

    # 3. Sub-result comparisons
    print("\n--- Sub-result comparisons ---")
    sub_results = {}
    for res_id, sub1, res_id2, sub2, label in SUB_RESULT_PAIRS:
        r = compare_sub_results(afr_dir, ply_groups, res_id, sub1, sub2, label)
        sub_results[f"{res_id}_{sub1}_vs_{sub2}"] = r
        status = r.get("frame_diagnosis", r.get("status", "?"))
        print(f"  {label}: {status}")
    results["sub_result_comparisons"] = sub_results

    # 4. Fingerprint all AFR fields at final increment
    print("\n--- Full field fingerprints (final forming increment) ---")
    field_fingerprints = {}
    for afr_path in sorted(afr_dir.glob("model_*.afr")):
        parts = afr_path.stem.split("_")
        res_id_name = int(parts[1]) if len(parts) >= 2 else -1
        try:
            r = load_final_increment(afr_path, ply_groups)
            fp_per_group = {}
            for grp, vals in r.get("data", {}).items():
                fp_per_group[grp] = fingerprint(vals, f"group_{grp}")
            field_fingerprints[afr_path.name] = {
                "res_type": r.get("res_type"),
                "last_incr": r.get("last_incr"),
                "fingerprints": fp_per_group,
            }
            if fp_per_group:
                g0 = ply_groups[0]
                fp = fp_per_group.get(g0, {})
                print(f"  {afr_path.name:30s} mean={fp.get('mean', 0):10.4g}  "
                      f"std={fp.get('std', 0):10.4g}  "
                      f"always_pos={fp.get('always_positive', '?')}")
        except Exception as e:
            field_fingerprints[afr_path.name] = {"error": str(e)}
            print(f"  {afr_path.name:30s} ERROR: {e}")

    results["field_fingerprints"] = field_fingerprints
    return results


def main():
    parser = argparse.ArgumentParser(description="WP1.2 Field Survey")
    parser.add_argument("--resolve-stress-frame", action="store_true",
                        help="Quick mode: only resolve stress frame for Batch A")
    parser.add_argument("--batch", choices=["A", "B"], default=None,
                        help="Survey only this batch")
    args = parser.parse_args()

    REPORT_OUT.parent.mkdir(exist_ok=True)

    # Sample directories — one fine run per batch
    samples = {
        "A": BATCH_A_ROOT / "geom 0_0.Results",
        "B": BATCH_B_ROOT / "mold set 000.Results",
    }

    if args.resolve_stress_frame:
        # Quick mode: just resolve stress frame for Batch A
        search_dir = samples["A"]
        run_dir = find_best_run(search_dir)
        afr_dir = find_afr_dir(run_dir)
        result = resolve_stress_frame(afr_dir, PLY_GROUPS["A"])
        print(f"\nStress frame: {result['frame']} ({result.get('confidence', '?')} confidence)")
        print(f"  s11/s22 ratio = {result.get('s11_s22_ratio', 'N/A'):.2f}")
        print(f"  Action required: {result.get('action_required', True)}")
        return

    report = {}
    batches_to_run = ["A", "B"] if args.batch is None else [args.batch]

    for batch_label in batches_to_run:
        search_dir = samples[batch_label]
        if not search_dir.exists():
            print(f"WARNING: {search_dir} not found — skipping Batch {batch_label}")
            continue
        report[batch_label] = survey_batch(batch_label, search_dir)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY — Open Questions Status")
    print(f"{'=' * 80}")

    if "A" in report:
        sf = report["A"].get("stress_frame", {})
        print(f"  Stress frame:  {sf.get('frame', 'not run')} "
              f"(ratio={sf.get('s11_s22_ratio', 'N/A')}, "
              f"action_required={sf.get('action_required', True)})")

        f214 = report["A"].get("field_214", {})
        print(f"  Field 214:     status={f214.get('status', 'not run')}, "
              f"res_type={f214.get('res_type', '?')}")

        for key, sr in report["A"].get("sub_result_comparisons", {}).items():
            diag = sr.get("frame_diagnosis", sr.get("status", "?"))
            print(f"  {key:35s}: {diag}")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"\n✓ Report written to: {REPORT_OUT}")
    print("\nNext step: update config/field_registry.yaml with resolved statuses.")


if __name__ == "__main__":
    main()
