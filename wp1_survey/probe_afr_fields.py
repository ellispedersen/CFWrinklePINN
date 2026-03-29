"""
WP1 Survey — Probe AFR field headers and fingerprint unknown fields.

Probes one fine-mesh run from each batch to compare field availability:
  - Batch A sample: geom 0_0 (UD, 2 plies, 20 AFR files)
  - Batch B sample: mold set 000 (Twintex, 3 plies, 19 AFR files)
Both batches store AFR files in <run>/results/ — the find_afr_dir() helper handles detection.

Usage:
    python -m wp1_survey.probe_afr_fields

Writes: reports/afr_field_survey.json
"""

import sys
import json
import numpy as np
from pathlib import Path

# Make io/aniform_readers importable
sys.path.insert(0, str(Path(__file__).parent.parent / "io"))
from aniform_readers.ReadAFResult import ReadAFResult
from aniform_readers.ReadAFSFile import ReadAFSFile


# ── Configuration ─────────────────────────────────────────────────────────────

BATCH_A_ROOT = Path(r"C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\aniform_raw")
BATCH_B_ROOT = Path(r"C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\simulation_batch_271125")

# Ply element groups by batch (tool groups excluded)
PLY_GROUPS = {
    "A": [6, 10],        # UD 2-ply: group 6 (0°) + group 10 (90°)
    "B": [4, 8, 12],     # Twintex 3-ply (confirmed from data — NOT [6,10,14])
}

REPORT_OUT = Path(__file__).parent.parent / "reports" / "afr_field_survey.json"


# ── Type / field tables ───────────────────────────────────────────────────────

RES_TYPE_NAMES = {
    0:  "UnKnownT",
    10: "ScalarT (1 float: s)",
    20: "VectorT (3 floats: vx,vy,vz)",
    30: "TensorUNIT (1 float: txx)",
    31: "STensorPSST (3 floats: txx,tyy,txy)",
    32: "TensorPSST (4 floats: txx,tyy,txy,tyx)",
    33: "STensorPSNT (4 floats: txx,tyy,tzz,txy)",
    34: "TensorPSNT (5 floats)",
    35: "STensorT (6 floats)",
    36: "TensorT (9 floats)",
    40: "EulerT (3 floats: rx,ry,rz)",
}

KNOWN_FIELDS = {
    40:  "Displacement [dx,dy,dz]",
    42:  "Rotation (tool DOF, near-zero on plies)",
    44:  "Temperature [T1,T2,T3]",
    48:  "Temperature 1 (top surface)",
    49:  "Temperature 2 (bottom surface)",
    102: "Green-Lagrange strain [E11,E22,E12]",
    105: "Thickness [t] (mm)",
    106: "Eq shear rate (1/s)",
    200: "Stress [s11,s22,s12] (MPa)",
    203: "Fiber direction 1/2 (unit vector)",
    204: "Shear angle f1_f2 (deg)",
    205: "Fiber strain 1/2 (scalar)",
    206: "Fiber stress 1/2 (MPa) — PRIMARY WRINKLE PRECURSOR",
    214: "Rel. crystallinity Nakamura (Batch A only)",
    302: "Penetration depth (CONTACT field)",
    304: "Slip path length (CONTACT field)",
    401: "Traction (CONTACT field, all zeros)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_best_run(search_dir: Path) -> Path:
    """Return the fine-mesh run (largest Part.section 1.1.msh)."""
    runs = [d for d in search_dir.iterdir()
            if d.is_dir() and (d / "meshes").exists()]
    if not runs:
        raise FileNotFoundError(f"No runs with meshes/ under {search_dir}")

    def ply_size(run: Path) -> int:
        msh = run / "meshes" / "Part.section 1.1.msh"
        return msh.stat().st_size if msh.exists() else 0

    return max(runs, key=ply_size)


def find_afr_dir(run_dir: Path) -> Path:
    """
    Return the directory containing .afr files for a run.

    Batch A stores AFRs in run/results/; Batch B stores them at run top-level.
    Falls back to run_dir itself if results/ subdir has no .afr files.
    Both batches confirmed to use <run>/results/ — fallback handles edge cases.
    """
    results_subdir = run_dir / "results"
    if results_subdir.is_dir() and any(results_subdir.glob("model_*.afr")):
        return results_subdir
    if any(run_dir.glob("model_*.afr")):
        return run_dir
    raise FileNotFoundError(f"No model_*.afr files found under {run_dir}")


def fingerprint(data_arr: np.ndarray) -> dict:
    """Basic statistics for field identification (skip index column 0)."""
    vals = data_arr[:, 1:]
    flat = vals.flatten()
    nonzero = flat[flat != 0]
    return {
        "shape": list(data_arr.shape),
        "n_components": int(vals.shape[1]),
        "mean":  float(np.nanmean(flat)),
        "std":   float(np.nanstd(flat)),
        "min":   float(np.nanmin(flat)),
        "max":   float(np.nanmax(flat)),
        "always_positive": bool((flat >= 0).all()),
        "always_negative": bool((flat <= 0).all()),
        "all_zero":        bool(np.allclose(flat, 0)),
        "pct_nonzero":     float(100.0 * len(nonzero) / max(len(flat), 1)),
        "order_of_magnitude": float(np.log10(np.abs(flat).mean() + 1e-12)),
    }


# ── Probing functions ─────────────────────────────────────────────────────────

def probe_all_afr(afr_dir: Path, ply_groups: list) -> dict:
    afr_files = sorted(afr_dir.glob("model_*.afr"))
    survey = {}

    col_widths = (30, 6, 6, 35, 14, 6)
    header = (f"{'File':<{col_widths[0]}} {'ResID':>{col_widths[1]}} {'SubID':>{col_widths[2]}} "
              f"{'Type':<{col_widths[3]}} {'Groups':>{col_widths[4]}} {'Incrs':>{col_widths[5]}}  Known meaning")
    print(f"\nProbing {len(afr_files)} .afr files in:\n  {afr_dir}")
    print(header)
    print("-" * 120)

    for afr_path in afr_files:
        stem  = afr_path.stem  # e.g. "model_302_1"
        parts = stem.split("_")
        res_id_from_name = int(parts[1]) if len(parts) >= 2 else -1
        sub_id_from_name = int(parts[2]) if len(parts) >= 3 else -1

        try:
            ResultsIncr, Groups, Increments, res_type, indices_included, res_id = ReadAFResult(
                str(afr_path),
                elemGrNrs=ply_groups,
                IncsToExport=[],
                silent=True,
            )

            unique_groups = sorted(int(g) for g in set(Groups))
            unique_incrs  = sorted(set(Increments))
            # res_type can be [] (empty list) if no data for requested groups
            rt = int(res_type) if not isinstance(res_type, list) else -1
            type_name = RES_TYPE_NAMES.get(rt, f"UNKNOWN ({res_type})")
            known     = KNOWN_FIELDS.get(res_id_from_name, "*** UNKNOWN ***")

            print(f"{afr_path.name:<{col_widths[0]}} {res_id_from_name:>{col_widths[1]}} "
                  f"{sub_id_from_name:>{col_widths[2]}} {type_name:<{col_widths[3]}} "
                  f"{str(unique_groups):>{col_widths[4]}} {len(unique_incrs):>{col_widths[5]}}  {known}")

            # Fingerprint first and last increment for each ply group
            fp = {}
            for grp in unique_groups:
                for ts_key in ["first", "last"]:
                    incr = unique_incrs[0] if ts_key == "first" else unique_incrs[-1]
                    if incr in ResultsIncr and grp in ResultsIncr[incr]:
                        fp[f"group_{grp}_{ts_key}_incr_{incr}"] = fingerprint(
                            ResultsIncr[incr][grp]
                        )

            survey[afr_path.name] = {
                "res_id":          res_id_from_name,
                "sub_id":          sub_id_from_name,
                "res_type":        rt,
                "res_type_name":   type_name,
                "groups":          unique_groups,
                "n_increments":    len(unique_incrs),
                "increment_range": [int(unique_incrs[0]), int(unique_incrs[-1])] if unique_incrs else [],
                "indices_included": bool(indices_included),
                "known_meaning":   known,
                "fingerprint":     fp,
            }

        except Exception as e:
            print(f"{afr_path.name:<{col_widths[0]}} ERROR: {e}")
            survey[afr_path.name] = {"error": str(e)}

    return survey


def probe_afs(run_dir: Path) -> dict:
    """Try run/results/model.afs then run/model.afs."""
    for candidate in [run_dir / "results" / "model.afs", run_dir / "model.afs"]:
        if candidate.exists():
            afs_path = candidate
            break
    else:
        print(f"  [AFS] Not found under {run_dir}")
        return {}

    try:
        Name, Description, IncrementInfo, Version, Core, SimulationInfo, Timing = ReadAFSFile(str(afs_path))
        # IncrementInfo is a pandas DataFrame: index=field names, columns=increment numbers
        times      = IncrementInfo.loc['t_end'].values.astype(float)
        loadblocks = IncrementInfo.loc['current_loadblock'].values.astype(int)
        converged  = IncrementInfo.loc['has_converged'].values.astype(bool)

        n_not_converged = int((~converged).sum())
        print(f"\n  [AFS] {Name} | version={Version} | {len(times)} increments")
        print(f"        Time range: {times.min():.3f} → {times.max():.3f} s")
        print(f"        Load blocks: {sorted(set(loadblocks.tolist()))}")
        if n_not_converged:
            print(f"        WARNING: {n_not_converged} non-converged increments")

        return {
            "name":            Name,
            "version":         str(Version),
            "n_increments":    int(len(times)),
            "time_min":        float(times.min()),
            "time_max":        float(times.max()),
            "loadblocks":      sorted(set(loadblocks.tolist())),
            "n_not_converged": n_not_converged,
            "times_sample":    times[:10].tolist(),
        }
    except Exception as e:
        print(f"  [AFS] ERROR: {e}")
        return {"error": str(e)}


# ── Main ─────────────────────────────────────────────────────────────────────

def probe_sample(batch_label: str, search_dir: Path) -> dict:
    """Find the fine run of a simulation folder and probe all its AFR files."""
    print(f"\n{'=' * 120}")
    print(f"Batch {batch_label} sample: {search_dir.name}")
    print(f"{'=' * 120}")

    run_dir  = find_best_run(search_dir)
    afr_dir  = find_afr_dir(run_dir)
    ply_grps = PLY_GROUPS[batch_label]

    print(f"  Run dir:  {run_dir}")
    print(f"  AFR dir:  {afr_dir}")
    print(f"  Ply groups: {ply_grps}")

    afs_info = probe_afs(run_dir)
    afr_data = probe_all_afr(afr_dir, ply_grps)

    # Report unknowns
    unknowns = [(fname, info.get("res_type_name", "?"))
                for fname, info in afr_data.items()
                if "UNKNOWN" in info.get("known_meaning", "")]
    if unknowns:
        print(f"\n  *** UNKNOWNS for Batch {batch_label}: ***")
        for fname, tname in unknowns:
            print(f"    {fname}: {tname}")

    return {
        "batch":        batch_label,
        "search_dir":   str(search_dir),
        "run_dir":      str(run_dir),
        "afr_dir":      str(afr_dir),
        "ply_groups":   ply_grps,
        "afs_summary":  afs_info,
        "afr_fields":   afr_data,
    }


def main():
    REPORT_OUT.parent.mkdir(exist_ok=True)

    # Batch A sample: geom 0_0
    batch_a_dir = BATCH_A_ROOT / "geom 0_0.Results"
    result_a = probe_sample("A", batch_a_dir)

    # Batch B sample: mold set 000
    batch_b_dir = BATCH_B_ROOT / "mold set 000.Results"
    result_b = probe_sample("B", batch_b_dir)

    # Cross-batch field comparison
    fields_a = set(result_a["afr_fields"].keys())
    fields_b = set(result_b["afr_fields"].keys())
    only_in_a = sorted(fields_a - fields_b)
    only_in_b = sorted(fields_b - fields_a)
    in_both   = sorted(fields_a & fields_b)

    print(f"\n{'=' * 120}")
    print("Cross-batch field comparison")
    print(f"{'=' * 120}")
    print(f"  In both:    {len(in_both)} files")
    print(f"  Only in A:  {only_in_a}")
    print(f"  Only in B:  {only_in_b}")

    report = {
        "batch_A": result_a,
        "batch_B": result_b,
        "field_comparison": {
            "in_both":   in_both,
            "only_in_A": only_in_a,
            "only_in_B": only_in_b,
        },
    }

    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report written to: {REPORT_OUT}")


if __name__ == "__main__":
    main()
