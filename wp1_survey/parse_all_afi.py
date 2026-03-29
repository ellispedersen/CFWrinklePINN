"""
WP1 Survey — Parse all .afi input files across both batches.

Batch A: aniform_raw/geom 0_*.Results  (20 sims, UD thermoplastic, 2 plies, 75mm stroke)
Batch B: simulation_batch_271125/mold set NNN  (45 mold sets, Twintex 2x2 twill, 3 plies, ~6-12mm stroke)

Extracts per-simulation:
  - batch label (A or B)
  - Geometry type and punch stroke depth
  - N plies and ply orientations
  - Ply thickness as declared in .afi  (WARNING: use model_102_1.afr for ground truth)
  - Process temperatures
  - Solve step configuration

Writes: reports/input_param_registry.json

Usage:
    python -m wp1_survey.parse_all_afi
"""

import re
import json
from pathlib import Path

BATCH_A_ROOT = Path(r"C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\aniform_raw")
BATCH_B_ROOT = Path(r"C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\simulation_batch_271125")
REPORT_OUT   = Path(__file__).parent.parent / "reports" / "input_param_registry.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_runs_in_dir(parent: Path) -> list:
    """Return all timestamped run directories under parent that contain meshes/."""
    return [d for d in parent.iterdir() if d.is_dir() and (d / "meshes").exists()]


def find_best_run(search_dir: Path) -> Path:
    """Return the fine-mesh run (largest Part.section 1.1.msh file)."""
    runs = find_runs_in_dir(search_dir)
    if not runs:
        raise FileNotFoundError(f"No run dirs with meshes/ under {search_dir}")

    def ply_size(run: Path) -> int:
        msh = run / "meshes" / "Part.section 1.1.msh"
        return msh.stat().st_size if msh.exists() else 0

    return max(runs, key=ply_size)


def parse_afi(afi_path: Path) -> dict:
    """Extract forming parameters from one .afi file."""
    text = afi_path.read_text(encoding="utf-8", errors="replace")

    record = {
        "afi_path": str(afi_path),
        "parse_status": "complete",
        "parse_notes": "",
        # geometry
        "geometry_name": None,
        "punch_stroke_loadset1_mm": None,
        "punch_stroke_loadset2_mm": None,
        "punch_stroke_total_mm": None,
        # layup
        "n_plies": 0,
        "ply_orientations_deg": [],
        "ply_thickness_afi_mm": None,
        # process
        "tool_temp_die_C": None,
        "tool_temp_punch_C": None,
        "blank_initial_temp_C": None,
        "n_loadsets": 0,
        "solve_increments": [],
        "solve_dt": [],
    }

    # Geometry name from *title keyword or parent folder name
    title_m = re.search(r'^\*title\s+"([^"]+)"', text, re.MULTILINE)
    record["geometry_name"] = (
        title_m.group(1).strip() if title_m else afi_path.parent.parent.stem
    )

    # Tool names (informational)
    tool_names = re.findall(r'% Tool ([\w\s\-\.]+)', text)
    if tool_names:
        record["tool_names"] = [t.strip() for t in tool_names]

    # Ply element groups — identified by Tri3LDT shell elements referencing Part.section
    eg_blocks = re.split(r'(?=\*elementgroup)', text)
    n_plies = 0
    orientations = []
    for block in eg_blocks:
        if "Tri3LDT\n" in block and "Part.section" in block:
            n_plies += 1
            cs_m = re.search(r'localcs (\d+)', block)
            if cs_m:
                cs_id = int(cs_m.group(1))
                # localcs 1 = dir [1,0,0] = 0°; localcs 2 = dir [0,1,0] = 90°
                orientations.append(0.0 if cs_id == 1 else 90.0)
            # Capture ply thickness from first ply block encountered
            t_m = re.search(r'\nthickness ([\d\.]+)', block)
            if t_m and record["ply_thickness_afi_mm"] is None:
                record["ply_thickness_afi_mm"] = float(t_m.group(1))

    record["n_plies"] = n_plies
    record["ply_orientations_deg"] = orientations

    # Punch stroke: UZ displacement prescribed on tool 3 in each loadset
    # Batch A uses 2 loadsets (heat + form); Batch B may use 1 loadset
    loadset_strokes = re.findall(r'UZ ([\d\.]+) tool 3', text)
    if len(loadset_strokes) >= 1:
        record["punch_stroke_loadset1_mm"] = float(loadset_strokes[0])
    if len(loadset_strokes) >= 2:
        record["punch_stroke_loadset2_mm"] = float(loadset_strokes[1])
        record["punch_stroke_total_mm"]    = float(loadset_strokes[1])
    elif len(loadset_strokes) == 1:
        record["punch_stroke_total_mm"] = float(loadset_strokes[0])

    # Temperatures — try multiple patterns (differ between batch A and B)
    die_t_m = re.search(r'T (\d+) tool 1\b', text)
    if die_t_m:
        record["tool_temp_die_C"] = int(die_t_m.group(1))

    punch_t_m = re.search(r'T (\d+) tool 3\b', text)
    if punch_t_m:
        record["tool_temp_punch_C"] = int(punch_t_m.group(1))

    # Blank initial temperature — group 6 is always the first ply element group
    blank_t_m = re.search(r'T0 (\d+) group 6', text)
    if blank_t_m:
        record["blank_initial_temp_C"] = int(blank_t_m.group(1))

    # Solve steps
    solve_blocks = re.split(r'(?=\*solve)', text)
    for block in solve_blocks[1:]:
        record["n_loadsets"] += 1
        incr_m = re.search(r'increments (\d+)', block)
        dt_m   = re.search(r'timeincrement ([\d\.eE\+\-]+)', block)
        record["solve_increments"].append(int(incr_m.group(1)) if incr_m else None)
        record["solve_dt"].append(float(dt_m.group(1)) if dt_m else None)

    # Flag partial records (exclude optional fields from check)
    optional = {"parse_notes", "tool_names", "punch_stroke_loadset2_mm", "tool_temp_punch_C"}
    missing = [k for k, v in record.items() if v is None and k not in optional]
    if missing:
        record["parse_status"] = "partial"
        record["parse_notes"] = f"Missing: {missing}"

    return record


# ── Batch collectors ──────────────────────────────────────────────────────────

def collect_batch_a(data_root: Path) -> list:
    geom_dirs = sorted(data_root.glob("geom 0_*.Results"))
    records = []
    for results_dir in geom_dirs:
        sim_id = results_dir.stem.replace(".Results", "").replace(" ", "_")
        try:
            best_run = find_best_run(results_dir)
            afi_path = best_run / "model.afi"
            if not afi_path.exists():
                print(f"  [A] {sim_id}: no model.afi in {best_run.name}")
                records.append({"sim_id": sim_id, "batch": "A",
                                 "parse_status": "failed", "parse_notes": "no model.afi"})
                continue
            rec = parse_afi(afi_path)
            rec["sim_id"]   = sim_id
            rec["batch"]    = "A"
            rec["run_used"] = best_run.name
            records.append(rec)
            print(f"  [A] {sim_id}: {rec['parse_status']:8s} | plies={rec['n_plies']} "
                  f"| stroke={rec['punch_stroke_total_mm']}mm | {rec['ply_orientations_deg']}")
        except Exception as e:
            print(f"  [A] {sim_id}: ERROR — {e}")
            records.append({"sim_id": sim_id, "batch": "A",
                             "parse_status": "failed", "parse_notes": str(e)})
    return records


def collect_batch_b(data_root: Path) -> list:
    mold_dirs = sorted(data_root.glob("mold set *.Results"))
    records = []
    for mold_dir in mold_dirs:
        sim_id = mold_dir.name.replace(" ", "_")
        try:
            best_run = find_best_run(mold_dir)
            afi_path = best_run / "model.afi"
            if not afi_path.exists():
                print(f"  [B] {sim_id}: no model.afi in {best_run.name}")
                records.append({"sim_id": sim_id, "batch": "B",
                                 "parse_status": "failed", "parse_notes": "no model.afi"})
                continue
            rec = parse_afi(afi_path)
            rec["sim_id"]   = sim_id
            rec["batch"]    = "B"
            rec["run_used"] = best_run.name
            records.append(rec)
            print(f"  [B] {sim_id}: {rec['parse_status']:8s} | plies={rec['n_plies']} "
                  f"| stroke={rec['punch_stroke_total_mm']}mm | {rec['ply_orientations_deg']}")
        except Exception as e:
            print(f"  [B] {sim_id}: ERROR — {e}")
            records.append({"sim_id": sim_id, "batch": "B",
                             "parse_status": "failed", "parse_notes": str(e)})
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    REPORT_OUT.parent.mkdir(exist_ok=True)

    print("=" * 90)
    print("WP1 — Input Parameter Registry (Batch A + Batch B)")
    print("=" * 90)

    print(f"\n--- Batch A ({BATCH_A_ROOT.name}) ---")
    records_a = collect_batch_a(BATCH_A_ROOT)

    print(f"\n--- Batch B ({BATCH_B_ROOT.name}) ---")
    records_b = collect_batch_b(BATCH_B_ROOT)

    registry = records_a + records_b

    with open(REPORT_OUT, "w") as f:
        json.dump(registry, f, indent=2)

    # Summary
    n_complete = sum(1 for r in registry if r.get("parse_status") == "complete")
    n_partial  = sum(1 for r in registry if r.get("parse_status") == "partial")
    n_failed   = sum(1 for r in registry if r.get("parse_status") == "failed")
    print(f"\nTotal: {n_complete} complete, {n_partial} partial, {n_failed} failed "
          f"({len(registry)} sims)")

    for label, records in [("A", records_a), ("B", records_b)]:
        strokes = sorted({r.get("punch_stroke_total_mm") for r in records
                          if r.get("punch_stroke_total_mm") is not None})
        n_plies_set = sorted({r.get("n_plies") for r in records if r.get("n_plies")})
        print(f"  Batch {label}: strokes={strokes} mm | n_plies={n_plies_set}")

    print(f"\n✓ Written to {REPORT_OUT}")


if __name__ == "__main__":
    main()
