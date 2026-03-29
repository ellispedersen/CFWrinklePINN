"""
WP1 Survey — Read force-stroke curves from model.track.txt across both batches.

Track file format (one row per tracked tool per increment):
  UZ = punch displacement (mm down = negative or positive depending on convention)
  RFZ = punch reaction force (N) = forming force

Handles both batches:
  Batch A (geom 0_x): model.track.txt in run/results/ OR run/ top-level
  Batch B (mold set NNN): model.track.txt at run/ top-level

Usage:
    python -m wp1_survey.read_track_force

Writes: reports/force_stroke_summary.json
"""

import json
import numpy as np
from pathlib import Path

BATCH_A_ROOT = Path(r"C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\aniform_raw")
BATCH_B_ROOT = Path(r"C:\Users\ellis\Documents\VS Code\CFWrinklePredict2\data\simulation_batch_271125")
REPORT_OUT   = Path(__file__).parent.parent / "reports" / "force_stroke_summary.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_runs_in_dir(parent: Path) -> list:
    return [d for d in parent.iterdir() if d.is_dir() and (d / "meshes").exists()]


def find_coarse_and_fine_runs(search_dir: Path):
    """Return (coarse_dir, fine_dir) by ply mesh file size. Both may be None."""
    runs = find_runs_in_dir(search_dir)

    def ply_size(run: Path) -> int:
        msh = run / "meshes" / "Part.section 1.1.msh"
        return msh.stat().st_size if msh.exists() else 0

    valid = [(r, ply_size(r)) for r in runs if ply_size(r) > 0]
    valid.sort(key=lambda x: x[1])

    if len(valid) < 2:
        return None, None
    return valid[0][0], valid[-1][0]   # (coarse, fine)


def find_track_file(run_dir: Path):
    """Find model.track.txt — may be in run/results/ (Batch A) or run/ (Batch B)."""
    for candidate in [run_dir / "results" / "model.track.txt",
                      run_dir / "model.track.txt"]:
        if candidate.exists():
            return candidate
    return None


def parse_track_file(track_path: Path) -> dict:
    """
    Parse model.track.txt and extract punch UZ and RFZ series.

    Track file format:
      Line 1: "AniForm analysis track_file"
      Line 2: tool names (space-separated, each repeated 8 or 14 times)
      Line 3: column headers — starts with "Incr Time" then per-tool:
              UX RFX UY RFY UZ RFZ T RQ (8 cols for translation-only tools)
              or UX RFX UY RFY UZ RFZ RX RMX RY RMY RZ RMZ T RQ (14 cols for rotational)
      Line 4: "Incr Time N##### N##### ..." (node IDs)
      Lines 5+: data rows — one wide row per increment

    The punch is the tool with the largest |UZ| range.
    """
    if not track_path.exists():
        return {"error": f"not found: {track_path}"}

    text  = track_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Find the header line that starts with column labels
    header_line = None
    data_start  = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        # The column header line has UX, RFX, UY, RFY, UZ, RFZ etc.
        if len(tokens) > 4 and "UX" in tokens and "UZ" in tokens:
            header_line = stripped
            # Data starts after the node-ID line (next line after header)
            data_start = i + 2  # skip header + node-ID line
            break

    if header_line is None:
        return {"error": "could not find UX/UZ header line", "n_lines": len(lines)}

    # Parse header to find column indices — first 2 entries may be Incr, Time or not
    header_tokens = header_line.split()

    # Find all UZ and RFZ column positions in the header
    uz_cols  = [j for j, t in enumerate(header_tokens) if t == "UZ"]
    rfz_cols = [j for j, t in enumerate(header_tokens) if t == "RFZ"]

    if not uz_cols or not rfz_cols:
        return {"error": f"no UZ/RFZ columns in header (found {len(header_tokens)} columns)"}

    # Parse data rows
    data_rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        tokens = stripped.split()
        if len(tokens) < 4:
            continue
        try:
            vals = [float(t) for t in tokens]
            data_rows.append(vals)
        except ValueError:
            continue

    if not data_rows:
        return {"error": "no data rows found"}

    arr = np.array(data_rows, dtype=np.float64)
    # First two columns are Incr and Time
    n_header = 2  # Incr, Time prefix in data rows but NOT in the column header line

    # The header_tokens don't include Incr/Time (they start at UX) but data rows
    # start with Incr, Time. So column index in header = column index in data - 2.
    # But let's check: if header starts with "Incr" or a number
    # Actually, looking at the raw data: header line has just UX RFX... without Incr/Time
    # but the data lines have Incr Time first. The Incr/Time line is a separate line.

    # Let's detect: does the first data row have more columns than header tokens?
    if arr.shape[1] > len(header_tokens):
        # Data has Incr + Time prefix columns not in header
        offset = arr.shape[1] - len(header_tokens)
    else:
        offset = 0

    # Extract UZ and RFZ for each tool, find punch (max |UZ| range)
    best_uz  = None
    best_rfz = None
    best_range = 0.0

    for uz_col, rfz_col in zip(uz_cols, rfz_cols):
        actual_uz_col  = uz_col + offset
        actual_rfz_col = rfz_col + offset
        if actual_uz_col >= arr.shape[1] or actual_rfz_col >= arr.shape[1]:
            continue
        uz_series  = arr[:, actual_uz_col]
        rfz_series = arr[:, actual_rfz_col]
        uz_range = float(np.abs(uz_series).max() - np.abs(uz_series).min())
        total_range = float(uz_series.max() - uz_series.min())
        if total_range > best_range:
            best_range = total_range
            best_uz    = uz_series
            best_rfz   = rfz_series

    if best_uz is None or best_range < 0.01:
        return {"error": f"no tool with UZ range > 0.01mm (found {len(uz_cols)} tools)"}

    uz_min, uz_max = float(best_uz.min()), float(best_uz.max())
    stroke_fraction = (best_uz - uz_min) / max(uz_max - uz_min, 1e-6)
    max_force_idx   = int(np.argmax(np.abs(best_rfz)))

    return {
        "n_data_points":          len(best_uz),
        "n_tools_detected":       len(uz_cols),
        "uz_range_mm":            [uz_min, uz_max],
        "rfz_min_N":              float(best_rfz.min()),
        "rfz_max_N":              float(best_rfz.max()),
        "max_force_stroke_fraction": float(stroke_fraction[max_force_idx]),
        "punch_uz_mm":            best_uz.tolist(),
        "punch_rfz_N":            best_rfz.tolist(),
        "stroke_fraction":        stroke_fraction.tolist(),
    }


# ── Batch collectors ──────────────────────────────────────────────────────────

def process_sim_dir(sim_id: str, search_dir: Path, batch: str, summary: list):
    """Process one simulation folder (a .Results dir or mold set dir)."""
    coarse_dir, fine_dir = find_coarse_and_fine_runs(search_dir)

    for resolution, run_dir in [("fine", fine_dir), ("coarse", coarse_dir)]:
        if run_dir is None:
            print(f"  {sim_id} [{resolution}]: no run found")
            continue

        track_path = find_track_file(run_dir)
        if track_path is None:
            result = {"error": "model.track.txt not found"}
        else:
            result = parse_track_file(track_path)

        result["sim_id"]     = sim_id
        result["batch"]      = batch
        result["resolution"] = resolution
        result["run_dir"]    = run_dir.name

        if "error" in result:
            print(f"  {sim_id} [{resolution}]: ERROR — {result['error']}")
        else:
            uz_max   = result["uz_range_mm"][1]
            rfz_max  = result["rfz_max_N"]
            max_sf   = result["max_force_stroke_fraction"]
            n_pts    = result["n_data_points"]
            print(f"  {sim_id} [{resolution}]: n={n_pts:4d}  UZ_range={uz_max:7.2f}mm  "
                  f"max_F={rfz_max:10.1f}N  max_force_at={max_sf:.2f}")

        summary.append(result)


def main():
    REPORT_OUT.parent.mkdir(exist_ok=True)

    summary = []

    print("=" * 90)
    print("WP1 — Force-Stroke Survey (Batch A + Batch B)")
    print("=" * 90)

    # Batch A
    print(f"\n--- Batch A ---")
    geom_dirs = sorted(BATCH_A_ROOT.glob("geom 0_*.Results"))
    print(f"  {len(geom_dirs)} simulation folders")
    for results_dir in geom_dirs:
        sim_id = results_dir.stem.replace(".Results", "").replace(" ", "_")
        process_sim_dir(sim_id, results_dir, "A", summary)

    # Batch B
    print(f"\n--- Batch B ---")
    mold_dirs = sorted(BATCH_B_ROOT.glob("mold set *.Results"))
    print(f"  {len(mold_dirs)} mold set folders")
    for mold_dir in mold_dirs:
        sim_id = mold_dir.name.replace(" ", "_")
        process_sim_dir(sim_id, mold_dir, "B", summary)

    with open(REPORT_OUT, "w") as f:
        json.dump(summary, f, indent=2)

    # Batch-level summary
    for batch_label in ["A", "B"]:
        recs = [r for r in summary if r.get("batch") == batch_label and "error" not in r
                and r.get("resolution") == "fine"]
        if recs:
            ranges = [r["uz_range_mm"][1] - r["uz_range_mm"][0] for r in recs]
            print(f"\nBatch {batch_label} fine runs: "
                  f"stroke range {min(ranges):.1f}–{max(ranges):.1f} mm "
                  f"({len(recs)} sims)")

    n_errors = sum(1 for r in summary if "error" in r)
    print(f"\nTotal records: {len(summary)} ({n_errors} errors)")
    print(f"✓ Written to {REPORT_OUT}")
    if n_errors:
        print("\nNote: If punch_uz values are near-zero, inspect model.track.txt manually "
              "— column layout may differ from expected.")


if __name__ == "__main__":
    main()
