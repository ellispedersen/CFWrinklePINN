from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from .schema import (
    CHUNK_NODE_DIM,
    CHUNK_TIME_DIM,
    COMPRESSION,
    CV_N_FOLDS,
    CV_RANDOM_SEED,
    FIELD_DEFS,
    H5_PATH,
    SCHEMA_VERSION,
    SimRecord,
)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "io"))

from aniform_readers.ReadAFMesh import ReadAFMesh  # noqa: E402
from aniform_readers.ReadAFResult import ReadAFResult  # noqa: E402
from aniform_readers.ReadAFSFile import ReadAFSFile  # noqa: E402


def _sim_id_from_results_dir_name(name: str) -> str:
    return name.replace(".Results", "").replace(" ", "_")


def _sim_base_id(sim_id: str) -> str:
    return sim_id.replace("_pair1", "").replace("_pair2", "")


def _find_runs(results_dir: Path) -> list[Path]:
    runs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "meshes").exists()]
    return sorted(runs, key=lambda p: p.name)


def _mesh_size(run_dir: Path) -> int:
    msh = run_dir / "meshes" / "Part.section 1.1.msh"
    return msh.stat().st_size if msh.exists() else 0


def _find_afr_dir(run_dir: Path) -> Path:
    candidate = run_dir / "results"
    if candidate.is_dir() and any(candidate.glob("model_*.afr")):
        return candidate
    if any(run_dir.glob("model_*.afr")):
        return run_dir
    raise FileNotFoundError(f"No model_*.afr under {run_dir}")


def _find_afm_path(run_dir: Path) -> Path:
    for p in (run_dir / "results" / "model.afm", run_dir / "model.afm"):
        if p.exists():
            return p
    raise FileNotFoundError(f"No model.afm under {run_dir}")


def _find_afs_path(run_dir: Path) -> Path:
    for p in (run_dir / "results" / "model.afs", run_dir / "model.afs"):
        if p.exists():
            return p
    raise FileNotFoundError(f"No model.afs under {run_dir}")


def load_configs() -> dict[str, Any]:
    cfg_path = _ROOT / "config" / "pipeline_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        pipeline = yaml.safe_load(f)

    field_registry_path = Path(pipeline["field_registry"])
    with field_registry_path.open("r", encoding="utf-8") as f:
        fields = yaml.safe_load(f)

    with (_ROOT / "reports" / "wrinkle_onset_registry.json").open("r", encoding="utf-8") as f:
        wrinkle_raw = json.load(f)
    with (_ROOT / "reports" / "force_stroke_summary.json").open("r", encoding="utf-8") as f:
        force_raw = json.load(f)
    with (_ROOT / "reports" / "input_param_registry.json").open("r", encoding="utf-8") as f:
        input_raw = json.load(f)

    wrinkle_by_id = {r["sim_id"]: r for r in wrinkle_raw if r.get("sim_id")}

    force_index: dict[tuple[str, str], dict[str, Any]] = {}
    for r in force_raw:
        sim_id = r.get("sim_id")
        res = r.get("resolution")
        if sim_id and res:
            force_index[(sim_id, res)] = r

    input_by_sim = {r["sim_id"]: r for r in input_raw if r.get("sim_id")}

    return {
        "pipeline": pipeline,
        "fields": fields,
        "wrinkle_registry": wrinkle_by_id,
        "force_stroke": force_index,
        "input_params": input_by_sim,
    }


def build_sim_catalog(configs: dict[str, Any]) -> list[SimRecord]:
    pipeline = configs["pipeline"]
    wrinkle = configs["wrinkle_registry"]
    force = configs["force_stroke"]
    input_params = configs["input_params"]

    out: list[SimRecord] = []
    for batch in ("A", "B"):
        bcfg = pipeline["batches"][batch]
        root = Path(pipeline["data"]["batch_a_root"] if batch == "A" else pipeline["data"]["batch_b_root"])
        results_dirs = sorted(root.glob(bcfg["folder_glob"]))
        for results_dir in results_dirs:
            sim_base = _sim_id_from_results_dir_name(results_dir.name)

            runs = _find_runs(results_dir)
            valid_runs = [r for r in runs if _mesh_size(r) > 0]
            if not valid_runs:
                continue
            valid_runs.sort(key=_mesh_size)
            coarse_default = valid_runs[0] if len(valid_runs) >= 2 else None
            fine_default = valid_runs[-1]

            if batch == "A" and sim_base in {"geom_0_0", "geom_0_1"}:
                sim_candidates = [f"{sim_base}_pair1", f"{sim_base}_pair2"]
            else:
                sim_candidates = [sim_base]

            for sim_id in sim_candidates:
                if sim_id == "geom_0_1_pair2":
                    continue
                wr_key = sim_id if sim_id in wrinkle else _sim_base_id(sim_id)
                wr = wrinkle.get(wr_key, {})

                fine = fine_default
                fine_run = wr.get("fine_run")
                if isinstance(fine_run, str) and fine_run.strip():
                    matching = [r for r in valid_runs if r.name == fine_run.strip()]
                    if matching:
                        fine = matching[-1]

                coarse = coarse_default
                if coarse is not None and fine is not None and coarse == fine and len(valid_runs) >= 2:
                    coarse = valid_runs[-2]

                fs_key = (_sim_base_id(sim_id), "fine")
                fs = force.get(fs_key, {})

                in_key = _sim_base_id(sim_id)
                inp = input_params.get(in_key, {})

                orientations = inp.get("ply_orientations_deg")
                if not orientations:
                    orientations = bcfg.get("ply_orientations_deg") or []

                max_sev = float(wr.get("max_compound_severity", 0.0) or 0.0)
                onset_increment = int(wr.get("onset_increment")) if wr.get("onset_increment") is not None else -1
                onset_stroke = (
                    float(wr["onset_stroke_fraction"]) if wr.get("onset_stroke_fraction") is not None else float("nan")
                )

                rec = SimRecord(
                    sim_id=sim_id,
                    batch=batch,
                    fine_dir=fine,
                    coarse_dir=coarse,
                    ply_groups=[int(x) for x in bcfg["ply_groups"]],
                    bending_groups=[int(x) for x in bcfg["bending_groups"]],
                    compound_severity=max_sev,
                    is_wrinkled=bool(max_sev > 0.10),
                    onset_increment=onset_increment,
                    onset_stroke_frac=onset_stroke,
                    uz_range_mm=list(fs.get("uz_range_mm", [float("nan"), float("nan")])),
                    max_rfz_N=float(fs.get("rfz_max_N", float("nan"))),
                    n_plies=int(inp.get("n_plies", bcfg["n_plies"])),
                    material=str(bcfg["material"]),
                    ply_orientations_deg=[float(x) for x in orientations],
                )
                out.append(rec)

    return sorted(out, key=lambda r: r.sim_id)


def read_reference_mesh(run_dir: Path, groups: list[int]) -> tuple[np.ndarray, np.ndarray]:
    afm_path = _find_afm_path(run_dir)
    nodes_by_group, elems_by_group, _ = ReadAFMesh(str(afm_path), elemGrNrsToExport=groups, silent=True)

    all_nodes = []
    for grp in groups:
        if grp in nodes_by_group:
            all_nodes.append(nodes_by_group[grp])
    if not all_nodes:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    nodes_cat = np.concatenate(all_nodes, axis=0)
    node_ids = nodes_cat[:, 0].astype(np.int64)
    xyz = nodes_cat[:, 1:4].astype(np.float32)
    order = np.argsort(node_ids)
    node_ids = node_ids[order]
    xyz = xyz[order]
    unique_ids, unique_idx = np.unique(node_ids, return_index=True)
    xyz = xyz[unique_idx]

    id_to_zero = {int(nid): i for i, nid in enumerate(unique_ids.tolist())}

    all_elems = []
    for grp in groups:
        if grp in elems_by_group:
            all_elems.append(elems_by_group[grp])
    if not all_elems:
        return xyz, np.zeros((0, 3), dtype=np.int32)

    mapped_parts: list[np.ndarray] = []
    n_total = 0
    n_discarded = 0
    for grp in groups:
        if grp not in elems_by_group or grp not in nodes_by_group:
            continue
        grp_nodes = nodes_by_group[grp]
        grp_node_ids = grp_nodes[:, 0].astype(np.int64)
        grp_local0_to_global = np.array([id_to_zero[int(nid)] for nid in grp_node_ids], dtype=np.int32)

        grp_conn = elems_by_group[grp][:, 1:4].astype(np.int64)
        n_total += int(grp_conn.shape[0])
        if grp_conn.size == 0:
            continue

        cmin = int(grp_conn.min())
        cmax = int(grp_conn.max())
        n_grp_nodes = int(grp_nodes.shape[0])

        # AniForm .afm connectivity is typically local node indexing per group.
        if cmin >= 0 and cmax < n_grp_nodes:
            mapped_grp = grp_local0_to_global[grp_conn]
            mapped_parts.append(mapped_grp.astype(np.int32))
            continue
        if cmin >= 1 and cmax <= n_grp_nodes:
            mapped_grp = grp_local0_to_global[grp_conn - 1]
            mapped_parts.append(mapped_grp.astype(np.int32))
            continue

        # Fallback for connectivity encoded as global node IDs.
        mapped_grp = np.array([[id_to_zero.get(int(n), -1) for n in tri] for tri in grp_conn], dtype=np.int32)
        keep_mask = (mapped_grp >= 0).all(axis=1)
        n_discarded += int((~keep_mask).sum())
        kept_grp = mapped_grp[keep_mask]
        if kept_grp.size:
            mapped_parts.append(kept_grp)

    kept = np.concatenate(mapped_parts, axis=0) if mapped_parts else np.zeros((0, 3), dtype=np.int32)
    n_kept = int(kept.shape[0])
    n_discarded += max(0, n_total - n_kept - n_discarded)
    print(f"[mesh] {run_dir.name}: Stored {n_kept}/{n_total} elements ({n_discarded} had out-of-group nodes)")
    return xyz, kept


def read_field_all_increments(
    afr_path: Path, afr_id: int, sub_id: int, groups: list[int]
) -> tuple[np.ndarray, list[int]]:
    if not afr_path.exists():
        return np.zeros((0, 0), dtype=np.float32), []

    results, _, increments, res_type, indices_included, _ = ReadAFResult(
        str(afr_path), elemGrNrs=groups, IncsToExport=[], silent=True
    )
    if isinstance(res_type, list):
        return np.zeros((0, 0), dtype=np.float32), []

    uniq_incr = sorted(set(int(i) for i in increments))
    frames = []
    for incr in uniq_incr:
        chunks = []
        for grp in groups:
            arr = results.get(incr, {}).get(grp)
            if arr is None:
                continue
            chunks.append(arr)
        if not chunks:
            continue
        cat = np.concatenate(chunks, axis=0)
        if indices_included:
            node_idx = cat[:, 0].astype(np.int64)
            vals = cat[:, 1:]
            order = np.argsort(node_idx)
            node_idx = node_idx[order]
            vals = vals[order]
            _, first_idx = np.unique(node_idx, return_index=True)
            vals = vals[first_idx]
        else:
            # ReadAFResult prepends a zero-padding column when
            # indices_included=False.  Strip it so downstream shapes
            # match the expected component counts (3 for VectorT /
            # STensorPSST, 1 for ScalarT).
            if cat.shape[1] > 1 and np.all(cat[:, 0] == 0):
                vals = cat[:, 1:]
            else:
                vals = cat
        frames.append(vals.astype(np.float32))

    if not frames:
        return np.zeros((0, 0), dtype=np.float32), []

    n_nodes = min(f.shape[0] for f in frames)
    frames = [f[:n_nodes] for f in frames]
    data = np.stack(frames, axis=0)
    if data.shape[-1] == 1:
        data = data[..., 0]
    return data.astype(np.float32), uniq_incr[: data.shape[0]]


def read_increment_times(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    afs_path = _find_afs_path(run_dir)
    _, _, incr_info, _, _, _, _ = ReadAFSFile(str(afs_path), silent=True)
    incr_numbers = incr_info.loc["incr_nr"].values.astype(np.int32)
    times = incr_info.loc["t_end"].values.astype(np.float32)
    loadblock = incr_info.loc["current_loadblock"].values.astype(np.int32)

    forming = times[loadblock == 2]
    if forming.size >= 2 and float(forming.max() - forming.min()) > 0.0:
        t0, t1 = float(forming.min()), float(forming.max())
    else:
        t0, t1 = float(times.min()), float(times.max())
    denom = max(t1 - t0, 1e-8)
    stroke = np.clip((times - t0) / denom, 0.0, 1.0).astype(np.float32)
    return incr_numbers, times, stroke


def _severity_from_components(batch: str, comp_frac: np.ndarray, dz_var: np.ndarray, shear: np.ndarray) -> np.ndarray:
    w_fiber, w_var, w_shear = 0.50, 0.35, 0.15
    comp_scale = 0.50
    dz_scale = 10.0 if batch == "A" else 0.40
    shear_scale = 90.0 if batch == "A" else 45.0
    s_fiber = np.clip(comp_frac / max(comp_scale, 1e-8), 0.0, 1.0)
    s_var = np.clip(dz_var / max(dz_scale, 1e-8), 0.0, 1.0)
    s_shear = np.clip(shear / max(shear_scale, 1e-8), 0.0, 1.0)
    return (w_fiber * s_fiber + w_var * s_var + w_shear * s_shear).astype(np.float32)


def compute_derived(
    batch: str, displacement: np.ndarray, fiber_stress_1: np.ndarray, shear_angle: np.ndarray
) -> dict[str, np.ndarray]:
    n_incr = displacement.shape[0]

    # dz is always the LAST component (handles both 3-col and legacy 4-col layouts)
    if displacement.ndim == 3 and displacement.shape[2] >= 3:
        dz = displacement[:, :, -1]
    else:
        dz = np.zeros((n_incr, 0), dtype=np.float32)

    # fiber_stress_1: scalar field — may be (n_incr, n_nodes) or (n_incr, n_nodes, 2) legacy
    if fiber_stress_1.ndim == 3:
        fs = fiber_stress_1[:, :, -1]  # last col = actual value
    elif fiber_stress_1.ndim == 2:
        fs = fiber_stress_1
    else:
        fs = np.zeros((n_incr, 0), dtype=np.float32)
    comp_frac = (fs < -0.05).mean(axis=1).astype(np.float32) if fs.size else np.zeros((n_incr,), dtype=np.float32)

    dz_variance = np.var(dz, axis=1).astype(np.float32) if dz.size else np.zeros((n_incr,), dtype=np.float32)

    # shear_angle: scalar — may be (n_incr, n_nodes) or (n_incr, n_nodes, 2) legacy
    if shear_angle.ndim == 3:
        sa = shear_angle[:, :, -1]
    elif shear_angle.ndim == 2:
        sa = shear_angle
    else:
        sa = np.zeros((n_incr, 0), dtype=np.float32)
    max_shear = np.max(np.abs(sa), axis=1).astype(np.float32) if sa.size else np.zeros((n_incr,), dtype=np.float32)

    severity = _severity_from_components(batch, comp_frac, dz_variance, max_shear)
    return {"comp_frac_f1": comp_frac, "dz_variance": dz_variance, "severity": severity}


def _write_dataset(grp: h5py.Group, name: str, arr: np.ndarray, compress: bool = True) -> h5py.Dataset:
    kwargs: dict[str, Any] = {}
    if compress:
        kwargs["compression"] = COMPRESSION
    if arr.ndim >= 2:
        if arr.ndim == 3:
            kwargs["chunks"] = (CHUNK_TIME_DIM, min(arr.shape[1], CHUNK_NODE_DIM), arr.shape[2])
        elif arr.ndim == 2:
            kwargs["chunks"] = (CHUNK_TIME_DIM, min(arr.shape[1], CHUNK_NODE_DIM))
    return grp.create_dataset(name, data=arr, **kwargs)


def write_sim_to_h5(h5: h5py.File, rec: SimRecord, field_data: dict[str, Any]) -> None:
    sim_grp = h5.require_group("simulations").create_group(rec.sim_id)
    sim_grp.attrs["batch"] = rec.batch
    sim_grp.attrs["material"] = rec.material
    sim_grp.attrs["n_plies"] = int(rec.n_plies)
    sim_grp.attrs["ply_groups"] = np.array(rec.ply_groups, dtype=np.int32)
    sim_grp.attrs["fine_run_dir"] = str(rec.fine_dir)
    sim_grp.attrs["n_nodes_fine"] = int(field_data["fine"]["mesh_nodes"].shape[0])
    sim_grp.attrs["n_nodes_coarse"] = int(field_data["coarse"]["mesh_nodes"].shape[0]) if field_data["coarse"] else 0
    sim_grp.attrs["uz_range_mm"] = np.array(rec.uz_range_mm, dtype=np.float32)
    sim_grp.attrs["max_rfz_N"] = float(rec.max_rfz_N)
    sim_grp.attrs["ply_orientations_deg"] = np.array(rec.ply_orientations_deg, dtype=np.float32)
    sim_grp.attrs["compound_severity"] = float(rec.compound_severity)
    sim_grp.attrs["is_wrinkled"] = bool(rec.is_wrinkled)
    sim_grp.attrs["onset_increment"] = int(rec.onset_increment)
    sim_grp.attrs["onset_stroke_frac"] = float(rec.onset_stroke_frac) if not math.isnan(rec.onset_stroke_frac) else np.nan

    mesh_grp = sim_grp.create_group("mesh")
    fine_mesh = mesh_grp.create_group("fine")
    fine_mesh.create_dataset("nodes", data=field_data["fine"]["mesh_nodes"], compression=COMPRESSION)
    fine_mesh.create_dataset("elements", data=field_data["fine"]["mesh_elements"], compression=COMPRESSION)

    if field_data["coarse"]:
        coarse_mesh = mesh_grp.create_group("coarse")
        coarse_mesh.create_dataset("nodes", data=field_data["coarse"]["mesh_nodes"], compression=COMPRESSION)
        coarse_mesh.create_dataset("elements", data=field_data["coarse"]["mesh_elements"], compression=COMPRESSION)

    for level in ("fine", "coarse"):
        if not field_data[level]:
            continue
        lvl_grp = sim_grp.create_group(level)
        _write_dataset(lvl_grp, "increments", field_data[level]["increments"].astype(np.int32), compress=False)
        _write_dataset(lvl_grp, "times_s", field_data[level]["times_s"].astype(np.float32), compress=False)
        _write_dataset(lvl_grp, "stroke_frac", field_data[level]["stroke_frac"].astype(np.float32), compress=False)

        for fname in FIELD_DEFS:
            arr = field_data[level]["fields"].get(fname)
            if arr is None:
                if fname == "crystallinity" and rec.batch == "B":
                    ds = lvl_grp.create_dataset("crystallinity", shape=(field_data[level]["increments"].shape[0], 0), dtype=np.float32)
                    ds.attrs["available"] = False
                continue
            ds = _write_dataset(lvl_grp, fname, arr.astype(np.float32), compress=True)
            if fname == "crystallinity":
                ds.attrs["available"] = True

        drv = lvl_grp.create_group("derived")
        for k, v in field_data[level]["derived"].items():
            _write_dataset(drv, k, v.astype(np.float32), compress=True)


def write_metadata(h5: h5py.File, configs: dict[str, Any], n_pairs: int) -> None:
    meta = h5.create_group("metadata")
    meta.create_dataset("field_registry", data=json.dumps(configs["fields"]))
    meta.create_dataset("pipeline_config", data=json.dumps(configs["pipeline"]))
    build = meta.create_group("build_info")
    build.attrs["date"] = datetime.now().isoformat()
    build.attrs["n_pairs"] = int(n_pairs)
    build.attrs["n_fields"] = int(len(FIELD_DEFS))
    build.attrs["schema_version"] = SCHEMA_VERSION
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_ROOT), stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_hash = ""
    build.attrs["git_hash"] = git_hash


def write_splits(h5: h5py.File, records: list[SimRecord]) -> None:
    sim_ids = np.array([r.sim_id for r in records], dtype=object)
    severities = np.array([float(r.compound_severity) for r in records], dtype=np.float32)
    batches = np.array([r.batch for r in records], dtype=object)

    quantiles = np.quantile(severities, [0.25, 0.5, 0.75])
    sev_bins = np.digitize(severities, quantiles, right=True)
    labels = np.array([f"{b}_Q{int(q)}" for b, q in zip(batches, sev_bins)], dtype=object)

    splits = h5.create_group("splits")
    splits.attrs["strategy"] = "stratified_batch_severity_quartile"
    splits.attrs["seed"] = CV_RANDOM_SEED
    splits.attrs["n_folds"] = CV_N_FOLDS
    dtype = h5py.string_dtype("utf-8")

    skf = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=CV_RANDOM_SEED)
    for k, (train_idx, test_idx) in enumerate(skf.split(sim_ids, labels)):
        fold = splits.create_group(f"fold_{k}")
        train_pool_ids = sim_ids[train_idx]
        train_pool_labels = labels[train_idx]
        try:
            tr, va = train_test_split(
                train_pool_ids,
                test_size=0.2,
                random_state=CV_RANDOM_SEED + k,
                stratify=train_pool_labels,
            )
        except ValueError:
            tr, va = train_test_split(
                train_pool_ids,
                test_size=0.2,
                random_state=CV_RANDOM_SEED + k,
                stratify=None,
            )
        fold.create_dataset("train", data=tr, dtype=dtype)
        fold.create_dataset("val", data=va, dtype=dtype)
        fold.create_dataset("test", data=sim_ids[test_idx], dtype=dtype)


def _collect_level_data(rec: SimRecord, run_dir: Path | None, batch: str) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    afr_dir = _find_afr_dir(run_dir)
    mesh_nodes, mesh_elems = read_reference_mesh(run_dir, rec.ply_groups)
    incr_numbers, times_s, stroke_frac = read_increment_times(run_dir)

    fields: dict[str, np.ndarray | None] = {}
    field_increments: dict[str, list[int]] = {}
    for field_name, (afr_id, sub_id, _shape_suffix, groups_key) in FIELD_DEFS.items():
        if field_name == "crystallinity" and batch == "B":
            fields[field_name] = None
            field_increments[field_name] = []
            continue
        groups = rec.bending_groups if groups_key == "bending" else rec.ply_groups
        afr_path = afr_dir / f"model_{afr_id}_{sub_id}.afr"
        data, incs = read_field_all_increments(afr_path, afr_id, sub_id, groups)
        fields[field_name] = data
        field_increments[field_name] = [int(i) for i in incs]

    ref_increments = field_increments.get("displacement", [])
    if not ref_increments:
        for incs in field_increments.values():
            if incs:
                ref_increments = incs
                break

    if ref_increments:
        afs_by_incr = {int(i): j for j, i in enumerate(incr_numbers.tolist())}
        aligned_incr = [i for i in ref_increments if i in afs_by_incr]
        if not aligned_incr:
            aligned_incr = ref_increments
        pick_time = [afs_by_incr[i] for i in aligned_incr if i in afs_by_incr]
        if pick_time:
            incr_numbers = incr_numbers[pick_time]
            times_s = times_s[pick_time]
            stroke_frac = stroke_frac[pick_time]
        else:
            n = len(aligned_incr)
            incr_numbers = np.asarray(aligned_incr, dtype=np.int32)
            times_s = np.zeros((n,), dtype=np.float32)
            stroke_frac = np.zeros((n,), dtype=np.float32)
    else:
        aligned_incr = incr_numbers.tolist()

    for field_name, data in list(fields.items()):
        if data is None:
            continue
        incs = field_increments.get(field_name, [])
        if data.shape[0] and incs:
            by_incr = {int(i): j for j, i in enumerate(incs)}
            pick = [by_incr[i] for i in incr_numbers.tolist() if i in by_incr]
            if pick:
                data = data[pick]
            else:
                data = np.zeros((len(incr_numbers),) + tuple(data.shape[1:]), dtype=np.float32)
        if data.shape[0] != len(incr_numbers):
            data = data[: len(incr_numbers)]
        fields[field_name] = data

    displacement = fields.get("displacement")
    fiber_stress_1 = fields.get("fiber_stress_1")
    shear_angle = fields.get("shear_angle")
    if displacement is None or fiber_stress_1 is None or shear_angle is None:
        raise RuntimeError(f"Missing required fields for derived features in {rec.sim_id}")

    derived = compute_derived(batch, displacement, fiber_stress_1, shear_angle)
    return {
        "mesh_nodes": mesh_nodes,
        "mesh_elements": mesh_elems,
        "increments": incr_numbers,
        "times_s": times_s,
        "stroke_frac": stroke_frac,
        "fields": fields,
        "derived": derived,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WP2 HDF5 dataset builder")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sim", type=str, default=None)
    parser.add_argument("--batch", choices=["A", "B"], default=None)
    args = parser.parse_args()

    configs = load_configs()
    catalog = build_sim_catalog(configs)
    if args.batch:
        catalog = [r for r in catalog if r.batch == args.batch]
    if args.sim:
        catalog = [r for r in catalog if r.sim_id == args.sim]

    if args.dry_run:
        print(f"Dry-run catalog size: {len(catalog)}")
        errors = 0
        for rec in catalog:
            try:
                afr_dir = _find_afr_dir(rec.fine_dir)
                for _, (afr_id, sub_id, _, _) in FIELD_DEFS.items():
                    p = afr_dir / f"model_{afr_id}_{sub_id}.afr"
                    if rec.batch == "B" and afr_id == 214:
                        continue
                    if not p.exists():
                        raise FileNotFoundError(str(p))
                print(f"[OK] {rec.sim_id}")
            except Exception as exc:
                errors += 1
                print(f"[ERR] {rec.sim_id}: {exc}")
        print(f"Dry-run complete: errors={errors}")
        return

    H5_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5_PATH, "w") as h5:
        write_metadata(h5, configs, n_pairs=len(catalog))
        for i, rec in enumerate(catalog, 1):
            print(f"[{i}/{len(catalog)}] {rec.sim_id}")
            fine_data = _collect_level_data(rec, rec.fine_dir, rec.batch)
            coarse_data = _collect_level_data(rec, rec.coarse_dir, rec.batch) if rec.coarse_dir else None
            write_sim_to_h5(h5, rec, {"fine": fine_data, "coarse": coarse_data})
        write_splits(h5, catalog)
    print(f"Built: {H5_PATH}")


if __name__ == "__main__":
    main()

