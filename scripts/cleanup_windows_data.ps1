<#
.SYNOPSIS
    Delete intermediate NPZ files and duplicate AniForm result datasets from Windows.

.DESCRIPTION
    Two phases:
      Phase 1 — CFWrinklePredict2/data/ intermediate outputs (NPZ, visualisations,
                 train/val splits). These are all regenerable from the canonical HDF5.
      Phase 2 — Duplicate AniForm result folders in CFAIMesh that are copies of the
                 canonical data already in CFWrinklePredict2/data/aniform_raw and
                 simulation_batch_271125.

    What is NOT touched:
      - CFAIMesh/Aniform/laminate setup.Results/   (setup experiments, not canonical)
      - CFAIMesh/Aniform/testing setup.Results/    (setup experiments, not canonical)
      - CFAIMesh/sim batch 261125/                 (different batch number — check before deleting)
      - CFWrinklePredict2/data/aniform_raw/        (canonical Batch A — kept for archiving)
      - CFWrinklePredict2/data/simulation_batch_271125/ (canonical Batch B)
      - All .py source files in CFWrinklePredict2/data/

.PARAMETER Execute
    Actually delete. Without this flag the script is a dry run (WhatIf mode).
    Always run without -Execute first to confirm what will be removed.

.EXAMPLE
    # See what would be deleted (safe — no changes)
    .\scripts\cleanup_windows_data.ps1

    # Actually delete
    .\scripts\cleanup_windows_data.ps1 -Execute
#>

param([switch]$Execute)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Predict2 = "C:\Users\ellis\Documents\VS Code\CFWrinklePredict2"
$CFAIMesh  = "C:\Users\ellis\Documents\VS Code\CFAIMesh"

# ---------------------------------------------------------------------------
# Phase 1 — Intermediate derived dirs in CFWrinklePredict2/data/
# ---------------------------------------------------------------------------
$NpzDirs = @(
    "extended_npz",
    "extended_npz_v2",
    "extended_train",
    "extended_train_bad",
    "extended_train_v2",
    "extended_val",
    "processed_test",
    "test_complete_npz",
    "test_full_npz",
    "ud_tape_npz",
    "visualizations",
    "visualizations_complete"
) | ForEach-Object { Join-Path "$Predict2\data" $_ }

# ---------------------------------------------------------------------------
# Phase 2 — Duplicate AniForm results in CFAIMesh
# ---------------------------------------------------------------------------
# All geom 0_N.Results dirs under CFAIMesh/Aniform (Batch A duplicates)
$AniFormGeomResults = Get-ChildItem -Path "$CFAIMesh\Aniform" -Directory -Filter "geom 0_*.Results" -ErrorAction SilentlyContinue

# Batch B partial duplicate and stray geom copy
$AniFormOtherDups = @(
    "$CFAIMesh\sim batch 271125",
    "$CFAIMesh\punch_die_sets\geom 0_5.Results"
) | Where-Object { Test-Path $_ }

# ---------------------------------------------------------------------------
# Helper: measure size
# ---------------------------------------------------------------------------
function Get-DirSize([string]$path) {
    if (-not (Test-Path $path)) { return 0L }
    (Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue |
     Measure-Object -Property Length -Sum).Sum ?? 0L
}

function Format-GB([long]$bytes) {
    if ($bytes -lt 1MB) { return "$([int]($bytes/1KB)) KB" }
    if ($bytes -lt 1GB) { return "{0:N1} MB" -f ($bytes/1MB) }
    "{0:N1} GB" -f ($bytes/1GB)
}

function Remove-IfExists([string]$path, [string]$label) {
    if (-not (Test-Path $path)) {
        Write-Host "  (not found — skip) $label" -ForegroundColor DarkGray
        return 0L
    }
    $size = Get-DirSize $path
    $sizeStr = Format-GB $size
    if ($Execute) {
        Remove-Item $path -Recurse -Force
        Write-Host "  DELETED [$sizeStr]  $label" -ForegroundColor Red
    } else {
        Write-Host "  would delete [$sizeStr]  $label" -ForegroundColor Yellow
    }
    return $size
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
$mode = if ($Execute) { "EXECUTE" } else { "DRY RUN — pass -Execute to actually delete" }
Write-Host ""
Write-Host "=== CFWrinkle Windows Data Cleanup — $mode ===" -ForegroundColor Cyan
Write-Host ""

$totalBytes = 0L

Write-Host "-- Phase 1: CFWrinklePredict2 intermediate NPZ/visualisation dirs --" -ForegroundColor Cyan
foreach ($dir in $NpzDirs) {
    $totalBytes += Remove-IfExists $dir (Split-Path $dir -Leaf)
}

Write-Host ""
Write-Host "-- Phase 2: Duplicate AniForm results in CFAIMesh --" -ForegroundColor Cyan
foreach ($dir in $AniFormGeomResults) {
    $totalBytes += Remove-IfExists $dir.FullName $dir.Name
}
foreach ($dir in $AniFormOtherDups) {
    $totalBytes += Remove-IfExists $dir (Split-Path $dir -Leaf)
}

Write-Host ""
Write-Host "-- NOT touched (review separately) --" -ForegroundColor DarkCyan
@(
    "$CFAIMesh\Aniform\laminate setup.Results  ← setup experiments (Nov 2025, not canonical dataset)",
    "$CFAIMesh\Aniform\testing setup.Results   ← setup experiments (Nov 2025, not canonical dataset)",
    "$CFAIMesh\sim batch 261125                ← different batch number — confirm before deleting"
) | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkCyan }

Write-Host ""
if ($Execute) {
    Write-Host "Done. Total freed: $(Format-GB $totalBytes)" -ForegroundColor Green
} else {
    Write-Host "Dry run complete. Total to free: $(Format-GB $totalBytes)" -ForegroundColor Yellow
    Write-Host "Re-run with -Execute to delete." -ForegroundColor Yellow
}
Write-Host ""
