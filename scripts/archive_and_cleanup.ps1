<#
.SYNOPSIS
    Staged archive and cleanup -- 103 GB free disk, ~214 GB to archive.

.DESCRIPTION
    Runs five phases in sequence. Each phase reports free disk space.
    Archives are verified before their source is deleted.
    Phases that have already completed are skipped automatically.

    Phase 0 -- Delete extraneous (~152 GB freed, no archiving):
                NPZ intermediates, CFAIMesh duplicate results,
                laminate/testing setup exploration runs.
    Phase 1 -- Archive Batch A  -> verify -> delete source   (~48 GB)
    Phase 2 -- Archive Batch B  -> verify -> delete source   (~52 GB)
    Phase 3 -- Archive WP3 HDF5 -> verify WSL -> delete Windows copy (~59 GB)
    Phase 4 -- Archive WP2 HDF5 -> verify WSL -> delete Windows copy (~55 GB)

    NOT deleted at any phase:
      CFAIMesh/sim batch 261125/         (setup reference for batch 271125)
      CFWrinklePredict2/data/*.py        (source code)
      CFAIMesh project/tooling files

.PARAMETER ArchiveDir
    Where to write archives. Default: C:\Users\ellis\Documents\CFWrinkle_Archive

.PARAMETER StartPhase
    Resume from this phase (0-4). Default: 0 (run all).

.PARAMETER DryRun
    Print what each phase would do without making changes.

.EXAMPLE
    .\scripts\archive_and_cleanup.ps1 -DryRun
    .\scripts\archive_and_cleanup.ps1
    .\scripts\archive_and_cleanup.ps1 -StartPhase 2
#>

param(
    [string]$ArchiveDir  = "C:\Users\ellis\Documents\CFWrinkle_Archive",
    [int]   $StartPhase  = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoDir  = Split-Path -Parent $PSScriptRoot
$Predict2 = "C:\Users\ellis\Documents\VS Code\CFWrinklePredict2"
$CFAIMesh = "C:\Users\ellis\Documents\VS Code\CFAIMesh"
$DataDir  = Join-Path $RepoDir "data"

# ---------------------------------------------------------------------------
# 7-Zip (required)
# ---------------------------------------------------------------------------
$SevenZip = $null
$_7zCmd = Get-Command "7z" -ErrorAction SilentlyContinue
$_7zFromPath = if ($_7zCmd) { $_7zCmd.Source } else { $null }
foreach ($c in @(
    "C:\Program Files\7-Zip\7z.exe",
    "C:\Program Files (x86)\7-Zip\7z.exe",
    $_7zFromPath
)) { if ($c -and (Test-Path $c)) { $SevenZip = $c; break } }

if (-not $SevenZip) {
    Write-Error "7-Zip not found. Install from https://7-zip.org"
    exit 1
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Format-Size([long]$bytes) {
    if ($bytes -ge 1GB) { return "{0:N1} GB" -f ($bytes / 1GB) }
    if ($bytes -ge 1MB) { return "{0:N0} MB" -f ($bytes / 1MB) }
    return "$([int]($bytes / 1KB)) KB"
}

function Get-FreeGB {
    $bytes = (Get-PSDrive C).Free
    return "{0:N1} GB" -f ($bytes / 1GB)
}

function Get-ItemSize([string]$path) {
    if (-not (Test-Path $path)) { return 0L }
    if (Test-Path $path -PathType Leaf) { return (Get-Item $path).Length }
    $sum = Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue |
           Measure-Object -Property Length -Sum |
           Select-Object -ExpandProperty Sum
    if ($sum) { [long]$sum } else { 0L }
}

function Remove-Target([string]$path, [string]$label = "") {
    $tag = if ($label) { $label } else { Split-Path $path -Leaf }
    if (-not (Test-Path $path)) {
        Write-Host "  (already gone) $tag" -ForegroundColor DarkGray
        return
    }
    $sz = Get-ItemSize $path
    if ($DryRun) {
        Write-Host "  [DRY RUN] would delete $(Format-Size $sz)  $tag" -ForegroundColor Yellow
    } else {
        Remove-Item $path -Recurse -Force
        Write-Host "  DELETED $(Format-Size $sz)  $tag" -ForegroundColor Red
    }
}

function Invoke-7z([string[]]$CmdArgs) {
    if ($DryRun) {
        Write-Host "  [DRY RUN] 7z $($CmdArgs -join ' ')" -ForegroundColor Yellow
        return
    }
    & $SevenZip @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "7-Zip failed (exit $LASTEXITCODE)" }
}

function New-Archive([string]$label, [string]$source, [string]$outPath, [bool]$compress) {
    if (-not (Test-Path $source)) {
        Write-Warning "  [$label] Source not found: $source"
        return
    }
    if (Test-Path $outPath) {
        Write-Host "  [$label] Archive exists -- skipping creation." -ForegroundColor DarkYellow
        return
    }

    $srcSz = Get-ItemSize $source
    Write-Host "  [$label] Archiving $(Format-Size $srcSz) -> $outPath" -ForegroundColor Cyan
    $level = if ($compress) { "3" } else { "0" }   # 0=store for HDF5 (already compressed)
    $t0 = Get-Date
    Invoke-7z @("a", "-t7z", "-mx=$level", "-mmt=on", $outPath, $source)
    if (-not $DryRun) {
        $elapsed = (Get-Date) - $t0
        $archSz = (Get-Item $outPath).Length
        Write-Host "  [$label] Done $([int]$elapsed.TotalMinutes)m$($elapsed.Seconds)s -- $(Format-Size $archSz)" -ForegroundColor Green
    }
}

function Test-Archive([string]$label, [string]$outPath) {
    if ($DryRun) {
        Write-Host "  [DRY RUN] would verify $outPath" -ForegroundColor Yellow
        return
    }
    if (-not (Test-Path $outPath)) { throw "Archive not found: $outPath" }
    Write-Host "  [$label] Verifying..." -ForegroundColor DarkGray
    Invoke-7z @("t", $outPath)
    Write-Host "  [$label] Verified OK" -ForegroundColor Green
}

function Test-WslHdf5([string]$defaultPath, [string]$label) {
    if ($DryRun) {
        Write-Host "  [DRY RUN] would verify WSL $label" -ForegroundColor Yellow
        return
    }
    Write-Host "  Checking WSL copy of $label ..." -ForegroundColor DarkGray
    $pyCode = "import h5py,sys`ntry:`n with h5py.File('$defaultPath','r') as f: n=len(list(f['simulations'].keys()))`n print('OK '+str(n)+' sims')`nexcept Exception as e:`n print('FAIL '+str(e))`n sys.exit(1)"
    $result = wsl -d Ubuntu-24.04 -- python3 -c $pyCode
    Write-Host "  WSL: $result"
    if ($result -notmatch "^OK") { throw "WSL HDF5 check failed for $label" }
}

function Write-Phase([int]$n, [string]$title) {
    Write-Host ""
    Write-Host "=== Phase $n -- $title  [free: $(Get-FreeGB)] ===" -ForegroundColor Cyan
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== CFWrinkle Staged Archive & Cleanup ===" -ForegroundColor Cyan
if ($DryRun) { Write-Host "DRY RUN -- no changes will be made" -ForegroundColor Yellow }
Write-Host "Archive dir: $ArchiveDir"
Write-Host "Start phase: $StartPhase"
Write-Host "Free now:    $(Get-FreeGB)"
Write-Host ""

if (-not $DryRun -and -not (Test-Path $ArchiveDir)) {
    New-Item -ItemType Directory -Path $ArchiveDir -Force | Out-Null
}

# ---------------------------------------------------------------------------
# Phase 0 -- Delete extraneous (no archiving, just free space)
# ---------------------------------------------------------------------------
if ($StartPhase -le 0) {
    Write-Phase 0 "Delete extraneous data"

    Write-Host "-- Material files (archive before anything is deleted) --"
    $matOut = "$ArchiveDir\cfwrinkle_materials.7z"
    if (Test-Path $matOut) {
        Write-Host "  Material archive already exists -- skipping." -ForegroundColor DarkYellow
    } elseif ($DryRun) {
        Write-Host "  [DRY RUN] would archive CFWrinklePredict2/data/materials/ + CFAIMesh/General material data -> $matOut" -ForegroundColor Yellow
    } else {
        New-Item -ItemType Directory -Path $ArchiveDir -Force | Out-Null
        Invoke-7z @("a", "-t7z", "-mx=5", "-mmt=on", $matOut,
            "$Predict2\data\materials",
            "$CFAIMesh\General material data 2025-04-14")
        Write-Host "  Material archive created: $matOut" -ForegroundColor Green
    }
    Write-Host ""

    Write-Host "-- NPZ / derived dirs in CFWrinklePredict2/data --"
    @(
        "extended_npz", "extended_npz_v2",
        "extended_train", "extended_train_bad", "extended_train_v2",
        "extended_val", "processed_test",
        "test_complete_npz", "test_full_npz",
        "ud_tape_npz", "visualizations", "visualizations_complete"
    ) | ForEach-Object { Remove-Target "$Predict2\data\$_" }

    Write-Host ""
    Write-Host "-- Duplicate Batch A results in CFAIMesh/Aniform --"
    Get-ChildItem "$CFAIMesh\Aniform" -Directory -Filter "geom 0_*.Results" -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Target $_.FullName $_.Name }

    Write-Host ""
    Write-Host "-- Duplicate Batch B in CFAIMesh --"
    Remove-Target "$CFAIMesh\sim batch 271125"
    Remove-Target "$CFAIMesh\punch_die_sets\geom 0_5.Results"

    Write-Host ""
    Write-Host "-- Exploration/setup runs in CFAIMesh --"
    Remove-Target "$CFAIMesh\Aniform\laminate setup.Results"
    Remove-Target "$CFAIMesh\Aniform\testing setup.Results"

    Write-Host ""
    Write-Host "Phase 0 complete. Free: $(Get-FreeGB)" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Phase 1 -- Archive Batch A, delete source
# ---------------------------------------------------------------------------
if ($StartPhase -le 1) {
    Write-Phase 1 "Archive Batch A -> delete source"
    $out = "$ArchiveDir\cfwrinkle_aniform_batchA.7z"
    New-Archive "Batch A" "$Predict2\data\aniform_raw" $out $true
    Test-Archive "Batch A" $out
    Remove-Target "$Predict2\data\aniform_raw" "aniform_raw"
    Write-Host "Phase 1 complete. Free: $(Get-FreeGB)" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Phase 2 -- Archive Batch B, delete source
# ---------------------------------------------------------------------------
if ($StartPhase -le 2) {
    Write-Phase 2 "Archive Batch B -> delete source"
    $out = "$ArchiveDir\cfwrinkle_aniform_batchB.7z"
    New-Archive "Batch B" "$Predict2\data\simulation_batch_271125" $out $true
    Test-Archive "Batch B" $out
    Remove-Target "$Predict2\data\simulation_batch_271125" "simulation_batch_271125"
    Write-Host "Phase 2 complete. Free: $(Get-FreeGB)" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Phase 3 -- Archive WP3 HDF5, verify WSL, delete Windows copy
# ---------------------------------------------------------------------------
if ($StartPhase -le 3) {
    Write-Phase 3 "Archive WP3 HDF5 -> verify WSL -> delete Windows copy"
    $src = "$DataDir\cfwrinkle_wp3_features.h5"
    $out = "$ArchiveDir\cfwrinkle_hdf5_wp3.7z"
    New-Archive "WP3 HDF5" $src $out $false
    Test-Archive "WP3 HDF5" $out
    Test-WslHdf5 "/home/ellis/cfwrinkle/data/cfwrinkle_wp3_features.h5" "WP3"
    Remove-Target $src "cfwrinkle_wp3_features.h5"
    Write-Host "Phase 3 complete. Free: $(Get-FreeGB)" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Phase 4 -- Archive WP2 HDF5, verify WSL, delete Windows copy
# ---------------------------------------------------------------------------
if ($StartPhase -le 4) {
    Write-Phase 4 "Archive WP2 HDF5 -> verify WSL -> delete Windows copy"
    $src = "$DataDir\cfwrinkle_dataset.h5"
    $out = "$ArchiveDir\cfwrinkle_hdf5_wp2.7z"
    New-Archive "WP2 HDF5" $src $out $false
    Test-Archive "WP2 HDF5" $out
    Test-WslHdf5 "/home/ellis/cfwrinkle/data/cfwrinkle_dataset.h5" "WP2"
    Remove-Target $src "cfwrinkle_dataset.h5"
    Write-Host "Phase 4 complete. Free: $(Get-FreeGB)" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== All phases complete ===" -ForegroundColor Green
Write-Host "Free disk: $(Get-FreeGB)"
Write-Host ""
Write-Host "Archives in $ArchiveDir :"
Get-ChildItem $ArchiveDir -Filter "*.7z" -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host "  $(Format-Size $_.Length)  $($_.Name)" }
Write-Host ""
Write-Host "Kept (not deleted):" -ForegroundColor DarkCyan
Write-Host "  $CFAIMesh\sim batch 261125   (setup reference)" -ForegroundColor DarkCyan
Write-Host "  $Predict2\data\*.py          (source code)" -ForegroundColor DarkCyan
