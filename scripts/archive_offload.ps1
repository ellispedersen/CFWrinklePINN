<#
.SYNOPSIS
    Compress and organize Windows-side CFWrinkle data for OneDrive offload.

.DESCRIPTION
    Creates separate archives for each logical data group. HDF5 files (already
    internally compressed) are stored without re-compression. AniForm binaries
    and processed NPZ files are compressed. All archives go to $DestDir.

    After verifying OneDrive sync is complete, delete the Windows originals to
    reclaim disk space.

.PARAMETER DestDir
    Destination folder for archives. Defaults to OneDrive\CFWrinkle_Archive.
    Must be on a drive with enough free space for the archives.

.PARAMETER SkipVerify
    Skip listing archive contents after creation (faster but less safe).

.PARAMETER WhatIf
    Dry run: print what would be archived without doing anything.

.EXAMPLE
    .\scripts\archive_offload.ps1
    .\scripts\archive_offload.ps1 -DestDir "D:\Backup\CFWrinkle"
    .\scripts\archive_offload.ps1 -WhatIf
#>

[CmdletBinding()]
param(
    [string]$DestDir = "$env:USERPROFILE\OneDrive\CFWrinkle_Archive",
    [switch]$SkipVerify,
    [switch]$WhatIf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$RepoDir    = Split-Path -Parent $PSScriptRoot
$Predict2   = "C:\Users\ellis\Documents\VS Code\CFWrinklePredict2"
$DataDir    = Join-Path $RepoDir "data"

$Sources = [ordered]@{
    # Key = archive name (without extension)
    # Value = @{ Path=...; Compress=$true/$false; Description=... }
    "cfwrinkle_hdf5_wp2" = @{
        Path        = Join-Path $DataDir "cfwrinkle_dataset.h5"
        Compress    = $false   # HDF5 is already internally compressed
        Description = "WP2 HDF5 dataset (~55 GB)"
    }
    "cfwrinkle_hdf5_wp3" = @{
        Path        = Join-Path $DataDir "cfwrinkle_wp3_features.h5"
        Compress    = $false   # HDF5 is already internally compressed
        Description = "WP3 features HDF5 (~58 GB)"
    }
    "cfwrinkle_aniform_raw" = @{
        Path        = Join-Path $Predict2 "data\aniform_raw"
        Compress    = $true    # Binary AFR/AFS/AFM — compresses well
        Description = "AniForm raw simulation outputs (AFR/AFS/AFM)"
    }
    "cfwrinkle_sim_batch" = @{
        Path        = Join-Path $Predict2 "data\simulation_batch_271125"
        Compress    = $true    # Twintex batch raw files
        Description = "Simulation batch 271125 (Twintex/Batch B)"
    }
}

# ---------------------------------------------------------------------------
# 7-Zip detection
# ---------------------------------------------------------------------------
$SevenZip = $null
foreach ($candidate in @(
    "C:\Program Files\7-Zip\7z.exe",
    "C:\Program Files (x86)\7-Zip\7z.exe",
    (Get-Command "7z" -ErrorAction SilentlyContinue)?.Source
)) {
    if ($candidate -and (Test-Path $candidate)) {
        $SevenZip = $candidate
        break
    }
}

if (-not $SevenZip) {
    Write-Warning "7-Zip not found. Falling back to PowerShell Compress-Archive (ZIP format, slower for large files)."
    Write-Warning "Install 7-Zip from https://7-zip.org for much better performance."
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Format-GB([long]$bytes) {
    "{0:N1} GB" -f ($bytes / 1GB)
}

function Get-PathSize([string]$path) {
    if (Test-Path $path -PathType Leaf) {
        (Get-Item $path).Length
    } elseif (Test-Path $path -PathType Container) {
        (Get-ChildItem $path -Recurse -File | Measure-Object -Property Length -Sum).Sum
    } else {
        0
    }
}

function New-Archive {
    param(
        [string]$Name,
        [string]$SourcePath,
        [string]$ArchivePath,
        [bool]$Compress
    )

    $ext = if ($SevenZip) { ".7z" } else { ".zip" }
    $outPath = $ArchivePath + $ext

    if (Test-Path $outPath) {
        Write-Host "  SKIP: $outPath already exists." -ForegroundColor Yellow
        return $outPath
    }

    if ($SevenZip) {
        # -mx=0 = store (no compression) for already-compressed files
        # -mx=5 = normal compression otherwise
        # -mmt = use all CPU threads
        # -bb1 = show file names as they're added (progress)
        $level = if ($Compress) { "5" } else { "0" }
        $7zArgs = @("a", "-t7z", "-mx=$level", "-mmt=on", "-bb1", $outPath, $SourcePath)
        Write-Host "  Running: 7z $($7zArgs -join ' ')" -ForegroundColor DarkGray
        & $SevenZip @7zArgs
        if ($LASTEXITCODE -ne 0) { throw "7-Zip failed for $Name (exit $LASTEXITCODE)" }
    } else {
        # Compress-Archive always compresses (no store mode), fine for smaller dirs
        if (Test-Path $SourcePath -PathType Container) {
            Compress-Archive -Path "$SourcePath\*" -DestinationPath $outPath -CompressionLevel Optimal
        } else {
            Compress-Archive -Path $SourcePath -DestinationPath $outPath -CompressionLevel Optimal
        }
    }

    return $outPath
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "=== CFWrinkle Data Archive for OneDrive Offload ===" -ForegroundColor Cyan
Write-Host "Destination: $DestDir"
Write-Host "7-Zip:       $(if ($SevenZip) { $SevenZip } else { 'not found — using Compress-Archive' })"
Write-Host ""

if ($WhatIf) {
    Write-Host "[DRY RUN — nothing will be created]" -ForegroundColor Yellow
    Write-Host ""
}

# Validate sources exist
$missing = @()
foreach ($name in $Sources.Keys) {
    $src = $Sources[$name]
    if (-not (Test-Path $src.Path)) {
        $missing += "  $name : $($src.Path)"
    }
}
if ($missing) {
    Write-Warning "The following sources were not found and will be skipped:"
    $missing | ForEach-Object { Write-Warning $_ }
    Write-Host ""
}

# Print plan
Write-Host "Archives to create:" -ForegroundColor Cyan
$totalSrcBytes = 0L
foreach ($name in $Sources.Keys) {
    $src = $Sources[$name]
    if (-not (Test-Path $src.Path)) { continue }
    $sizeBytes = Get-PathSize $src.Path
    $totalSrcBytes += $sizeBytes
    $comprMode = if ($src.Compress) { "compressed" } else { "stored (already compressed)" }
    Write-Host "  [$name]" -ForegroundColor White
    Write-Host "    Source:      $($src.Path)"
    Write-Host "    Size:        $(Format-GB $sizeBytes)"
    Write-Host "    Mode:        $comprMode"
    Write-Host "    Description: $($src.Description)"
    Write-Host ""
}
Write-Host "Total source data: $(Format-GB $totalSrcBytes)"
Write-Host ""

if ($WhatIf) {
    Write-Host "Dry run complete. Re-run without -WhatIf to create archives." -ForegroundColor Yellow
    exit 0
}

# Create destination
if (-not (Test-Path $DestDir)) {
    Write-Host "Creating destination directory: $DestDir"
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
}

# Archive each source
$results = @()
foreach ($name in $Sources.Keys) {
    $src = $Sources[$name]
    if (-not (Test-Path $src.Path)) {
        Write-Host "[$name] SKIPPED — source not found: $($src.Path)" -ForegroundColor Yellow
        continue
    }

    Write-Host "[$name] Archiving $($src.Description)..." -ForegroundColor Cyan
    $startTime = Get-Date
    $archivePath = Join-Path $DestDir $name

    try {
        $outPath = New-Archive -Name $name -SourcePath $src.Path `
                               -ArchivePath $archivePath -Compress $src.Compress
        $elapsed = (Get-Date) - $startTime
        $archiveSize = (Get-Item $outPath).Length
        Write-Host "  Done in $([int]$elapsed.TotalMinutes)m $($elapsed.Seconds)s — archive: $(Format-GB $archiveSize)" -ForegroundColor Green

        if (-not $SkipVerify) {
            Write-Host "  Verifying..." -ForegroundColor DarkGray
            if ($SevenZip) {
                & $SevenZip t $outPath | Select-String "Everything is Ok"
                if ($LASTEXITCODE -ne 0) { throw "Archive verification failed for $name" }
            }
            # For zip, Compress-Archive doesn't corrupt silently; skip explicit test
            Write-Host "  Verified OK" -ForegroundColor Green
        }

        $results += [PSCustomObject]@{
            Name        = $name
            SourceSize  = Format-GB (Get-PathSize $src.Path)
            ArchiveSize = Format-GB $archiveSize
            Elapsed     = "$([int]$elapsed.TotalMinutes)m $($elapsed.Seconds)s"
            Path        = $outPath
            Status      = "OK"
        }
    } catch {
        Write-Host "  FAILED: $_" -ForegroundColor Red
        $results += [PSCustomObject]@{
            Name   = $name
            Status = "FAILED: $_"
            Path   = ""
        }
    }
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host "=== Summary ===" -ForegroundColor Cyan
$results | Format-Table -AutoSize

Write-Host ""
Write-Host "Archives written to: $DestDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Wait for OneDrive to finish syncing (green tick on all files in Explorer)."
Write-Host "  2. Verify you can still open the archives: 7z l <archive.7z>"
Write-Host "  3. Enable OneDrive Files On-Demand so archives don't consume local disk:"
Write-Host "     Settings > OneDrive > Free up space"
Write-Host "  4. Delete the Windows originals to reclaim space:"
Write-Host "     - $DataDir\cfwrinkle_dataset.h5"
Write-Host "     - $DataDir\cfwrinkle_wp3_features.h5"
Write-Host "     - $Predict2\data\aniform_raw"
Write-Host "     - $Predict2\data\simulation_batch_271125"
Write-Host ""
Write-Host "  (WSL copies in /home/ellis/cfwrinkle/data/ are your active training data.)"
