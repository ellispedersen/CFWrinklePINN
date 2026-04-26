param(
    [switch]$IncludeIntegration
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$nativeCommandPrefAvailable = $null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)
if ($nativeCommandPrefAvailable) {
    $oldNativeCommandPref = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    $env:PYTHONPATH = $repoRoot
    $pythonExe = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

    Write-Host "== Step 1: dataset typing/value + computed input checks =="
    & $pythonExe -m pytest `
        tests/test_dataset_type_audit.py `
        tests/test_computed_inputs.py `
        tests/test_wp3_unit.py -q
    if ($LASTEXITCODE -ne 0) {
        throw "Step 1 validation checks failed."
    }

    Write-Host "== Step 2: fast model/smoke checks =="
    & $pythonExe -m pytest tests/test_model_unit.py tests/test_training_smoke.py -q -m "not slow"
    if ($LASTEXITCODE -ne 0) {
        throw "Step 2 model/smoke checks failed."
    }

    $wp3Path = if ($env:WP3_H5) { $env:WP3_H5 } else { "data/cfwrinkle_wp3_features.h5" }
    $canReadWp3 = $false
    if (Test-Path $wp3Path) {
        try {
            & $pythonExe -c "import h5py, pathlib; p=pathlib.Path(r'$wp3Path'); h5py.File(p, 'r').close()" > $null 2>&1
            if ($LASTEXITCODE -eq 0) {
                $canReadWp3 = $true
            }
        }
        catch {
            $global:LASTEXITCODE = 0
            $canReadWp3 = $false
        }
    }
    if ($canReadWp3) {
        Write-Host "== Step 3: WP3 feature dataset validation =="
        & $pythonExe -m wp3_features.validate_features --path $wp3Path --verbose
        if ($LASTEXITCODE -ne 0) {
            throw "WP3 feature dataset validation failed."
        }
    }
    else {
        Write-Host "== Step 3: skipped (WP3 file missing or unreadable at $wp3Path) =="
        $global:LASTEXITCODE = 0
    }

    if ($IncludeIntegration) {
        Write-Host "== Step 4: integration checks =="
        & $pythonExe -m pytest tests/test_wp1_wp2_integration.py tests/test_real_data_integration.py -q -m integration
        if ($LASTEXITCODE -ne 0) {
            throw "Step 4 integration checks failed."
        }
    }
    else {
        Write-Host "== Step 4: skipped (pass -IncludeIntegration to run integration checks) =="
    }

    Write-Host "Validation workflow completed."
}
finally {
    if ($nativeCommandPrefAvailable) {
        $PSNativeCommandUseErrorActionPreference = $oldNativeCommandPref
    }
    Pop-Location
}
