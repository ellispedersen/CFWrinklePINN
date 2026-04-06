Set-Location "C:\Users\ellis\Documents\VS Code\CFWrinklePINN"
. .\.venv\Scripts\Activate.ps1

# Use bundle defaults: do not force architecture overrides unless required.
Remove-Item Env:HSA_OVERRIDE_GFX_VERSION -ErrorAction SilentlyContinue
Remove-Item Env:PYTORCH_ROCM_ARCH -ErrorAction SilentlyContinue

# Ensure ROCm SDK console scripts from Python 3.12 are discoverable.
$py312Scripts = "C:\Users\ellis\AppData\Local\Programs\Python\Python312\Scripts"
if (Test-Path $py312Scripts) {
    $env:PATH = "$py312Scripts;$env:PATH"
}

$env:TORCH_BLAS_PREFER_HIPBLASLT = "1"
$env:PYTORCH_TUNABLE_OP_ENABLED = "1"

Write-Output "Activated AMD bundle environment"
python -c "import torch; print('torch', torch.__version__); print('hip', torch.version.hip); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
