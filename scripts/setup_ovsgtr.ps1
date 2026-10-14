param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$originalLocation = Get-Location

try {
    Set-Location -LiteralPath $projectRoot
    if (-not (Test-Path -LiteralPath "data\OvSGTR\GroundingDINO\setup.py")) {
        throw "Clone https://github.com/gpt4vision/OvSGTR.git into data/OvSGTR first."
    }
    # Install runtime dependencies directly into the selected host Python.
    & $Python -c "import torch, torchvision; assert torch.cuda.is_available(), 'CUDA PyTorch is required'; print(torch.__version__, torchvision.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch validation failed." }
    $pythonExecutable = (& $Python -c "import sys; print(sys.executable)").Trim()
    if ($LASTEXITCODE -ne 0) { throw "Python interpreter lookup failed." }
    $env:PIP_CACHE_DIR = Join-Path $projectRoot ".cache\pip"
    $env:HF_HOME = Join-Path $projectRoot ".cache\huggingface"
    $env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
    & $pythonExecutable -m pip install -r data\ovsgtr_requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    & $pythonExecutable scripts\patch_ovsgtr.py
    if ($LASTEXITCODE -ne 0) { throw "Compatibility patch failed." }
    & $pythonExecutable scripts\download_ovsgtr_assets.py
    if ($LASTEXITCODE -ne 0) { throw "Model asset download failed." }
    & cmd /c scripts\build_groundingdino.cmd $pythonExecutable
    if ($LASTEXITCODE -ne 0) { throw "GroundingDINO CUDA build failed." }
    & $pythonExecutable -u -c "from pathlib import Path; from data.osg_builder import OvSGTRBuilder; graph = OvSGTRBuilder(device='cuda').build(Path('Dataset/AmsterTime/old/0000.jpg')); print('Objects:', len(graph['objects']), 'Relations:', len(graph['relations']))"
    if ($LASTEXITCODE -ne 0) { throw "AmsterTime inference check failed." }
    Write-Host "OvSGTR setup and inference check completed."
} finally {
    Set-Location -LiteralPath $originalLocation.Path
}
