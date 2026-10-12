<#
Builds a self-contained copy of the app that runs on any 64-bit Windows PC:
embedded Python 3.11 + CPU PyTorch + all packages + app code + trained weights.
Copy the result folder (or zip) to the laptop and double-click "Start Cell Segmentation.bat".

Run on the development PC (needs internet once, ~1 GB download):
    powershell -ExecutionPolicy Bypass -File portable\build_portable.ps1
    powershell -ExecutionPolicy Bypass -File portable\build_portable.ps1 -Dest E:\CellSeg_Portable -Zip
#>
param(
    [string]$Dest = "D:\CellSeg_Portable",
    [switch]$Zip
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$PyVersion = "3.11.9"                      # last 3.11 release with an embeddable zip
$TorchPins = @("torch==2.6.0", "torchvision==0.21.0")
$Here = $PSScriptRoot
$AppSrc = Join-Path (Split-Path $Here) "cell_seg"
$Weights = Join-Path $AppSrc "weights\unet_cells.pth"
$PyDir = Join-Path $Dest "python"
$AppDir = Join-Path $Dest "app"
$Py = Join-Path $PyDir "python.exe"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $exe $($argv -join ' ')" }
}

if (-not (Test-Path $Weights)) { throw "Trained weights missing: $Weights" }
if (Test-Path $Dest) { throw "$Dest already exists. Delete it or pass -Dest <new folder>." }

Step "Downloading embedded Python $PyVersion"
New-Item -ItemType Directory -Force $PyDir | Out-Null
$tmp = Join-Path $env:TEMP "cellseg_build"
New-Item -ItemType Directory -Force $tmp | Out-Null
$pyZip = Join-Path $tmp "python-embed.zip"
Invoke-WebRequest "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip" -OutFile $pyZip -UseBasicParsing
Expand-Archive $pyZip -DestinationPath $PyDir

# ._pth replaces sys.path: stdlib zip, site-packages, and the app folder (so "from config import" works).
@"
python311.zip
.
Lib\site-packages
..\app
import site
"@ | Set-Content (Join-Path $PyDir "python311._pth") -Encoding ascii

# Ship the MSVC runtime next to python.exe so torch DLLs load on PCs without the VC++ redistributable.
foreach ($dll in "msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll") {
    $src = Join-Path $env:WINDIR "System32\$dll"
    if ((Test-Path $src) -and -not (Test-Path (Join-Path $PyDir $dll))) { Copy-Item $src $PyDir }
}

Step "Installing pip"
$getPip = Join-Path $tmp "get-pip.py"
Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip -UseBasicParsing
Run $Py @($getPip, "--no-warn-script-location", "--disable-pip-version-check")

Step "Installing PyTorch (CPU build, works on every laptop)"
Run $Py (@("-m", "pip", "install", "--no-warn-script-location", "--disable-pip-version-check") + $TorchPins +
         @("--index-url", "https://download.pytorch.org/whl/cpu"))

Step "Installing app packages"
Run $Py @("-m", "pip", "install", "--no-warn-script-location", "--disable-pip-version-check",
          "-r", (Join-Path $Here "requirements-portable.txt"))

Step "Copying app code and trained weights"
# robocopy exit codes 0-7 mean success
robocopy $AppSrc $AppDir /E /NFL /NDL /NJH /NJS /NP /XD outputs __pycache__ .pytest_cache tests | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
New-Item -ItemType Directory -Force (Join-Path $AppDir "outputs") | Out-Null
Copy-Item (Join-Path $Here "Start Cell Segmentation.bat") $Dest
Copy-Item (Join-Path $Here "HOW_TO_RUN.txt") $Dest

Step "Smoke test: segment the bundled samples on CPU"
$smoke = Join-Path $tmp "smoke_test.py"
@"
import cv2, gradio
from config import Params, SAMPLES_DIR
from core.pipeline import CellCountingPipeline
p = CellCountingPipeline()
for f in sorted(SAMPLES_DIR.glob('*.png')):
    img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
    print('  ', f.name, p.run(img, Params()).count, 'cells')
print('   gradio', gradio.__version__, '- OK')
"@ | Set-Content $smoke -Encoding ascii
Push-Location $AppDir
try { Run $Py @($smoke) } finally { Pop-Location }
Remove-Item (Join-Path $AppDir "outputs\*") -Recurse -Force -ErrorAction SilentlyContinue

if ($Zip) {
    Step "Zipping"
    $zipPath = "$Dest.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath }
    Push-Location (Split-Path $Dest)
    try { Run "tar.exe" @("-a", "-c", "-f", $zipPath, (Split-Path $Dest -Leaf)) } finally { Pop-Location }
    Write-Host "Zip: $zipPath ($([math]::Round((Get-Item $zipPath).Length / 1MB)) MB)"
}

Remove-Item $tmp -Recurse -Force
$size = (Get-ChildItem $Dest -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host "`nDone. Portable app: $Dest ($([math]::Round($size)) MB)" -ForegroundColor Green
Write-Host "Copy it to the laptop and double-click 'Start Cell Segmentation.bat'."
