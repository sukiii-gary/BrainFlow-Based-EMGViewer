param(
    [ValidateSet('install','run','demo','test','selftest','build')]
    [string]$Action = 'run',
    [string]$Python = '',
    [string]$ShieldIP = '',
    [ValidateRange(1,65535)][int]$Port = 6677
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython) { $Python = $venvPython }
    else { $Python = 'python' }
}
Push-Location $projectRoot
try {
    switch ($Action) {
        'install' { & $Python -m pip install -r requirements-build.txt }
        'run' {
            if ($ShieldIP) { & $Python app/main.py --wifi $ShieldIP $Port }
            else { & $Python app/main.py }
        }
        'demo' { & $Python app/main.py --synthetic }
        'test' { & $Python scripts/run_tests.py }
        'selftest' { & $Python app/main.py --package-self-test }
        'build' {
            & $Python -m PyInstaller --noconfirm --distpath dist --workpath build EMG_Viewer_UDP.spec
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Action '$Action' failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
