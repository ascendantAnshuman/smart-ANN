<#
.SYNOPSIS
    One command from a fresh checkout to the running demonstration.

.DESCRIPTION
    Creates a virtual environment, installs the project, builds the phone-care
    index, calibrates both thresholds from measurements, proves the pipeline
    runs offline, and opens the dashboard.

    Every step prints what it produced. Steps that need a result from an earlier
    step check for it and stop with the command to run, rather than carrying on
    and producing a page full of blanks.

    Python 3.12 is required: chroma-hnswlib publishes no cp313 wheel, so the
    install cannot succeed on 3.13 (ADR-0017).

.PARAMETER VenvPath
    Where to create the environment. The default is short and outside OneDrive,
    and both properties matter.

    Outside OneDrive, because a synced virtual environment is slow and a source
    of file-locking failures mid-install.

    Short, because Windows limits a path to 260 characters unless long-path
    support is enabled, and torch ships header files with very deep names. An
    install into a nested path fails partway through with a confusing
    'No such file or directory' naming a header nobody asked for.

.PARAMETER SkipInstall
    Use an environment that already exists and go straight to the demo steps.

.PARAMETER NoDashboard
    Build everything but do not launch Streamlit.

.EXAMPLE
    .\scripts\run_demo.ps1
#>
[CmdletBinding()]
param(
    [string]$VenvPath = "$env:USERPROFILE\.venvs\smartann-demo",
    [switch]$SkipInstall,
    [switch]$NoDashboard
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Step($n, $text) {
    Write-Host ""
    Write-Host "=== [$n] $text" -ForegroundColor Cyan
}

function Fail($text) {
    Write-Host ""
    Write-Host "STOPPED: $text" -ForegroundColor Red
    exit 1
}

$py = Join-Path $VenvPath "Scripts\python.exe"

# ---------------------------------------------------------------- environment
if (-not $SkipInstall) {
    Step 1 "Python 3.12 environment at $VenvPath"

    if (-not (Test-Path $py)) {
        # py -3.12 is the launcher's own selector; it is the only reliable way to
        # ask for a specific minor version on Windows.
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if (-not $launcher) {
            Fail "the 'py' launcher was not found. Install Python 3.12 from python.org."
        }
        & py -3.12 -c "import sys; print(sys.version)" 2>$null
        if (-not $?) {
            Fail "Python 3.12 is not installed. This project cannot run on 3.13: chroma-hnswlib has no cp313 wheel (ADR-0017)."
        }
        & py -3.12 -m venv $VenvPath
        if (-not $?) { Fail "could not create the virtual environment at $VenvPath" }
    }

    & $py -m pip install --quiet --disable-pip-version-check --upgrade pip
    Write-Host "    installing the project and its demo extras (this pulls torch; allow a few minutes)"
    & $py -m pip install --quiet --disable-pip-version-check -e ".[demo]"
    if (-not $?) { Fail "the install failed. The output above says why." }
    & $py -c "import smartann, hnswlib, streamlit, plotly; print('    imports OK')"
    if (-not $?) { Fail "the environment installed but does not import" }
} else {
    if (-not (Test-Path $py)) { Fail "-SkipInstall was passed but $py does not exist" }
    Write-Host "=== [1] using the existing environment at $VenvPath" -ForegroundColor Cyan
}

# ------------------------------------------------------------------- the data
Step 2 "Build the demonstration index"
& $py scripts\phase11_build_demo_index.py
if (-not $?) { Fail "the index build failed" }

Step 3 "Evaluate retrieval and measure the effort ladder"
& $py scripts\phase11_demo_eval.py --allow-busy
if (-not $?) { Fail "the evaluation failed" }

Step 4 "Calibrate the refusal and cache thresholds from that evaluation"
& $py scripts\phase12_calibrate.py
if (-not $?) { Fail "calibration failed" }

Step 5 "Prove the pipeline answers with the network blocked"
& $py scripts\phase12_offline_proof.py
if (-not $?) {
    Write-Host "    the offline proof did not pass; the dashboard will show that honestly" -ForegroundColor Yellow
}

Step 6 "Fixed depth against the controller, over a recorded handset trace"
& $py scripts\phase12_stress_test.py --allow-busy
if (-not $?) { Fail "the stress test failed" }

Step 7 "On-device against a cloud API"
& $py scripts\phase13_comparison.py
if (-not $?) { Fail "the comparison failed" }

Step 8 "Assemble the viva pack"
& $py scripts\phase15_viva_pack.py
if (-not $?) { Fail "the viva pack failed" }

# -------------------------------------------------------------- the dashboard
if ($NoDashboard) {
    Write-Host ""
    Write-Host "Everything is built. Start the dashboard with:" -ForegroundColor Green
    Write-Host "    & `"$py`" -m streamlit run app\dashboard.py"
    exit 0
}

Step 9 "Open the dashboard"
Write-Host "    Ctrl+C here stops it."
& $py -m streamlit run app\dashboard.py
