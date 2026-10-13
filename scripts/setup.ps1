# EbookText Studio setup: Python (via winget if missing), venv + libraries,
# Tesseract OCR (via winget if missing), Japanese/English language data, NDLOCR-Lite, shortcuts.
# Usage: setup.bat            (full setup)
#        install_ocr.bat      (-OcrOnly: Tesseract + language data only, for the exe version)
param([switch]$OcrOnly, [switch]$NoShortcut, [switch]$NoNdl)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}
function Have-Winget { return [bool](Get-Command winget -ErrorAction SilentlyContinue) }
function Step($m) { Write-Host ""; Write-Host "== $m" -ForegroundColor Cyan }

function Find-Python {
    foreach ($v in '3.12', '3.13', '3.11') {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py "-$v" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { return @('py', "-$v") }
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $ok = & python -c "import sys; print(int((3,11) <= sys.version_info[:2] <= (3,13)))" 2>$null
        if ($ok -eq '1') { return @('python') }
    }
    return $null
}

function Find-Tesseract {
    $c = Get-Command tesseract -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in "$env:ProgramFiles\Tesseract-OCR\tesseract.exe", "${env:ProgramFiles(x86)}\Tesseract-OCR\tesseract.exe") {
        if (Test-Path $p) { return $p }
    }
    return $null
}

try {
    if (-not $OcrOnly) {
        Step 'Python'
        $py = Find-Python
        if (-not $py) {
            if (-not (Have-Winget)) { throw 'Python 3.11-3.13 not found and winget is unavailable. Install Python 3.12 from python.org and run setup.bat again.' }
            Write-Host 'Installing Python 3.12 with winget...'
            winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
            Refresh-Path
            $py = Find-Python
            if (-not $py) { throw 'Python was installed but is not visible yet. Close this window and run setup.bat again.' }
        }
        Write-Host ("Using: " + ($py -join ' '))

        Step 'Virtual environment and libraries'
        $pyArgs = @()
        if ($py.Length -gt 1) { $pyArgs = $py[1..($py.Length - 1)] }
        & $py[0] @pyArgs -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv' }
        $venvPy = Join-Path $root '.venv\Scripts\python.exe'
        & $venvPy -m pip install --upgrade pip
        & $venvPy -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'pip install failed (check the internet connection).' }
    }

    Step 'Tesseract OCR'
    $tess = Find-Tesseract
    if (-not $tess) {
        if (-not (Have-Winget)) { throw 'Tesseract not found and winget is unavailable. See README for manual installation.' }
        Write-Host 'Installing Tesseract with winget (a permission prompt may appear)...'
        winget install -e --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
        Refresh-Path
        $tess = Find-Tesseract
    }
    if ($tess) { Write-Host "Tesseract: $tess" } else { Write-Warning 'Tesseract is still not found. Install it manually (see README).' }

    Step 'Language data (eng, jpn, jpn_vert, osd)'
    $tessdata = Join-Path $env:LOCALAPPDATA 'EbookTextStudio\tessdata'
    New-Item -ItemType Directory -Force -Path $tessdata | Out-Null
    foreach ($lang in 'eng', 'jpn', 'jpn_vert', 'osd') {
        $dest = Join-Path $tessdata "$lang.traineddata"
        if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 100000)) { Write-Host "$lang: already present"; continue }
        Write-Host "Downloading $lang ..."
        $tmp = "$dest.download"
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/tesseract-ocr/tessdata_best/raw/main/$lang.traineddata" -OutFile $tmp
        Move-Item -Force $tmp $dest
    }
    Write-Host "Saved to $tessdata"

    if (-not $NoNdl) {
        Step 'NDLOCR-Lite (OCR specialised for Japanese books; recommended)'
        try {
            $support = Join-Path $env:LOCALAPPDATA 'EbookTextStudio'
            $ndlHome = Join-Path $support 'ndlocr-lite'
            $ndlVenv = Join-Path $support 'ndl-venv'
            $pyNdl = $py
            if ($OcrOnly) { $pyNdl = Find-Python }
            if (-not $pyNdl) {
                Write-Warning 'Python 3.11-3.13 not found: skipping NDLOCR-Lite (Tesseract still works).'
            } else {
                $pyNdlArgs = @()
                if ($pyNdl.Length -gt 1) { $pyNdlArgs = $pyNdl[1..($pyNdl.Length - 1)] }
                if (-not (Test-Path (Join-Path $ndlHome 'src\ocr.py'))) {
                    Write-Host 'Downloading NDLOCR-Lite from github.com/ndl-lab/ndlocr-lite (about 150 MB, CC BY 4.0) ...'
                    $ProgressPreference = 'SilentlyContinue'
                    $zip = Join-Path $env:TEMP 'ndlocr-lite.zip'
                    $tmp = Join-Path $env:TEMP 'ndlocr-lite-extract'
                    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
                    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/ndl-lab/ndlocr-lite/archive/refs/heads/master.zip' -OutFile $zip
                    Expand-Archive -Path $zip -DestinationPath $tmp -Force
                    $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
                    New-Item -ItemType Directory -Force -Path $ndlHome | Out-Null
                    Copy-Item -Recurse -Force (Join-Path $inner.FullName '*') $ndlHome
                    Remove-Item -Recurse -Force $tmp, $zip
                }
                $ndlPy = Join-Path $ndlVenv 'Scripts\python.exe'
                if (-not (Test-Path $ndlPy)) {
                    & $pyNdl[0] @pyNdlArgs -m venv $ndlVenv
                    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the NDLOCR-Lite virtual environment' }
                }
                & $ndlPy -m pip install --upgrade pip
                & $ndlPy -m pip install onnxruntime opencv-python-headless lxml tqdm pyyaml numpy pillow networkx dill ordered-set reportlab pypdfium2 pypdf protobuf pyparsing
                if ($LASTEXITCODE -ne 0) { throw 'pip install failed for NDLOCR-Lite' }
                Push-Location (Join-Path $ndlHome 'src')
                & $ndlPy -c "import ocr"
                $rc = $LASTEXITCODE
                Pop-Location
                if ($rc -ne 0) { throw 'NDLOCR-Lite import test failed' }
                Write-Host "NDLOCR-Lite ready: $ndlHome"
            }
        } catch {
            Write-Warning "NDLOCR-Lite setup failed: $($_.Exception.Message). Tesseract still works; run setup.bat again to retry."
        }
    }

    if (-not $OcrOnly -and -not $NoShortcut) {
        Step 'Shortcuts'
        $pyw = Join-Path $root '.venv\Scripts\pythonw.exe'
        $shell = New-Object -ComObject WScript.Shell
        $targets = @(
            (Join-Path ([Environment]::GetFolderPath('Desktop')) 'EbookText Studio.lnk'),
            (Join-Path ([Environment]::GetFolderPath('Programs')) 'EbookText Studio.lnk')
        )
        foreach ($t in $targets) {
            $sc = $shell.CreateShortcut($t)
            $sc.TargetPath = $pyw
            $sc.Arguments = '"' + (Join-Path $root 'app.py') + '"'
            $sc.WorkingDirectory = $root
            $sc.Description = 'EbookText Studio'
            $sc.Save()
            Write-Host "Created: $t"
        }
    }

    Write-Host ''
    Write-Host 'Setup complete.' -ForegroundColor Green
    if ($OcrOnly) { Write-Host 'Start EbookTextStudio.exe.' } else { Write-Host 'Start from the desktop shortcut or run.bat.' }
    exit 0
}
catch {
    Write-Host ''
    Write-Host "Setup FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
