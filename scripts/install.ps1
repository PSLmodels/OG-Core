<#
.SYNOPSIS
    OG-* universal installer for Windows (uv-based).

.DESCRIPTION
    Takes a user from zero (only git installed) to a working OG-* model env:
      1. Install uv (if not present)
      2. Clone the chosen repo
      3. uv sync --extra dev (installs Python + project + deps)
      4. Verify import

    Only repos that have migrated to uv (pyproject.toml + uv.lock) are offered.

.PARAMETER Repo
    Skip the model menu. One of: og-core, og-eth.

.PARAMETER RepoUrl
    Use a custom Git URL (e.g. your fork or SSH URL). Bypasses the menu.

.PARAMETER Dest
    Parent directory where the clone is created. Default: current directory.
    The clone always lands in <Dest>\<RepoName>.

.PARAMETER Branch
    For development: clone a non-default branch (default: repo's default
    branch). Useful for testing forks/PRs before they merge.

.PARAMETER Yes
    Auto-confirm every prompt (non-interactive).

.PARAMETER NoDevDeps
    Install runtime deps only (skip dev/test tooling).

.PARAMETER SkipUvInstall
    Don't install uv; assume it's already on PATH.

.PARAMETER NoLog
    Don't write a log file.

.EXAMPLE
    .\scripts\install.ps1
    Fully interactive: pick a model and a destination, then install.

.EXAMPLE
    .\scripts\install.ps1 -Repo og-eth -Dest C:\work -Yes
    Unattended install of OG-ETH into C:\work\OG-ETH.
#>

[CmdletBinding()]
param(
    [string]$Repo = "",
    [string]$RepoUrl = "",
    [string]$Branch = "",
    [string]$Dest = "",
    [switch]$Yes,
    [switch]$NoDevDeps,
    [switch]$SkipUvInstall,
    [switch]$NoLog
)

$ErrorActionPreference = 'Stop'
# $PSScriptRoot is the script's directory when run from a file; it's empty when
# run via the brew-style one-liner (& ([scriptblock]::Create((irm ...)))), in
# which case we fall back to the current working directory so the log file
# lands somewhere predictable instead of crashing Split-Path on the script's
# own source text.
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$WithDevDeps = -not $NoDevDeps

# -- Repo catalog (only uv-migrated repos) -------------------------------------
$Repos = @(
    [pscustomobject]@{ Key="og-core"; Owner="PSLmodels"; Name="OG-Core"; Pkg="ogcore"; Desc="base model (no country calibration)" },
    [pscustomobject]@{ Key="og-eth";  Owner="EAPD-DRB";  Name="OG-ETH";  Pkg="ogeth";  Desc="Ethiopia" }
)

# -- Colors --------------------------------------------------------------------
$UseAnsi = $Host.UI.SupportsVirtualTerminal -or ($env:WT_SESSION -ne $null)
if ($UseAnsi) {
    $E = [char]27
    $BOLD = "$E[1m"; $DIM = "$E[2m"
    $RED = "$E[91m"; $GREEN = "$E[92m"; $YELLOW = "$E[93m"; $RESET = "$E[0m"
} else {
    $BOLD = ""; $DIM = ""; $RED = ""; $GREEN = ""; $YELLOW = ""; $RESET = ""
}

# -- Logging -------------------------------------------------------------------
$Ts = Get-Date -Format "yyyyMMdd-HHmmss"
$LogFile = Join-Path $ScriptDir ".install-$Ts.log"
$WriteLog = -not $NoLog
if ($WriteLog) {
    try { Start-Transcript -Path $LogFile -Append | Out-Null }
    catch {
        Write-Host "WARN: could not start transcript at $LogFile : $($_.Exception.Message)"
        $WriteLog = $false
    }
}
function Stop-TranscriptIfActive {
    if ($script:WriteLog) { try { Stop-Transcript | Out-Null } catch {} }
}

# -- Helpers -------------------------------------------------------------------
function Write-Hr       { Write-Host "--------------------------------------------------------------" }
function Write-HrThick  { Write-Host "==============================================================" }

function Write-Pass($label, $detail = "") {
    $suffix = if ($detail) { "  $DIM($detail)$RESET" } else { "" }
    Write-Host "  $GREEN[PASS]$RESET $label$suffix"
}
function Write-Fail($label, $detail = "") {
    $suffix = if ($detail) { "  $DIM($detail)$RESET" } else { "" }
    Write-Host "  $RED[FAIL]$RESET $label$suffix"
}
function Write-Warn2($label, $detail = "") {
    $suffix = if ($detail) { "  $DIM($detail)$RESET" } else { "" }
    Write-Host "  $YELLOW[WARN]$RESET $label$suffix"
}
function Write-Skip($label, $detail = "") {
    $suffix = if ($detail) { "  $DIM($detail)$RESET" } else { "" }
    Write-Host "  $YELLOW[SKIP]$RESET $label$suffix"
}
function Write-Cmd($cmd) { Write-Host "  $DIM`$ $cmd$RESET" }

$TotalSteps = 4
function Step-Banner($n, $title) {
    Write-Host ""
    Write-Hr
    Write-Host "  ${BOLD}Step $n of ${TotalSteps}: $title${RESET}"
    Write-Hr
}

function Test-Interactive {
    try { return -not [Console]::IsInputRedirected }
    catch { return [Environment]::UserInteractive }
}

function Prompt-YN($prompt, $default = 'y') {
    $opts = if ($default -eq 'y') { '[Y/n/q]' } else { '[y/N/q]' }
    if ($Yes) {
        Write-Host "$prompt $opts ${DIM}(auto: yes)${RESET}"
        return $true
    }
    if (-not (Test-Interactive)) {
        Write-Host "${RED}ERROR:${RESET} non-interactive session and -Yes not given." -ForegroundColor Red
        return $false
    }
    while ($true) {
        $ans = Read-Host "$prompt $opts"
        if ([string]::IsNullOrWhiteSpace($ans)) { $ans = $default }
        switch -Regex ($ans) {
            '^(y|yes)$' { return $true }
            '^(n|no)$'  { return $false }
            '^(q|quit)$' {
                Write-Host "${YELLOW}Aborted by user.${RESET}"
                Stop-TranscriptIfActive
                exit 130
            }
            default { Write-Host "  Please answer y, n, or q." }
        }
    }
}

# -- Pre-flight ----------------------------------------------------------------
if ($env:CONDA_DEFAULT_ENV) {
    Write-Host "${RED}ERROR:${RESET} conda env '$($env:CONDA_DEFAULT_ENV)' is active. Run 'conda deactivate' first." -ForegroundColor Red
    Stop-TranscriptIfActive; exit 1
}
if ($env:VIRTUAL_ENV) {
    Write-Host "${RED}ERROR:${RESET} virtualenv '$($env:VIRTUAL_ENV)' is active. Run 'deactivate' first." -ForegroundColor Red
    Stop-TranscriptIfActive; exit 1
}
$GitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $GitCmd) {
    Write-Host "${RED}ERROR:${RESET} 'git' is not installed (or not on PATH)." -ForegroundColor Red
    Write-Host "Install Git on Windows, then re-run:"
    Write-Host "  winget install -e --id Git.Git"
    Write-Host "  choco install git -y                      # if you use Chocolatey"
    Write-Host "  https://git-scm.com/download/win          # MSI installer"
    Stop-TranscriptIfActive; exit 1
}
$GitBin = $GitCmd.Source

# -- Detect existing uv --------------------------------------------------------
$UvBin = $null
function Detect-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { $script:UvBin = $cmd.Source; return $true }
    $candidates = @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:USERPROFILE\.cargo\bin\uv.exe",
        "$env:LOCALAPPDATA\Programs\uv\uv.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $script:UvBin = $c; return $true }
    }
    return $false
}
[void](Detect-Uv)
$UvPresent = [bool]$UvBin

# -- Pick repo -----------------------------------------------------------------
$RepoOwner = $null; $RepoName = $null; $PkgName = $null; $RepoDesc = $null

function Choose-RepoInteractive {
    if (-not (Test-Interactive)) {
        Write-Host "${RED}ERROR:${RESET} no -Repo given and session is non-interactive." -ForegroundColor Red
        Stop-TranscriptIfActive; exit 2
    }
    Write-Host ""
    Write-HrThick
    Write-Host "  ${BOLD}OG-* Installer (uv-based)${RESET}"
    Write-HrThick
    Write-Host "  Which OG country model do you want to install?"
    Write-Host "  ${DIM}(only repos that have migrated to uv are listed)${RESET}"
    Write-Host ""
    $i = 1
    foreach ($r in $script:Repos) {
        Write-Host ("    {0}) {1,-9} ({2,-10}) -- {3}" -f $i, $r.Name, $r.Owner, $r.Desc)
        $i++
    }
    Write-Host ("    {0}) Other (paste a Git URL)" -f $i)
    Write-Host ""
    while ($true) {
        $choice = Read-Host ("  Choice [1-{0}]" -f $i)
        if (-not ($choice -match '^[0-9]+$')) { Write-Host "  Please enter a number."; continue }
        $n = [int]$choice
        if ($n -lt 1 -or $n -gt $i) { Write-Host "  Out of range."; continue }
        if ($n -eq $i) {
            $url = Read-Host "  Git URL"
            if ([string]::IsNullOrWhiteSpace($url)) { Write-Host "  No URL given."; continue }
            $script:RepoUrl = $url
            return
        }
        $r = $script:Repos[$n - 1]
        $script:Repo = $r.Key
        $script:RepoOwner = $r.Owner; $script:RepoName = $r.Name
        $script:PkgName = $r.Pkg; $script:RepoDesc = $r.Desc
        return
    }
}

if ($RepoUrl -and -not $Repo) {
    # custom URL via CLI; menu skipped
} elseif ($Repo) {
    $match = $Repos | Where-Object { $_.Key -eq $Repo }
    if (-not $match) {
        Write-Host "${RED}ERROR:${RESET} unknown -Repo '$Repo'. Use -? for the list." -ForegroundColor Red
        Stop-TranscriptIfActive; exit 2
    }
    $RepoOwner = $match.Owner; $RepoName = $match.Name
    $PkgName = $match.Pkg; $RepoDesc = $match.Desc
} else {
    Choose-RepoInteractive
}

# Custom URL: derive repo name + package name
if ($RepoUrl -and -not $RepoName) {
    $leaf = Split-Path -Leaf $RepoUrl
    $RepoName = $leaf -replace '\.git$', ''
    $RepoOwner = "(custom URL)"; $RepoDesc = "custom repo"
    $PkgName = ($RepoName.ToLower() -replace '-', '')
    $Repo = $PkgName
}
if (-not $RepoUrl) {
    $RepoUrl = "https://github.com/$RepoOwner/$RepoName.git"
}

# -- Pick destination ----------------------------------------------------------
if (-not $Dest) {
    if (-not (Test-Interactive)) {
        $Dest = "."
    } else {
        Write-Host ""
        Write-Host ("  Where would you like to install {0}?" -f $RepoName)
        Write-Host ("  Enter the PARENT directory; {0} will be cloned as a subfolder inside." -f $RepoName)
        Write-Host ("  Default: current directory ({0})" -f (Get-Location).Path)
        $entered = Read-Host "  Parent directory [.]"
        if ([string]::IsNullOrWhiteSpace($entered)) { $Dest = "." } else { $Dest = $entered }
    }
}

# Expand env vars + ~
$Dest = [Environment]::ExpandEnvironmentVariables($Dest)
if ($Dest.StartsWith("~")) {
    $Dest = Join-Path $env:USERPROFILE $Dest.Substring(1).TrimStart('\','/')
}

# $Dest is the parent directory. Must exist; resolve to absolute.
if (-not (Test-Path $Dest)) {
    Write-Host "${RED}ERROR:${RESET} parent directory does not exist: $Dest" -ForegroundColor Red
    Write-Host "Create it first (mkdir $Dest) or pick a different -Dest."
    Stop-TranscriptIfActive; exit 1
}
$ParentAbs = (Resolve-Path $Dest).Path
$DestAbs = Join-Path $ParentAbs $RepoName

# Refuse if PARENT is a dangerous system dir. (User-home is fine -- clone lands
# inside it, not overwriting it.)
$parentNorm = $ParentAbs.TrimEnd('\').ToLower()
$dangerous = @(
    $env:WINDIR.ToLower(),
    "${env:ProgramFiles}".ToLower(),
    "${env:ProgramFiles(x86)}".ToLower()
)
if ($dangerous -contains $parentNorm -or $parentNorm -match '^[a-z]:[\\/]?$') {
    Write-Host "${RED}ERROR:${RESET} refusing to install into '$ParentAbs' (system dir)." -ForegroundColor Red
    Stop-TranscriptIfActive; exit 1
}

# -- Banner / plan -------------------------------------------------------------
Write-Host ""
Write-HrThick
Write-Host "  ${BOLD}OG-* Installer (uv-based)${RESET}"
Write-HrThick
Write-Host ("  Platform     : Windows {0}" -f $env:PROCESSOR_ARCHITECTURE)
Write-Host ("  Model        : {0}" -f $RepoName)
Write-Host ("  Description  : {0}" -f $RepoDesc)
Write-Host ("  Source       : {0}" -f $RepoUrl)
if ($Branch) { Write-Host ("  Branch       : {0}" -f $Branch) }
Write-Host ("  Destination  : {0}" -f $DestAbs)
Write-Host ("  Package      : {0}" -f $PkgName)
Write-Host ("  Dev/test deps: {0}" -f $(if ($WithDevDeps) { "yes" } else { "no" }))
if ($UvPresent) {
    Write-Host ("  uv           : {0} {1}detected{2}" -f $UvBin, $GREEN, $RESET)
} else {
    Write-Host ("  uv           : {0}will install{1} (~5MB, official installer)" -f $YELLOW, $RESET)
}
if ($WriteLog) { Write-Host ("  Log file     : {0}" -f $LogFile) }
Write-Host ""
Write-Host "  ${BOLD}Plan ($TotalSteps steps):${RESET}"
if ($UvPresent -or $SkipUvInstall) {
    Write-Host "    1. Install uv                      ${DIM}skipped${RESET}"
} else {
    Write-Host "    1. Install uv                      ${DIM}~5MB, seconds${RESET}"
}
Write-Host ("    2. Clone {0,-25} ${DIM}depends on network${RESET}" -f $RepoName)
Write-Host "    3. uv sync (Python + deps)         ${DIM}~30s, ~500MB${RESET}"
Write-Host "    4. Verify installation             ${DIM}a few seconds${RESET}"
Write-Host ""
Write-Host "  You will be asked to confirm before each mutating step."
Write-Host ""

if (-not (Prompt-YN "Proceed with installation?" 'y')) {
    Write-Host "${YELLOW}Aborted by user.${RESET}"
    Stop-TranscriptIfActive; exit 0
}

# -- Result tracking -----------------------------------------------------------
$StepResults = New-Object System.Collections.Generic.List[object]
function Record-Step($name, $state, $detail = "") {
    $script:StepResults.Add([pscustomobject]@{ Name=$name; State=$state; Detail=$detail })
}
$StartTime = Get-Date

# -- Step 1: Install uv --------------------------------------------------------
Step-Banner 1 "Install uv"
if ($UvPresent) {
    Write-Pass "uv already present" $UvBin
    Record-Step "uv" "SKIP" "already present"
} elseif ($SkipUvInstall) {
    Write-Fail "-SkipUvInstall given but no uv found"
    Record-Step "uv" "FAIL" "no uv and -SkipUvInstall"
    Stop-TranscriptIfActive; exit 1
} else {
    Write-Host "  Will install uv via the official installer:"
    Write-Host "    Source : https://astral.sh/uv/install.ps1"
    Write-Host "    Method : irm | iex -- installs to your user profile, no admin."
    Write-Host ""
    if (-not (Prompt-YN "Download and install uv?" 'y')) {
        Record-Step "uv" "SKIP" "declined"
        Write-Host "${RED}Cannot continue without uv. Aborting.${RESET}"
        Stop-TranscriptIfActive; exit 1
    }
    Write-Cmd "irm https://astral.sh/uv/install.ps1 | iex"
    $ProgressPreference = 'SilentlyContinue'
    # Run in a -ExecutionPolicy Bypass subprocess so this works regardless of
    # the current ExecutionPolicy (uv install script is unsigned).
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -Command `
        "[Net.ServicePointManager]::SecurityProtocol = 'Tls12'; iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "uv install subprocess returned $LASTEXITCODE"
        Record-Step "uv" "FAIL" "install subprocess failed"
        Stop-TranscriptIfActive; exit 1
    }
    [void](Detect-Uv)
    if (-not $UvBin) {
        # Documented default install location
        $candidate = "$env:USERPROFILE\.local\bin\uv.exe"
        if (Test-Path $candidate) { $UvBin = $candidate }
    }
    if (-not $UvBin -or -not (Test-Path $UvBin)) {
        Write-Fail "uv install completed but binary not found"
        Record-Step "uv" "FAIL" "binary not found post-install"
        Stop-TranscriptIfActive; exit 1
    }
    $uvVer = ""
    try { $uvVer = ((& $UvBin --version 2>$null) | Select-Object -First 1).Trim() } catch {}
    Write-Pass "uv installed" "$UvBin ($uvVer)"
    Record-Step "uv" "PASS" $UvBin
}

# -- Step 2: Clone the repo ----------------------------------------------------
Step-Banner 2 "Clone $RepoName"

function Normalize-RemoteUrl($u) {
    return ($u -replace '\.git/?$', '').ToLower()
}

$DestHasRepo = $false; $DestEmpty = $false
if (Test-Path $DestAbs) {
    if ((Get-ChildItem -Force $DestAbs -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
        $DestEmpty = $true
    } elseif (Test-Path (Join-Path $DestAbs ".git")) {
        $existingUrl = ""
        try { $existingUrl = (& $GitBin -C $DestAbs config --get remote.origin.url 2>$null).Trim() } catch {}
        if ((Normalize-RemoteUrl $existingUrl) -eq (Normalize-RemoteUrl $RepoUrl)) {
            $DestHasRepo = $true
        } else {
            Write-Fail "Destination is a git repo for a different remote" $existingUrl
            Write-Host "  Either remove $DestAbs or pick a different destination with -Dest."
            Record-Step "Clone" "FAIL" "wrong remote"
            Stop-TranscriptIfActive; exit 1
        }
    } else {
        Write-Fail "Destination exists and is not empty (and not a git clone)" $DestAbs
        Write-Host "  Either remove $DestAbs or pick a different destination with -Dest."
        Record-Step "Clone" "FAIL" "destination not empty"
        Stop-TranscriptIfActive; exit 1
    }
}

if ($DestHasRepo) {
    $branch = "?"
    try { $branch = (& $GitBin -C $DestAbs rev-parse --abbrev-ref HEAD 2>$null).Trim() } catch {}
    Write-Host ("  Existing clone of {0} found at {1} (branch: {2})." -f $RepoName, $DestAbs, $branch)
    Write-Host "  Will run 'git pull --ff-only' to bring it up to date."
    Write-Host ""
    if (Prompt-YN "Update existing clone?" 'y') {
        Write-Cmd "git -C $DestAbs pull --ff-only"
        & $GitBin -C $DestAbs pull --ff-only
        if ($LASTEXITCODE -eq 0) {
            Write-Pass "Repo updated" $DestAbs
            Record-Step "Clone" "PASS" "updated ($branch)"
        } else {
            Write-Warn2 "git pull failed; continuing with existing state"
            Record-Step "Clone" "WARN" "pull failed; existing state used"
        }
    } else {
        Write-Skip "Update" "using existing clone as-is"
        Record-Step "Clone" "SKIP" "existing clone used as-is ($branch)"
    }
} else {
    if ($Branch) {
        Write-Host ("  Will clone {0} (branch: {1}) into {2}." -f $RepoUrl, $Branch, $DestAbs)
    } else {
        Write-Host ("  Will clone {0} into {1}." -f $RepoUrl, $DestAbs)
    }
    Write-Host ""
    if (Prompt-YN "Clone now?" 'y') {
        if ($Branch) {
            Write-Cmd "git clone --branch $Branch $RepoUrl $DestAbs"
            & $GitBin clone --branch $Branch $RepoUrl $DestAbs
        } else {
            Write-Cmd "git clone $RepoUrl $DestAbs"
            & $GitBin clone $RepoUrl $DestAbs
        }
        if ($LASTEXITCODE -ne 0) { throw "git clone failed (exit $LASTEXITCODE)" }
        $branch = "?"
        try { $branch = (& $GitBin -C $DestAbs rev-parse --abbrev-ref HEAD 2>$null).Trim() } catch {}
        Write-Pass "Cloned" "$DestAbs (branch: $branch)"
        Record-Step "Clone" "PASS" $branch
    } else {
        Write-Fail "Clone declined; cannot continue."
        Record-Step "Clone" "FAIL" "declined"
        Stop-TranscriptIfActive; exit 1
    }
}

# Verify the repo is uv-native.
if (-not (Test-Path (Join-Path $DestAbs "pyproject.toml"))) {
    Write-Fail "$DestAbs has no pyproject.toml"
    Write-Host "  This installer requires repos that have migrated to uv."
    Record-Step "Clone" "FAIL" "no pyproject.toml"
    Stop-TranscriptIfActive; exit 1
}
if (-not (Test-Path (Join-Path $DestAbs "uv.lock"))) {
    Write-Warn2 "$DestAbs has no uv.lock; uv sync will create one"
}

# -- Step 3: uv sync -----------------------------------------------------------
Step-Banner 3 "Install $RepoName (uv sync)"
$syncArgs = @("sync")
if ($WithDevDeps) { $syncArgs += @("--extra", "dev") }
Write-Host ("  Will install {0} + Python + all deps into {1}\.venv" -f $RepoName, $DestAbs)
Write-Host "  Working directory : $DestAbs"
Write-Host ("  Command           : uv {0}" -f ($syncArgs -join ' '))
Write-Host ""
if (Prompt-YN "Run uv sync now?" 'y') {
    Push-Location $DestAbs
    try {
        Write-Cmd ("uv {0}" -f ($syncArgs -join ' '))
        & $UvBin @syncArgs
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed (exit $LASTEXITCODE)" }
        Write-Pass "$RepoName installed (editable)"
        Record-Step "$RepoName install" "PASS" "uv sync"
    } finally { Pop-Location }
} else {
    Write-Fail "uv sync declined; package will not be importable."
    Record-Step "$RepoName install" "FAIL" "declined"
}

# -- Step 4: Verify ------------------------------------------------------------
Step-Banner 4 "Verify installation"
$VenvPython = Join-Path $DestAbs ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Fail "Venv python not found; skipping import check." $VenvPython
    Record-Step "Verification" "FAIL" "no venv python"
} else {
    $pyver = ""
    try { $pyver = (& $VenvPython -W ignore -c "import sys; print(sys.version.split()[0])" 2>&1 | Select-Object -Last 1).ToString().Trim() } catch {}
    Write-Pass "Python in .venv" $pyver

    # Run import as a discrete process; merge stderr into stdout so PS doesn't
    # treat upstream deprecation warnings (e.g. from pygam) as halting errors.
    # -W ignore silences Python's own warnings.
    & $VenvPython -W ignore -c "import $PkgName" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $ver = ""
        try { $ver = (& $VenvPython -W ignore -c "import $PkgName; print(getattr($PkgName, '__version__', '?'))" 2>&1 | Select-Object -Last 1).ToString().Trim() } catch {}
        Write-Pass "import $PkgName" $ver
        Record-Step "Verification" "PASS" "import $PkgName ($ver)"
    } else {
        Write-Fail "import $PkgName" "package not importable; check log above"
        Record-Step "Verification" "FAIL" "import $PkgName failed"
    }
}

# -- Summary -------------------------------------------------------------------
$Elapsed = (Get-Date) - $StartTime
$ElapsedMin = [int]$Elapsed.TotalMinutes
$ElapsedSec = [int]($Elapsed.TotalSeconds - ($ElapsedMin * 60))

Write-Host ""
Write-HrThick
Write-Host ("  ${BOLD}Installation Summary -- {0}${RESET}" -f $RepoName)
Write-HrThick

$AllOk = $true
foreach ($r in $StepResults) {
    switch ($r.State) {
        "PASS" { Write-Pass  $r.Name $r.Detail }
        "SKIP" { Write-Skip  $r.Name $r.Detail }
        "WARN" { Write-Warn2 $r.Name $r.Detail }
        "FAIL" { Write-Fail  $r.Name $r.Detail; $AllOk = $false }
        default { Write-Warn2 $r.Name "unknown: $($r.State)" }
    }
}
Write-Host ""
Write-Host ("  Elapsed  : {0}m {1}s" -f $ElapsedMin, $ElapsedSec)
Write-Host ("  Location : {0}" -f $DestAbs)
Write-Host ("  Venv     : {0}\.venv" -f $DestAbs)
if ($WriteLog) { Write-Host ("  Log      : {0}" -f $LogFile) }
Write-Host ""

if ($AllOk) {
    Write-Host "  ${GREEN}${BOLD}All steps completed successfully.${RESET}"
    Write-Host ""
    Write-Host "  ${BOLD}To start using ${RepoName}:${RESET}"
    Write-Host "    cd $DestAbs"
    Write-Host "    .\.venv\Scripts\Activate.ps1     # activate venv"
    Write-Host ("    python -W ignore -c `"import {0}; print({0}.__file__)`"" -f $PkgName)
    Write-Host "  Or run commands without activating:"
    Write-Host ("    uv run python -W ignore -c `"import {0}; print({0}.__file__)`"" -f $PkgName)
    $exDir = Join-Path $DestAbs "examples"
    if (Test-Path $exDir) {
        Write-Host ""
        Write-Host ("  Example scripts: {0}" -f $exDir)
    }
    Stop-TranscriptIfActive
    exit 0
} else {
    Write-Host "  ${RED}${BOLD}One or more steps failed.${RESET}"
    Write-Host ""
    Write-Host "  Review the [FAIL] entries above."
    if ($WriteLog) { Write-Host "  Full output is in: $LogFile" }
    Stop-TranscriptIfActive
    exit 1
}
