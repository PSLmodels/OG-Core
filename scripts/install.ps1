<#
.SYNOPSIS
    OG-* universal installer for Windows (uv-based).

.DESCRIPTION
    Takes a user from zero (only git installed) to a working OG-* model env:
      1. Install uv (if not present)
      2. Clone the chosen repo(s)
      3. uv sync --extra dev (installs Python + project + deps)
      4. Verify import

    Installs one repo, several (-Repo og-zaf,og-idn), or all (-All).
    Only repos that have migrated to uv (pyproject.toml + uv.lock) are offered.

.PARAMETER Repo
    Catalog repo key(s) to install. Comma-separate for several:
    -Repo og-zaf,og-idn  (see -List for valid keys).

.PARAMETER All
    Install every repo in the catalog.

.PARAMETER List
    Print the repo catalog (human-readable) and exit.

.PARAMETER ListJson
    Print the repo catalog as JSON and exit.

.PARAMETER RepoUrl
    Install one custom Git URL (single repo only). Bypasses the menu.

.PARAMETER Dest
    Parent directory for the clone(s). Default: current directory.
    Each repo lands in <Dest>\<RepoName>.

.PARAMETER Branch
    For development: clone a non-default branch. Single repo only (not valid
    with -All or multiple -Repo).

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
    .\scripts\install.ps1 -Repo og-zaf,og-idn -Dest C:\OG
    Install several models at once into C:\OG.

.EXAMPLE
    .\scripts\install.ps1 -All -Dest C:\OG -Yes
    Hands-free: install every repo into C:\OG with no prompts.
#>

[CmdletBinding()]
param(
    [string[]]$Repo = @(),
    [switch]$All,
    [string]$RepoUrl = "",
    [string]$Branch = "",
    [string]$Dest = "",
    [switch]$List,
    [switch]$ListJson,
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
    [pscustomobject]@{ Key="og-eth";  Owner="EAPD-DRB";  Name="OG-ETH";  Pkg="ogeth";  Desc="Ethiopia" },
    [pscustomobject]@{ Key="og-zaf";  Owner="EAPD-DRB";  Name="OG-ZAF";  Pkg="ogzaf";  Desc="South Africa" },
    [pscustomobject]@{ Key="og-idn";  Owner="EAPD-DRB";  Name="OG-IDN";  Pkg="ogidn";  Desc="Indonesia" },
    [pscustomobject]@{ Key="og-phl";  Owner="EAPD-DRB";  Name="OG-PHL";  Pkg="ogphl";  Desc="Philippines" },
    [pscustomobject]@{ Key="og-bra";  Owner="PSLmodels"; Name="OG-BRA";  Pkg="ogbra";  Desc="Brazil" }
)

# -- Catalog listing (-List / -ListJson) ---------------------------------------
# Emit from the embedded $Repos (runtime source of truth) so it works even when
# only the script is fetched. scripts/repos.json holds the same data as a file;
# a CI check keeps them in sync.
if ($ListJson) {
    $catalog = [pscustomobject]@{
        schema_version = 1
        repos = @($Repos | ForEach-Object {
            [pscustomobject]@{ key = $_.Key; owner = $_.Owner; repo = $_.Name; package = $_.Pkg; description = $_.Desc }
        })
    }
    $catalog | ConvertTo-Json -Depth 5
    exit 0
}
if ($List) {
    "{0,-9}  {1,-22}  {2,-8}  {3}" -f "KEY", "REPO", "PACKAGE", "DESCRIPTION"
    foreach ($r in $Repos) {
        "{0,-9}  {1,-22}  {2,-8}  {3}" -f $r.Key, "$($r.Owner)/$($r.Name)", $r.Pkg, $r.Desc
    }
    exit 0
}

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

function Write-Section($title) {
    Write-Host ""
    Write-Hr
    Write-Host "  ${BOLD}$title${RESET}"
    Write-Hr
}

function Fail-Arg($msg) {
    Write-Host "${RED}ERROR:${RESET} $msg" -ForegroundColor Red
    Stop-TranscriptIfActive
    exit 2
}

function Normalize-RemoteUrl($u) {
    return ($u -replace '\.git/?$', '').ToLower()
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
        Stop-TranscriptIfActive
        exit 2
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

# -- Pick repo interactively (single repo; multi is opt-in via -Repo/-All) -----
$RepoOwner = $null; $RepoName = $null; $PkgName = $null; $RepoDesc = $null

function Choose-RepoInteractive {
    if (-not (Test-Interactive)) {
        Write-Host "${RED}ERROR:${RESET} no -Repo/-All given and session is non-interactive." -ForegroundColor Red
        Stop-TranscriptIfActive; exit 2
    }
    Write-Host ""
    Write-HrThick
    Write-Host "  ${BOLD}OG-* Installer (uv-based)${RESET}"
    Write-HrThick
    Write-Host "  Which OG country model do you want to install?"
    Write-Host "  ${DIM}(only repos that have migrated to uv are listed; use -All for every one)${RESET}"
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
        $script:RepoOwner = $r.Owner; $script:RepoName = $r.Name
        $script:PkgName = $r.Pkg; $script:RepoDesc = $r.Desc
        return
    }
}

# -- Resolve the list of repos to install --------------------------------------
$Targets = @()
if ($RepoUrl) {
    if ($All -or $Repo.Count -gt 0) {
        Fail-Arg "-RepoUrl installs a single repo; it cannot be combined with -All or -Repo."
    }
    $leaf = Split-Path -Leaf $RepoUrl
    $cuName = $leaf -replace '\.git$', ''
    $cuPkg = ($cuName.ToLower() -replace '-', '')
    $Targets += [pscustomobject]@{ Owner="(custom URL)"; Name=$cuName; Pkg=$cuPkg; Desc="custom repo"; Url=$RepoUrl }
} elseif ($All) {
    if ($Repo.Count -gt 0) { Fail-Arg "-All installs every repo; do not also pass -Repo." }
    foreach ($r in $Repos) {
        $Targets += [pscustomobject]@{ Owner=$r.Owner; Name=$r.Name; Pkg=$r.Pkg; Desc=$r.Desc; Url="https://github.com/$($r.Owner)/$($r.Name).git" }
    }
} elseif ($Repo.Count -gt 0) {
    $seen = @{}
    foreach ($key in $Repo) {
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true
        $match = $Repos | Where-Object { $_.Key -eq $key }
        if (-not $match) { Fail-Arg "unknown -Repo '$key'. Run -List to see valid keys." }
        $Targets += [pscustomobject]@{ Owner=$match.Owner; Name=$match.Name; Pkg=$match.Pkg; Desc=$match.Desc; Url="https://github.com/$($match.Owner)/$($match.Name).git" }
    }
} else {
    Choose-RepoInteractive
    if ($RepoUrl) {
        $leaf = Split-Path -Leaf $RepoUrl
        $cuName = $leaf -replace '\.git$', ''
        $cuPkg = ($cuName.ToLower() -replace '-', '')
        $Targets += [pscustomobject]@{ Owner="(custom URL)"; Name=$cuName; Pkg=$cuPkg; Desc="custom repo"; Url=$RepoUrl }
    } else {
        $Targets += [pscustomobject]@{ Owner=$RepoOwner; Name=$RepoName; Pkg=$PkgName; Desc=$RepoDesc; Url="https://github.com/$RepoOwner/$RepoName.git" }
    }
}

$TotalTargets = $Targets.Count
if ($Branch -and $TotalTargets -gt 1) {
    Fail-Arg "-Branch can only be used with a single repo (selected: $TotalTargets)."
}
# One repo keeps the per-step confirmations; multiple run as a batch.
$PerStep = ($TotalTargets -eq 1)

# -- Pick destination (PARENT directory; each repo lands at PARENT\RepoName) ---
if (-not $Dest) {
    if (-not (Test-Interactive)) {
        $Dest = "."
    } else {
        Write-Host ""
        Write-Host "  Where would you like to install?"
        Write-Host "  Enter the PARENT directory; each repo is cloned into its own subfolder."
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

if (-not (Test-Path $Dest)) {
    Write-Host "${RED}ERROR:${RESET} parent directory does not exist: $Dest" -ForegroundColor Red
    Write-Host "Create it first (mkdir $Dest) or pick a different -Dest."
    Stop-TranscriptIfActive; exit 1
}
$ParentAbs = (Resolve-Path $Dest).Path

# Refuse if PARENT is a dangerous system dir. (User-home is fine -- clones land
# inside it, not overwriting it.)
$parentNorm = $ParentAbs.TrimEnd('\').ToLower()
$dangerous = @(
    "${env:WINDIR}".ToLower(),
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
Write-Host ("  Destination  : {0}" -f $ParentAbs)
if ($Branch) { Write-Host ("  Branch       : {0}" -f $Branch) }
Write-Host ("  Dev/test deps: {0}" -f $(if ($WithDevDeps) { "yes" } else { "no" }))
if ($UvPresent) {
    Write-Host ("  uv           : {0} {1}detected{2}" -f $UvBin, $GREEN, $RESET)
} else {
    Write-Host ("  uv           : {0}will install{1} (~5MB, official installer)" -f $YELLOW, $RESET)
}
if ($WriteLog) { Write-Host ("  Log file     : {0}" -f $LogFile) }
Write-Host ""
Write-Host "  ${BOLD}Repos to install ($TotalTargets):${RESET}"
foreach ($t in $Targets) {
    Write-Host "    - $($t.Name)  ${DIM}$($t.Desc)${RESET} -> $ParentAbs\$($t.Name)"
}
Write-Host ""
Write-Host "  Each repo: clone, uv sync (Python + deps, ~30s/~500MB), verify import."
if ($PerStep) {
    Write-Host "  You will be asked to confirm before each mutating step."
} else {
    Write-Host "  ${BOLD}$TotalTargets repos${RESET} will be installed after one confirmation below."
}
Write-Host ""

$proceedPrompt = if ($TotalTargets -gt 1) { "Proceed with installing $TotalTargets repos?" } else { "Proceed with installation?" }
if (-not (Prompt-YN $proceedPrompt 'y')) {
    Write-Host "${YELLOW}Aborted by user.${RESET}"
    Stop-TranscriptIfActive; exit 0
}

# -- Per-repo result tracking --------------------------------------------------
$RepoResults = New-Object System.Collections.Generic.List[object]
function Record-Repo($name, $state, $detail = "") {
    $script:RepoResults.Add([pscustomobject]@{ Name=$name; State=$state; Detail=$detail })
}

# Install ONE repo: clone (or update) + uv sync + verify. Never throws to the
# caller; records a per-repo result so the loop can continue on failure. Uses
# script-scope $ParentAbs, $UvBin, $GitBin, $WithDevDeps, $Branch, $PerStep.
function Install-OneRepo($owner, $name, $pkg, $desc, $url, $idx, $total) {
    $destAbs = Join-Path $ParentAbs $name
    $repoState = "PASS"
    Write-Host ""
    Write-Hr
    if ($total -gt 1) {
        Write-Host "  ${BOLD}[$idx/$total] $name${RESET}  ${DIM}$owner/$name${RESET}"
    } else {
        Write-Host "  ${BOLD}$name${RESET}  ${DIM}$owner/$name${RESET}"
    }
    Write-Hr

    try {
        # --- Clone or update existing ---
        $destHasRepo = $false
        if (Test-Path $destAbs) {
            if (@(Get-ChildItem -Force $destAbs -ErrorAction SilentlyContinue).Count -eq 0) {
                # empty dir; clone into it
            } elseif (Test-Path (Join-Path $destAbs ".git")) {
                $existingUrl = ""
                try { $existingUrl = (& $GitBin -C $destAbs config --get remote.origin.url 2>$null).Trim() } catch {}
                if ((Normalize-RemoteUrl $existingUrl) -eq (Normalize-RemoteUrl $url)) {
                    $destHasRepo = $true
                } else {
                    Write-Fail "${name}: destination is a git repo for a different remote" $existingUrl
                    Record-Repo $name "FAIL" "wrong remote at $destAbs"
                    return
                }
            } else {
                Write-Fail "${name}: destination exists and is not empty" $destAbs
                Record-Repo $name "FAIL" "destination not empty"
                return
            }
        }

        if ($destHasRepo) {
            $branchNow = "?"
            try { $branchNow = (& $GitBin -C $destAbs rev-parse --abbrev-ref HEAD 2>$null).Trim() } catch {}
            Write-Host ("  Existing clone found at {0} (branch: {1})." -f $destAbs, $branchNow)
            $doUpdate = $true
            if ($PerStep) { if (-not (Prompt-YN "Update existing clone (git pull --ff-only)?" 'y')) { $doUpdate = $false } }
            if ($doUpdate) {
                Write-Cmd "git -C $destAbs pull --ff-only"
                & $GitBin -C $destAbs pull --ff-only
                if ($LASTEXITCODE -eq 0) { Write-Pass "$name updated" }
                else { Write-Warn2 "${name}: git pull failed; using existing state"; $repoState = "WARN" }
            } else { Write-Skip "${name}: using existing clone as-is" }
        } else {
            if ($Branch) { Write-Host ("  Will clone {0} (branch: {1}) into {2}." -f $url, $Branch, $destAbs) }
            else { Write-Host ("  Will clone {0} into {1}." -f $url, $destAbs) }
            $doClone = $true
            if ($PerStep) { if (-not (Prompt-YN "Clone now?" 'y')) { $doClone = $false } }
            if (-not $doClone) { Write-Fail "${name}: clone declined"; Record-Repo $name "FAIL" "clone declined"; return }
            if ($Branch) {
                Write-Cmd "git clone --branch $Branch $url $destAbs"
                & $GitBin clone --branch $Branch $url $destAbs
            } else {
                Write-Cmd "git clone $url $destAbs"
                & $GitBin clone $url $destAbs
            }
            if ($LASTEXITCODE -ne 0) { Write-Fail "${name}: git clone failed"; Record-Repo $name "FAIL" "git clone failed"; return }
            $branchNow = "?"
            try { $branchNow = (& $GitBin -C $destAbs rev-parse --abbrev-ref HEAD 2>$null).Trim() } catch {}
            Write-Pass "$name cloned" "$destAbs (branch: $branchNow)"
        }

        # --- Verify the repo is uv-native ---
        if (-not (Test-Path (Join-Path $destAbs "pyproject.toml"))) {
            Write-Fail "${name}: no pyproject.toml (not a uv-native repo)"
            Record-Repo $name "FAIL" "no pyproject.toml"
            return
        }
        if (-not (Test-Path (Join-Path $destAbs "uv.lock"))) { Write-Warn2 "${name}: no uv.lock; uv sync will create one" }

        # --- uv sync ---
        $syncArgs = @("sync")
        if ($WithDevDeps) { $syncArgs += @("--extra", "dev") }
        $doSync = $true
        if ($PerStep) {
            Write-Host "  Will run: uv $($syncArgs -join ' ')  (in $destAbs)"
            if (-not (Prompt-YN "Run uv sync now?" 'y')) { $doSync = $false }
        }
        if (-not $doSync) { Write-Fail "${name}: uv sync declined"; Record-Repo $name "FAIL" "uv sync declined"; return }
        Write-Cmd "uv $($syncArgs -join ' ')  (cwd: $destAbs)"
        Push-Location $destAbs
        $syncRc = 1
        try { & $UvBin @syncArgs; $syncRc = $LASTEXITCODE } finally { Pop-Location }
        if ($syncRc -ne 0) { Write-Fail "${name}: uv sync failed"; Record-Repo $name "FAIL" "uv sync failed"; return }
        Write-Pass "$name installed (uv sync)"

        # --- Verify import ---
        $venvPy = Join-Path $destAbs ".venv\Scripts\python.exe"
        if (-not (Test-Path $venvPy)) {
            Write-Fail "${name}: venv python not found" $venvPy
            Record-Repo $name "FAIL" "no venv python"
            return
        }
        & $venvPy -W ignore -c "import $pkg" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $ver = "?"
            try { $ver = (& $venvPy -W ignore -c "import $pkg; print(getattr($pkg, '__version__', '?'))" 2>&1 | Select-Object -Last 1).ToString().Trim() } catch {}
            Write-Pass "${name}: import $pkg" $ver
            if ($repoState -eq "WARN") { Record-Repo $name "WARN" "import $pkg ($ver); pull warning" }
            else { Record-Repo $name "PASS" "import $pkg ($ver)" }
        } else {
            Write-Fail "${name}: import $pkg failed" "package not importable; check log above"
            Record-Repo $name "FAIL" "import $pkg failed"
        }
    } catch {
        Write-Fail "${name}: unexpected error" $_.Exception.Message
        Record-Repo $name "FAIL" "error: $($_.Exception.Message)"
    }
}

$StartTime = Get-Date

# -- Install uv (once, shared by all repos) ------------------------------------
Write-Section "Install uv"
$UvState = "PASS"; $UvDetail = ""
if ($UvPresent) {
    Write-Pass "uv already present" $UvBin
    $UvState = "SKIP"; $UvDetail = "already present"
} elseif ($SkipUvInstall) {
    Write-Fail "-SkipUvInstall given but no uv found"
    Stop-TranscriptIfActive; exit 1
} else {
    Write-Host "  Will install uv via the official installer:"
    Write-Host "    Source : https://astral.sh/uv/install.ps1"
    Write-Host "    Method : irm | iex -- installs to your user profile, no admin."
    Write-Host ""
    if (-not (Prompt-YN "Download and install uv?" 'y')) {
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
        Stop-TranscriptIfActive; exit 1
    }
    [void](Detect-Uv)
    if (-not $UvBin) {
        $candidate = "$env:USERPROFILE\.local\bin\uv.exe"
        if (Test-Path $candidate) { $UvBin = $candidate }
    }
    if (-not $UvBin -or -not (Test-Path $UvBin)) {
        Write-Fail "uv install completed but binary not found"
        Stop-TranscriptIfActive; exit 1
    }
    $uvVer = ""
    try { $uvVer = ((& $UvBin --version 2>$null) | Select-Object -First 1).Trim() } catch {}
    Write-Pass "uv installed" "$UvBin ($uvVer)"
    $UvDetail = $UvBin
}

# -- Install each repo (continue on failure) -----------------------------------
$idx = 0
foreach ($t in $Targets) {
    $idx++
    Install-OneRepo $t.Owner $t.Name $t.Pkg $t.Desc $t.Url $idx $TotalTargets
}

# -- Summary -------------------------------------------------------------------
$Elapsed = (Get-Date) - $StartTime
$ElapsedMin = [int]$Elapsed.TotalMinutes
$ElapsedSec = [int]($Elapsed.TotalSeconds - ($ElapsedMin * 60))

Write-Host ""
Write-HrThick
Write-Host "  ${BOLD}Installation Summary${RESET}"
Write-HrThick
switch ($UvState) {
    "PASS" { Write-Pass "uv" $UvDetail }
    "SKIP" { Write-Skip "uv" $UvDetail }
    default { Write-Warn2 "uv" $UvDetail }
}
$AllOk = $true
foreach ($r in $RepoResults) {
    switch ($r.State) {
        "PASS" { Write-Pass  $r.Name $r.Detail }
        "WARN" { Write-Warn2 $r.Name $r.Detail }
        "SKIP" { Write-Skip  $r.Name $r.Detail }
        "FAIL" { Write-Fail  $r.Name $r.Detail; $AllOk = $false }
        default { Write-Warn2 $r.Name "unknown: $($r.State)"; $AllOk = $false }
    }
}
Write-Host ""
Write-Host ("  Elapsed  : {0}m {1}s" -f $ElapsedMin, $ElapsedSec)
Write-Host ("  Location : {0}" -f $ParentAbs)
if ($WriteLog) { Write-Host ("  Log      : {0}" -f $LogFile) }
Write-Host ""

if ($AllOk) {
    Write-Host "  ${GREEN}${BOLD}All repos installed successfully.${RESET}"
    Write-Host ""
    Write-Host "  ${BOLD}To start using them:${RESET}"
    foreach ($r in $RepoResults) {
        Write-Host ("    cd {0}\{1}; .\.venv\Scripts\Activate.ps1" -f $ParentAbs, $r.Name)
    }
    Stop-TranscriptIfActive
    exit 0
} else {
    Write-Host "  ${RED}${BOLD}One or more repos failed.${RESET}"
    Write-Host ""
    Write-Host "  Review the [FAIL] entries above."
    if ($WriteLog) { Write-Host "  Full output is in: $LogFile" }
    Stop-TranscriptIfActive
    exit 1
}
