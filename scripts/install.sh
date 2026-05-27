#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# OG-* universal installer for macOS and Linux (uv-based).
#
# Takes a user from zero (only git installed) to a working OG-* model env:
#   1. Install uv (if not present)
#   2. Clone the chosen repo
#   3. uv sync --extra dev (installs Python + project + deps)
#   4. Verify import
#
# Only repos that have migrated to uv (pyproject.toml + uv.lock) are offered.
#
# Usage:
#   ./scripts/install.sh                   # interactive
#   ./scripts/install.sh --repo og-eth     # skip the model menu
#   ./scripts/install.sh --help
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Repo catalog ──────────────────────────────────────────────────────────────
# Each entry: KEY|OWNER|REPO_NAME|PKG_NAME|DESCRIPTION
# Only repos that have migrated to uv (pyproject.toml + uv.lock).
# Add entries as other OG-* repos migrate.
REPOS=(
    "og-core|PSLmodels|OG-Core|ogcore|base model (no country calibration)"
    "og-eth|EAPD-DRB|OG-ETH|ogeth|Ethiopia"
)

# ── Defaults ──────────────────────────────────────────────────────────────────
REPO_KEY=""
REPO_URL=""
BRANCH=""
DEST=""
ASSUME_YES=0
SKIP_UV_INSTALL=0
WITH_DEV_DEPS=1
WRITE_LOG=1

usage() {
    cat <<EOF
Universal OG-* installer (uv-based, macOS and Linux).

Usage:
  $0 [options]

Options:
  -h, --help              Show this message and exit.
  -y, --yes               Auto-confirm every prompt (non-interactive).
      --repo KEY          Skip menu; one of: og-core, og-eth
      --repo-url URL      Use a custom Git URL. Bypasses the menu.
      --branch BRANCH     For development: clone a non-default branch (default:
                          repo's default branch). Useful for testing forks/PRs.
      --dest DIR          Parent directory where the clone is created
                          (default: current directory). The clone always
                          lands in <DIR>/\${REPO_NAME}.
      --no-dev-deps       Install runtime deps only (skip dev/test tooling).
      --skip-uv-install   Don't install uv; assume it's already on PATH.
      --no-log            Don't write a log file.

Examples:
  $0                                        # fully interactive
  $0 --repo og-eth                          # menu skipped; prompt for dest
  $0 --repo og-eth --dest ~/Projects --yes        # clones to ~/Projects/OG-ETH
  $0 --repo-url git@github.com:me/OG-USA.git --dest .
  $0 --repo-url https://github.com/me/OG-ETH.git --branch my-branch --dest /tmp
EOF
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0;;
        -y|--yes) ASSUME_YES=1; shift;;
        --repo) REPO_KEY="$2"; shift 2;;
        --repo=*) REPO_KEY="${1#*=}"; shift;;
        --repo-url) REPO_URL="$2"; shift 2;;
        --repo-url=*) REPO_URL="${1#*=}"; shift;;
        --branch) BRANCH="$2"; shift 2;;
        --branch=*) BRANCH="${1#*=}"; shift;;
        --dest) DEST="$2"; shift 2;;
        --dest=*) DEST="${1#*=}"; shift;;
        --no-dev-deps) WITH_DEV_DEPS=0; shift;;
        --skip-uv-install) SKIP_UV_INSTALL=1; shift;;
        --no-log) WRITE_LOG=0; shift;;
        *) echo "Unknown option: $1" >&2; echo; usage >&2; exit 2;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'
    RED=$'\033[91m'; GREEN=$'\033[92m'; YELLOW=$'\033[93m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi

# ── Logging ───────────────────────────────────────────────────────────────────
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${SCRIPT_DIR}/.install-${TS}.log"
if [ "$WRITE_LOG" = 1 ]; then
    if : > "$LOG_FILE" 2>/dev/null; then
        exec > >(tee -a "$LOG_FILE") 2>&1
    else
        printf "WARN: cannot write to %s; logging disabled.\n" "$LOG_FILE" >&2
        WRITE_LOG=0
    fi
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
hr()       { printf '%s\n' "──────────────────────────────────────────────────────────────"; }
hr_thick() { printf '%s\n' "══════════════════════════════════════════════════════════════"; }

print_pass() { printf "  ${GREEN}[PASS]${RESET} %s%s\n" "$1" "${2:+  ${DIM}($2)${RESET}}"; }
print_fail() { printf "  ${RED}[FAIL]${RESET} %s%s\n" "$1" "${2:+  ${DIM}($2)${RESET}}"; }
print_warn() { printf "  ${YELLOW}[WARN]${RESET} %s%s\n" "$1" "${2:+  ${DIM}($2)${RESET}}"; }
print_skip() { printf "  ${YELLOW}[SKIP]${RESET} %s%s\n" "$1" "${2:+  ${DIM}($2)${RESET}}"; }
echo_cmd()   { printf "  ${DIM}$ %s${RESET}\n" "$*"; }

TOTAL_STEPS=4
step_banner() {
    echo
    hr
    printf "  ${BOLD}Step %s of %s: %s${RESET}\n" "$1" "$TOTAL_STEPS" "$2"
    hr
}

prompt_yn() {
    local prompt="$1" default="${2:-y}" opts
    if [ "$default" = "y" ]; then opts="[Y/n/q]"; else opts="[y/N/q]"; fi
    if [ "$ASSUME_YES" = 1 ]; then
        printf "%s %s ${DIM}(auto: yes)${RESET}\n" "$prompt" "$opts"
        return 0
    fi
    if [ ! -t 0 ]; then
        printf "${RED}ERROR:${RESET} stdin is not a terminal and --yes was not given.\n" >&2
        return 2
    fi
    while true; do
        printf "%s %s " "$prompt" "$opts"
        local ans=""
        IFS= read -r ans || true
        ans="${ans:-$default}"
        case "$ans" in
            [Yy]|[Yy][Ee][Ss]) return 0;;
            [Nn]|[Nn][Oo]) return 1;;
            [Qq]|[Qq][Uu][Ii][Tt])
                printf "${YELLOW}Aborted by user.${RESET}\n"; exit 130;;
            *) printf "  Please answer y, n, or q.\n";;
        esac
    done
}

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [ "$(id -u)" = "0" ]; then
    printf "${RED}ERROR:${RESET} do not run this script as root.\n" >&2
    exit 1
fi
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    printf "${RED}ERROR:${RESET} conda env '${CONDA_DEFAULT_ENV}' is active.\n" >&2
    printf "Run 'conda deactivate' first, then re-run.\n" >&2
    exit 1
fi
if [ -n "${VIRTUAL_ENV:-}" ]; then
    printf "${RED}ERROR:${RESET} virtualenv '${VIRTUAL_ENV}' is active.\n" >&2
    printf "Run 'deactivate' first, then re-run.\n" >&2
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    printf "${RED}ERROR:${RESET} 'git' is not installed (or not on PATH).\n" >&2
    case "$(uname -s)" in
        Darwin) printf "Install: xcode-select --install (or: brew install git)\n" >&2;;
        Linux) printf "Install: sudo apt-get install -y git  (or dnf/pacman equivalent)\n" >&2;;
    esac
    exit 1
fi
GIT_BIN="$(command -v git)"
if ! command -v curl >/dev/null 2>&1; then
    printf "${RED}ERROR:${RESET} 'curl' is required to install uv.\n" >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin) OS_NAME="macOS";;
    Linux)  OS_NAME="Linux";;
    *) printf "${RED}ERROR:${RESET} unsupported OS '%s'.\n" "$(uname -s)" >&2; exit 1;;
esac

# ── Detect existing uv ────────────────────────────────────────────────────────
UV_BIN=""
detect_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"; return 0
    fi
    local c
    for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "/usr/local/bin/uv" "/opt/homebrew/bin/uv"; do
        if [ -x "$c" ]; then UV_BIN="$c"; return 0; fi
    done
    return 1
}
detect_uv || true
UV_PRESENT=0; [ -n "$UV_BIN" ] && UV_PRESENT=1

# ── Look up repo by key ───────────────────────────────────────────────────────
lookup_repo() {
    local key="$1" entry k owner name pkg desc
    for entry in "${REPOS[@]}"; do
        IFS='|' read -r k owner name pkg desc <<< "$entry"
        if [ "$k" = "$key" ]; then
            REPO_OWNER="$owner"; REPO_NAME="$name"; PKG_NAME="$pkg"; REPO_DESC="$desc"
            return 0
        fi
    done
    return 1
}

# ── Pick repo (interactive menu or from --repo flag) ──────────────────────────
choose_repo_interactive() {
    if [ ! -t 0 ]; then
        printf "${RED}ERROR:${RESET} no --repo given and stdin is not a terminal.\n" >&2
        exit 2
    fi
    echo
    hr_thick
    printf "  ${BOLD}OG-* Installer (uv-based)${RESET}\n"
    hr_thick
    printf "  Which OG country model do you want to install?\n"
    printf "  ${DIM}(only repos that have migrated to uv are listed)${RESET}\n"
    echo
    local i=1 entry k owner name pkg desc
    for entry in "${REPOS[@]}"; do
        IFS='|' read -r k owner name pkg desc <<< "$entry"
        printf "    %d) %-9s (%-10s) -- %s\n" "$i" "$name" "$owner" "$desc"
        i=$((i + 1))
    done
    printf "    %d) Other (paste a Git URL)\n" "$i"
    echo
    while true; do
        printf "  Choice [1-%d]: " "$i"
        local choice=""
        IFS= read -r choice || true
        if ! printf '%s' "$choice" | grep -Eq '^[0-9]+$'; then
            printf "  Please enter a number.\n"; continue
        fi
        if [ "$choice" -lt 1 ] || [ "$choice" -gt "$i" ]; then
            printf "  Out of range.\n"; continue
        fi
        if [ "$choice" -eq "$i" ]; then
            printf "  Git URL : "
            local url=""
            IFS= read -r url || true
            if [ -z "$url" ]; then printf "  No URL given.\n"; continue; fi
            REPO_URL="$url"; return 0
        fi
        local idx=$((choice - 1))
        IFS='|' read -r REPO_KEY REPO_OWNER REPO_NAME PKG_NAME REPO_DESC <<< "${REPOS[$idx]}"
        return 0
    done
}

if [ -n "$REPO_URL" ] && [ -z "$REPO_KEY" ]; then
    : # custom URL; menu skipped
elif [ -n "$REPO_KEY" ]; then
    if ! lookup_repo "$REPO_KEY"; then
        printf "${RED}ERROR:${RESET} unknown --repo '%s'. Use --help for the list.\n" "$REPO_KEY" >&2
        exit 2
    fi
else
    choose_repo_interactive
fi

# Custom URL: derive repo name + package name from it.
if [ -n "$REPO_URL" ] && [ -z "${REPO_NAME:-}" ]; then
    base="$(basename "$REPO_URL")"
    REPO_NAME="${base%.git}"
    REPO_OWNER="(custom URL)"
    REPO_DESC="custom repo"
    PKG_NAME="$(printf '%s' "$REPO_NAME" | tr '[:upper:]' '[:lower:]' | tr -d '-')"
    REPO_KEY="$PKG_NAME"
fi

if [ -z "$REPO_URL" ]; then
    REPO_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}.git"
fi

# ── Pick destination (PARENT directory; clone lands at PARENT/REPO_NAME) ──────
if [ -z "$DEST" ]; then
    if [ ! -t 0 ]; then
        DEST="."
    else
        printf "\n  Where would you like to install %s?\n" "$REPO_NAME"
        printf "  Enter the PARENT directory; %s will be cloned as a subfolder inside.\n" "$REPO_NAME"
        printf "  Default: current directory (%s)\n" "$(pwd)"
        printf "  Parent directory [.]: "
        IFS= read -r DEST || true
        DEST="${DEST:-.}"
    fi
fi

# Expand ~
case "$DEST" in
    "~") DEST="$HOME";;
    "~/"*) DEST="$HOME/${DEST#~/}";;
esac

# DEST is the parent directory. Must exist; resolve to absolute.
if [ ! -d "$DEST" ]; then
    printf "${RED}ERROR:${RESET} parent directory does not exist: %s\n" "$DEST" >&2
    printf "Create it first (mkdir -p %s) or pick a different --dest.\n" "$DEST" >&2
    exit 1
fi
PARENT_ABS="$(cd "$DEST" && pwd)"
DEST_ABS="${PARENT_ABS}/${REPO_NAME}"

# Refuse if PARENT is a dangerous system dir. (User-home is fine -- clone lands
# inside it, not overwriting it.)
case "$PARENT_ABS" in
    "/"|"/usr"|"/etc"|"/var"|"/bin"|"/sbin"|"/opt")
        printf "${RED}ERROR:${RESET} refusing to install into '%s' (system dir).\n" "$PARENT_ABS" >&2
        exit 1;;
esac

# ── Banner / plan ─────────────────────────────────────────────────────────────
echo
hr_thick
printf "  ${BOLD}OG-* Installer (uv-based)${RESET}\n"
hr_thick
printf "  Platform     : %s %s\n" "$OS_NAME" "$(uname -m)"
printf "  Model        : %s\n" "$REPO_NAME"
printf "  Description  : %s\n" "$REPO_DESC"
printf "  Source       : %s\n" "$REPO_URL"
[ -n "$BRANCH" ] && printf "  Branch       : %s\n" "$BRANCH"
printf "  Destination  : %s\n" "$DEST_ABS"
printf "  Package      : %s\n" "$PKG_NAME"
printf "  Dev/test deps: %s\n" "$([ "$WITH_DEV_DEPS" = 1 ] && echo yes || echo no)"
if [ "$UV_PRESENT" = 1 ]; then
    printf "  uv           : %s ${GREEN}detected${RESET}\n" "$UV_BIN"
else
    printf "  uv           : ${YELLOW}will install${RESET} (~5MB, official installer)\n"
fi
[ "$WRITE_LOG" = 1 ] && printf "  Log file     : %s\n" "$LOG_FILE"
echo
printf "  ${BOLD}Plan (%d steps):${RESET}\n" "$TOTAL_STEPS"
if [ "$UV_PRESENT" = 1 ] || [ "$SKIP_UV_INSTALL" = 1 ]; then
    printf "    1. Install uv                      ${DIM}skipped${RESET}\n"
else
    printf "    1. Install uv                      ${DIM}~5MB, seconds${RESET}\n"
fi
printf "    2. Clone %-25s ${DIM}depends on network${RESET}\n" "$REPO_NAME"
printf "    3. uv sync (Python + deps)         ${DIM}~30s, ~500MB${RESET}\n"
printf "    4. Verify installation             ${DIM}a few seconds${RESET}\n"
echo
printf "  You will be asked to confirm before each mutating step.\n"
echo

if ! prompt_yn "Proceed with installation?" y; then
    printf "${YELLOW}Aborted by user.${RESET}\n"; exit 0
fi

# ── Result tracking ───────────────────────────────────────────────────────────
declare -a STEP_NAMES=() STEP_STATES=() STEP_DETAILS=()
record_step() { STEP_NAMES+=("$1"); STEP_STATES+=("$2"); STEP_DETAILS+=("$3"); }
START_TS=$(date +%s)

# ── Step 1: Install uv ────────────────────────────────────────────────────────
step_banner 1 "Install uv"
if [ "$UV_PRESENT" = 1 ]; then
    print_pass "uv already present" "$UV_BIN"
    record_step "uv" SKIP "already present"
elif [ "$SKIP_UV_INSTALL" = 1 ]; then
    print_fail "--skip-uv-install given but no uv found"
    record_step "uv" FAIL "no uv and --skip-uv-install"
    exit 1
else
    printf "  Will install uv via the official installer:\n"
    printf "    Source : https://astral.sh/uv/install.sh\n"
    printf "    Target : \$HOME/.local/bin/uv (the installer's default)\n"
    printf "    Method : curl | sh -- no sudo, installs to your home directory.\n"
    echo
    if ! prompt_yn "Download and install uv?" y; then
        record_step "uv" SKIP "declined"
        printf "${RED}Cannot continue without uv. Aborting.${RESET}\n"; exit 1
    fi
    echo_cmd "curl -LsSf https://astral.sh/uv/install.sh | sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    detect_uv || true
    if [ -z "$UV_BIN" ] || [ ! -x "$UV_BIN" ]; then
        if [ -x "$HOME/.local/bin/uv" ]; then UV_BIN="$HOME/.local/bin/uv"
        elif [ -x "$HOME/.cargo/bin/uv" ]; then UV_BIN="$HOME/.cargo/bin/uv"
        fi
    fi
    if [ -z "$UV_BIN" ] || [ ! -x "$UV_BIN" ]; then
        print_fail "uv install completed but binary not found"
        record_step "uv" FAIL "binary not found post-install"
        exit 1
    fi
    print_pass "uv installed" "$UV_BIN ($("$UV_BIN" --version 2>/dev/null | head -1))"
    record_step "uv" PASS "$UV_BIN"
fi

# ── Step 2: Clone the repo ────────────────────────────────────────────────────
step_banner 2 "Clone ${REPO_NAME}"

DEST_HAS_REPO=0; DEST_EMPTY=0
if [ -d "$DEST_ABS" ]; then
    if [ -z "$(ls -A "$DEST_ABS" 2>/dev/null || true)" ]; then
        DEST_EMPTY=1
    elif [ -d "$DEST_ABS/.git" ]; then
        existing_url="$("$GIT_BIN" -C "$DEST_ABS" config --get remote.origin.url 2>/dev/null || true)"
        norm() { printf '%s' "$1" | sed -E 's#\.git/?$##' | tr '[:upper:]' '[:lower:]'; }
        if [ "$(norm "$existing_url")" = "$(norm "$REPO_URL")" ]; then
            DEST_HAS_REPO=1
        else
            print_fail "Destination is a git repo for a different remote" "$existing_url"
            printf "  Either remove %s or pick a different destination with --dest.\n" "$DEST_ABS"
            record_step "Clone" FAIL "wrong remote"; exit 1
        fi
    else
        print_fail "Destination exists and is not empty (and not a git clone)" "$DEST_ABS"
        printf "  Either remove %s or pick a different destination with --dest.\n" "$DEST_ABS"
        record_step "Clone" FAIL "destination not empty"; exit 1
    fi
fi

if [ "$DEST_HAS_REPO" = 1 ]; then
    branch="$("$GIT_BIN" -C "$DEST_ABS" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")"
    printf "  Existing clone of %s found at %s (branch: %s).\n" "$REPO_NAME" "$DEST_ABS" "$branch"
    printf "  Will run 'git pull --ff-only' to bring it up to date.\n"
    echo
    if prompt_yn "Update existing clone?" y; then
        echo_cmd "git -C $DEST_ABS pull --ff-only"
        if "$GIT_BIN" -C "$DEST_ABS" pull --ff-only; then
            print_pass "Repo updated" "$DEST_ABS"
            record_step "Clone" PASS "updated ($branch)"
        else
            print_warn "git pull failed; continuing with existing state"
            record_step "Clone" WARN "pull failed; existing state used"
        fi
    else
        print_skip "Update" "using existing clone as-is"
        record_step "Clone" SKIP "existing clone used as-is ($branch)"
    fi
else
    if [ -n "$BRANCH" ]; then
        printf "  Will clone %s (branch: %s) into %s.\n" "$REPO_URL" "$BRANCH" "$DEST_ABS"
    else
        printf "  Will clone %s into %s.\n" "$REPO_URL" "$DEST_ABS"
    fi
    echo
    if prompt_yn "Clone now?" y; then
        if [ -n "$BRANCH" ]; then
            echo_cmd "git clone --branch $BRANCH $REPO_URL $DEST_ABS"
            "$GIT_BIN" clone --branch "$BRANCH" "$REPO_URL" "$DEST_ABS"
        else
            echo_cmd "git clone $REPO_URL $DEST_ABS"
            "$GIT_BIN" clone "$REPO_URL" "$DEST_ABS"
        fi
        branch="$("$GIT_BIN" -C "$DEST_ABS" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")"
        print_pass "Cloned" "$DEST_ABS (branch: $branch)"
        record_step "Clone" PASS "$branch"
    else
        print_fail "Clone declined; cannot continue."
        record_step "Clone" FAIL "declined"; exit 1
    fi
fi

# Verify the repo is uv-native (has pyproject.toml and uv.lock).
if [ ! -f "$DEST_ABS/pyproject.toml" ]; then
    print_fail "$DEST_ABS has no pyproject.toml"
    printf "  This installer requires repos that have migrated to uv.\n"
    record_step "Clone" FAIL "no pyproject.toml"; exit 1
fi
if [ ! -f "$DEST_ABS/uv.lock" ]; then
    print_warn "$DEST_ABS has no uv.lock; uv sync will create one"
fi

# ── Step 3: uv sync ───────────────────────────────────────────────────────────
step_banner 3 "Install ${REPO_NAME} (uv sync)"
sync_args=("sync")
[ "$WITH_DEV_DEPS" = 1 ] && sync_args+=("--extra" "dev")
printf "  Will install %s + Python + all deps into %s/.venv\n" "$REPO_NAME" "$DEST_ABS"
printf "  Working directory : %s\n" "$DEST_ABS"
printf "  Command           : uv %s\n" "${sync_args[*]}"
echo
if prompt_yn "Run uv sync now?" y; then
    cd "$DEST_ABS"
    echo_cmd "uv ${sync_args[*]}"
    "$UV_BIN" "${sync_args[@]}"
    print_pass "${REPO_NAME} installed (editable)"
    record_step "${REPO_NAME} install" PASS "uv sync"
else
    print_fail "uv sync declined; package will not be importable."
    record_step "${REPO_NAME} install" FAIL "declined"
fi

# ── Step 4: Verify ────────────────────────────────────────────────────────────
step_banner 4 "Verify installation"
if [ ! -x "$DEST_ABS/.venv/bin/python" ]; then
    print_fail "Venv python not found; skipping import check." "$DEST_ABS/.venv/bin/python"
    record_step "Verification" FAIL "no venv python"
else
    # -W ignore silences upstream deprecation warnings (e.g. from pygam) so
    # the verification output stays clean.
    pyver="$("$DEST_ABS/.venv/bin/python" -W ignore -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown)"
    print_pass "Python in .venv" "$pyver"
    if "$DEST_ABS/.venv/bin/python" -W ignore -c "import $PKG_NAME" >/dev/null 2>&1; then
        ver="$("$DEST_ABS/.venv/bin/python" -W ignore -c "import $PKG_NAME; print(getattr($PKG_NAME, '__version__', '?'))" 2>/dev/null)"
        print_pass "import $PKG_NAME" "$ver"
        record_step "Verification" PASS "import $PKG_NAME ($ver)"
    else
        print_fail "import $PKG_NAME" "package not importable; check log above"
        record_step "Verification" FAIL "import $PKG_NAME failed"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
END_TS=$(date +%s); ELAPSED=$((END_TS - START_TS))
ELAPSED_MIN=$((ELAPSED / 60)); ELAPSED_SEC=$((ELAPSED % 60))

echo
hr_thick
printf "  ${BOLD}Installation Summary -- %s${RESET}\n" "$REPO_NAME"
hr_thick
all_ok=1
for i in "${!STEP_NAMES[@]}"; do
    name="${STEP_NAMES[$i]}"; state="${STEP_STATES[$i]}"; detail="${STEP_DETAILS[$i]}"
    case "$state" in
        PASS) print_pass "$name" "$detail";;
        SKIP) print_skip "$name" "$detail";;
        WARN) print_warn "$name" "$detail";;
        FAIL) print_fail "$name" "$detail"; all_ok=0;;
        *)    print_warn "$name" "unknown: $state";;
    esac
done
echo
printf "  Elapsed  : %dm %ds\n" "$ELAPSED_MIN" "$ELAPSED_SEC"
printf "  Location : %s\n" "$DEST_ABS"
printf "  Venv     : %s/.venv\n" "$DEST_ABS"
[ "$WRITE_LOG" = 1 ] && printf "  Log      : %s\n" "$LOG_FILE"
echo

if [ "$all_ok" = 1 ]; then
    printf "  ${GREEN}${BOLD}All steps completed successfully.${RESET}\n"
    echo
    printf "  ${BOLD}To start using ${REPO_NAME}:${RESET}\n"
    printf "    cd %s\n" "$DEST_ABS"
    printf "    source .venv/bin/activate         # activate venv\n"
    printf "    python -W ignore -c \"import %s; print(%s.__file__)\"\n" "$PKG_NAME" "$PKG_NAME"
    printf "  Or run commands without activating:\n"
    printf "    uv run python -W ignore -c \"import %s; print(%s.__file__)\"\n" "$PKG_NAME" "$PKG_NAME"
    if [ -d "$DEST_ABS/examples" ]; then
        printf "\n  Example scripts: %s/examples\n" "$DEST_ABS"
    fi
    exit 0
else
    printf "  ${RED}${BOLD}One or more steps failed.${RESET}\n"
    echo
    printf "  Review the [FAIL] entries above.\n"
    [ "$WRITE_LOG" = 1 ] && printf "  Full output is in: %s\n" "$LOG_FILE"
    exit 1
fi
