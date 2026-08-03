#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# OG-* universal installer for macOS and Linux (uv-based).
#
# Takes a user from zero (only git installed) to a working OG-* model env:
#   1. Install uv (if not present)
#   2. Clone the chosen repo(s)
#   3. uv sync --extra dev (installs Python + project + deps)
#   4. Verify import
#
# Installs one repo, several (--repo a,b or repeated --repo), or all (--all).
# Only repos that have migrated to uv (pyproject.toml + uv.lock) are offered.
#
# Usage:
#   ./scripts/install.sh                            # interactive (one repo)
#   ./scripts/install.sh --repo og-eth              # skip the model menu
#   ./scripts/install.sh --all --dest ~/OG --yes    # install everything, no prompts
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
    "og-zaf|EAPD-DRB|OG-ZAF|ogzaf|South Africa"
    "og-idn|EAPD-DRB|OG-IDN|ogidn|Indonesia"
    "og-phl|EAPD-DRB|OG-PHL|ogphl|Philippines"
    "og-bra|PSLmodels|OG-BRA|ogbra|Brazil"
)

# ── Defaults ──────────────────────────────────────────────────────────────────
SELECTED_KEYS=()
INSTALL_ALL=0
REPO_URL=""
BRANCH=""
DEST=""
ASSUME_YES=0
SKIP_UV_INSTALL=0
WITH_DEV_DEPS=1
WRITE_LOG=1
# Scratch fields populated by lookup_repo / choose_repo_interactive.
REPO_OWNER=""; REPO_NAME=""; PKG_NAME=""; REPO_DESC=""

usage() {
    cat <<EOF
Universal OG-* installer (uv-based, macOS and Linux).

Usage:
  $0 [options]

Options:
  -h, --help              Show this message and exit.
      --list              Print the repo catalog (human-readable) and exit.
      --list-json         Print the repo catalog as JSON and exit.
  -y, --yes               Auto-confirm every prompt (non-interactive).
      --repo KEY          Install a catalog repo (see --list). Repeat for
                          several (--repo og-zaf --repo og-idn); a comma-
                          separated list (--repo og-zaf,og-idn) also works.
      --all               Install every repo in the catalog.
      --repo-url URL      Install one custom Git URL (single repo only).
      --branch BRANCH     For development: clone a non-default branch. Single
                          repo only (not valid with --all or multiple --repo).
      --dest DIR          Parent directory for the clone(s) (default: current
                          directory). Each repo lands in <DIR>/<REPO_NAME>.
      --no-dev-deps       Install runtime deps only (skip dev/test tooling).
      --skip-uv-install   Don't install uv; assume it's already on PATH.
      --no-log            Don't write a log file.

Examples:
  $0                                          # interactive (one repo)
  $0 --repo og-eth                            # menu skipped; prompt for dest
  $0 --repo og-zaf --repo og-idn --dest ~/OG  # install several
  $0 --all --dest ~/OG --yes                  # hands-free: install all, no prompts
  $0 --repo og-eth --dest ~/OG --yes          # hands-free: one repo, no prompts
  $0 --repo-url https://github.com/me/OG-ETH.git --branch my-branch --dest /tmp
EOF
}

# ── Catalog listing (--list / --list-json) ────────────────────────────────────
# Both emit from the embedded REPOS array (the runtime source of truth) so they
# work even when only the script is fetched (the curl one-liner). scripts/repos.json
# is the same data as a standalone file; a CI check keeps the two in sync.
print_catalog_human() {
    printf '%-9s  %-22s  %-8s  %s\n' "KEY" "REPO" "PACKAGE" "DESCRIPTION"
    local entry k owner name pkg desc
    for entry in "${REPOS[@]}"; do
        IFS='|' read -r k owner name pkg desc <<< "$entry"
        printf '%-9s  %-22s  %-8s  %s\n' "$k" "$owner/$name" "$pkg" "$desc"
    done
}

print_catalog_json() {
    printf '{\n  "schema_version": 1,\n  "repos": [\n'
    local entry k owner name pkg desc first=1
    for entry in "${REPOS[@]}"; do
        IFS='|' read -r k owner name pkg desc <<< "$entry"
        if [ "$first" -eq 0 ]; then printf ',\n'; fi
        first=0
        printf '    {"key": "%s", "owner": "%s", "repo": "%s", "package": "%s", "description": "%s"}' \
            "$k" "$owner" "$name" "$pkg" "$desc"
    done
    printf '\n  ]\n}\n'
}

# ── Argument parsing ──────────────────────────────────────────────────────────
# Split a comma-separated value and append each key to SELECTED_KEYS.
add_repo_keys() {
    local value="$1" part
    local IFS=','
    for part in $value; do
        [ -n "$part" ] && SELECTED_KEYS+=("$part")
    done
}

# Ensure a value-taking option actually has its value (avoids a raw "$2:
# unbound variable" crash under set -u when a flag is the last argument).
need_arg() {
    if [ "$1" -lt 2 ]; then
        echo "Option $2 requires a value." >&2
        echo >&2
        usage >&2
        exit 2
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0;;
        --list) print_catalog_human; exit 0;;
        --list-json) print_catalog_json; exit 0;;
        -y|--yes) ASSUME_YES=1; shift;;
        --all) INSTALL_ALL=1; shift;;
        --repo) need_arg $# "--repo"; add_repo_keys "$2"; shift 2;;
        --repo=*) add_repo_keys "${1#*=}"; shift;;
        --repo-url) need_arg $# "--repo-url"; REPO_URL="$2"; shift 2;;
        --repo-url=*) REPO_URL="${1#*=}"; shift;;
        --branch) need_arg $# "--branch"; BRANCH="$2"; shift 2;;
        --branch=*) BRANCH="${1#*=}"; shift;;
        --dest) need_arg $# "--dest"; DEST="$2"; shift 2;;
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

section() { echo; hr; printf "  ${BOLD}%s${RESET}\n" "$1"; hr; }

err() { printf "${RED}ERROR:${RESET} %s\n" "$1" >&2; exit 2; }

# Normalize a Git URL for comparison (strip trailing .git, lowercase).
norm_url() { printf '%s' "$1" | sed -E 's#\.git/?$##' | tr '[:upper:]' '[:lower:]'; }

prompt_yn() {
    local prompt="$1" default="${2:-y}" opts
    if [ "$default" = "y" ]; then opts="[Y/n/q]"; else opts="[y/N/q]"; fi
    if [ "$ASSUME_YES" = 1 ]; then
        printf "%s %s ${DIM}(auto: yes)${RESET}\n" "$prompt" "$opts"
        return 0
    fi
    if [ ! -t 0 ]; then
        printf "${RED}ERROR:${RESET} stdin is not a terminal and --yes was not given.\n" >&2
        exit 2
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

# ── Pick repo interactively (single repo; multi is opt-in via --repo/--all) ────
choose_repo_interactive() {
    if [ ! -t 0 ]; then
        printf "${RED}ERROR:${RESET} no --repo/--all given and stdin is not a terminal.\n" >&2
        exit 2
    fi
    echo
    hr_thick
    printf "  ${BOLD}OG-* Installer (uv-based)${RESET}\n"
    hr_thick
    printf "  Which OG country model do you want to install?\n"
    printf "  ${DIM}(only repos that have migrated to uv are listed; use --all for every one)${RESET}\n"
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
        IFS='|' read -r _ REPO_OWNER REPO_NAME PKG_NAME REPO_DESC <<< "${REPOS[$idx]}"
        return 0
    done
}

# ── Per-repo result tracking ──────────────────────────────────────────────────
REPO_RESULT_NAMES=(); REPO_RESULT_STATES=(); REPO_RESULT_DETAILS=()
record_repo() { REPO_RESULT_NAMES+=("$1"); REPO_RESULT_STATES+=("$2"); REPO_RESULT_DETAILS+=("$3"); }

# install_one_repo OWNER NAME PKG DESC URL IDX TOTAL
# Clone (or update) + uv sync + verify for ONE repo. Never exits; records a
# per-repo result and returns 0 (ok/warn) or 1 (failed) so the caller can
# continue to the next repo. Uses globals: PARENT_ABS, UV_BIN, GIT_BIN,
# WITH_DEV_DEPS, BRANCH (single-repo only), PER_STEP (1 = ask before each step).
install_one_repo() {
    local owner="$1" name="$2" pkg="$3" desc="$4" url="$5" idx="$6" total="$7"
    local dest_abs="${PARENT_ABS}/${name}"
    local repo_state="PASS" branch_now=""

    echo
    hr
    if [ "$total" -gt 1 ]; then
        printf "  ${BOLD}[%d/%d] %s${RESET}  ${DIM}%s${RESET}\n" "$idx" "$total" "$name" "$owner/$name"
    else
        printf "  ${BOLD}%s${RESET}  ${DIM}%s${RESET}\n" "$name" "$owner/$name"
    fi
    hr

    # --- Clone or update existing ---
    local dest_has_repo=0
    if [ -d "$dest_abs" ]; then
        if [ -z "$(ls -A "$dest_abs" 2>/dev/null || true)" ]; then
            : # empty dir; clone into it
        elif [ -d "$dest_abs/.git" ]; then
            local existing_url
            existing_url="$("$GIT_BIN" -C "$dest_abs" config --get remote.origin.url 2>/dev/null || true)"
            if [ "$(norm_url "$existing_url")" = "$(norm_url "$url")" ]; then
                dest_has_repo=1
            else
                print_fail "$name: destination is a git repo for a different remote" "$existing_url"
                record_repo "$name" FAIL "wrong remote at $dest_abs"
                return 1
            fi
        else
            print_fail "$name: destination exists and is not empty" "$dest_abs"
            record_repo "$name" FAIL "destination not empty"
            return 1
        fi
    fi

    if [ "$dest_has_repo" = 1 ]; then
        branch_now="$("$GIT_BIN" -C "$dest_abs" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")"
        printf "  Existing clone found at %s (branch: %s).\n" "$dest_abs" "$branch_now"
        local do_update=1
        if [ "$PER_STEP" = 1 ]; then
            prompt_yn "Update existing clone (git pull --ff-only)?" y || do_update=0
        fi
        if [ "$do_update" = 1 ]; then
            echo_cmd "git -C $dest_abs pull --ff-only"
            if "$GIT_BIN" -C "$dest_abs" pull --ff-only; then
                print_pass "$name updated"
            else
                print_warn "$name: git pull failed; using existing state"
                repo_state="WARN"
            fi
        else
            print_skip "$name: using existing clone as-is"
        fi
    else
        if [ -n "$BRANCH" ]; then
            printf "  Will clone %s (branch: %s) into %s.\n" "$url" "$BRANCH" "$dest_abs"
        else
            printf "  Will clone %s into %s.\n" "$url" "$dest_abs"
        fi
        local do_clone=1
        if [ "$PER_STEP" = 1 ]; then
            prompt_yn "Clone now?" y || do_clone=0
        fi
        if [ "$do_clone" = 0 ]; then
            print_fail "$name: clone declined"
            record_repo "$name" FAIL "clone declined"
            return 1
        fi
        if [ -n "$BRANCH" ]; then
            echo_cmd "git clone --branch $BRANCH $url $dest_abs"
            if ! "$GIT_BIN" clone --branch "$BRANCH" "$url" "$dest_abs"; then
                print_fail "$name: git clone failed"; record_repo "$name" FAIL "git clone failed"; return 1
            fi
        else
            echo_cmd "git clone $url $dest_abs"
            if ! "$GIT_BIN" clone "$url" "$dest_abs"; then
                print_fail "$name: git clone failed"; record_repo "$name" FAIL "git clone failed"; return 1
            fi
        fi
        branch_now="$("$GIT_BIN" -C "$dest_abs" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")"
        print_pass "$name cloned" "$dest_abs (branch: $branch_now)"
    fi

    # --- Verify the repo is uv-native ---
    if [ ! -f "$dest_abs/pyproject.toml" ]; then
        print_fail "$name: no pyproject.toml (not a uv-native repo)"
        record_repo "$name" FAIL "no pyproject.toml"
        return 1
    fi
    [ ! -f "$dest_abs/uv.lock" ] && print_warn "$name: no uv.lock; uv sync will create one"

    # --- uv sync ---
    local sync_args
    sync_args=("sync")
    [ "$WITH_DEV_DEPS" = 1 ] && sync_args+=("--extra" "dev")
    local do_sync=1
    if [ "$PER_STEP" = 1 ]; then
        printf "  Will run: uv %s  (in %s)\n" "${sync_args[*]}" "$dest_abs"
        prompt_yn "Run uv sync now?" y || do_sync=0
    fi
    if [ "$do_sync" = 0 ]; then
        print_fail "$name: uv sync declined"
        record_repo "$name" FAIL "uv sync declined"
        return 1
    fi
    echo_cmd "uv ${sync_args[*]}  (cwd: $dest_abs)"
    if ! ( cd "$dest_abs" && "$UV_BIN" "${sync_args[@]}" ); then
        print_fail "$name: uv sync failed"
        record_repo "$name" FAIL "uv sync failed"
        return 1
    fi
    print_pass "$name installed (uv sync)"

    # --- Verify import ---
    local venv_py="$dest_abs/.venv/bin/python"
    if [ ! -x "$venv_py" ]; then
        print_fail "$name: venv python not found" "$venv_py"
        record_repo "$name" FAIL "no venv python"
        return 1
    fi
    if "$venv_py" -W ignore -c "import $pkg" >/dev/null 2>&1; then
        local ver
        ver="$("$venv_py" -W ignore -c "import $pkg; print(getattr($pkg, '__version__', '?'))" 2>/dev/null || echo '?')"
        print_pass "$name: import $pkg" "$ver"
        if [ "$repo_state" = "WARN" ]; then
            record_repo "$name" WARN "import $pkg ($ver); pull warning"
        else
            record_repo "$name" PASS "import $pkg ($ver)"
        fi
        return 0
    else
        print_fail "$name: import $pkg failed" "package not importable; check log above"
        record_repo "$name" FAIL "import $pkg failed"
        return 1
    fi
}

# ── Resolve the list of repos to install ──────────────────────────────────────
# Each TARGETS entry: OWNER|NAME|PKG|DESC|URL
TARGETS=()
if [ -n "$REPO_URL" ]; then
    if [ "$INSTALL_ALL" = 1 ] || [ "${#SELECTED_KEYS[@]}" -gt 0 ]; then
        err "--repo-url installs a single repo; it cannot be combined with --all or --repo."
    fi
    base="$(basename "$REPO_URL")"
    cu_name="${base%.git}"
    cu_pkg="$(printf '%s' "$cu_name" | tr '[:upper:]' '[:lower:]' | tr -d '-')"
    TARGETS+=("(custom URL)|$cu_name|$cu_pkg|custom repo|$REPO_URL")
elif [ "$INSTALL_ALL" = 1 ]; then
    if [ "${#SELECTED_KEYS[@]}" -gt 0 ]; then
        err "--all installs every repo; do not also pass --repo."
    fi
    for entry in "${REPOS[@]}"; do
        IFS='|' read -r k owner name pkg desc <<< "$entry"
        TARGETS+=("$owner|$name|$pkg|$desc|https://github.com/$owner/$name.git")
    done
elif [ "${#SELECTED_KEYS[@]}" -gt 0 ]; then
    seen=" "
    for key in "${SELECTED_KEYS[@]}"; do
        case "$seen" in *" $key "*) continue;; esac
        seen="$seen$key "
        if lookup_repo "$key"; then
            TARGETS+=("$REPO_OWNER|$REPO_NAME|$PKG_NAME|$REPO_DESC|https://github.com/$REPO_OWNER/$REPO_NAME.git")
        else
            err "unknown --repo '$key'. Run --list to see valid keys."
        fi
    done
else
    choose_repo_interactive
    if [ -n "$REPO_URL" ]; then
        base="$(basename "$REPO_URL")"
        cu_name="${base%.git}"
        cu_pkg="$(printf '%s' "$cu_name" | tr '[:upper:]' '[:lower:]' | tr -d '-')"
        TARGETS+=("(custom URL)|$cu_name|$cu_pkg|custom repo|$REPO_URL")
    else
        TARGETS+=("$REPO_OWNER|$REPO_NAME|$PKG_NAME|$REPO_DESC|https://github.com/$REPO_OWNER/$REPO_NAME.git")
    fi
fi

TOTAL_TARGETS="${#TARGETS[@]}"
if [ -n "$BRANCH" ] && [ "$TOTAL_TARGETS" -gt 1 ]; then
    err "--branch can only be used with a single repo (selected: $TOTAL_TARGETS)."
fi
# One repo keeps the per-step confirmations; multiple run as a batch.
PER_STEP=1; [ "$TOTAL_TARGETS" -gt 1 ] && PER_STEP=0

# ── Pick destination (PARENT directory; each repo lands at PARENT/REPO_NAME) ──
if [ -z "$DEST" ]; then
    if [ ! -t 0 ]; then
        DEST="."
    else
        printf "\n  Where would you like to install?\n"
        printf "  Enter the PARENT directory; each repo is cloned into its own subfolder.\n"
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

if [ ! -d "$DEST" ]; then
    printf "${RED}ERROR:${RESET} parent directory does not exist: %s\n" "$DEST" >&2
    printf "Create it first (mkdir -p %s) or pick a different --dest.\n" "$DEST" >&2
    exit 1
fi
PARENT_ABS="$(cd "$DEST" && pwd)"

# Refuse if PARENT is a dangerous system dir. (User-home is fine -- clones land
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
printf "  Destination  : %s\n" "$PARENT_ABS"
[ -n "$BRANCH" ] && printf "  Branch       : %s\n" "$BRANCH"
printf "  Dev/test deps: %s\n" "$([ "$WITH_DEV_DEPS" = 1 ] && echo yes || echo no)"
if [ "$UV_PRESENT" = 1 ]; then
    printf "  uv           : %s ${GREEN}detected${RESET}\n" "$UV_BIN"
else
    printf "  uv           : ${YELLOW}will install${RESET} (~5MB, official installer)\n"
fi
[ "$WRITE_LOG" = 1 ] && printf "  Log file     : %s\n" "$LOG_FILE"
echo
printf "  ${BOLD}Repos to install (%d):${RESET}\n" "$TOTAL_TARGETS"
for t in "${TARGETS[@]}"; do
    IFS='|' read -r t_owner t_name t_pkg t_desc t_url <<< "$t"
    printf "    - %-9s ${DIM}%s${RESET} -> %s/%s\n" "$t_name" "$t_desc" "$PARENT_ABS" "$t_name"
done
echo
printf "  Each repo: clone, uv sync (Python + deps, ~30s/~500MB), verify import.\n"
if [ "$PER_STEP" = 1 ]; then
    printf "  You will be asked to confirm before each mutating step.\n"
else
    printf "  ${BOLD}%d repos${RESET} will be installed after one confirmation below.\n" "$TOTAL_TARGETS"
fi
echo

proceed_prompt="Proceed with installation?"
[ "$TOTAL_TARGETS" -gt 1 ] && proceed_prompt="Proceed with installing $TOTAL_TARGETS repos?"
if ! prompt_yn "$proceed_prompt" y; then
    printf "${YELLOW}Aborted by user.${RESET}\n"; exit 0
fi

START_TS=$(date +%s)

# ── Install uv (once, shared by all repos) ────────────────────────────────────
section "Install uv"
UV_STATE="PASS"; UV_DETAIL=""
if [ "$UV_PRESENT" = 1 ]; then
    print_pass "uv already present" "$UV_BIN"
    UV_STATE="SKIP"; UV_DETAIL="already present"
elif [ "$SKIP_UV_INSTALL" = 1 ]; then
    print_fail "--skip-uv-install given but no uv found"
    exit 1
else
    printf "  Will install uv via the official installer:\n"
    printf "    Source : https://astral.sh/uv/install.sh\n"
    printf "    Target : \$HOME/.local/bin/uv (the installer's default)\n"
    printf "    Method : curl | sh -- no sudo, installs to your home directory.\n"
    echo
    if ! prompt_yn "Download and install uv?" y; then
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
        exit 1
    fi
    print_pass "uv installed" "$UV_BIN ($("$UV_BIN" --version 2>/dev/null | head -1))"
    UV_DETAIL="$UV_BIN"
fi

# ── Install each repo (continue on failure) ───────────────────────────────────
i=0
for t in "${TARGETS[@]}"; do
    i=$((i + 1))
    IFS='|' read -r t_owner t_name t_pkg t_desc t_url <<< "$t"
    install_one_repo "$t_owner" "$t_name" "$t_pkg" "$t_desc" "$t_url" "$i" "$TOTAL_TARGETS" || true
done

# ── Summary ───────────────────────────────────────────────────────────────────
END_TS=$(date +%s); ELAPSED=$((END_TS - START_TS))
ELAPSED_MIN=$((ELAPSED / 60)); ELAPSED_SEC=$((ELAPSED % 60))

echo
hr_thick
printf "  ${BOLD}Installation Summary${RESET}\n"
hr_thick
case "$UV_STATE" in
    PASS) print_pass "uv" "$UV_DETAIL";;
    SKIP) print_skip "uv" "$UV_DETAIL";;
    *)    print_warn "uv" "$UV_DETAIL";;
esac
all_ok=1
for idx in "${!REPO_RESULT_NAMES[@]}"; do
    name="${REPO_RESULT_NAMES[$idx]}"; state="${REPO_RESULT_STATES[$idx]}"; detail="${REPO_RESULT_DETAILS[$idx]}"
    case "$state" in
        PASS) print_pass "$name" "$detail";;
        WARN) print_warn "$name" "$detail";;
        SKIP) print_skip "$name" "$detail";;
        FAIL) print_fail "$name" "$detail"; all_ok=0;;
        *)    print_warn "$name" "unknown: $state"; all_ok=0;;
    esac
done
echo
printf "  Elapsed  : %dm %ds\n" "$ELAPSED_MIN" "$ELAPSED_SEC"
printf "  Location : %s\n" "$PARENT_ABS"
[ "$WRITE_LOG" = 1 ] && printf "  Log      : %s\n" "$LOG_FILE"
echo

if [ "$all_ok" = 1 ]; then
    printf "  ${GREEN}${BOLD}All repos installed successfully.${RESET}\n"
    echo
    printf "  ${BOLD}To start using them:${RESET}\n"
    for idx in "${!REPO_RESULT_NAMES[@]}"; do
        name="${REPO_RESULT_NAMES[$idx]}"
        printf "    cd %s/%s && source .venv/bin/activate\n" "$PARENT_ABS" "$name"
    done
    exit 0
else
    printf "  ${RED}${BOLD}One or more repos failed.${RESET}\n"
    echo
    printf "  Review the [FAIL] entries above.\n"
    [ "$WRITE_LOG" = 1 ] && printf "  Full output is in: %s\n" "$LOG_FILE"
    exit 1
fi
