# Quick install

A single installer for OG-Core and its country calibrations. Pre-req: **git** installed. Nothing else.

It installs uv if needed, clones the repo(s) you choose, runs `uv sync --extra dev`, and verifies the import. Pick one from a menu, name several with `--repo` / `-Repo`, or install everything with `--all` / `-All`.

You can run it two ways — paste a one-line command, or download the script and run it. Both do the same thing.

## Option 1 — One-line

### macOS / Linux

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/PSLmodels/OG-Core/master/scripts/install.sh)"
```

### Windows (PowerShell)

```powershell
$f = "$env:TEMP\og-install.ps1"; irm https://raw.githubusercontent.com/PSLmodels/OG-Core/master/scripts/install.ps1 -OutFile $f; powershell -ExecutionPolicy Bypass -File $f
```

(On Windows the installer is saved to a temp file and run from there, so it executes as a normal script.)

## Option 2 — Download, then run

Handy if you'd rather read the script first, or keep it to re-run later.

### macOS / Linux

```bash
curl -fsSL https://raw.githubusercontent.com/PSLmodels/OG-Core/master/scripts/install.sh -o install.sh
bash install.sh
```

### Windows (PowerShell)

```powershell
Invoke-WebRequest -UseBasicParsing -Uri https://raw.githubusercontent.com/PSLmodels/OG-Core/master/scripts/install.ps1 -OutFile install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

## Choosing a repo and skipping prompts

By default the installer shows a menu of repos and prompts for a destination. Flags let you go straight there — they work with either method above:

- `--repo` / `-Repo` — a short key for a repo in the built-in catalog. To install several at once: on macOS/Linux repeat the flag (`--repo og-zaf --repo og-idn`), or comma-separate; on Windows comma-separate (`-Repo og-zaf,og-idn`). Keys:
  - `og-core` — base model ([PSLmodels/OG-Core](https://github.com/PSLmodels/OG-Core))
  - `og-eth` — Ethiopia ([EAPD-DRB/OG-ETH](https://github.com/EAPD-DRB/OG-ETH))
  - `og-zaf` — South Africa ([EAPD-DRB/OG-ZAF](https://github.com/EAPD-DRB/OG-ZAF))
  - `og-idn` — Indonesia ([EAPD-DRB/OG-IDN](https://github.com/EAPD-DRB/OG-IDN))
  - `og-phl` — Philippines ([EAPD-DRB/OG-PHL](https://github.com/EAPD-DRB/OG-PHL))
  - `og-bra` — Brazil ([PSLmodels/OG-BRA](https://github.com/PSLmodels/OG-BRA))
- `--all` / `-All` — install every repo in the catalog (each into its own subfolder).
- `--list` / `-List` and `--list-json` / `-ListJson` — print the repo catalog (human-readable or JSON) and exit, without installing anything. The same catalog is also published as a static file you can fetch directly: [`scripts/repos.json`](repos.json).
- `--repo-url` / `-RepoUrl` — a full git URL for a single custom repo (a fork, or a country repo not yet in the catalog). Clones the default branch.
- `--branch` / `-Branch` — **for development work**: install a non-default branch. Single repo only (not valid with `--all` or multiple `--repo`).
- `--dest` / `-Dest` and `--yes` / `-Yes` — set the parent directory and skip every confirmation prompt. Combine with `--repo` / `--all` for a fully hands-free install.

```bash
# macOS / Linux
# one repo, no prompts
bash install.sh --repo og-core --dest ~/Projects --yes
# several at once (repeat the flag, or comma-separate)
bash install.sh --repo og-zaf --repo og-idn --dest ~/Projects
# hands-free: install everything, no prompts
bash install.sh --all --dest ~/Projects --yes
# a single custom repo on a specific branch (development)
bash install.sh --repo-url https://github.com/OWNER/OG-XYZ.git --branch my-feature
```

```powershell
# Windows
# one repo, no prompts
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Repo og-core -Dest C:\Users\$env:USERNAME\Projects -Yes
# several at once
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Repo og-zaf,og-idn -Dest C:\OG
# hands-free: install everything, no prompts
powershell -ExecutionPolicy Bypass -File .\install.ps1 -All -Dest C:\OG -Yes
```

More country calibrations get added to the catalog as they migrate to uv.

## After install

Activate the venv and you're set:

```bash
# macOS / Linux
cd <destination>
source .venv/bin/activate
python -W ignore -c "import ogcore; print(ogcore.__version__)"
```

```powershell
# Windows
cd <destination>
.\.venv\Scripts\Activate.ps1
python -W ignore -c "import ogcore; print(ogcore.__version__)"
```

(Swap `ogcore` for the package of the repo you installed — e.g. `ogeth` for OG-ETH.)
