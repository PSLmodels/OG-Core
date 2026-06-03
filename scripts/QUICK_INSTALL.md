# Quick install

A single installer for OG-Core and its country calibrations. Pre-req: **git** installed. Nothing else.

It installs uv if needed, clones the repo you choose, runs `uv sync --extra dev`, and verifies the import. Pick the repo from a menu, or pass `--repo` / `-Repo`.

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

- `--repo` / `-Repo` — a short key for a repo in the built-in catalog:
  - `og-core` — base model ([PSLmodels/OG-Core](https://github.com/PSLmodels/OG-Core))
  - `og-eth` — Ethiopia calibration ([EAPD-DRB/OG-ETH](https://github.com/EAPD-DRB/OG-ETH)); works once its uv migration lands
- `--repo-url` / `-RepoUrl` — a full git URL, for any other uv-based repo (a fork, or a country repo not yet in the catalog). Clones the default branch.
- `--branch` / `-Branch` — **for development work**: install a non-default branch (e.g. a fork or a migration branch before it merges).
- `--dest` / `-Dest` and `--yes` / `-Yes` — set the parent directory and skip the confirmation prompt.

```bash
# macOS / Linux -- OG-Core to ~/Projects/OG-Core, no prompts
bash install.sh --repo og-core --dest ~/Projects --yes
# any repo by URL (default branch)
bash install.sh --repo-url https://github.com/OWNER/OG-XYZ.git
# a specific branch (development)
bash install.sh --repo-url https://github.com/OWNER/OG-XYZ.git --branch my-feature
```

```powershell
# Windows -- OG-Core to C:\Users\<you>\Projects\OG-Core, no prompts
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Repo og-core -Dest C:\Users\$env:USERNAME\Projects -Yes
# any repo by URL (default branch)
powershell -ExecutionPolicy Bypass -File .\install.ps1 -RepoUrl https://github.com/OWNER/OG-XYZ.git
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
