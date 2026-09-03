"""Fast-forward a local master or main branch from upstream/master."""

from __future__ import annotations

import subprocess
import sys


def git(
    *args: str, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        check=True,
        text=True,
        capture_output=capture_output,
    )


def current_branch() -> str:
    return git("branch", "--show-current", capture_output=True).stdout.strip()


def main() -> int:
    branch = current_branch()
    if branch not in {"master", "main"}:
        print(
            "STOP: switch to your local master or main branch first.",
            file=sys.stderr,
        )
        return 1

    git("fetch", "upstream")
    # A sync should not silently create a merge commit when branches diverge.
    git("merge", "--ff-only", "upstream/master")
    git("push", "origin", branch)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
