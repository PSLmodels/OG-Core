"""Create or update a local branch for an OG-Core upstream pull request."""

from __future__ import annotations

import subprocess
import sys


def git(*args: str, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        check=check,
        text=True,
        capture_output=capture_output,
    )


def current_branch() -> str:
    return git("branch", "--show-current", capture_output=True).stdout.strip()


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].isdigit() or int(sys.argv[1]) < 1:
        print("ERROR: specify one positive pull-request number: make git-pr N=123", file=sys.stderr)
        return 1

    branch = current_branch()
    if branch not in {"master", "main"}:
        print("STOP: switch to your local master or main branch first.", file=sys.stderr)
        return 1

    number = sys.argv[1]
    pr_branch = f"pr-{number}"
    git("fetch", "upstream", f"refs/pull/{number}/head")

    exists = git("show-ref", "--verify", "--quiet", f"refs/heads/{pr_branch}", check=False).returncode == 0
    if exists:
        git("switch", pr_branch)
        result = git("merge", "--ff-only", "FETCH_HEAD", check=False)
        if result.returncode:
            print(
                f"STOP: {pr_branch} cannot fast-forward to upstream PR #{number}. "
                "Delete or reconcile the local branch, then try again.",
                file=sys.stderr,
            )
            return result.returncode
    else:
        git("switch", "--create", pr_branch, "FETCH_HEAD")

    git("status", "--short", "--branch")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
