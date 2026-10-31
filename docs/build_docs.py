"""Build Jupyter Book 2 and the Sphinx autodoc API as one static website.

The MyST source is staged below _build so URL-prefix configuration and
generated pages never overwrite authored files. Existing API URLs and Python
object anchors are retained, including links into the checked-out source.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlsplit
import zlib

from pybtex.backends.markdown import Backend
from pybtex.database import parse_file
from pybtex.style.formatting.alpha import Style
import yaml

DOCS = Path(__file__).resolve().parent
BOOK = DOCS / "book"
BUILD = BOOK / "_build"


class BibliographyBackend(Backend):
    """Keep source URLs available without printing them across the page."""

    def format_href(self, url, text, external=False):
        parsed = urlsplit(url)
        if parsed.hostname in {"doi.org", "dx.doi.org"}:
            label = "DOI"
        elif parsed.path.lower().endswith(".pdf"):
            label = "PDF"
        else:
            label = "Source"
        return super().format_href(url, label, external=False)


def citation_anchor(source, key):
    """Use stable, project-unique IDs rather than bibliography positions."""
    prefix = "reference" if source == "OGCore_references" else "usecase"
    slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
    return f"{prefix}-{slug}"


def run(*args, cwd=DOCS.parent, env=None):
    """Run a build step with the same interpreter/environment as uv run."""
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def generate_reference_pages(stage):
    """Generate parameter/variable pages from the same JSON as the model."""
    for name in ("make_params", "make_vars"):
        spec = importlib.util.spec_from_file_location(
            name, DOCS / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUTPUT_PATH = str(stage / "content" / "intro")
        module.main()


def generate_bibliographies(stage):
    """Keep complete, separate bibliographies, including uncited use cases."""
    generated = stage / "_generated"
    generated.mkdir(exist_ok=True)
    for source, target, prefix in (
        ("OGCore_references.bib", "references.md", ""),
        ("citations.bib", "citations.md", "C-"),
    ):
        data = parse_file(BOOK / source)
        for entry in data.entries.values():
            # Empty optional fields must not produce links such as doi.org/.
            for key, value in list(entry.fields.items()):
                if key.lower() in {"doi", "url"} and not value.strip():
                    del entry.fields[key]
        entries = Style().format_bibliography(data)
        blocks = []
        for entry in entries:
            anchor = citation_anchor(Path(source).stem, entry.key)
            description = entry.text.render(BibliographyBackend())
            description = description.replace("\n", "\n    ")
            blocks.append(
                f"({anchor})=\n**[{prefix}{entry.label}]**\n:   {description}"
            )
        text = "\n\n".join(blocks)
        (generated / target).write_text(
            ":::{div}\n:class: ogcore-bibliography\n\n" + text + "\n\n:::\n"
        )


def prefix_api_urls(entries, base_url):
    """Resolve API navigation for both local hosting and GitHub Pages."""
    for entry in entries:
        if entry.get("url", "").startswith("/content/api/"):
            entry["url"] = base_url + entry["url"]
        prefix_api_urls(entry.get("children", []), base_url)


def stage_book(stage, base_url):
    """Copy authored inputs, keeping build caches outside the source tree."""
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copytree(BOOK / "content", stage / "content")
    for path in BOOK.glob("*.bib"):
        shutil.copy2(path, stage / path.name)
    shutil.copy2(DOCS / "OG-Core_logo.png", stage / "OG-Core_logo.png")
    shutil.copy2(BOOK / "ogcore.css", stage / "ogcore.css")
    config = yaml.safe_load((BOOK / "myst.yml").read_text())
    prefix_api_urls(config["project"]["toc"], base_url)
    # MyST adds BASE_URL to header navigation itself. URL entries in the
    # project TOC are external links and need the explicit prefix above.
    (stage / "myst.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    generate_reference_pages(stage)
    generate_bibliographies(stage)


def write_redirects(stage, output):
    """Preserve legacy narrative .html URLs and section/equation fragments."""
    legacy = json.loads((DOCS / "bibliography_anchors.json").read_text())
    for path in (stage / "content").rglob("*.md"):
        relative = path.relative_to(stage)
        # MyST uses the project root for the introduction.
        target = (
            "/"
            if str(relative) == "content/intro/intro.md"
            else "/"
            + relative.with_suffix("").as_posix().replace("_", "-").lower()
            + "/"
        )
        redirect = output / relative.with_suffix(".html")
        redirect.parent.mkdir(parents=True, exist_ok=True)
        # Resolve from the legacy file so BASE_URL need not be hard-coded.
        destination = os.path.relpath(
            output / target.lstrip("/"), redirect.parent
        )
        destination = destination.rstrip("/") + "/"
        aliases = {
            "#" + old: "#" + citation_anchor(path.stem, key)
            for old, key in legacy.get(path.stem, {}).items()
        }
        redirect.write_text(
            '<!doctype html><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="0;url={destination}">'
            f'<link rel="canonical" href="{destination}">'
            f'<a href="{destination}">Continue to the documentation</a>'
            "<script>const aliases = " + json.dumps(aliases) + ";"
            "const hash = location.hash.toLowerCase()"
            ".replace(/^#equation-/, '#').replace(/_/g, '-');"
            f"location.replace({json.dumps(destination)} + location.search + "
            "(aliases[hash] || hash));"
            "</script>\n"
        )


def merge_inventories(output):
    """Expose Python objects at the existing root intersphinx URL as well."""
    root = output / "objects.inv"
    api = output / "content" / "api" / "objects.inv"
    parts = root.read_bytes().split(b"\n", 4)
    header = b"\n".join(parts[:4]) + b"\n"
    entries = zlib.decompress(parts[4]).decode().splitlines()
    # The book keeps its narrative inventory entries. Python definitions come
    # from Sphinx; their relative URLs must resolve from the published root.
    pattern = re.compile(r"(.+?)\s+(py:\S+)\s+(-?\d+)\s+(\S+)\s+(.*)")
    entries = [line for line in entries if not pattern.fullmatch(line)]
    api_body = zlib.decompress(api.read_bytes().split(b"\n", 4)[4]).decode()
    count = 0
    for line in api_body.splitlines():
        match = pattern.fullmatch(line)
        if not match:
            continue
        name, domain, priority, uri, display = match.groups()
        entries.append(
            f"{name} {domain} {priority} content/api/{uri} {display}"
        )
        count += 1
    if not count:
        raise RuntimeError(
            "Sphinx did not generate any Python inventory entries"
        )
    root.write_bytes(
        header + zlib.compress(("\n".join(entries) + "\n").encode())
    )


def prepare_static_navigation(output, base_url):
    """Keep API links out of the MyST client-side router."""
    shutil.copy2(BOOK / "navigation.js", output / "ogcore-navigation.js")
    script = f'<script src="{base_url}/ogcore-navigation.js" defer></script>'
    for page in output.rglob("*.html"):
        if page.is_relative_to(output / "content" / "api"):
            continue
        html = page.read_text()
        if "</head>" in html:
            page.write_text(html.replace("</head>", script + "</head>", 1))


def verify_api(output):
    """Catch a successful book build that accidentally omits the API."""
    api = output / "content" / "api"
    for source in (BOOK / "content" / "api").glob("*.rst"):
        if not (api / source.with_suffix(".html").name).is_file():
            raise RuntimeError(f"API page missing: {source.name}")
    for page, anchor in (
        ("pensions", "ogcore.pensions.NDC_amount"),
        (
            "parameters",
            "ogcore.parameters.Specifications.update_specifications",
        ),
        ("solvers", "ogcore.solvers.AndersonAccelerator.update"),
    ):
        html = (api / f"{page}.html").read_text()
        if f'id="{anchor}"' not in html or "[source]" not in html:
            raise RuntimeError(f"API object or source link missing: {anchor}")
    if not (api / "_modules" / "ogcore" / "pensions.html").is_file():
        raise RuntimeError("API source-code pages missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", ""))
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    if base_url and not base_url.startswith("/"):
        parser.error("--base-url must be a URL path starting with /")
    env = dict(os.environ, BASE_URL=base_url)
    stage = BUILD / "source"
    stage_book(stage, base_url)
    # Keep template downloads and MyST caches separate from copied source.
    myst_build = BUILD / "myst"
    myst_build.mkdir(exist_ok=True)
    (stage / "_build").symlink_to(myst_build, target_is_directory=True)
    run(shutil.which("jupyter-book"), "clean", "--execute", "--yes", cwd=stage)
    run(
        shutil.which("jupyter-book"),
        "build",
        "--html",
        "--execute",
        "--strict",
        "--ci",
        cwd=stage,
        env=env,
    )
    output = BUILD / "html"
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(myst_build / "html", output)
    # Build directly into the final tree so all relative asset/viewcode links
    # remain valid. -E prevents cached state from hiding source changes.
    run(
        sys.executable,
        "-m",
        "sphinx",
        "-E",
        "-b",
        "html",
        "-W",
        "--keep-going",
        "-c",
        DOCS / "api",
        BOOK / "content" / "api",
        output / "content" / "api",
        env=env,
    )
    write_redirects(stage, output)
    merge_inventories(output)
    prepare_static_navigation(output, base_url)
    verify_api(output)
    # GitHub Pages must serve Sphinx's _static and _modules directories.
    (output / ".nojekyll").touch()
    print(f"Documentation built and API verified: {output}")


if __name__ == "__main__":
    main()
