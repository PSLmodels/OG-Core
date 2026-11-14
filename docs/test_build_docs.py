"""Regression checks for documentation content and source-derived APIs.

Run with: uv run --extra docs --extra dev pytest docs/test_build_docs.py
"""

import re
import json
import subprocess
import sys
from io import BytesIO
import zlib

from sphinx.util.inventory import InventoryFile
import yaml

import build_docs


def test_static_api_navigation_bypasses_the_book_router():
    """API links must work after SPA navigation and under a hosting prefix."""
    script = json.dumps(str(build_docs.DOCS / "book" / "navigation.js"))
    code = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(SCRIPT_PATH, 'utf8');
for (const base of ['', '/OG-Core']) {
  let handler;
  let assigned;
  const document = {
    currentScript: {src: `https://example.org${base}/ogcore-navigation.js`},
    baseURI: `https://example.org${base}/content/theory/government/`,
    addEventListener(name, callback, capture) {
      assert.equal(name, 'click');
      assert.equal(capture, true);
      handler = callback;
    },
  };
  const window = {location: {
    origin: 'https://example.org',
    assign(url) { assigned = url; },
  }};
  vm.runInNewContext(source, {document, window, URL});
  function click(path, modifiers = {}) {
    assigned = undefined;
    const link = {
      href: new URL(path, document.baseURI).href,
      hasAttribute: () => false,
    };
    let prevented = false;
    let stopped = false;
    handler({button: 0, target: {closest: () => link},
      preventDefault() { prevented = true; },
      stopPropagation() { stopped = true; }, ...modifiers});
    assert.equal(prevented, stopped);
    return assigned;
  }
  const pension = `${base}/content/api/pensions.html` +
    '#ogcore.pensions.NDC_amount';
  assert.equal(click(pension), 'https://example.org' + pension);
  const search = `${base}/content/api/search.html?q=NDC`;
  assert.equal(click(search), 'https://example.org' + search);
  assert.equal(click(`${base}/index.html`), `https://example.org${base}/index.html`);
  assert.equal(click(`${base}/content/theory/government`), undefined);
  assert.equal(click(pension, {ctrlKey: true}), undefined);
  assert.equal(
    click('https://other.example/content/api/pensions.html'), undefined);
}
""".replace("SCRIPT_PATH", script)
    subprocess.run(["node", "-e", code], check=True)


def test_api_links_support_local_and_prefixed_hosting():
    entries = [
        {"file": "intro.md"},
        {"children": [{"url": "/content/api/pensions.html"}]},
        {"url": "https://github.com/PSLmodels/OG-Core"},
    ]
    build_docs.prefix_api_urls(entries, "/OG-Core")
    assert entries[1]["children"][0]["url"] == (
        "/OG-Core/content/api/pensions.html"
    )
    assert entries[0] == {"file": "intro.md"}
    assert entries[2]["url"].startswith("https://")


def test_staged_navigation_does_not_double_prefix_header_links(
    tmp_path, monkeypatch
):
    book = tmp_path / "book"
    (book / "content").mkdir(parents=True)
    (book / "ogcore.css").write_text("/* theme */\n")
    config = {
        "version": 1,
        "project": {"toc": [{"url": "/content/api/pensions.html"}]},
        "site": {"nav": [{"title": "Book", "url": "/index.html"}]},
    }
    (book / "myst.yml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(build_docs, "BOOK", book)
    monkeypatch.setattr(
        build_docs, "generate_reference_pages", lambda stage: None
    )
    monkeypatch.setattr(
        build_docs, "generate_bibliographies", lambda stage: None
    )
    stage = tmp_path / "stage"
    build_docs.stage_book(stage, "/OG-Core")
    result = yaml.safe_load((stage / "myst.yml").read_text())
    assert result["project"]["toc"][0]["url"] == (
        "/OG-Core/content/api/pensions.html"
    )
    assert result["site"]["nav"][0]["url"] == "/index.html"


def test_bibliographies_preserve_uncited_entries_and_separate_lists(
    tmp_path, monkeypatch
):
    book = tmp_path / "book"
    book.mkdir()
    # Uppercase fields and empty DOI values occur in the actual bibliographies.
    template = (
        "@article{{{key}, AUTHOR={{A. Author}}, TITLE={{{title}}}, "
        "JOURNAL={{Journal}}, YEAR={{2026}}, DOI={{}}, "
        "URL={{https://example.org/a/very/long/report.pdf}}}}"
    )
    (book / "OGCore_references.bib").write_text(
        template.format(key="theory", title="A theory reference")
    )
    (book / "citations.bib").write_text(
        template.format(key="uncited", title="An uncited use case")
    )
    monkeypatch.setattr(build_docs, "BOOK", book)
    stage = tmp_path / "stage"
    stage.mkdir()
    build_docs.generate_bibliographies(stage)
    references = (stage / "_generated" / "references.md").read_text()
    citations = (stage / "_generated" / "citations.md").read_text()
    assert "A theory reference" in references
    assert "An uncited use case" in citations
    assert "An uncited use case" not in references
    assert "A theory reference" not in citations
    assert "[C-" in citations
    assert "https://doi.org/" not in references + citations
    assert "[PDF](https://example.org/a/very/long/report.pdf)" in citations
    assert "(usecase-uncited)=" in citations
    assert "(reference-theory)=" in references
    assert ":class: ogcore-bibliography" in references
    assert "\n:   " in references
    assert "- **[" not in references


def test_root_inventory_retains_narrative_and_python_objects(tmp_path):
    """A normal Sphinx client must resolve both domains from the book root."""
    api = tmp_path / "content" / "api"
    api.mkdir(parents=True)
    header = (
        b"# Sphinx inventory version 2\n# Project: OG-Core\n# Version: \n"
        b"# The remainder of this file is compressed using zlib.\n"
    )
    book_entries = (
        b"government std:label -1 content/theory/government#gov Government\n"
    )
    (tmp_path / "objects.inv").write_bytes(
        header + zlib.compress(book_entries)
    )
    api_entries = (
        b"ogcore.pensions.NDC_amount py:function 1 pensions.html#$ -\n"
        b"ogcore.pensions py:module 0 pensions.html#module-$ -\n"
    )
    original_api = header + zlib.compress(api_entries)
    (api / "objects.inv").write_bytes(original_api)
    build_docs.merge_inventories(tmp_path)
    result = (tmp_path / "objects.inv").read_bytes()
    entries = zlib.decompress(result.split(b"\n", 4)[4])
    assert book_entries in entries
    # Test the standard consumer, including its $ URI expansion.
    inventory = InventoryFile.load(
        BytesIO(result), "https://example.org/OG-Core/", None
    )
    assert inventory["py:function"]["ogcore.pensions.NDC_amount"].uri == (
        "https://example.org/OG-Core/content/api/pensions.html"
        "#ogcore.pensions.NDC_amount"
    )
    assert "government" in inventory["std:label"]
    assert (api / "objects.inv").read_bytes() == original_api
    build_docs.merge_inventories(tmp_path)
    assert (tmp_path / "objects.inv").read_bytes() == result


def test_legacy_redirects_resolve_with_either_hosting_prefix(tmp_path):
    stage = tmp_path / "source"
    output = tmp_path / "html"
    for name in (
        "intro/intro.md",
        "theory/market_clearing.md",
        "OGCore_references.md",
    ):
        path = stage / "content" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Title\n")
    build_docs.write_redirects(stage, output)
    redirect = (output / "content/theory/market_clearing.html").read_text()
    assert "market-clearing/" in redirect
    assert "location.search" in redirect
    assert "location.hash.toLowerCase()" in redirect
    assert "#equation-" in redirect
    assert ".replace(/_/g, '-')" in redirect
    assert "OG-Core" not in redirect
    assert "../../" in (output / "content/intro/intro.html").read_text()
    assert (
        "ogcore-references/"
        in (output / "content/OGCore_references.html").read_text()
    )
    legacy = (output / "content/OGCore_references.html").read_text()
    assert '"#id2": "#reference-armington-1969"' in legacy


def test_autodoc_rebuilds_changed_source_signatures_and_docstrings(tmp_path):
    """Exercise our real Sphinx config against a changing Python module."""
    module = tmp_path / "fixture_api.py"
    source = tmp_path / "rst"
    conf = tmp_path / "config"
    output = tmp_path / "html"
    source.mkdir()
    conf.mkdir()
    api_conf = str(build_docs.DOCS / "api/conf.py")
    (conf / "conf.py").write_text(
        "import runpy, sys\n"
        f"globals().update(runpy.run_path({api_conf!r}))\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "root_doc = 'index'\nhtml_logo = None\n"
    )
    (source / "index.rst").write_text(
        "Fixture API\n===========\n\n"
        ".. automodule:: fixture_api\n   :members: benefit\n\n"
        ".. autoclass:: fixture_api.Account\n   :members: credit\n"
    )
    for signature, docstring in (
        ("wage, rate", "Original benefit documentation."),
        (
            "wage, rate, adjustment=0",
            "Revised source-derived benefit documentation.",
        ),
    ):
        module.write_text(
            f"def benefit({signature}):\n"
            f'    """{docstring}\n\n'
            "    Args:\n        wage (float): Pensionable earnings.\n"
            "        rate (float): Contribution rate.\n\n"
            "    Returns:\n        float: The pension benefit.\n"
            '    """\n    return wage * rate\n\n'
            "class Account:\n"
            '    """A notional account."""\n'
            "    def credit(self, amount):\n"
            '        """Credit a contribution."""\n'
            "        return amount\n"
        )
        build_docs.run(
            sys.executable,
            "-m",
            "sphinx",
            "-E",
            "-b",
            "html",
            "-W",
            "--keep-going",
            "-c",
            conf,
            source,
            output,
        )
        html = (output / "index.html").read_text()
        assert docstring in html
        assert 'id="fixture_api.Account.credit"' in html
        assert "_modules/fixture_api.html#benefit" in html
        assert "Pensionable earnings" in html
        search = (output / "search.html").read_text()
        assert "This search covers the Python API." in search
        assert 'aria-label="Search the Python API' in search
        if "adjustment" in signature:
            assert "Original benefit documentation" not in html
            assert re.search(r'class="sig-param".*?adjustment', html)
            assert (
                "Revised source-derived benefit documentation"
                in (output / "_modules" / "fixture_api.html").read_text()
            )
