"""Sphinx configuration for the source-generated Python API reference.

Jupyter Book 2 builds the narrative book with MyST, which cannot run Sphinx
autodoc. Keep the API's existing autodoc declarations and render them with
Sphinx into the same published site instead.
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "OG-Core API"
author = "Jason DeBacker and Richard W. Evans"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_math_dollar",
]
root_doc = "public_api"
source_suffix = ".rst"
html_theme = "pydata_sphinx_theme"
html_title = "OG-Core API"
html_logo = "../OG-Core_logo.png"
html_static_path = []
templates_path = [str(Path(__file__).resolve().parent / "_templates")]
book_url = os.environ.get("BASE_URL", "").rstrip("/")
html_context = {
    "book_url": book_url + "/",
    "documentation_links": [
        ("Book", book_url + "/"),
        ("Theory", book_url + "/content/theory/intro/"),
        ("API reference", book_url + "/content/api/public_api.html"),
    ],
}
html_theme_options = {
    "logo": {
        "link": (os.environ.get("BASE_URL", "").rstrip("/") or "") + "/",
    },
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_center": ["documentation-nav"],
    "search_bar_text": "Search the Python API …",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/PSLmodels/OG-Core",
            "icon": "fa-brands fa-github",
        }
    ],
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
