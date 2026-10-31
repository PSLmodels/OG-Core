# Building the documentation

Install [uv](https://docs.astral.sh/uv/) and Node.js 22 or newer, then run from
the repository root:

```sh
make documentation
```

The command installs the `docs` dependencies, executes the book's code cells,
and writes the complete static site to `docs/book/_build/html`. Both the
Jupyter Book and Sphinx stages use strict checking. Template downloads require
network access on the first build. Generated inputs and caches live under
`docs/book/_build`; authored Markdown is not overwritten.

To preview the site locally:

```sh
uv run python -m http.server 8000 --directory docs/book/_build/html
```

Open <http://localhost:8000/>. Use an HTTP server rather than opening the HTML
files directly, because the MyST site loads JavaScript and JSON resources.

For hosting under a path, set that path at build time:

```sh
BASE_URL=/OG-Core make documentation
```

The deployment workflow sets this automatically for GitHub Pages. The output
folder remains `docs/book/_build/html`; publish its entire contents, including
the `content/api` directory. The build also writes redirects for the previous
narrative `.html` URLs and retains existing API URLs and Python object anchors.

## Automatic Python API reference

Jupyter Book 2 uses the MyST engine, which does not support Sphinx autodoc.
We therefore keep the existing reStructuredText API declarations and build
them with Sphinx as part of the same `make documentation` command:

- `docs/book/myst.yml` configures the narrative book and links to the API.
- `docs/book/content/api/*.rst` selects the documented modules, classes,
  functions, and methods using `automodule`, `autoclass`, and `:members:`.
- `docs/api/conf.py` enables autodoc, Napoleon, MathJax, and viewcode.
- `docs/build_docs.py` assembles the MyST and Sphinx output into one site.

Every build imports the local `ogcore` source and regenerates signatures and
docstrings. There are no manually copied function descriptions or signatures.
The API retains source-code links, equations, class methods, its Python object
inventory, and its own search. The API uses a Sphinx theme and search; the
narrative book uses the MyST theme and search. Both sections provide Book,
Theory, and API reference navigation. The book also links directly to API
search, and the API search identifies its scope and links back to the book.
The small `docs/book/navigation.js` handler makes API links load a new document
instead of being handled as MyST chapter routes, including after navigation
within the book.

The root `objects.inv` combines the book's narrative references with Python
objects from the API, with their URLs adjusted to `content/api/`. Existing
intersphinx configurations can keep using the book root. The API-only inventory
also remains available at `content/api/objects.inv`.

When changing an existing function or docstring, rebuild the documentation.
To expose an additional function, add its name to the relevant `:members:`
list. For a new module, add an `.rst` page, add it to the `public_api.rst`
toctree, and add its API URL to `myst.yml`. The build checks that every API
page exists and that representative functions, methods, and source links are
present, so a book-only build cannot silently drop the API.

## Generated tables and bibliographies

Parameter and variable tables are generated from `ogcore/default_parameters.json`
and `ogcore/model_variables.json` using `make_params.py` and `make_vars.py`.
The build stages their output rather than editing checked-in Markdown files.

The theory references and the list of OG-Core use cases are generated separately
from `OGCore_references.bib` and `citations.bib`, preserving uncited use cases
and their `C-` labels. Inline citations and per-page references are rendered by
MyST. Edit the BibTeX inputs rather than generated bibliography text.

The bibliographies use stable anchors derived from BibTeX keys, aligned labels,
and compact Source/PDF/DOI links. `docs/bibliography_anchors.json` records the
published Jupyter Book 1 citation IDs; the legacy-page redirects map those IDs
to the stable anchors. Keep that mapping when adding or reordering references.

Equations, figures, and tables are numbered continuously across the book.
`docs/book/ogcore.css` styles the reference lists, stacks their labels on narrow
screens, and hides the repeated metadata/download toolbar. Author metadata
is retained, with author blocks hidden through the MyST theme configuration.

Documentation-build regression tests can be run with:

```sh
uv run --extra docs --extra dev pytest docs/test_build_docs.py
```

## GitHub social preview image

The images in this directory also include
[`OG-Core_logo_gitfig.png`](OG-Core_logo_gitfig.png), used for the GitHub
social preview. GitHub recommends 1280×640 pixels. The original
[`OG-Core_logo_long.png`](OG-Core_logo_long.png) is 500×346 pixels.

To recreate the social preview in Adobe Photoshop:

1. Open `OG-Core_logo_long.png` with **File → Open**.
2. Select **Image → Canvas Size** and increase the width to 692 pixels,
   retaining the height of 346 pixels. This produces the required 2:1 ratio.
3. Select **Image → Image Size** and resize to 1280×640 pixels.
4. Save the result as `OG-Core_logo_gitfig.png`.
5. Upload it under **Social preview** in the repository’s
   [GitHub settings](https://github.com/PSLmodels/OG-Core/settings).
