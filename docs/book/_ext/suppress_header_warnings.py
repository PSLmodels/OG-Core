"""Suppress MyST ``myst.header`` warnings for the auto-generated pages.

``docs/book/content/intro/parameters.md`` and ``variables.md`` are written by
``docs/make_params.py`` and ``docs/make_vars.py``.  Both deliberately place the
parameter/variable headings two levels below their section heading (H2 to H4),
which MyST reports as "Non-consecutive header level increase".  MyST reads
Sphinx's global ``suppress_warnings`` setting, so silencing ``myst.header``
there would silence it for the whole book.  This extension instead drops those
records only when they come from the two generated files.
"""

import logging
import os

TARGETS = (
    "content/intro/parameters.md",
    "content/intro/variables.md",
)
SUBTYPE = "[myst.header]"


class HeaderWarningFilter(logging.Filter):
    """Drop ``myst.header`` warnings raised by the generated pages."""

    def filter(self, record):
        if SUBTYPE not in str(record.msg):
            return True
        location = str(getattr(record, "location", "") or "")
        location = location.replace(os.sep, "/")
        return not any(target in location for target in TARGETS)


def _add_filter(app):
    log_filter = HeaderWarningFilter()
    for handler in logging.getLogger("sphinx").handlers:
        # Insert ahead of Sphinx's own filters: WarningSuppressor increments
        # the build's warning count and WarningIsErrorFilter raises under
        # ``-W``, and both run before any filter added with addFilter().
        handler.filters.insert(0, log_filter)


def setup(app):
    app.connect("builder-inited", _add_filter)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
