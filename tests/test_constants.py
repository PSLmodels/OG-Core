"""
Tests of constants.py module and the label metadata loaded from
model_variables.json.
"""

from ogcore import constants


def test_labels_have_no_control_characters():
    r"""
    Guard against LaTeX label strings in model_variables.json being
    written with a single backslash (e.g. "$\theta$"), which JSON parses
    into a control character (a tab, in the case of "\t") and which then
    fails to render in plots.  Backslashes must be escaped ("$\\theta$").
    """
    control_chars = set("\t\r\n\x08\x0c\x0b")
    for label_map in (constants.VAR_LABELS, constants.ToGDP_LABELS):
        for key, label in label_map.items():
            assert isinstance(label, str)
            bad = control_chars.intersection(label)
            assert not bad, (
                f"Label for {key!r} contains control character(s) "
                f"{[c.encode('unicode_escape').decode() for c in bad]}: "
                f"{label!r}. Escape backslashes in model_variables.json."
            )
