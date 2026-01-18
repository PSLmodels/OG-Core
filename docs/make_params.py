import numpy as np
import pandas as pd
from collections import OrderedDict
import ogcore
import os
import sys

CURDIR_PATH = os.path.abspath(os.path.dirname(__file__))
OGCORE_PATH = os.path.join(CURDIR_PATH, "..", "ogcore")
TEMPLATE_PATH = os.path.join(CURDIR_PATH, "templates")
OUTPUT_PATH = os.path.join(CURDIR_PATH, "./book/content/intro")

# Order for policy_params.md.
SECTION_1_ORDER = [
    "Household Parameters",
    "Firm Parameters",
    "Government Parameters",
    "Fiscal Policy Parameters",
    "Tax Functions",
    "Open Economy Parameters",
    "Economic Assumptions",
    "Demographic Parameters",
    "Model Solution Parameters",
    "Other Parameters",
]


def make_params(path):
    """Make string with all parameter information.

    Args:
        path: Path to parameter file.

    Returns:
        Single string with all parameter information.
    """
    with open(path) as pfile:
        json_text = pfile.read()
    params = ogcore.utils.json_to_dict(json_text)
    df = pd.DataFrame(params).transpose().drop("schema")
    # Add parameter text
    df["content"] = paramtextdf(df)
    # Organize sections
    df.section_1 = np.where(
        df.section_1 == "", "Other Parameters", df.section_1
    )
    section_1_order_index = dict(
        zip(SECTION_1_ORDER, range(len(SECTION_1_ORDER)))
    )
    df["section_1_order"] = df.section_1.map(section_1_order_index)
    df.sort_values(["section_1_order", "section_2"], inplace=True)
    # Add section titles when they change.
    df["new_section_1"] = ~df.section_1.eq(df.section_1.shift())
    df["new_section_2"] = ~df.section_2.eq(df.section_2.shift()) & (
        df.section_2 > ""
    )
    df["section_1_content"] = np.where(
        df.new_section_1, "## " + df.section_1 + "\n\n", ""
    )
    df["section_2_content"] = np.where(
        df.new_section_2, "### " + df.section_2 + "\n\n", ""
    )
    # Concatenate section titles with content for each parameter.
    df.content = df.section_1_content + df.section_2_content + df.content

    # Return a single string.
    return "\n\n".join(df.content)


def boolstr(b):
    """Return a bool value or Series as 'True'/'False' strings.

    Args:
        b: Bool value or pandas Series.

    Returns:
        If b is a value, returns a single 'True'/'False' value.
        If b is a Series, returns a Series of 'True'/'False' values.
    """
    if isinstance(b, pd.Series):
        return pd.Series(np.where(b, "True", "False"), index=b.index)
    if b:
        return "True"
    return "False"


def paramtextdf(df):
    """Don't include sections - do that later.

    Args:
        df: DataFrame representing parameters.
        ptype:
    """

    def title(df):
        return "####  `" + df.index + "`  \n"

    def long_name(df):
        return "_Long Name:_ " + df.title + "  \n"

    def description(df):
        return "_Description:_ " + df.description + "  \n"

    def notes(df):
        return np.where(df.notes == "", "", "_Notes:_ " + df.notes + "  \n")

    def value_type(df):
        return "_Value Type:_ " + df.type + "  \n"

    # def default_value_one(row):
    #     return '_Default Value:_ ' + str(row.value[0]['value']) + '  \n'

    # def default_value(df):
    #     return df.apply(default_value_one, axis=1)

    def valid_range_one(row):
        try:
            r = row.validators["range"]
            return (
                "_Valid Range:_"
                + " min = "
                + str(r["min"])
                + " and max = "
                + str(r["max"])
                + "  \n"
                + "_Out-of-Range Action:_ "
                + r.get("level", "error")
                + "  \n"
            )
        except KeyError:  # case of no validators, or also non-numeric ones?
            try:
                r = row.validators["choice"]
                return "_Valid Choices:_" + str(r["choices"]) + "  \n"
            except KeyError:
                return ""

    def valid_range(df):
        return df.apply(valid_range_one, axis=1)

    text = title(df)
    text += description(df)
    text += notes(df)
    text += value_type(df)
    text += valid_range(df)
    return text


def write_file(text, file):
    """Writes the concatenation of a template and calculated text to a file.

    Args:
        text: String with calculated documentation.
        file: Filename (without '.md' or a path). Must also match the filename
            of the template.

    Returns:
        Nothing. Result is written to file.
    """
    template = os.path.join(TEMPLATE_PATH, file + "_template.md")
    outfile = os.path.join(OUTPUT_PATH, file + ".md")
    with open(template, "r") as f:
        template_text = f.read()
    with open(outfile, "w") as f:
        f.write(template_text + "\n\n" + text)


def main():
    # Model parameters.
    param_text = make_params(
        os.path.join(OGCORE_PATH, "default_parameters.json")
    )
    write_file(param_text, "parameters")
    # Normal return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
