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

# Order for model_variables.md.
SECTION_ORDER = [
    "Economic Aggregates",
    "Production Sector/Consumption Good Aggregates",
    "Government Revenues, Outlays, and Debt",
    "Prices",
    "Household Variables",
    "Model Scaling and Fit",
]


def make_vars(path):
    """Make string with all model variable information.

    Args:
        path: Path to variable metadata file.

    Returns:
        Single string with all parameter information.
    """
    with open(path) as pfile:
        json_text = pfile.read()
    params = ogcore.utils.json_to_dict(json_text)
    df = pd.DataFrame(params).T
    df.reset_index(names="varname", inplace=True)
    df.reset_index(names="var_order", inplace=True)
    # Add parameter text
    df["content"] = paramtextdf(df)
    # Organize sections
    section_order_index = dict(zip(SECTION_ORDER, range(len(SECTION_ORDER))))
    df["section_order"] = df.section.map(section_order_index)
    df.sort_values(["section_order", "var_order"], inplace=True)
    # Add section titles when they change.
    df["new_section"] = ~df.section.eq(df.section.shift())
    df["section_content"] = np.where(
        df.new_section, "## " + df.section + "\n\n", ""
    )
    # Concatenate section titles with content for each parameter.
    df.content = df.section_content + df.content

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
        return "####  `" + df.varname + "`  \n"

    def description(df):
        return "_Description:_ " + df.desc + "  \n"

    def tpi_dim(df):
        return "_TPI dimensions:_ " + df["TPI dimensions"] + "  \n"

    def ss_dim(df):
        return "_SS dimensions:_ " + df["SS dimensions"] + "  \n"

    text = title(df)
    text += description(df)
    text += tpi_dim(df)
    text += ss_dim(df)

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
    param_text = make_vars(os.path.join(OGCORE_PATH, "model_variables.json"))
    write_file(param_text, "variables")
    # Normal return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
