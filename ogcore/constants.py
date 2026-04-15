import os
import ogcore.utils as utils

# Read in json file
cur_dir = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(cur_dir, "model_variables.json")
with open(file) as f:
    json_text = f.read()
var_metadata = utils.json_to_dict(json_text)

SHOW_RUNTIME = False  # Flag to display RuntimeWarnings when run model

REFORM_DIR = "OUTPUT_REFORM"
BASELINE_DIR = "OUTPUT_BASELINE"

# Default year for model runs
DEFAULT_START_YEAR = 2025

VAR_LABELS = dict([(k, v["label"]) for k, v in var_metadata.items()])

ToGDP_LABELS = dict([(k, v["toGDP_label"]) for k, v in var_metadata.items()])

GROUP_LABELS = {
    7: {
        0: "0-25%",
        1: "25-50%",
        2: "50-70%",
        3: "70-80%",
        4: "80-90%",
        5: "90-99%",
        6: "Top 1%",
    },
    9: {
        0: "0-25%",
        1: "25-50%",
        2: "50-70%",
        3: "70-80%",
        4: "80-90%",
        5: "90-99%",
        6: "99-99.5%",
        7: "99.5-99.9%",
        8: "Top 0.1%",
    },
    10: {
        0: "0-25%",
        1: "25-50%",
        2: "50-70%",
        3: "70-80%",
        4: "80-90%",
        5: "90-99%",
        6: "99-99.5%",
        7: "99.5-99.9%",
        8: "99.9-99.99%",
        9: "Top 0.01%",
    },
}

# List of deviation factors for initial guesses of r and TR used in
# SS.run_SS for a more robust SS solution
DEV_FACTOR_LIST = [
    [1.00, 1.0],
    [0.95, 1.0],
    [1.05, 1.0],
    [0.90, 1.0],
    [1.10, 1.0],
    [0.85, 1.0],
    [1.15, 1.0],
    [0.80, 1.0],
    [1.20, 1.0],
    [0.75, 1.0],
    [1.25, 1.0],
    [0.70, 1.0],
    [1.30, 1.0],
    [1.00, 0.2],
    [0.95, 0.2],
    [1.05, 0.2],
    [0.90, 0.2],
    [1.10, 0.2],
    [0.85, 0.2],
    [1.15, 0.2],
    [0.80, 0.2],
    [1.20, 0.2],
    [0.75, 0.2],
    [1.25, 0.2],
    [0.70, 0.2],
    [1.30, 0.2],
    [1.00, 0.6],
    [0.95, 0.6],
    [1.05, 0.6],
    [0.90, 0.6],
    [1.10, 0.6],
    [0.85, 0.6],
    [1.15, 0.6],
    [0.80, 0.6],
    [1.20, 0.6],
    [0.75, 0.6],
    [1.25, 0.6],
    [0.70, 0.6],
    [1.30, 0.6],
]
