# OG-Core

| | |
| --- | --- |
| Org | [![PSL cataloged](https://img.shields.io/badge/PSL-cataloged-a0a0a0.svg)](https://www.PSLmodels.org) [![OS License: CCO-1.0](https://img.shields.io/badge/OS%20License-CCO%201.0-yellow)](https://github.com/PSLmodels/OG-Core/blob/master/LICENSE) [![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://pslmodels.github.io/OG-Core/) |
| Package | [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/) [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3108/) [![Python 3.11](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3118/) [![PyPI Latest Release](https://img.shields.io/pypi/v/ogcore.svg)](https://pypi.org/project/ogcore/) [![PyPI Downloads](https://img.shields.io/pypi/dm/ogcore.svg?label=PyPI%20downloads)](https://pypi.org/project/ogcore/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
| Testing | ![example event parameter](https://github.com/PSLmodels/OG-Core/actions/workflows/build_and_test.yml/badge.svg?branch=master) ![example event parameter](https://github.com/PSLmodels/OG-Core/actions/workflows/deploy_docs.yml/badge.svg?branch=master) ![example event parameter](https://github.com/PSLmodels/OG-Core/actions/workflows/check_black.yml/badge.svg?branch=master) [![Codecov](https://codecov.io/gh/PSLmodels/OG-Core/branch/master/graph/badge.svg)](https://codecov.io/gh/PSLmodels/OG-Core) |


OG-Core is an overlapping-generations (OG) model core theory, logic, and solution method algorithms that allow for dynamic general equilibrium analysis of fiscal policy. OG-Core provides a general framework and is a dependency of several country-specific OG models, such as [OG-USA](https://github.com/PSLmodels/OG-USA) and [OG-UK](https://github.com/PSLmodels/OG-UK). The model output includes changes in macroeconomic aggregates (GDP, investment, consumption), wages, interest rates, and the stream of tax revenues over time. Regularly updated documentation of the model theory--its output, and solution method--and the Python API is available [here](https://pslmodels.github.io/OG-Core).


## Disclaimer

The model is constantly under development, and model components could change significantly. The package will have released versions, which will be checked against existing code prior to release. Stay tuned for an upcoming release!


## Using/contributing to OG-Core

There are two primary methods for installing and running OG-Core on your computer locally. The first and simplest method is to download the most recent `ogcore` Python package from the Python Package Index ([PyPI.org]()). A second option is to fork and clone the most recent version of OG-Core from its GitHub repository and create the conda environment for the `ogcore` package. We detail both of these methods below.


### Installing and Running OG-Core from Python Package Index (PyPI.org)

* Open your terminal (or Conda command prompt), and make sure you have the most recent version of `pip` (the Python Index Package manager) by typing on a Unix/macOS machine `python3 -m pip install --upgrade pip` or on a Windows machine `py -m pip install --upgrade pip`.
* Install the [`ogcore`](https://pypi.org/project/ogcore/) package from the Python Package Index by typing `pip install ogcore`.
* Navigate to a folder `./YourFolderName/` where you want to save scripts to run OG-Core and output from the simulations in those scripts.
* Save the python script [`run_ogcore_example.py`](https://github.com/PSLmodels/OG-Core/blob/master/run_examples/run_ogcore_example.py) from the OG-Core GitHub repository in the folder where you are working on your local machine `./YourFolderName/run_ogcore_example.py`.
* Run the model with an example reform from terminal/command prompt by typing `python run_ogcore_example.py`
* You can adjust the `run_ogcore_example.py` script by modifying model parameters specified in the `og_spec` dictionary.
* Model outputs will be saved in the following files:
  * `./run_example_plots`
    * This folder will contain a number of plots generated from OG-Core to help you visualize the output from your run
  * `./ogcore_example_output.csv`
    * This is a summary of the percentage changes in macro variables over the first ten years and in the steady-state.
  * `./OUTPUT_BASELINE/model_params.pkl`
    * Model parameters used in the baseline run
    * See [`execute.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/execute.py) in the OG-Core repository for items in the dictionary object in this pickle file
  * `./OUTPUT_BASELINE/SS/SS_vars.pkl`
    * Outputs from the model steady state solution under the baseline policy
    * See [`SS.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/SS.py) in the OG-Core repository for what is in the dictionary object in this pickle file
  * `./OUTPUT_BASELINE/TPI/TPI_vars.pkl`
    * Outputs from the model timepath solution under the baseline policy
    * See [`TPI.py`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/TPI.py) in the OG-Core repository for what is in the dictionary object in this pickle file
  * An analogous set of files in the `./OUTPUT_REFORM` directory, which represent objects from the simulation of the reform policy

Note that, depending on your machine, a full model run (solving for the full time path equilibrium for the baseline and reform policies) can take more than two hours of compute time.

If you run into errors running the example script, please open a new issue in the OG-Core repo with a description of the issue and any relevant tracebacks you receive.

The CSV output file `./ogcore_example_output.csv` can be compared to the [`./run_examples/expected_ogcore_example_output.csv`](https://github.com/PSLmodels/OG-Core/blob/master/run_examples/expected_ogcore_example_output.csv) file in the OG-Core repository to confirm that you are generating the expected output. The easiest way to do this is to copy the [`example-diffs`](https://github.com/PSLmodels/OG-Core/blob/master/run_examples/example-diffs) and [`example-diffs.bat`](https://github.com/PSLmodels/OG-Core/blob/master/run_examples/example-diffs.bat) files from the OG-Core repository and use the `sh example-diffs` command (or `example-diffs` on Windows) from the `run_examples` directory. If you run into errors running the example script, please open a new issue in the OG-Core repo with a description of the issue and any relevant tracebacks you receive.


### Installing and Running OG-Core from GitHub repository

* Install the [Anaconda distribution](https://www.anaconda.com/distribution/) of Python
* Clone this repository to a directory on your computer
* From the terminal (or Conda command prompt), navigate to the directory to which you cloned this repository and run `conda env create -f environment.yml`
* Then, `conda activate ogcore-dev`
* Then install by `pip install -e .`
* Navigate to `./run_examples`
* Run the model with an example reform from terminal/command prompt by typing `python run_ogcore_example.py`
* You can adjust the `./run_examples/run_ogcore_example.py` script by modifying model parameters specified in the `og_spec` dictionary.
* Model outputs will be saved in the following files:
  * `./run_examples/run_example_plots`
    * This folder will contain a number of plots generated from OG-Core to help you visualize the output from your run
  * `./run_examples/ogcore_example_output.csv`
    * This is a summary of the percentage changes in macro variables over the first ten years and in the steady-state.
  * `./run_examples/OUTPUT_BASELINE/model_params.pkl`
    * Model parameters used in the baseline run
    * See `execute.py` for items in the dictionary object in this pickle file
  * `./run_examples/OUTPUT_BASELINE/SS/SS_vars.pkl`
    * Outputs from the model steady state solution under the baseline policy
    * See `SS.py` for what is in the dictionary object in this pickle file
  * `./run_examples/OUTPUT_BASELINE/TPI/TPI_vars.pkl`
    * Outputs from the model timepath solution under the baseline policy
    * See `TPI.py` for what is in the dictionary object in this pickle file
  * An analogous set of files in the `./run_examples/OUTPUT_REFORM` directory, which represent objects from the simulation of the reform policy

Note that, depending on your machine, a full model run (solving for the full time path equilibrium for the baseline and reform policies) can take more than two hours of compute time.

If you run into errors running the example script, please open a new issue in the OG-Core repo with a description of the issue and any relevant tracebacks you receive.

The CSV output file `./run_examples/ogcore_example_output.csv` can be compared to the `./run_examples/expected_ogcore_example_output.csv` file that is checked into the repository to confirm that you are generating the expected output. The easiest way to do this is to use the `sh example-diffs` command (or `example-diffs` on Windows) from the `run_examples` directory. If you run into errors running the example script, please open a new issue in the OG-Core repo with a description of the issue and any relevant tracebacks you receive.


## Core Maintainers

The core maintainers of the OG-Core repository are:

* [Jason DeBacker](https://www.jasondebacker.com/) (GitHub handle: [jdebacker](https://github.com/jdebacker)), Associate Professor, Department of Economics, Darla Moore School of Business, University of South Carolina; President, PSL Foundation; Vice President of Research and Co-founder, Open Research Group, Inc.
* [Richard W. Evans](https://sites.google.com/site/rickecon/) (GitHub handle: [rickecon](https://github.com/rickecon)), Advisory Board Visiting Fellow, Center for Public Finance, Baker Institute for Public Policy at Rice University; President, Open Research Group, Inc.; Director, Open Source Economics Laboratory


## Citing OG-Core

OG-Core (Version #.#.#)[Source code], https://github.com/PSLmodels/OG-Core
