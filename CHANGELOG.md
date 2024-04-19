# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.11.6] - 2024-04-17 14:00:00

### Added

- Scatters parameters once in `TPI.py`
- Removes Python 3.9 tests from `build_and_test.yml`


## [0.11.5] - 2024-04-11 12:00:00

### Added

- Adds a list of file change event triggers to `build_and_test.yml` so that those tests only run when one of those files is changed.
- Updates the codecov GH Action to version 4 and adds a secret token.
- Adds a list of file change event triggers to `deploy_docs.yml` and `docs_check.yml`, and limits `docs_check.yml` to only run on pull requests.


## [0.11.4] - 2024-04-03 22:00:00

### Added

- Add a function to `utils.py` to shift lifetime profiles of parameters
- Add a function to `utils.py` to compute percentage changes in non-stationary variables
- Add more functionality to `parameters_plots.py`, allowing the user to plot parameters from multiple parameters objects together


## [0.11.3] - 2024-03-08 12:00:00

### Added

- Allow for `demographics.py` to save downloaded data directly.
- Retrieve population data from the UN World Population Prospects database through CSV rather than JSON to avoid rate limit errors.


## [0.11.2] - 2024-02-17 12:00:00

### Added

- Updates `demographics.py` with more functionality across the time path.
- Allow the user to have the population distribution from the initial period forward inferred from given fertility, mortality, and immigration rates (functionality to infer immigration from a given evolution of the population is retained, the user specifies what they want to do via arguments to the relevant function calls).
- Extends all series returned from the get_pop_objs() function over the full transition path of T+S periods (except those that apply only to a single period).
- Addresses Issues #900 and #899


## [0.11.1] - 2024-02-12 15:00:00

### Added

- Updated `setup.py` Python version requirement to be `python_requires=">=3.7.7, <3.12"`


## [0.11.0] - 2024-02-06 15:00:00

### Added

- Allow `chi_n` parameter to vary over the time path (PR #897)
- Create a demographics module in OG-Core (PR #896)
- Create a time varying ability matrix (PR # 895)
- Simplify the extrapolation of arrays over the time path (PR #891)
- Update the copyright year of documentation to 2024


## [0.10.10] - 2023-10-25 17:00:00

### Added

- Remove `surve_rate` parameter (PR #886)
- Updates to `plot_2D_taxfunc` (PR #881)


## [0.10.9] - 2023-09-08 12:00:00

### Added

- PR #880 standardize the time path output length
- PR #878 fix tax function indexing, dimensions, and plotting. This PR also enables Python 3.11.
- PR #875 remove unused dependency


## [0.10.8] - 2023-04-22 12:00:00

### Added

- Adds a 2D monotonic smoothing spline tax function estimation to `txfunc.py`
- Changes the tax function parameters objects from NumPy arrays to lists in order to accomodate the nonparametric functions that get passed with the `mono` and `mono2D` options


## [0.10.7] - 2023-03-31 12:00:00

### Added

- Uses lists to pass and access effective tax rate objects `etr_params` and marginal tax rate objects `mtrx_params` and `mtry_params`


## [0.10.6] - 2023-02-15 12:00:00

### Added

- Uses 300 dpi when saving plots to disk
- Better labels of the `plot_industry_aggregates` plots


## [0.10.5] - 2023-02-14 12:00:00

### Added

- Fix to `SS.py` to use baseline solution on reform run if dimensions match
- Fix to `test_basic.py` dimensions for `r_gov_scale`


## [0.10.4] - 2023-02-06 12:00:00

### Added

- New calibration section to documentation (PR #850)
- Allow government risk premia to vary across time path for parameters `r_gov_shift` and `r_gov_scale` (PR #852)


## [0.10.3] - 2023-01-21 12:00:00

### Added

- Bug fixes for new tax function parameter estimation


## [0.10.2] - 2023-01-12 12:00:00

### Added

- Adds a new minimum value to the `r_gov_shift` parameter of -0.3


## [0.10.1] - 2023-01-05 12:00:00

### Added

- Removes hard coded year label in parameter_plots.plot_population_path() (PR #825)
- Fixes documentation (PR # 827)
- Adds "mono" specification to default_parameter.json and test_parameters.py (PR #830)
- Restricts Python version to be < 3.11 and removes the mkl dependency in environment.yml and setup.py (PR #833 and #840)
- Updates CI testing to include Mac, Windows, and Linux operating systems and Python 3.9 and 3.10, and solves some CI test issues (PR #836)
- Increases the maximum values for r_gov_shift and r_gov_scale in default_parameters.json (PR #838)
- Removes the mkl dependency from environment.yml and setup.py (PR #840)


## [0.10.0] - 2022-09-27 12:00:00

### Added

- Adds matrix of tax noncompliance parameters to households (PR #816)
- Incorporate input/output matrix mapping production goods to consumption goods (PR #818)
- Adds a new monotonic tax function estimation method to txfunc.py (PR #819)


## [0.9.2] - 2022-08-21 12:00:00

### Added

- Updates the form of the investments tax credit to be on a proxy for investment (depreciated capital) in order to satisfy theoretical requirements of static firms in each industry.
- Update the documentation
- Update the requirement for the m_wealth parameter in the wealth tax function to be strictly greater than zero.


## [0.9.1] - 2022-07-22 12:00:00

### Added

- Adds an investment tax credit parameter to the model
- Adds a boolean that allows the option to compute a reform using a stored baseline solution rather than recomputing the baseline


## [0.9.0] - 2022-06-30 12:00:00

### Added

- Adds multiple production industries to the model


## [0.8.2] - 2022-06-01 12:00:00

### Added

- Formatting of source code with black
- Last tag before extension of model to include multiple industries


## [0.8.1] - 2022-04-01 12:00:00

### Added

- Updates PyPI.org packaging setup and includes auto-publishing GH Action (PRs #790, #795, and #797)
- Cleans up documentation issues and old erroneous references to ogusa package (PR #797)


## [0.8.0] - 2022-02-18 12:00:00

### Added

- Adds a public capital good (i.e., infrastructure) used to produce private goods and services
- Adds a financial intermediary that links domestic and foreign savings to investment
- Improves multiprocessing with Dask
- Updates documentation
- Moves testing files outside of the ogcore package source files directory
- Tests functionality with Python 3.10.


## [0.7.0] - 2021-08-30 12:00:00

### Added

- This is the first release of the OG-Core model (formerly the OG-USA model)


## Previous versions

### Summary

- Version [0.7.0] on August 30, 2021 was the first time that the OG-USA repository was detached from all of the core model logic, which was named OG-Core. Before this version, OG-USA was part of what is now the [`OG-Core`](https://github.com/PSLmodels/OG-Core) repository. In the next version of OG-USA, we adjusted the version numbering to begin with 0.1.0. This initial version of 0.7.0, was sequential from what OG-USA used to be when the OG-Core project was called OG-USA.
- Any earlier versions of OG-USA can be found in the [`OG-Core`](https://github.com/PSLmodels/OG-Core) repository [release history](https://github.com/PSLmodels/OG-Core/releases) from [v.0.6.4](https://github.com/PSLmodels/OG-Core/releases/tag/v0.6.4) (Jul. 20, 2021) or earlier.


[0.11.6]: https://github.com/PSLmodels/OG-Core/compare/v0.11.5...v0.11.6
[0.11.5]: https://github.com/PSLmodels/OG-Core/compare/v0.11.4...v0.11.5
[0.11.4]: https://github.com/PSLmodels/OG-Core/compare/v0.11.3...v0.11.4
[0.11.3]: https://github.com/PSLmodels/OG-Core/compare/v0.11.2...v0.11.3
[0.11.2]: https://github.com/PSLmodels/OG-Core/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/PSLmodels/OG-Core/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/PSLmodels/OG-Core/compare/v0.10.10...v0.11.0
[0.10.10]: https://github.com/PSLmodels/OG-Core/compare/v0.10.9...v0.10.10
[0.10.9]: https://github.com/PSLmodels/OG-Core/compare/v0.10.8...v0.10.9
[0.10.8]: https://github.com/PSLmodels/OG-Core/compare/v0.10.7...v0.10.8
[0.10.7]: https://github.com/PSLmodels/OG-Core/compare/v0.10.6...v0.10.7
[0.10.6]: https://github.com/PSLmodels/OG-Core/compare/v0.10.5...v0.10.6
[0.10.5]: https://github.com/PSLmodels/OG-Core/compare/v0.10.4...v0.10.5
[0.10.4]: https://github.com/PSLmodels/OG-Core/compare/v0.10.3...v0.10.4
[0.10.3]: https://github.com/PSLmodels/OG-Core/compare/v0.10.2...v0.10.3
[0.10.2]: https://github.com/PSLmodels/OG-Core/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/PSLmodels/OG-Core/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/PSLmodels/OG-Core/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/PSLmodels/OG-Core/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/PSLmodels/OG-Core/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/PSLmodels/OG-Core/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/PSLmodels/OG-Core/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/PSLmodels/OG-Core/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/PSLmodels/OG-Core/compare/v0.7.0...v0.8.0
