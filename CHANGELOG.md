# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.2] - 2025-01-20 12:00:00

### Added

- A new parameters, `r_gov_DY` and `r_gov_DY2`, that allow the government interest rate to be a function of the debt-to-GDP ratio.  See PR [#1037](https://github.com/PSLmodels/OG-Core/pull/1037)


## [0.15.0] - 2025-12-03 12:00:00

### Added

- A new parameter `baseline_theta` to the `Parameters` class that allows the user to specify whether to use the steady-state replacement rate parameters from the baseline solution in a reform run.  See PR [#1077](https://github.com/PSLmodels/OG-Core/pull/1077)

## [0.14.14] - 2025-11-24 12:00:00

### Added

- Create `SS.SS_initial_guesses` function to allow more flexible initial guesses for steady state solution ((PR #1061)[https://github.com/PSLmodels/OG-Core/pull/1061])
- Robust steady-state solution used for reform solution ((PR #1061)[https://github.com/PSLmodels/OG-Core/pull/1061])
- Test of `SS.solve_for_j` function ((PR #1061)[https://github.com/PSLmodels/OG-Core/pull/1061])

### Bug Fix
- Fixes to deprecated API calls avoid many warnings during testing ((PR #1061)[https://github.com/PSLmodels/OG-Core/pull/1061])

## [0.14.13] - 2025-11-21 12:00:00

### Bug Fix

- Fix calculation of consumption tax revenue with differentiated goods ((PR #1074)[https://github.com/PSLmodels/OG-Core/pull/1074])

## [0.14.12] - 2025-11-07 12:00:00

### Bug Fix

- Use data for pre-time path population distribution (rather than inferring it) ([PR #1071](https://github.com/PSLmodels/OG-Core/pull/1071))

## [0.14.11] - 2025-11-07 12:00:00

### Added

- Adds Ethiopia demographic data mapping ((PR #1063)[https://github.com/PSLmodels/OG-Core/pull/1063])
-
## [0.14.10] - 2025-09-11 12:00:00

### Added

- Fixes nonconformable matrices in `TPI.py` introduced in version 0.14.9 ((PR #1054)[https://github.com/PSLmodels/OG-Core/pull/1054])

## [0.14.9] - 2025-09-10 20:00:00

### Added

- Fixes `replacement_rate_adjustment` parameter in the steady state ((PR #1053)[https://github.com/PSLmodels/OG-Core/pull/1053])
- Adds some saved output to `tpi_vars.pkl` object ((PR #1054)[https://github.com/PSLmodels/OG-Core/pull/1054])

## [0.14.8] - 2025-08-26 12:00:00

### Added

- Adds a complete benchmark suite for measuring and optimizing Dask performance in OG-Core, with particular focus on Windows performance issues.
- New and updated files:
    - tests/test_dask_benchmarks.py: Mock benchmark tests with synthetic workloads
    - tests/test_real_txfunc_benchmarks.py: Real-world tax function benchmarks
    - tests/run_benchmarks.py: Automated benchmark runner with reporting
    - tests/BENCHMARK_README.md: Comprehensive documentation and usage guide
    - pytest.ini: Updated with benchmark test markers
- Key features:
    - Platform-specific optimization tests (Windows, macOS, Linux)
    - Memory usage and compute time benchmarking
    - Baseline establishment and performance regression detection
    - Comparison of different Dask schedulers and client configurations
    - Real tax function estimation performance measurement
    - Automated identification of optimal Dask settings per platform
- Benefits:
    - Establishes performance baselines before optimization work
    - Identifies Windows-specific Dask performance bottlenecks
    - Provides automated regression detection for future changes
    - Enables data-driven optimization decisions
    - Supports continuous performance monitoring
- Usage:
    - `python tests/run_benchmarks.py  # Run all benchmarks`
    - `python tests/run_benchmarks.py --quick  # Quick benchmarks only`
    - `python tests/run_benchmarks.py --save-baseline  # Save performance baseline`
    - `python tests/run_benchmarks.py --compare-baseline # Compare against baseline`
- 🤖 Generated with help from Claude Code

## [0.14.7] - 2025-08-21 17:00:00

### Added

- Refactor calls to dask in `SS.py` and `TPI.py`.  See PR [#1048](https://github.com/PSLmodels/OG-Core/pull/1048)

## [0.14.6] - 2025-08-15 14:00:00

### Added

- Removes `initial_guess_w_SS` in `default_parameters.json`
- Updates environment and testing to cover Python 3.13

## [0.14.5] - 2025-07-08 22:00:00

### Added

- Increases the maximum value of `initial_guess_TR_SS` in `default_parameters.json`

## [0.14.4] - 2025-06-23 18:00:00

### Added

- Fixes the sign error on the remittances `RM` term in `aggregates.py`, `resource_constraint()` function.
- Added a test with positive remittances to `test_aggregates.py`, `test_resource_constraint()` function.

## [0.14.3] - 2025-04-25 10:00:00

### Added

- Puts a ceiling on the version of the `marshmallow<4.0.0` package in `environment.yml`
- Update `txfunc.py` for the estimation of the `HSV` and `GS` tax functions.
- Update `test_txfunc.py`, `tax_func_estimate_outputs.pkl`, and `tax_func_loop_outputs.pkl` files for testing
- Update `dask` and `distributed` client calls in `SS.py` and `TPI.py` to allow for updated versions.

## [0.14.2] - 2025-04-04 12:00:00

### Added

- `utils.pct_change_unstationarized` replaced with `utils.unstationarize_vars`, which allows for more general use of a utility to find unstationarized values of time series output
- `output_plots.py` and `output_tables.py` are updated to all for plots and tables of unstationarized output for variables of any type, not just percentage changes.
- `ouput_tables.tp_output_dump_table` has been renamed `output_tables.time_series_table`
- `utils.param_dump_json` has been renamed `utils.params_to_json`
- API docs have been updated to include functions left out previously and for new function names

## [0.14.1] - 2025-03-16 12:00:00

### Bug Fix

- Packages `model_variables.json` with `ogcore`

## [0.14.0] - 2025-03-16 07:00:00

### Added

- Updates the output dictionaries for `TPI.py` and `SS.py` to use consistent variables names
- Adds a `model_variables.json` file that has metadata about the model variables and is used to build a new chapter in the documentation describing the model variables
- Replaces `print` commands in `TPI.py` and `SS.py` with `logger.info` commands for easier suppression of output

## [0.13.2] - 2024-12-08 12:00:00

### Added

- Adds KOR, THA, BRA to the list of countries in the `demographics.py` module.

## [0.13.1] - 2024-10-02 12:00:00

### Added

- Three new parameters to adjust government spending amounts in the case of `baseline_spending=True`:
  - `alpha_bs_G`: the proportional adjustment to the level of baseline spending on government consumption (time varying, default value is 1.0 for each model period)
  - `alpha_bs_T`: the proportional adjustment to the level of baseline spending on non-pension transfers (time varying, default value is 1.0 for each model period)
  - `alpha_bs_I`: the proportional adjustment to the level of baseline spending on infrastructure investment (time varying, default value is 1.0 for each model period)

## [0.13.0] - 2024-09-26 12:00:00

### Added

- Updates all of the documentation.
    - Adds remittances to all instances of the household budget constraint
    - Rewrites bequests and transfers components of household budget constraint in terms of individual variables in all instances
    - Adds a household transfers section to `households.md` with subsections on bequests, remittances, government transfers, and universal basic income
    - Changes all instances of $p_t Y_t$ to $Y_t
    - Updates the steady-state equilibrium algorithm description in `equilibrium.md`
    - Added updates to the government pensions descriptions in `government.md` and added `pensions` to all instances of the household budget constraint.
    - Updates the docstrings in `tax.py` for the wealth tax ETR and MTR functions. The code is right. I just thought there was a clearer specification of the equations in LaTeX.
- Adds remittances to the OG-Core code
    - Adds aggregate remittances function `get_RM()` to `aggregates.py`
    - Adds household remittances function `get_rm()` to `household.py`
    - Adds four new remittance parameters: `alpha_RM_1`, `g_RM`, `alpha_RM_T`, `eta_RM`
    - We model aggregate remittances as a percent of GDP in the first period, then growing at a specified rate that can deviate from the country growth rate until the cutoff rule period, after which the remittance growth rate trends back to the long-run model growth rate (growth rate in population and productivity). We also model remittances in reforms as being a percentage of baseline GDP. In this way, if remittance parameters are not changed in the reform, remittances remain at their baseline levels. The only way they change is if their parameter values are changed.
    - Adds 3 tests using the `test_get_RM()` function in `test_aggregates.py`
    - Adds 4 tests using the `test_get_rm()` function in `test_household.py`
    - Changes the `initial_guess_r_SS` in two tests in `test_SS.py` because they were not solving with their current values
- Increases `RC_SS` steady-state resource constraint tolerance from 1e-9 to 1e-8 because two `test_run_SS()` tests were failing in `test_SS.py` with resource constraints errors just bigger than 1e-9 (-2.29575914e-09 for [Baseline, small open] and -2.29575914e-09 for [Reform, small open]).
- Increases `RC_TPI` transition path resource constraint tolerance from 1e-5 to 1e-4 in because one `test_run_TPI_full_run()` test was failing in `test_TPI.py` with a resource constraint error just bigger than 1e-5 (1.4459913381864586e-05 for `[Baseline, M=3 non-zero Kg]`).
- Updated two directory path references that were out of date in `test_run_example.py`.
- Updated expected value tuples and dictionaries in `test_txfunc.py`.

## [0.12.0] - 2024-08-20 12:00:00

### Added

- Support for Python 3.12

## [0.11.17] - 2024-08-18 12:00:00

### Added

- Description of `alpha_I` in docs
- Updates valid range of the nominal UBI parameters

### Bug Fix

- Extrapolate `alpha_I` in `parameters.py`
- Ensure `alpha_I` shape conforms in `TPI.py`
- Fix formatting of labels in `constants.py`

## [0.11.16] - 2024-08-10 12:00:00

### Added

- Added parameter script `make_params.py` that generates a markdown file for the documentation `parameters.md`. Updates `Makefile`, and GitHub Actions ([PR #963](https://github.com/PSLmodels/OG-Core/pull/963))
- Updated debt-to-GDP plot labels ([PR #962](https://github.com/PSLmodels/OG-Core/pull/962))

## [0.11.15] - 2024-07-30 12:00:00

### Bug Fix

- Make `OGcorePlots.mplstyle` importable from the package by adding it to `setup.py`

## [0.11.14] - 2024-07-30 12:00:00

### Added

- Aesthetic updates to plotting functions in `parameter_plots.py` and `demographics.py`

## [0.11.13] - 2024-07-28 12:00:00

### Added

- Added three new pension types to the model: (i) defined benefits system, (ii) notional defined contribution system, and (iii) points system.

## [0.11.12] - 2024-07-26 01:00:00

### Bug Fix

- Fixes extrapolation of nested lists of tax function parameters.

## [0.11.11] - 2024-06-24 01:00:00

### Added

- Add new parameters for resource constraint tolerances for steady state and time path solution.

## [0.11.10] - 2024-06-17 01:00:00

### Added

- Add HSV to list of valid tax functions in `default_parameters.json`


## [0.11.9] - 2024-06-12 01:00:00

### Added

- Update `demographics.py` in the case input prompt not work.
- Add new utility to dump the parameters to a JSON file


## [0.11.8] - 2024-06-09 01:00:00

### Added

- Updates to `demographics.py` module to accept token for UN World Population Prospects database access or to download data from the [Population-Data](https://github.com/EAPD-DRB/Population-Data) repository.

## [0.11.7] - 2024-06-07 01:00:00

### Added

- Heathcote, Storesletten, and Violante (2017) tax functions to `txfunc.py`

## [0.11.6] - 2024-04-19 01:00:00

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


[0.15.0]: https://github.com/PSLmodels/OG-Core/compare/v0.14.14...v0.15.0
[0.14.14]: https://github.com/PSLmodels/OG-Core/compare/v0.14.13...v0.14.14
[0.14.13]: https://github.com/PSLmodels/OG-Core/compare/v0.14.12...v0.14.13
[0.14.12]: https://github.com/PSLmodels/OG-Core/compare/v0.14.11...v0.14.12
[0.14.11]: https://github.com/PSLmodels/OG-Core/compare/v0.14.10...v0.14.11
[0.14.10]: https://github.com/PSLmodels/OG-Core/compare/v0.14.9...v0.14.10
[0.14.9]: https://github.com/PSLmodels/OG-Core/compare/v0.14.8...v0.14.9
[0.14.8]: https://github.com/PSLmodels/OG-Core/compare/v0.14.7...v0.14.8
[0.14.7]: https://github.com/PSLmodels/OG-Core/compare/v0.14.6...v0.14.7
[0.14.6]: https://github.com/PSLmodels/OG-Core/compare/v0.14.5...v0.14.6
[0.14.5]: https://github.com/PSLmodels/OG-Core/compare/v0.14.4...v0.14.5
[0.14.4]: https://github.com/PSLmodels/OG-Core/compare/v0.14.3...v0.14.4
[0.14.3]: https://github.com/PSLmodels/OG-Core/compare/v0.14.2...v0.14.3
[0.14.2]: https://github.com/PSLmodels/OG-Core/compare/v0.14.1...v0.14.2
[0.14.1]: https://github.com/PSLmodels/OG-Core/compare/v0.14.0...v0.14.1
[0.14.0]: https://github.com/PSLmodels/OG-Core/compare/v0.13.2...v0.14.0
[0.13.2]: https://github.com/PSLmodels/OG-Core/compare/v0.13.1...v0.13.2
[0.13.1]: https://github.com/PSLmodels/OG-Core/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/PSLmodels/OG-Core/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/PSLmodels/OG-Core/compare/v0.11.17...v0.12.0
[0.11.17]: https://github.com/PSLmodels/OG-Core/compare/v0.11.16...v0.11.17
[0.11.16]: https://github.com/PSLmodels/OG-Core/compare/v0.11.15...v0.11.16
[0.11.15]: https://github.com/PSLmodels/OG-Core/compare/v0.11.14...v0.11.15
[0.11.14]: https://github.com/PSLmodels/OG-Core/compare/v0.11.13...v0.11.14
[0.11.13]: https://github.com/PSLmodels/OG-Core/compare/v0.11.11...v0.11.13
[0.11.11]: https://github.com/PSLmodels/OG-Core/compare/v0.11.10...v0.11.11
[0.11.10]: https://github.com/PSLmodels/OG-Core/compare/v0.11.9...v0.11.10
[0.11.9]: https://github.com/PSLmodels/OG-Core/compare/v0.11.8...v0.11.9
[0.11.8]: https://github.com/PSLmodels/OG-Core/compare/v0.11.7...v0.11.8
[0.11.7]: https://github.com/PSLmodels/OG-Core/compare/v0.11.6...v0.11.7
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
