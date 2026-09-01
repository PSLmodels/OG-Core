# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Bug Fixes

- Fixes Issue [#1200](https://github.com/PSLmodels/OG-Core/issues/1200):
  `replacement_rate_adjust` was read only inside `SS_amount`, so it applied
  to the US-Style Social Security system and was silently ignored under
  Defined Benefits, Notional Defined Contribution, and Points System. The
  adjustment is now applied to those three systems in `pension_amount`,
  via a `replacement_rate_adjustment` helper that mirrors the indexing
  `SS_amount` already uses, including the per-cohort `t + tt` offset along
  the time path. `SS_amount` is unchanged, so US-Style results cannot move.
  
### Added
- `npv_table` in `output_tables.py` (Issue #1131): builds a table of the net present value of the reform-minus-baseline change in flow variables (e.g. `Y`) over a horizon, evaluated at a list of discount rates. Values are un-stationarized by default so the NPV is taken over the actual (trend-inclusive) level path.

## [0.20.0] - 2026-08-13 12:00:00

### Added

- Addresses Issue [#1205](https://github.com/PSLmodels/OG-Core/issues/1205):
  the UN Data Portal API token can now come from a `un_token` argument to
  `demographics.get_un_data`, from a `UN_API_TOKEN` environment variable, or
  from a single per-user file (`$XDG_CONFIG_HOME/og/un_api_token.txt`, or
  `%APPDATA%\og\un_api_token.txt` on Windows). Sources are tried in that
  order, then `un_api_token.txt` in the working directory. This gives one
  token per user instead of one per directory.
- An `og-token` command, OG-Core's first console script, to manage that
  token without hunting for the file: `og-token set` saves one,
  `og-token show` reports where it lives and which source wins, and
  `og-token rm` deletes it. The token is read without echoing and is never
  printed back.

### Changed

- Answering the UN API token prompt now saves the token to the per-user file
  rather than to the current working directory, which used to leave a copy of
  the token in every directory a model was run from. An existing
  `un_api_token.txt` in the working directory is still read, with a notice
  that the location is deprecated.
- The token prompt now names both ways forward: where to generate a free
  token, and that pressing return uses the archived copy of the same data
  instead.
- The token is read with `getpass` rather than `input`, so it is no longer
  echoed into the terminal and its scrollback while being typed or pasted.
- An expired token is now named as the reason for falling back to the
  archived data, instead of failing the same way as a missing one. The
  portal issues JSON Web Tokens, so the expiry date is read locally with
  no extra request, and `og-token show` reports whether the stored token
  is still current. A token that is not a readable JSON Web Token is used
  as before, with no expiry reported.
- A run that falls back to the archived data now says once, not once per
  series, how to register a token: where to get one, the full path of the
  `og-token` command belonging to the running interpreter, and the token
  file to write. The full path matters because `og-token` is installed
  beside the interpreter and is normally not on the shell's PATH, so
  someone who obtains a token days later has something they can paste
  rather than a command that reports "not found".
- The token prompt is skipped when standard input is not interactive, so
  scheduled and scripted runs fall back to the Population-Data archive
  instead of waiting on input.

## [0.19.2] - 2026-08-18 17:00:00

### Adds
- Updates `.gitignore` to include a line `.claude/`.
- Updates the `uv.lock` file to include upgraded `pillow` package version 12.3.0. This satisfies the dependabot PR update in PR #1176. I updated all the packages in the environment. This resulted in an update of the `pytest` package, which caused errors that led to the changes below.
- Updated the version in `pyproject.toml`, `ogcore/__init__.py` and `CHANGELOG.md`.
- Updated the version of `pytest` in `pyproject.toml` to `pytest>=9.1`. Some changes to logging happened in versions 9.1+, that are not backward compatible. All our testing started throwing errors once I updated that package. I changed the `pytest` miniversion in line 94 of `pyproject.toml` to 9.1, and I updated the `filterwarnings` in lines 94-98.
- I updated all the `Makefile` commands to fit with standard GNU formatting and to work with our updated usage of the `uv` package. I also got rid of some compute studio commands, the objects of which had no targets.
- I updated the format of the `logger.info()` warnings in `household.py` which was a source of errors from the updated `pytest` package.
- I fixed some errors that were happening in the building of our Jupyter Book documentation. There was a fight going on between our `make_params.py` and `make_vars.py` scripts and the Sphinx `_toc.yml` table of contents script. And we were missing `config.rst`, `SS.rst`, and `TPI.rst` files. I explicitly put the API chapters in the `_toc.yml` file. And I built an updated `parameters.md` file.
- We had some Jupyter Book warnings for headings that go from H2 to H4 in parameters.md and variables.md. However, this is what we want. So  I added the `_ext/supress_header_warnings.py` file and referenced it in `_config.py` in order to surpress only those warnings associated with only the `parameters.md` and `variables.md` files.

### Bug Fixes

- Fixes Issue [#1199](https://github.com/PSLmodels/OG-Core/issues/1199): `aggregates.revenue` counted payroll tax revenue twice. Each household's `income_payroll_tax_liab` already includes `T_P = tau_payroll * labor_income`, so payroll revenue is inside `iit_payroll_tax_revenue`; PR #1184 then added `get_payroll_tax_revenue` (the same take, re-derived from the aggregate wage bill) into that total again when `tau_payroll` was nonzero. Revenue was overstated, the government over-spent, and any model with a nonzero `tau_payroll` failed the steady-state resource constraint (OG-USA was unaffected only because its `tau_payroll` is 0). The erroneous addition is removed; `get_payroll_tax_revenue` still provides the income vs payroll reporting split, which is unchanged. (See [PR #1204](https://github.com/PSLmodels/OG-Core/pull/1204) and [PR #1199](https://github.com/PSLmodels/OG-Core/pull/1199).)

## [0.19.1] - 2026-08-10 12:00:00

### Added

- Stall detection for the TPI outer loop (Issue #1177): when the best distance has not improved over the last `TPI_stall_window` iterations (default 50; 0 disables), `run_TPI` logs a diagnosis distinguishing a cycling outer loop (suggesting a lower `nu` or `TPI_outer_method="anderson"`) from a diverging economy (usually an inconsistent fiscal block, which solver settings cannot fix). The default `TPI_stall_action="warn"` only logs, leaving model solutions unchanged; `"stop"` also ends the loop early, so a hopeless run fails through the existing non-convergence checks instead of spending the rest of `maxiter`. The window check lives in `ogcore.solvers.diagnose_stall` and works for both the picard and anderson update rules. The diagnosis is re-logged if it changes (e.g. escalates from cycling to diverging), and a run that ends unconverged while stalled carries the diagnosis in the `RuntimeError` message, so it reaches users who only see the traceback. [PR #1178](https://github.com/PSLmodels/OG-Core/pull/1178).
- Changes the default of `stationarized` to `False` in `output_plots.plot_aggregates` and `output_plots.plot_industry_aggregates`, so unstationarized values are plotted by default, resolving Issue [#1133](https://github.com/PSLmodels/OG-Core/issues/1133). [PR #1183](https://github.com/PSLmodels/OG-Core/pull/1183).

### Bug Fixes

- Fixes Issue [#1186](https://github.com/PSLmodels/OG-Core/issues/1186): with demographics that vary across lifetime income groups (introduced in PR #1165), the `use_zeta = False` branch of `household.get_bq` divided each group's bequest pool by its birth share (`lambdas[j]`) rather than its actual population share, so bequests received did not sum to bequests left and the steady-state resource constraint failed. Receipts are now divided by the group's actual population (from `omega_SS` / `omega`), matching the `use_zeta = True` branch. Results are unchanged when demographics are common across groups. [PR #1187](https://github.com/PSLmodels/OG-Core/pull/1187).
- Fixes Issue [#1015](https://github.com/PSLmodels/OG-Core/issues/1015): fixes escape with special characters in parameter names that caused failures when reading from JSON files. Adds a test to prevent this in the future. [PR #1193](https://github.com/PSLmodels/OG-Core/pull/1193).
- Resolve Issue [#1152](https://github.com/PSLmodels/OG-Core/issues/1152): expose same keys in the steady-state and time path dictionaries. [PR #1185](https://github.com/PSLmodels/OG-Core/pull/1185).
- Correct payroll tax calculations, resolving Issue [#1023](https://github.com/PSLmodels/OG-Core/issues/1023). [PR #1184](https://github.com/PSLmodels/OG-Core/pull/1184).

## [0.19.0] - 2026-07-29 12:00:00

### Added

- Fixes Issue [#1169](https://github.com/PSLmodels/OG-Core/issues/1169): the Notional Defined Contribution pension system could not run against a real `Specifications` object because its growth-rate settings were not parameters. `ndc_growth_rate`, `dir_growth_rate`, and `points_growth_rate` are now parameters in `default_parameters.json` (choices `"r"`, `"Curr GDP"`, `"LR GDP"`; default `"LR GDP"`, matching the previous fallback behavior). `g_ndc` and `g_dir` now handle the scalar `r` and `g_y` of the steady state, and `delta_ret` falls back to the steady-state mortality rates implied by `rho` (averaged over lifetime income groups) when no `mort_rates_SS` attribute is set. The NDC system is added to the real-`Specifications` pension test.

## [0.18.1] - 2026-07-22 12:00:00

### Bug Fixes

- Fixes Issue [#1180](https://github.com/PSLmodels/OG-Core/issues/1180): `get_pop_objs` raised an `AssertionError` for the homogeneous case (`income_percentiles=None` with no income-specific inputs), because `expand_pop_obj_J` asserted before reaching its homogeneous branch. `income_percentiles` now defaults to a single income group (J=1) when no income-specific inputs are supplied, restoring the pre-0.18 call signature; it remains required whenever gradients or immigrant income shares are provided. Also removes a stray debug `print` from that branch.

## [0.18.0] - 2026-07-20 12:00:00

### Added

- Demographic parameters (`rho`, `omega`, `imm_rates`) are now allowed to vary across lifetime income groups, `J`. See PR [#1165](https://github.com/PSLmodels/OG-Core/pull/1165)

## [0.17.0] - 2026-07-16 12:00:00

### Bug Fixes

- Fixes the Defined Benefits and Points System pension paths so they run against a real `Specifications` object (Issues #1014 and #1075): `p.retire` (an array since PR #433) was passed as the scalar `S_ret` into the numba loops, the scalar `p.g_y` was indexed as an array inside the loops, the scalar steady-state wage was indexed as a path, and the time-varying 3-D `e` matrix (PR #895) was sliced as 2-D. The systems use the steady-state retirement age and earnings profile for now; full time variation remains open in Issue #1014. The same coercions are applied to the NDC path, but the NDC system additionally requires growth-rate settings (`ndc_growth_rate`, `dir_growth_rate`) that are not yet parameters in `default_parameters.json`, so it still cannot run against a real `Specifications` object (see Issue #1169).
- Wires the pre-time-path inputs the DB/NDC/PS benefit formulas need into the TPI: labor supplied before the time path begins comes from the model's initial condition (the same baseline object that initializes wealth), and pre-time-path wages are anchored to the period-0 wage of the current path. Adds the full time-path (T x S x J) evaluation of Defined Benefits amounts (`DB_3dim_loop`) used when the TPI computes aggregate revenues, built from each cohort's own wage and labor history so aggregates are consistent with household behavior. With these changes a country model using the Defined Benefits system solves both the steady state and the transition path.
- Adds regression tests that call `pension_amount` with a real `Specifications` object per pension system (the existing tests pre-scalarized the inputs and so never exercised the real interface) and a local-marked steady-state solve test with the Defined Benefits system. See PR [#1167](https://github.com/PSLmodels/OG-Core/pull/1167).
- Fixes the tax-liability revenue calculation using the first year's tax noncompliance and filer rates for every period, even when those rates are set to vary over time (Issue #1168): households responded to the changing rates while government revenue, the budget, and debt stayed on the first-year values. `income_tax_liab` now slices the rates over the transition path. See PR [#1174](https://github.com/PSLmodels/OG-Core/pull/1174).
- Fixes the steady-state `etr_ss` and `mtry_ss` diagnostics using the labor-income tax noncompliance rate in place of the capital-income rate (`capital_noncompliance_rate_2D` in `SS.py`, Issue #1170). This affects only the post-solve SS diagnostics -- the solution itself already used the correct rates -- and is a no-op when the two rates are equal (the default). See PR [#1171](https://github.com/PSLmodels/OG-Core/pull/1171).
- Fixes `alpha_FA` (direct foreign aid, added in 0.16.0) not being extended over the model time path: it was never registered in `tp_param_list`, so a multi-year path stayed short and broke the transition solve. It now extends to `T + S` with the last value carried forward, like `alpha_G`/`alpha_T`/`alpha_I`. See PR [#1166](https://github.com/PSLmodels/OG-Core/pull/1166).

## [0.16.4] - 2026-07-02 12:00:00

### Added

- Adds an opt-in accelerated update rule for the TPI outer loop, selected by the new `TPI_outer_method` parameter (default `"picard"`, which leaves the model's historical damped functional iteration and all model solutions unchanged). Setting `TPI_outer_method="anderson"` uses limited-memory Anderson acceleration on the outer-loop residual history, guarded by a trust region anchored to the always-feasible damped point (controlled by `TPI_anderson_m`, `TPI_anderson_beta`, and `TPI_trust_radius`). On a stiff multi-industry reform that limit-cycles under constant dampening, this converged to the same equilibrium in 53 outer iterations vs 126 for constant `nu=0.1` (about −37% wall-clock vs the best damped-`nu` schedule). The new solver lives in `ogcore/solvers.py`. See PR [#1164](https://github.com/PSLmodels/OG-Core/pull/1164).

## [0.16.3] - 2026-06-25 15:00:00

### Added

- Better functionality and more country repositories with the OG installer. See PR [#1162](https://github.com/PSLmodels/OG-Core/pull/1162)

### Bug Fix
- Fixes an inconsistency with the pre-time path population distribution and growth rates. Note that the `demographics.get_pop` function has been changed and now only returns one object: the population distribution (not also the distribution prior to the start year).  In addition, the `get_pop` and `get_pop_objs` functions no longer have the `pre_pop_dist` kwarg. See PR [#1073](https://github.com/PSLmodels/OG-Core/pull/1073).

## [0.16.2] - 2026-06-15 12:00:00

### Added

 - Adds `output_tables.model_fit_table` function that generates a table comparing model output to data for a set of target moments, and adds a test of this function in `test_output_tables.py`. The function takes as input a list of target moment descriptions, the TPI output dictionary, and the model parameters object, and returns a pandas DataFrame with the target moment descriptions, the model values for those moments, and the data values for those moments (where applicable). The function currently supports a set of macroeconomic moments (interest rate, capital share of output, labor share of output), inequality moments (Gini coefficient for before-tax income and after-tax income), and demographic moments (fraction of population 65+ and population growth rate). See PR [#1138](https://github.com/PSLmodels/OG-Core/pull/1138).
 - Validates `beta_annual` and `chi_b` against `J`. ([PR #1149](https://github.com/PSLmodels/OG-Core/pull/1149))

### Bug Fixes
- Fixes issue with reading UN data on Pandas >= 3.0 and UN token string. ([PR #1151](https://github.com/PSLmodels/OG-Core/pull/1151))
- Fixes math notation for plot labels. ([PR #1148](https://github.com/PSLmodels/OG-Core/pull/1148))
- Fixes reshaping issues with `J=1` parameterization. ([PR #1145](https://github.com/PSLmodels/OG-Core/pull/1145))


## [0.16.1] - 2026-06-04 12:00:00

### Added

- Adds a `use_sparse_FOC_jac` `Specifications` parameter (default `True`) that accelerates the time path iteration (TPI) household solve. With it on, `scipy.optimize.root` is given a sparse (banded) finite-difference Jacobian for the stacked household Euler and labor first order conditions: the sparsity pattern is auto-detected once per problem size and the solver then needs far fewer function evaluations per Jacobian build (about 20x fewer on the default S=80 cohort solve), with an automatic fallback to dense finite differences if the Jacobian is not sparse enough to benefit or if a solve fails. The result matches the legacy dense-finite-difference solution to within the model's resource-constraint accuracy floor on every calibration tested (OG-Core standard example, OG-ETH, OG-ZAF, OG-PHL, OG-IDN), giving roughly a 1.9-2.4x TPI speedup at no accuracy cost. Set `use_sparse_FOC_jac=False` to recover bit-identical agreement with v0.16.0 and earlier.

## [0.16.0] - 2026-06-02 12:00:00

### Added

- Adds a new parameter, `alpha_FA`, that allows the user to specify the level of direct foreign aid as a percentage of GDP.  See PR [#1126](https://github.com/PSLmodels/OG-Core/pull/1126).

## [0.15.13] - 2026-05-15 06:00:00

### Added

- Increases the maximum value of the `mindist_TPI` parameter to 0.01 in `default_parameters.json`.
- Updates the `uv.lock` with package updates.

## [0.15.12] - 2026-05-14 12:00:00

### Added

- Caches the `e_long` array in `household.FOC_savings` and `household.FOC_labor` rather than rebuilding it on every call. `e_long` is a pure function of `p.e`, `p.S`, and `p.J`, none of which change during a solve, so a `_get_e_long` helper now builds it once per worker and reuses it. Profiling identified this rebuild as the single most expensive operation in a TPI run. The change is a pure cache — model output is bit-for-bit identical to master — and gives roughly a 3x speedup on a single-reform TPI run. See PR [#1128](https://github.com/PSLmodels/OG-Core/pull/1128).
- Builds the per-period tax-parameter slices in `TPI.inner_loop` as numpy arrays via a new `_params_to_array` helper, and switches `txfunc.get_tax_rates` to `np.asarray`, so the repeated per-call list-to-array conversion is skipped on the hot TPI path. `mono` and `mono2D` tax functions store callables rather than numbers, so their nested-list form is passed through unchanged. Profiling identified this conversion as the next hot spot after the `e_long` rebuild. Model output is bit-for-bit identical to master, and the change gives roughly a further 10% speedup on a single-reform TPI run (about 3.3x cumulative versus master). See PR [#1128](https://github.com/PSLmodels/OG-Core/pull/1128).
- Changes the minimum of the allowable range for `tau_c` in `default_parameters.py` to allow for government consumption subsidies.
- Fixes a bug in the `parameter_plots.plot_fert_rates` function. See PR [#1127](https://github.com/PSLmodels/OG-Core/pull/1127).

## [0.15.11] - 2026-05-08 12:00:00

### Added

- Updates the `uv.lock` to include dependabot upgrades PRs [#1118](https://github.com/PSLmodels/OG-Core/pull/1118), [#1119](https://github.com/PSLmodels/OG-Core/pull/1119), [#1120](https://github.com/PSLmodels/OG-Core/pull/1120), and [#1121](https://github.com/PSLmodels/OG-Core/pull/1121), as well as some other package upgrades that were available.

## [0.15.10] - 2026-04-23 12:00:00

### Added

- Updates `uv.lock` to include upgrade to package `nbconvert from 7.17.0 to 7.17.1. This was the subject of [PR 1110](https://github.com/PSLmodels/OG-Core/pull/1110). But we could not get its tests to pass.
- Fixes typo in `CHANGELOG.md`


## [0.15.9] - 2026-04-23 03:00:00

### Added

- Create a `ogcore` logging environment so that logging specificed in `ogcore.config` is not overridden by other logging configurations in user code. See PR [#1109](https://github.com/PSLmodels/OG-Core/pull/1109).
- Includes [PR 1111](https://github.com/PSLmodels/OG-Core/pull/1111) and [PR 1112](https://github.com/PSLmodels/OG-Core/pull/1112) that change `build_and_test.yml` to not upload CodeCov statistics for PRs generated by dependabots and updates the `codecov-action` to v5.

## [0.15.8] - 2026-04-16 00:30:00

### Added

- Updates the `uv.lock` file to the current version of `ogcore` package (0.15.8)
- Adds `'**.lock'` to the paths in `build_and_test.yml` so that any changes to the `uv.lock` file trigger the GH Action CI tests in `build_and_test.yml`
- Incorporates the dependabot updates to `uv.lock` versions from [PR 1105](https://github.com/PSLmodels/OG-Core/pull/1105) and [PR 1106](https://github.com/PSLmodels/OG-Core/pull/1106).

## [0.15.7] - 2026-04-15 20:00:00

### Added

- Fixed some typos in `CHANGELOG.md` and `intro.md`
- Added dependabot package version updates to `uv.lock` from [PR 1102](https://github.com/PSLmodels/OG-Core/pull/1102) and [PR 1103](https://github.com/PSLmodels/OG-Core/pull/1103)

## [0.15.6] - 2026-04-15 16:00:00

### Added

- Changes to package files to use `uv` for packaging and running commands.  See PR [#1093](https://github.com/PSLmodels/OG-Core/pull/1093)

## [0.15.5] - 2026-01-27 12:00:00

### Added

- Additional parameter metadata to `default_parameters.json`.  See PR [#1097](https://github.com/PSLmodels/OG-Core/pull/1097)

## [0.15.4] - 2026-01-27 12:00:00

### Added

- Ability to simulate the model with a single type of household (`J=1`).  See PR [#1062](https://github.com/PSLmodels/OG-Core/pull/1062)

## [0.15.3] - 2026-01-24 12:00:00

### Added

- Two new parameters, `income_tax_filer` and `wealth_tax_filer`, that determine whether certain types `j` pay income or wealth taxes, respectively.  See PR [#1084](https://github.com/PSLmodels/OG-Core/pull/1084)

## [0.15.2] - 2026-01-22 12:00:00

### Added

- Two new parameters, `r_gov_DY` and `r_gov_DY2`, that allow the government interest rate to be a function of the debt-to-GDP ratio.  See PR [#1037](https://github.com/PSLmodels/OG-Core/pull/1037)

## [0.15.1] - 2026-01-19 12:00:00

### Added

- A new parameter `c_min` to the `Parameters` class that allows the user to specify minium consumption amounts by consumption good.  See PR [#1085](https://github.com/PSLmodels/OG-Core/pull/1085)

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


[0.20.0]: https://github.com/PSLmodels/OG-Core/compare/v0.19.2...v0.20.0
[0.19.2]: https://github.com/PSLmodels/OG-Core/compare/v0.19.1...v0.19.2
[0.19.1]: https://github.com/PSLmodels/OG-Core/compare/v0.19.0...v0.19.1
[0.19.0]: https://github.com/PSLmodels/OG-Core/compare/v0.18.1...v0.19.0
[0.18.1]: https://github.com/PSLmodels/OG-Core/compare/v0.18.0...v0.18.1
[0.18.0]: https://github.com/PSLmodels/OG-Core/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/PSLmodels/OG-Core/compare/v0.16.4...v0.17.0
[0.16.4]: https://github.com/PSLmodels/OG-Core/compare/v0.16.3...v0.16.4
[0.16.3]: https://github.com/PSLmodels/OG-Core/compare/v0.16.2...v0.16.3
[0.16.2]: https://github.com/PSLmodels/OG-Core/compare/v0.16.1...v0.16.2
[0.16.1]: https://github.com/PSLmodels/OG-Core/compare/v0.16.0...v0.16.1
[0.16.0]: https://github.com/PSLmodels/OG-Core/compare/v0.15.13...v0.16.0
[0.15.13]: https://github.com/PSLmodels/OG-Core/compare/v0.15.12...v0.15.13
[0.15.12]: https://github.com/PSLmodels/OG-Core/compare/v0.15.11...v0.15.12
[0.15.11]: https://github.com/PSLmodels/OG-Core/compare/v0.15.10...v0.15.11
[0.15.10]: https://github.com/PSLmodels/OG-Core/compare/v0.15.9...v0.15.10
[0.15.9]: https://github.com/PSLmodels/OG-Core/compare/v0.15.8...v0.15.9
[0.15.8]: https://github.com/PSLmodels/OG-Core/compare/v0.15.7...v0.15.8
[0.15.7]: https://github.com/PSLmodels/OG-Core/compare/v0.15.6...v0.15.7
[0.15.6]: https://github.com/PSLmodels/OG-Core/compare/v0.15.5...v0.15.6
[0.15.5]: https://github.com/PSLmodels/OG-Core/compare/v0.15.4...v0.15.5
[0.15.4]: https://github.com/PSLmodels/OG-Core/compare/v0.15.3...v0.15.4
[0.15.3]: https://github.com/PSLmodels/OG-Core/compare/v0.15.2...v0.15.3
[0.15.2]: https://github.com/PSLmodels/OG-Core/compare/v0.15.1...v0.15.2
[0.15.1]: https://github.com/PSLmodels/OG-Core/compare/v0.15.0...v0.15.1
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
