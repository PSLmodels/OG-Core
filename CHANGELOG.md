# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2023-10-26 15:00:00

### Added

- Simple update of version in `setup.py` and `cs-config/cs_config/functions.py` to make sure that the `publish_to_pypi.yml` GitHub Action works
- Removes Windows OS tests from `build_and_test.yml`, which are not working right now for some reason.

## [0.1.1] - 2023-10-25 17:00:00

### Added

- Updates `README.md`
- Changes `check_black.yml` to `check_format.yml`
- Updates other GH Action files: `build_and_test.yml`, `docs_check.yml`, and `deploy_docs.yml`
- Updates `publish_to_pypi.yml`
- Adds changes from PRs [#73](https://github.com/PSLmodels/OG-USA/pull/73) and [#67](https://github.com/PSLmodels/OG-USA/pull/67)

## [0.1.0] - 2023-07-19 12:00:00

### Added

- Restarts the release numbering to follow semantic versioning and the OG-USA version numbering as separate from the OG-Core version numbering.
- Adds restriction `python<3.11` to `environment.yml` and `setup.py`.
- Changes the format of `setup.py`.
- Updates `build_and_test.yml` to test Python 3.9 and 3.10.
- Updates some GH Action script versions in `check_black.yml`.
- Updates the Python version to 3.10 in  `docs_check.yml` and `deploy_docs.yml`.
- Updated the `LICENSE` file to one that GitHub recognizes.
- Updates the `run_og_usa.py` run script.
- Updates some tests and associated data.
- Pins the version of `rpy2` package in `environment.yml` and `setup.py`


## Previous versions

### Summary

- Version [0.7.0] on August 30, 2021 was the first time that the OG-USA repository was detached from all of the core model logic, which was named OG-Core. Before this version, OG-USA was part of what is now the [`OG-Core`](https://github.com/PSLmodels/OG-Core) repository. In the next version of OG-USA, we adjusted the version numbering to begin with 0.1.0. This initial version of 0.7.0, was sequential from what OG-USA used to be when the OG-Core project was called OG-USA.
- Any earlier versions of OG-USA can be found in the [`OG-Core`](https://github.com/PSLmodels/OG-Core) repository [release history](https://github.com/PSLmodels/OG-Core/releases) from [v.0.6.4](https://github.com/PSLmodels/OG-Core/releases/tag/v0.6.4) (Jul. 20, 2021) or earlier.



[0.1.2]: https://github.com/PSLmodels/OG-USA/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/PSLmodels/OG-USA/compare/v0.1.0...v0.1.1
