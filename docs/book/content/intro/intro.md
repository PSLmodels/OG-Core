(Chap_Intro)=
# OG-Core

| | |
| --- | --- |
| Org | [![PSL cataloged](https://img.shields.io/badge/PSL-cataloged-a0a0a0.svg)](https://www.PSLmodels.org) [![OS License: CCO-1.0](https://img.shields.io/badge/OS%20License-CCO%201.0-yellow)](https://github.com/PSLmodels/OG-Core/blob/master/LICENSE) [![Jupyter Book Badge](https://raw.githubusercontent.com/jupyter-book/jupyter-book/next/docs/media/images/badge.svg)](https://pslmodels.github.io/OG-Core/) |
| Package | [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/) [![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3137/)  [![PyPI Latest Release](https://img.shields.io/pypi/v/ogcore.svg)](https://pypi.org/project/ogcore/) [![PyPI Downloads](https://img.shields.io/pypi/dm/ogcore.svg?label=PyPI%20downloads)](https://pypi.org/project/ogcore/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
| Testing | ![example event parameter](https://github.com/PSLmodels/OG-Core/actions/workflows/build_and_test.yml/badge.svg?branch=master) ![example event parameter](https://github.com/PSLmodels/OG-Core/actions/workflows/deploy_docs.yml/badge.svg?branch=master) ![example event parameter](https://github.com/PSLmodels/OG-Core/actions/workflows/check_black.yml/badge.svg?branch=master) [![Codecov](https://codecov.io/gh/PSLmodels/OG-Core/branch/master/graph/badge.svg)](https://codecov.io/gh/PSLmodels/OG-Core) |

`OG-Core` is the core logic for a country-agnostic overlapping-generations (OG) model of an economy that allows for dynamic general equilibrium analysis of fiscal policy. The source code is openly available for download or collaboration at the GitHub repository [www.github.com/PSLmodels/OG-Core](https://github.com/PSLmodels/OG-Core), or you can click on the GitHub icon at the top right of this page.

**Country calibrations of OG-Core**
|               |             |             |
| :-----------: | :---------: | :---------: |
| United States, [OG-USA](https://github.com/PSLmodels/OG-USA) | United Kingdom, [OG-UK](https://github.com/PSLmodels/OG-USA) | Phillipines, [OG-PHL](https://github.com/EAPD-DRB/OG-PHL) |
|  South Africa, [OG-ZAF](https://github.com/EAPD-DRB/OG-ZAF) | Indonesia, [OG-IDN](https://github.com/EAPD-DRB/OG-IDN) | Ethiopia, [OG-ETH](https://github.com/EAPD-DRB/OG-ETH) |
| India, [OG-IND](https://github.com/OpenSourceEcon/OG-IND) | Malaysia, [OG-MYS](https://github.com/OpenSourceEcon/OG-MYS) |  |

The model output focuses changes in macroeconomic aggregates (GDP, investment, consumption), wages, interest rates, and the stream of tax revenues over time. Although `OG-Core` can be run independently based on default parameter values (currently representing something similar to the United States), it is meant to be a dependency of a country-specific calibration. This documentation contains the following major sections, which are regularly updated.

* Contributing to `OG-Core`
* `OG-Core` API
* `OG-Core` Theory
* Appendix
* References
* Citations of `OG-Core`


(Sec_CoreMaintainers)=
## Core Maintainers

[Jason DeBacker](https://www.jasondebacker.com/) (GitHub handle [@jdebacker](https://github.com/jdebacker)) and [Richard W. Evans](https://sites.google.com/site/rickecon/) (GitHub handle [@rickecon](https://github.com/rickecon)) are the core maintainers of `OG-Core`. If you have questions about or contributions to the model or repository, please submit a GitHub "Issue" described in the {ref}`Sec_GitHubIssue` subsection or "Pull Request" as described in the {ref}`Sec_GitHubPR` subsection of the {ref}`Sec_Workflow` section of the `OG-Core` {ref}`Chap_ContribGuide`.


(Sec_Disclaimer)=
## Disclaimer

The OG-Core model is continuously under development. Users will be notified through [closed PR threads](https://github.com/PSLmodels/OG-Core/pulls?q=is%3Apr+is%3Aclosed) and through the [release notes](https://github.com/PSLmodels/OG-Core/releases) what changes have been implemented. The package will have released versions, which will be checked against existing code prior to release. Stay tuned for an upcoming release!


(Sec_CitingOGCore)=
## Citing OG-Core

`OG-Core` (Version #.#.#)[Source code], https://github.com/PSLmodels/OG-Core.
