(Chap_Intro)=
# OG-Core

`OG-Core` is the core logic for a country-agnostic overlapping-generations (OG) model of an economy that allows for dynamic general equilibrium analysis of fiscal policy. The source code is openly available for download or collaboration at the GitHub repository [www.github.com/PSLmodels/OG-Core](https://github.com/PSLmodels/OG-Core), or you can click on the GitHub icon at the top right of this page.

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

The model is continuously under development. Users will be notified through [closed PR threads](https://github.com/PSLmodels/OG-Core/pulls?q=is%3Apr+is%3Aclosed) and through the [release notes](https://github.com/PSLmodels/OG-Core/releases) what changes have been implemented. The package will have released versions, which will be checked against existing code prior to release. Stay tuned for an upcoming release!


(Sec_CitingOGCore)=
## Citing OG-Core

`OG-Core` (Version #.#.#)[Source code], https://github.com/PSLmodels/OG-Core.
