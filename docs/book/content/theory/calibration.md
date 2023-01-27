(Chap_Calib)=
# Calibrating OG-Core to a Country or Region

The `OG-Core` model represents all the general model solution code for any overlapping generations model of a country or region. Although `OG-Core` has a `default_parameters.json` file that allows it to run independently, the preferred method for using `OG-Core` is as a dependency to a country calibration repository. We recommend that another repository is made, such as [`OG-USA`](https://github.com/PSLmodels/OG-USA) or [`OG-ZAF`](https://github.com/EAPD-DRB/OG-ZAF/) that uses `OG-Core` as its main computational foundation and engine and calibrates country-specific variables and functions in its own respective source code. This approach results in a working overlapping generations model consisting of a country-specific calibration repository plus a dependency on the general `OG-Core` model logic and options.

{numref}`TabCountryModels` is a list of country-specific calibrations of overlapping generations models that use `OG-Core` as a dependency from oldest to newest. Note that these models are in varying stages of completeness and maturity. It is true that a model is never really fully calibrated. The modeler is always updating calibrated values as new data become available. And the modeler can always search for better fit and better targeting strategies. As such, the only measures of model maturity of the country calibrations below is the data the repository was created and whether or not the example run script runs completely through baseline and reform steady-state and transition-path equilibrium computation.

```{list-table} **Country-specific calibrated OG models based on OG-Core.**
:header-rows: 1
:name: TabCountryModels
* - **Country**
  - **Model name**
  - **GitHub repo**
  - **Documentation**
  - **Date created**
  - **Example script runs**
* - United States
  - `OG-USA`
  - https://github.com/PSLmodels/OG-USA
  - https://pslmodels.github.io/OG-USA
  - 2014-May-25
  - Yes
* - United Kingdom
  - `OG-UK`
  - https://github.com/PSLmodels/OG-UK
  - https://pslmodels.github.io/OG-UK
  - 2021-Feb-14
  - No
* - India
  - `OG-IND`
  - https://github.com/Revenue-Academy/OG-IND
  - https://revenue-academy.github.io/OG-IND
  - 2022-Jul-17
  - Yes
* - Malaysia
  - `OG-MYS`
  - https://github.com/Revenue-Academy/OG-MYS
  -
  - 2022-Jul-17
  - Yes
* - South Africa
  - `OG-ZAF`
  - https://github.com/EAPD-DRB/OG-ZAF
  - https://eapd-drb.github.io/OG-ZAF
  - 2022-Oct-09
  - Yes
```

In the following section, we detail a list of items to calibrate for a country and what types of data and approaches might be available for those calibrations. Each of the country-specific models listed in {numref}`TabCountryModels` will have varying degrees of calibration maturity and futher varying degrees of documentation of their calibration. But the following section details all the areas where each of these models should be calibrated.


(SecCalibList)=
## Detail of parameters, data, and approaches for calibration




(SecCalibFootnotes)=
## Footnotes

<!-- [^citation_note]: See {cite}`AuerbachEtAl:1981,AuerbachEtAl:1983`, {cite}`AuerbachKotlikoff:1983a,AuerbachKotlikoff:1983b,AuerbachKotlikoff:1983c`, and {cite}`AuerbachKotlikoff:1985`. -->
