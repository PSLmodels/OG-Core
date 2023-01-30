(Chap_Calib)=
# Calibrating OG-Core

The `OG-Core` model represents all the general model solution code for any overlapping generations model of a country or region. Although `OG-Core` has a `default_parameters.json` file that allows it to run independently, the preferred method for using `OG-Core` is as a dependency to a country calibration repository. We recommend that another repository is made, such as [`OG-USA`](https://github.com/PSLmodels/OG-USA) or [`OG-ZAF`](https://github.com/EAPD-DRB/OG-ZAF/) that uses `OG-Core` as its main computational foundation and engine and calibrates country-specific variables and functions in its own respective source code. This approach results in a working overlapping generations model consisting of a country-specific calibration repository plus a dependency on the general `OG-Core` model logic and options.

{numref}`TabCountryModels` is a list of country-specific calibrations of overlapping generations models that use `OG-Core` as a dependency from oldest to newest. Note that these models are in varying stages of completeness and maturity. It is true that a model is never really fully calibrated. The model maintainer is always updating calibrated values as new data become available. And the modele maintainer can always search for better fit and better targeting strategies. As such, the only measures of model maturity of the country calibrations below is the date the repository was created.

```{list-table} **Country-specific calibrated OG models based on OG-Core.**
:header-rows: 1
:name: TabCountryModels
* - **Country**
  - **Model name**
  - **GitHub repo**
  - **Documentation**
  - **Date created**
* - United States
  - `OG-USA`
  - https://github.com/PSLmodels/OG-USA
  - https://pslmodels.github.io/OG-USA
  - May 25, 2014
* - United Kingdom
  - `OG-UK`
  - https://github.com/PSLmodels/OG-UK
  - https://pslmodels.github.io/OG-UK
  - Feb. 14, 2021
* - India
  - `OG-IND`
  - https://github.com/Revenue-Academy/OG-IND
  - https://revenue-academy.github.io/OG-IND
  - Jul. 17, 2022
* - Malaysia
  - `OG-MYS`
  - https://github.com/Revenue-Academy/OG-MYS
  -
  - Jul. 17, 2022
* - South Africa
  - `OG-ZAF`
  - https://github.com/EAPD-DRB/OG-ZAF
  - https://eapd-drb.github.io/OG-ZAF
  - Oct. 9, 2022
```

In the following section, we detail a list of items to calibrate for a country and what types of data and approaches might be available for those calibrations. Each of the country-specific models listed in {numref}`TabCountryModels` will have varying degrees of calibration maturity and futher varying degrees of documentation of their calibration. But the following section details all the areas where each of these models should be calibrated.


(SecCalibList)=
## Detail of parameters, data, and approaches for calibration

{numref}`TabCalibStrategy` shows the data and calibration strategies for each parameter and parameter area of the model.

```{list-table} **Areas, parameters, and data strategies for calibrating country- or region-specific OG model based on OG-Core.**
:header-rows: 1
:name: TabCalibStrategy
* - **General item description**
  - **Specific item description**
  - **Data source**
* - Demographics
  - Using UN population data
  - Access to country demographics in UN Population Data Portal
* - Demographics
  - Other data source
  - Custom interface between OG model and other data source. Data source must have the number of people by age, fertility rates by age, mortality rates by age (age bins are suitable and interpolation can be used).
* - Macroeconomic parameters
  - Capital share of income, private/sovereign interest rate spread, long-run growth rate, debt-to-GDP ratios, transfer spending to GDP, government spending on goods and services to GDP, foreign purchases of government debt
  - Capital and Labor cost data by industry. Average private borrowing rate/corporate bond yields, GDP time series, publicly held government debt time series, government transfer program spending, government spending (total non-transfer and infrastructure spending separately)
* - Lifetime income profiles
  - Approximate US profiles rescaled by Gini coefficient
  - Gini coefficient for the country
* - Lifetime income profiles
  - Estimate from micro data
  - Individual panel data with earnings (wage, salaries, self-employment income before taxes), labor hours, and age (can impute labor hours if necessary)
* - Labor supply elasticities
  - Constant
  - Use existing estimates from the research literature. Or cross sectional or panel data with hours and wages.
* - Labor supply elasticities
  - Age varying
  - Use existing estimates from the research literature. Or cross sectional or panel data with hours and wages and age.
* - Bequest motive
  - Bequest motive
  - Data on bequests given and/or bequests received similar to the US Survey of Consumer Finances. Other forms of information could allow us to rescale the US bequest distribution to match some moment from the target country.
* - Rate of time preference
  - Constant
  - Research empiritcal literature
* - Rate of time preference
  - Heterogeneous (match to MPCs and wealth distribution)
  - Data on country marginal propensity to consume (e.g., US Consumer Expenditure Survey or PSID) and data on the distribution of wealth in the country
* - Composite consumption share parameters
  - Stone-Geary sub-utility function
  - Consumption by category data within the country (e.g., similar to the US Consumer Expenditure Survey)
* - Hand-to-mouth consumers
  - Calibrated separately from savers
  - Cross-sectional or panel data with measures of income, wealth, consumption
* - Link PIT microsimulation model, produces effective tax rates and marginal tax rates by total income (even better is has both labor income and capital income breakdown)
  - PIT model has Python API
  - Microsimulation model with Python API
* - Link PIT microsimulation model, produces effective tax rates and marginal tax rates by total income (even better is has both labor income and capital income breakdown)
  - PIT model has command line interface
  - Microsimulation model that can be executed from a terminal command line
* - Link PIT microsimulation model, produces effective tax rates and marginal tax rates by total income (even better is has both labor income and capital income breakdown)
  - PIT model has another way to interact with it
  - Microsimulation model is in another program like Excel that can be run with an executable or with other software
* - Consumption tax rates
  - Single rate
  - Average consumption tax rates (e.g., time series with total revenue from consumption taxes and time series on GDP/national income)
* - Consumption tax rates
  - Product-specific rates
  - Consumption tax rates by product or industry category
* - Public Pension system (exogenous retirement age)
  - If one of [notional defined contribution, defined benefits, points system, US Social Security]
  - Pension rules based on age, payout, retirement rules, spouse benefits
* - Public Pension system (exogenous retirement age)
  - If pension system not mentioned above
  - Pension rules based on age, payout, retirement rules, spouse benefits
* - Production functions by industry
  - More than one industry
  - Time series of capital and labor demand by industry, output by industry
* - Calibrate METRs, capital cost recovery by industry with Cost of Capital Calculator
  - Gather data on cost recovery policies and business tax system by country
  - Tax code treatment of business income, depreciation
* - Calibrate METRs, capital cost recovery by industry with Cost of Capital Calculator
  - Gather data on value of different types of assets by industry
  - Time series or recent snapshot  of investment or asset holdings by asset type, tax treatment (e.g., corporation, partnership), and industry
* - Calibrate METRs, capital cost recovery by industry with Cost of Capital Calculator
  - Link [`Cost-of-Capital-Calculator`](https://ccc.pslmodels.org/) to OG macro model
  - No additional data requirements
* - Infrastructure
  - As share of gov’t spending and as share of firm production
  - Current government infrastructure spending data plus time series of capital and labor demand by industry, output by industry
```


(SecCalibFootnotes)=
## Footnotes

<!-- [^citation_note]: See {cite}`AuerbachEtAl:1981,AuerbachEtAl:1983`, {cite}`AuerbachKotlikoff:1983a,AuerbachKotlikoff:1983b,AuerbachKotlikoff:1983c`, and {cite}`AuerbachKotlikoff:1985`. -->
