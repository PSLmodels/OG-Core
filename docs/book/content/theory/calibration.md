(Chap_Calib)=
# Calibrating OG-Core

The `OG-Core` model represents all the general model solution code for any overlapping generations model of a country or region. Although `OG-Core` has a `default_parameters.json` file that allows it to run independently, the preferred method for using `OG-Core` is as a dependency to a country calibration repository. We recommend that another repository is made, such as [`OG-USA`](https://github.com/PSLmodels/OG-USA) or [`OG-ZAF`](https://github.com/EAPD-DRB/OG-ZAF/) that uses `OG-Core` as its main computational foundation and engine and calibrates country-specific variables and functions in its own respective source code. This approach results in a working overlapping generations model consisting of a country-specific calibration repository plus a dependency on the general `OG-Core` model logic and options.

{numref}`TabCountryModels` is a list of country-specific calibrations of overlapping generations models that use `OG-Core` as a dependency from oldest to newest. Note that these models are in varying stages of completeness and maturity. It is true that a model is never really fully calibrated. The model maintainer is always updating calibrated values as new data become available. And the model maintainer can always search for better fit and better targeting strategies. As such, the only measures of model maturity of the country calibrations below is the date the repository was created.

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

In the following section, we detail a list of items to calibrate for a country and what types of data and approaches might be available for those calibrations. Each of the country-specific models listed in {numref}`TabCountryModels` will have varying degrees of calibration maturity and further varying degrees of documentation of their calibration. But the following section details all the areas where each of these models should be calibrated.


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
  - Research empirical literature
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


(SecCalibOther)=
## Other parameters to calibrate

(SecCalibOther_InitW)=
### Initial distribution of capital $\Gamma_1$ and aggregate household wealth $B_1$

  One of the initial state parameters in the transition path equilibrium solution algorithm is the initial distribution of wealth held by households $\mathbf{\hat{\Gamma}}_1 \equiv \{b_{j,s,1}\}_{j=1,s=E+1}^{J,E+S}$. Note also that the initial value of aggregate household wealth $\hat{B}_1$ is a function of the initial distribution of capital $\mathbf{\hat{\Gamma}}_1$ and the pre-initial period population distribution $\{\hat{\omega}_{s,0}\}_{s=E+1}^{E+S}$ (see equation {eq}`EqStnrz_Bt`).

  ```{math}
  :label: EqMarkClr_B1
    B_1 \equiv \frac{1}{1 + \tilde{g}_{n,1}}\sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\hat{\omega}_{s-1,0}\lambda_j\hat{b}_{j,s,1} + i_s\hat{\omega}_{s,0}\lambda_j\hat{b}_{j,s,1}\Bigr) \quad\text{where}\quad \frac{1}{1 + \tilde{g}_{n,1}} = \frac{\tilde{N}_0}{\tilde{N}_1}
  ```

  OG-Core has three parameter objects for calibrating this initial distribution of capital $\mathbf{\hat{\Gamma}}_1 \equiv \{b_{j,s,1}\}_{j=1,s=E+1}^{J,E+S}$:
  - `initial_wealth_factor_mat`
  - `use_initial_BY_ratio`
  - `initial_BY_ratio`

  The `initial_wealth_factor_mat` parameter object is an $S \times J$ matrix of factors for which the default is a matrix of ones. This parameter object tells the model what factor of the steady-state distribution of household wealth is the initial distribution of household wealth.

  ```{math}
  :label: EqInitHHwealthDist
    \begin{split}
      &\mathbf{\hat{\Gamma}}_1 = \texttt{initial_wealth_factor_mat}\:\times\:\mathbf{\bar{\Gamma}} \\
      &\Rightarrow\quad \hat{b}_{j,s,1} = \texttt{initial_wealth_factor_mat}_{s,j}\:\times\:\bar{b}_{j,s} \quad\forall j,s
    \end{split}
  ```

  The default parameterization of `initial_wealth_factor_mat` = 1 for all $s$ and $j$ sets the initial distribution of household wealth equal to the steady-state distribution.

  The other two parameters for calibrating the initial distribution of wealth are `use_initial_BY_ratio` and `initial_BY_ratio`. These are used if the modeler wants to shape the initial distribution of capital $\mathbf{\hat{\Gamma}}_1 \equiv \{b_{j,s,1}\}_{j=1,s=E+1}^{J,E+S}$ by targeting the ratio of initial aggregate household wealth $\hat{B}_1$ to GDP $\hat{Y}_1$. Because this is a ratio, the stationarized version is equal to the nonstationarized version.

  {numref}`TabWealthGDPcountries` shows the aggregate household wealth to GDP ratios in seven countries, with values ranging from 1.695 to 5.594. The range of valid values for the `initial_BY_ratio` parameter is 0.8 to 7.0.

  ```{list-table} **Aggregate household wealth as a percent of GDP in select countries**
  :header-rows: 1
  :name: TabWealthGDPcountries
  * - **Country**
    - **Aggr. HH wealth ($B_1$)**
    - **Nominal GDP ($Y_1$)**
    - **Aggr. HH wealth/GDP**
    - **Data year**
  * - United States
    - $163.9 T
    - $29.30 T
    - 559.4%
    - 2024
  * - United Kingdom
    - $18.06 T
    - $3.70 T
    - 488.1%
    - 2024
  * - India
    - $16.01 T
    - $3.76 T
    - 425.8%
    - 2024
  * - Indonesia
    - $3.59 T
    - $1.40 T
    - 256.4%
    - 2024
  * - South Africa
    - $1.03 T
    - $0.401 T
    - 256.2%
    - 2024
  * - Philippines
    - $1.011 T
    - $0.404 T
    - 250.2%
    - 2022
  * - Ethiopia
    - $0.300 T
    - $0.177 T
    - 169.5%
    - 2022
  ```

  If the modeler wants to calibrate the initial distribution of wealth $\mathbf{\hat{\Gamma}}_1$ to target the initial aggregate household wealth to GDP ration $\hat{B}_1/\hat{Y}_1$, he simply chooses the values for `initial_wealth_factor_mat`, which sets the shape of the initial distribution of wealth, then chooses `use_initial_BY_ratio=True` and sets a target value for `initial_BY_ratio`. The default value for `initial_BY_ratio` is the USA value of 5.594 from the table above. But this value is only used if `use_initial_BY_ratio=True`. Because the default value of `use_initial_BY_ratio=False`, the value of `initial_BY_ratio` is not used in this default case.

  The reason we need the boolean flag parameter `use_initial_BY_ratio` is because the denominator $\hat{Y}_1$ of the wealth-to-GDP target is endogenous. We don't know the equilibrium value of GDP in the initial period. So in the `TPI.py` solution algorithm, we choose $\hat{B}_1$ in each iteration that sets the initial aggregate household wealth to GDP ratio equal to its target. This is done by shifting the initial distribution of household wealth $\mathbf{\hat{\Gamma}}_1$ up or down by a constant factor.


<!--(SecCalibFootnotes)=
## Footnotes

[^citation_note]: See {cite}`AuerbachEtAl:1981,AuerbachEtAl:1983`, {cite}`AuerbachKotlikoff:1983a,AuerbachKotlikoff:1983b,AuerbachKotlikoff:1983c`, and {cite}`AuerbachKotlikoff:1985`. -->
