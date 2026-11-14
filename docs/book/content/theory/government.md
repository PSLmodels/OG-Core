
(Chap_UnbalGBC)=
# Government

In `OG-Core`, the government enters by levying taxes on households, providing transfers to households, levying taxes on firms, spending resources on public goods and infrastructure, and making rule-based adjustments to stabilize the economy in the long-run. It is this last activity that is the focus of this chapter.


(SecUnbalGBC_policy)=
## Government Tax and Transfer Policy

Government levies taxes on households and firms, funds public pensions, and makes other transfers to households.

### Taxes

#### Individual income taxes

Income taxes are modeled through the net tax liability function for each household $tax_{j,s,t}$, which can be decomposed into the effective tax rates times total income and wealth, respectively, {eq}`EqTaxCalcLiabETR2`. In this chapter, we detail the household tax component of government activity $tax_{j,s,t}$ in `OG-Core`, along with our method of incorporating detailed microsimulation data into a dynamic general equilibrium model.

```{math}
:label: EqHHBC2_gov
\begin{aligned}
  p_t c_{j,s,t} + &\sum_{i=1}^I (1 + \tau^{c}_{i,t})p_{i,t}c_{min,i} + b_{j,s+1,t+1} = \\
  &(1 + r_{p,t})b_{j,s,t} + w_t e_{j,s} n_{j,s,t} ... \\
  &\qquad +\: bq_{j,s,t} + rm_{j,s,t} + tr_{j,s,t} + ubi_{j,s,t} + pension_{j,s,t} - tax_{j,s,t}  \\
  &\qquad\quad\forall j,t\quad\text{and}\quad E+1\leq s\leq E+S \quad\text{where}\quad b_{j,E+1,t}=0
\end{aligned}
```

Incorporating realistic tax and incentive detail into a general equilibrium model is notoriously difficult for two reasons. First, it is impossible in a dynamic general equilibrium model to capture all of the dimensions of heterogeneity on which the real-world tax rate depends. For example, a household's tax liability in reality depends on filing status, number of dependents, many types of income, and some characteristics correlated with age. A good heterogeneous agent DGE model tries to capture the most important dimensions of heterogeneity, and necessarily neglects the other dimensions.

The second difficulty in modeling realistic tax and incentive detail is the need for good microeconomic data on the individuals who make up the economy from which to simulate behavioral responses and corresponding tax liabilities and tax rates.

`OG-Core` follows the method of {cite}`DeBackerEtAl:2019` of generating detailed tax data on effective tax rates and marginal tax rates for a sample of tax filers along with their respective income and demographic characteristics and then using that data to estimate parametric tax functions that can be incorporated into `OG-Core`.

(SecTaxCalcRateTheory)=
##### Effective and Marginal Tax Rates

  Before going into more detail regarding how we handle these two difficulties in `OG-Core`, we need to define some functions and make some notation. For notational simplicity, we will use the variable $x$ to summarize labor income, and we will use the variable $y$ to summarize capital income.

  ```{math}
  :label: EqTaxCalcLabInc
    x_{j,s,t} \equiv w_{t}e_{j,s}n_{j,s,t} \quad\forall j, t \quad\text{and}\quad E+1\leq s\leq E+S
  ```
  ```{math}
  :label: EqTaxCalcCapInc
    y_{j,s,t} \equiv r_{p,t}b_{j,s,t} \quad\forall j, t \quad\text{and}\quad E+1\leq s\leq E+S
  ```

  We can express household net tax liability $tax_{j,s,t}$ from the household budget constraint {eq}`EqHHBC` as an effective tax rate on income multiplied by total income plus an effective tax rate on wealth multiplied by wealth $b_{j,s,t}$,

  ```{math}
  :label: EqTaxCalcLiabETR2
  \begin{aligned}
    tax_{j,s,t} &= tax^{x+y}_{j,s,t} + tax^w_{j,s,t} \\
    &=\tau^{etr,xy}_{s,t}(x_{j,s,t}, y_{j,s,t})\left(x_{j,s,t} + y_{j,s,t}\right) + \tau^{etr,w}_{t}\bigl(b_{j,s,t}\bigr)b_{j,s,t}
  \end{aligned}
  ```

  where the wealth tax function is described in Section {ref}`SecGovWealthTax` of this chapter. Rearranging the first term on the right-hand-side of {eq}`EqTaxCalcLiabETR2` gives the definition of an effective tax rate on income ($\tau_{s,t}^{etr,xy}$) as total tax liability, minus the wealth tax liability, divided by unadjusted gross income. And the definition of the effective tax rate on wealth ($\tau_{t}^{etr,w}$) as total tax liability, minus the income tax liability, divided by wealth.

  A marginal tax rate ($\tau_{s,t}^{mtr}$) is defined as the change in total tax liability from a small change in either income or wealth. We allow these functions to vary by age $s$ and time $t$. In `OG-Core`, we differentiate between the marginal tax rate on labor income ($\tau_{s,t}^{mtrx}$), the marginal tax rate on capital income ($\tau_{s,t}^{mtry}$), and the marginal tax rate on wealth ($\tau_{t}^{mtrw}$).

  ```{math}
  :label: EqTaxCalcMTRx
    \tau^{mtrx} \equiv \frac{\partial tax^{x+y}_{j,s,t}}{\partial w_t e_{j,s}n_{j,s,t}} = \frac{\partial tax^{x+y}_{j,s,t}}{\partial x_{j,s,t}} \quad\forall j,t \quad\text{and}\quad E+1\leq s\leq E+S
  ```
  ```{math}
  :label: EqTaxCalcMTRy
    \tau^{mtry} \equiv \frac{\partial tax^{x+y}_{j,s,t}}{\partial r_{p,t}b_{j,s,t}} = \frac{\partial tax^{x+y}_{j,s,t}}{\partial y_{j,s,t}} \qquad\quad\forall j,t \quad\text{and}\quad E+1\leq s\leq E+S
  ```

  ```{math}
  :label: EqTaxCalcMTRw
    \tau^{mtrw} \equiv \frac{\partial tax^w_{j,s,t}}{\partial b_{j,s,t}} \qquad\qquad\qquad\quad\forall j,t \quad\text{and}\quad E+1\leq s\leq E+S
  ```

  Note that a change in wealth $b_{j,s,t}$ changes both income tax liability (through capital income) and wealth tax liability. However, it is both intuitively and computationally convenient to defined the marginal tax rate on capital income and the marginal tax rate on wealth to be separate, as we have done here.

  As we show in Section [Optimality Conditions](https://pslmodels.github.io/OG-Core/content/theory/households.html#optimality-conditions) of the Households chapter of the `OG-Core` repository documentation, the derivative of total tax liability with respect to labor supply $\frac{\partial tax_{j,s,t}}{n_{j,s,t}}$ and the derivative of total tax liability next period with respect to savings $\frac{\partial tax_{j,s+1,t+1}}{b_{j,s+1,t+1}}$ show up in the household Euler equations for labor supply and savings , respectively, in the `OG-Core` documentation. It is valuable to be able to express those marginal tax rates, for which we have no data, as marginal tax rates for which we do have data. The following two expressions show how the marginal tax rates of labor supply can be expressed as the marginal tax rate on labor income times the household-specific wage and how the marginal tax rate of savings can be expressed as the marginal tax rate of capital income times the interest rate.

  ```{math}
  :label: EqMTRx_derive
    \frac{\partial tax_{j,s,t}}{\partial n_{j,s,t}}  = \frac{\partial tax^{x+y}_{j,s,t}}{\partial w_t e_{j,s}n_{j,s,t}}\frac{\partial w_{t}e_{j,s}n_{j,s,t}}{\partial n_{j,s,t}} = \frac{\partial tax^{x+y}_{j,s,t}}{\partial w_{t}e_{j,s}n_{j,s,t}}w_t e_{j,s} = \tau^{mtrx}_{s,t}w_t e_{j,s}
  ```

  ```{math}
  :label: EqMTRy_derive
    \frac{\partial tax_{j,s,t}}{\partial b_{j,s,t}} = \frac{\partial tax^{x+y}_{j,s,t}}{\partial r_{p,t}b_{j,s,t}}\frac{\partial r_{p,t}b_{j,s,t}}{\partial b_{j,s,t}} + \frac{\partial tax^w_{j,s,t}}{b_{j,s,t}}  = \tau^{mtry}_{s,t}r_{p,t} + \tau^{mtrw}_{t}
  ```


(SecTaxCalcFuncs)=
##### Fitting Tax Functions

  In looking at the 2D scatter plot on effective tax rates as a function of total income in {numref}`Figure %s <FigTaxCalcETRtotinc>` and the 3D scatter plots of $ETR$, $MTRx$, and $MTRy$ in {numref}`Figure %s <FigTaxCalc3DvsPred>`, it is clear that all of these rates exhibit negative exponential or logistic shape. This empirical regularity allows us to make an important and nonrestrictive assumption. We can fit parametric tax rate functions to these data that are constrained to be monotonically increasing in labor income and capital income. This assumption of monotonicity is computationally important as it preserves a convex budget set for each household, which is important for being able to solve many household lifetime problems over a large number of periods.


(SecTaxCalcFuncs_DEP)=
###### Default Tax Functional Form

  For the default option, `OG-Core` follows the approach of {cite}`DeBackerEtAl:2019` in using the following functional form to estimate tax functions for each age $s=E+1, E+2, ... E+S$ in each time period $t$. This option can be manually selected by setting the parameter `tax_func_type="DEP"`. Alternative specifications are outlined in Section {ref}`SecTaxCalcFuncs_Alt` below. Equation {eq}`EqTaxCalcTaxFuncForm` is written as a generic tax rate, but we use this same functional form for $ETR$'s, $MTRx$'s, and $MTRy$'s.
  ```{math}
  :label: EqTaxCalcTaxFuncForm
  \begin{aligned}
    \tau(x,y) = &\Bigl[\tau(x) + shift_x\Bigr]^\phi\Bigl[\tau(y) + shift_y\Bigr]^{1-\phi} + shift \\
    &\text{where}\quad \tau(x) \equiv (max_x - min_x)\left(\frac{Ax^2 + Bx}{Ax^2 + Bx + 1}\right) + min_x \\
    &\quad\text{and}\quad \tau(y) \equiv (max_y - min_y)\left(\frac{Cy^2 + Dy}{Cy^2 + Dy + 1}\right) + min_y \\
    &\text{where}\quad A,B,C,D,max_x,max_y,shift_x,shift_y > 0 \quad\text{and}\quad\phi\in[0,1] \\
    &\quad\text{and}\quad max_x > min_x \quad\text{and}\quad max_y > min_y
  \end{aligned}
  ```

  The parameters values will, in general, differ across the different functions (effective and marginal rate functions) and by age, $s$, and tax year, $t$.  We drop the subscripts for age and year from the above exposition for clarity.

  By assuming each tax function takes the same form, we are breaking the analytical link between the the effective tax rate function and the marginal rate functions.  In particular, one could assume an effective tax rate function and then use the analytical derivative of that to find the marginal tax rate function.  However, we've found it useful to separately estimate the marginal and average rate functions.  One reason is that we want the tax functions to be able to capture policy changes that have differential effects on marginal and average rates.  For example, a change in the standard deduction for tax payers would have a direct effect on their average tax rates.  But it will have secondary effect on marginal rates as well, as some filers will find themselves in different tax brackets after the policy change. These are smaller and second order effects. When tax functions are are fit to the new policy, in this case a lower standard deduction, we want them to be able to represent this differential impact on the marginal and average tax rates. The second reason is related to the first. As the additional flexibility allows us to model specific aspects of tax policy more closely, it also allows us to better fit the parameterized tax functions to the data.

  The key building blocks of the functional form Equation {eq}`EqTaxCalcTaxFuncForm` are the $\tau(x)$ and $\tau(y)$ univariate functions. The ratio of polynomials in the $\tau(x)$ function $\frac{Ax^2 + Bx}{Ax^2 + Bx + 1}$ with positive coefficients $A,B>0$ and positive support for labor income $x>0$ creates a negative-exponential-shaped function that is bounded between 0 and 1, and the curvature is governed by the ratio of quadratic polynomials. The multiplicative scalar term $(max_x-min_x)$ on the ratio of polynomials and the addition of $min_x$ at the end of $\tau(x)$ expands the range of the univariate negative-exponential-shaped function to $\tau(x)\in[min_x, max_x]$. The $\tau(y)$ function is an analogous univariate negative-exponential-shaped function in capital income $y$, such that $\tau(y)\in[min_y,max_y]$.

  The respective $shift_x$ and $shift_y$ parameters in Equation {eq}`EqTaxCalcTaxFuncForm` are analogous to the additive constants in a Stone-Geary utility function. These constants ensure that the two sums $\tau(x) + shift_x$ and $\tau(y) + shift_y$ are both strictly positive. They allow for negative tax rates in the $\tau(\cdot)$ functions despite the requirement that the arguments inside the brackets be strictly positive. The general $shift$ parameter outside of the Cobb-Douglas brackets can then shift the tax rate function so that it can accommodate negative tax rates. The Cobb-Douglas share parameter $\phi\in[0,1]$ controls the shape of the function between the two univariate functions $\tau(x)$ and $\tau(y)$.

  This functional form for tax rates delivers flexible parametric functions that can fit the tax rate data shown in the scatterplot data in {numref}`Figure %s <FigTaxCalcETRtotinc>` and {numref}`Figure %s <FigTaxCalc3DvsPred>` as well as a wide variety of policy reforms. Further, these functional forms are monotonically increasing in both labor income $x$ and capital income $y$. This characteristic of monotonicity in $x$ and $y$ is essential for guaranteeing convex budget sets and thus uniqueness of solutions to the household Euler equations. The assumption of monotonicity does not appear to be a strong one when viewing the the tax rate data shown in {numref}`Figure %s <FigTaxCalc3DvsPred>`. While it does limit the potential tax systems to which one could apply our methodology, tax policies that do not satisfy this assumption would result in non-convex budget sets and thus require non-standard DGE model solutions methods and would not guarantee a unique equilibrium. The 12 parameters of our tax rate functional form from {eq}`EqTaxCalcTaxFuncForm` are summarized in {numref}`TabTaxCalcTfuncParams`.

  ```{list-table} **Description of tax rate function $\tau(x,y)$ parameters.**
  :header-rows: 1
  :name: TabTaxCalcTfuncParams
  * - Symbol
    - Description
  * - $A$
    - Coefficient on squared labor income term $x^2$ in $\tau(x)$
  * - $B$
    - Coefficient on labor income term $x$ in $\tau(x)$
  * - $C$
    - Coefficient on squared capital income term $y^2$ in $\tau(y)$
  * - $D$
    - Coefficient on capital income term *y* in $\tau(y)$
  * - $max_{x}$
    - Maximum tax rate on labor income $x$ given $y = 0$
  * - $min_{x}$
    - Minimum tax rate on labor income $x$ given $y = 0$
  * - $max_{y}$
    - Maximum tax rate on capital income $y$ given $x = 0$
  * - $min_{y}$
    - Minimum tax rate on capital income $y$ given $x = 0$
  * - $shift_{x}$
    - shifter $>\,\|min_{x}\|$ ensures that $\tau(x,y)$ + $shift_{x}\,\geq\,0$ despite potentially negative values for $\tau(x)$
  * - $shift_{y}$
    - shifter $>\,\|min_{y}\|$ ensures that $\tau(x,y)$ + $shift_{y}\,\geq\,0$ despite potentially negative values for $\tau(y)$
  * - $shift$
    - shifter (can be negative) allows for support of $\tau(x,y)$ to include negative tax rates
  * - $\phi$
    - Cobb-Douglas share parameter between 0 and 1
  ```

  ```{figure} ./images/Compare_ETR_functions.png
  ---
  height: 500px
  name: FigTaxCalcETRtotinc
  ---
  Plot of estimated $ETR$ functions: $t=2017$ and $s=43$ under current law
  ```

  ```{figure} ./images/Age42_2017_vsPred.png
  ---
  height: 500px
  name: FigTaxCalc3DvsPred
  ---
  Estimated tax rate functions of ETR, MTRx, MTRy, and histogram as functions of labor income and capital income from microsimulation model: $t=2017$ and $s=42$ under 2017 law in the United States. Note: Axes in the histogram in the lower-right panel have been switched relative to the other three figures in order to see the distribution.
  ```

  ```{list-table} **Estimated baseline current law tax rate function for $\tau_{s,t}(x,y)$ parameters for $s=42$, $t=2017$.**
  :header-rows: 1
  :name: TabTaxCalcEst42
  * - Parameter
    - $ETR$
    - $MTRx$
    - $MTRy$
  * - $A$
    - 6.28E-12
    - 3.43E-23
    - 4.32E-11
  * - $B$
    - 4.36E-05
    - 4.50E-04
    - 5.52E-05
  * - $C$
    - 1.04E-23
    - 9.81E-12
    - 5.62E-12
  * - $D$
    - 7.77E-09
    - 5.30E-08
    - 3.09E-06
  * - $max_{x}$
    - 0.80
    - 0.71
    - 0.44
  * - $min_{x}$
    - -0.14
    - -0.17
    - 0.00E+00
  * - $max_{y}$
    - 0.80
    - 0.80
    - 0.13
  * - $min_{y}$
    - -0.15
    - -0.42
    - 0.00E+00
  * - $shift_{x}$
    - 0.15
    - 0.18
    - 4.45E-03
  * - $shift_{y}$
    - 0.16
    - 0.43
    - 1.34E-03
  * - $shift$
    - -0.15
    - -0.42
    - 0.00E+00
  * - $\phi$
    - 0.84
    - 0.96
    - 0.86
  * - Obs. ($N$)
    - 3,105
    - 3,105
    - 1,990
  * - SSE
    - 9,122.68
    - 15,041.35
    - 7,756.54
  ```

  Let $\boldsymbol{\theta}_{s,t}=(A,B,C,D,max_x,min_x,max_y,min_y,shift_x,shift_y,shift,\phi)$ be the full vector of 12 parameters of the tax function for a particular type of tax rate, age of filers, and year. We first directly specify $min_x$ as the minimum tax rate and $max_x$ as the maximum tax rate in the data for age-$s$ and period-$t$ individuals for capital income close to 0 ($\$0<y<\$3,000$), and $min_y$ as the minimum tax rate and $max_y$ as the maximum tax rate for labor income close to 0 ($\$0<x<\$3,000$). We then set $shift_x = \min(0,|min_x|)+\epsilon$ and $shift_y = \min(0,|min_y|)+\epsilon$ so that the respective arguments in the brackets of {eq}`EqTaxCalcTaxFuncForm` are strictly positive. Then let $shift$ be be the minimum tax rate in the corresponding data minus $\epsilon$. Let $\bar{\boldsymbol{\theta}}_{s,t}=\{min_x,max_x,min_y,max_y,shift_x,shift_y, shift\}$ be the set of parameters we take directly from the data in this way.

  We then estimate five remaining parameters $\tilde{\boldsymbol{\theta}}_{s,t}=(A,B,C,D,\phi)$ using the following nonlinear weighted least squares criterion,

  ```{math}
  :label: EqTaxCalcThetaWSSQ
  \begin{aligned}
      \boldsymbol{\hat{\theta}}_{s,t} = \tilde{\boldsymbol{\theta}}_{s,t}:\quad &\min_{\tilde{\boldsymbol{\theta}}_{s,t}}\sum_{i=1}^{N} \Bigl[\tau_{i}-\tau_{s,t}\bigl(x_i,y_i|\tilde{\boldsymbol{\theta}}_{s,t},\bar{\boldsymbol{\theta}}_{s,t}\bigr)\Bigr]^{2} w_i, \\
      &\qquad\text{subject to}\quad A, B, C, D > 0 \quad\text{and}\quad \phi\in[0,1]
  \end{aligned}
  ```

  where $\tau_{i}$ is the tax rate for observation $i$ from the microsimulation output, $\tau_{s,t}(x_i,y_i|\tilde{\boldsymbol{\theta}}_{s,t},\bar{\boldsymbol{\theta}}_{s,t})$ is the predicted tax rate for filing-unit $i$ with $x_{i}$ labor income and $y_{i}$ capital income given parameters $\boldsymbol{\theta}_{s,t}$, and $w_{i}$ is the CPS sampling weight of this observation. The number $N$ is the total number of observations from the microsimulation output for age $s$ and year $t$. {numref}`Figure %s <FigTaxCalc3DvsPred>` shows the typical fit of an estimated tax function $\tau_{s,t}\bigl(x,y|\hat{\boldsymbol{\theta}}_{s,t}\bigr)$ to the data for $s=42$ and year $t=2017$.

  The underlying data can limit the number of tax functions that can be estimated. For example, we use the age of the primary filer from the PUF-CPS match to be equivalent to the age of the DGE model household. The DGE model we use allows for individuals up to age 100, however the data contain few primary filers with age above age 80. Because we cannot reliably estimate tax functions for $s>80$, we apply the tax function estimates for 80 year-olds to those with model ages 81 to 100. In the case certain ages below age 80 have too few observations to enable precise estimation of the model parameters, we use a linear interpolation method to find the values for those ages $21\leq s <80$ that cannot be precisely estimated. [^interpolation_note]

  In `OG-Core`, we estimate the 12-parameter functional form {eq}`EqTaxCalcTaxFuncForm` using weighted nonlinear least squares to fit an effective tax rate function $(\tau^{etr}_{s,t})$, a marginal tax rate of labor income function $(\tau^{mtrx}_{s,t})$, and a marginal tax rate of capital income function $(\tau^{mtry}_{s,t})$ for each age $E+1\leq s\leq E+S$ and each of the first 10 years from the current period. [^param_note] That means we have to perform 2,400 estimations of 12 parameters each. {numref}`Figure %s <FigTaxCalc3DvsPred>` shows the predicted surfaces for $\tau^{etr}_{s=42,t=2017}$, $\tau^{mtrx}_{s=42,t=2017}$, and $\tau^{mtry}_{s=42,t=2017}$ along with the underlying scatter plot data from which those functions were estimated. {numref}`TabTaxCalcEst42` shows the estimated values of those functional forms.


(SecTaxCalcFuncs_Alt)=
###### Alternative Functional Forms

  In addition to the default option using tax functions of the form developed by {cite}`DeBackerEtAl:2019`, `OG-Core` also allows users to specify alternative tax functions. Four alternatives are offered:

  1. Functions as in {cite}`DeBackerEtAl:2019`, but where $\tau^{etr}_{s,t}$, $\tau^{mtrx}_{s,t}$, and $\tau^{mtry}_{s,t}$ are functions of total income (i.e., $x+y$) and not labor and capital income separately.  Users can select this option by setting the parameter `tax_func_type="DEP_totalinc"`.

  2. Functions of the Gouveia and Strauss form {cite}`GouveiaStrauss:1994`:

     ```{math}
         \tau = \phi_{0}(1 - (x+y)^{(\phi_{1}-1)}((x+y)^{-\phi_{1}} + \phi_{2})^{(-1 - \phi_{1})/\phi_{1}})
     ```

     Users can select this option by setting the parameter `tax_func_type="GS"`.  The three parameters of this function ($\phi_{0}, \phi_{1}, \phi_{2}$) are estimated using the weighted sum of squares estimated described in Equation {eq}`EqTaxCalcThetaWSSQ`.

  3. Tax functions from {cite}`HeathcoteStoreslettenViolante:2017`:

     ```{math}
       \tau = 1 - \phi_{0}(x+y)^{1-phi_1}
     ```

     Users can select this option by setting the parameter `tax_func_type="HSV"`.  The two parameters of this function ($\phi_{0}, \phi_{1}$) are estimated using ordinary least squares.

  4. Linear tax functions (i.e., $\tau =$ a constant).  Users can select this option by setting the parameter `tax_func_type="linear"`.  The constant rate is found by taking the weighted average of the appropriate tax rate (effective tax rate, marginal tax rate on labor income, marginal tax rate on labor income) for each age and year, where the values are weighted by sampling weights and income.

  5. Monotonically increasing smoothing spline `tax_func_type="mono"`. This functional approach is based on {cite}`EilersMarx:1996`, {cite}`Eilers:2003` and {cite}`Eilers:2005`, which is a least squares smoothing spline that penalizes nonmonotonicities. We used k-fold cross validation to show that the optimal smoothing parameter for this function on our tax data is $\lambda=12$, which is set as the default. These tax functions are fit to total income data (labor income plus capital income).

  Among all of these tax functional forms, users can set the `age_specific` parameter to `False` if they wish to have one function for all ages $s$.  In addition, for the functions based on {cite}`DeBackerEtAl:2019` (`tax_func_type="DEP"` or `tax_func_type="DEP_totinc"`), one can set `analytical_mtrs=True` if they wish to have the $\tau^{mtrx}_{s,t}$ and $\tau^{mtry}_{s,t}$ derived from the $\tau^{etr}_{s,t}$ functions.  This provides theoretical consistency, but reduced fit of the functions (see {cite}`DeBackerEtAl:2019` for more details).

(SecTaxCompliance)=
###### Income Tax Noncompliance

The parameters of the tax functions used in `OG-Core` are typically estimated using data that miss the effects of compliance (either because they use administrative data from tax returns or because of similar under-reporting in survey data {cite}`HurstEtAl:2014`(see Hurst et al. (REStat, 2014))).  `OG-Core` therefore offers the user two parameters that scale the marginal and effective tax rate functions in order to account for income tax noncompliance.  These parameters are `labor_income_tax_noncompliance_rate` and `capital_income_tax_noncompliance_rate` and represent the rates of noncompliance on the respective sources of income.  The implementation of these parameters is as follows.  Let represent $\eta$ the noncompliance rate (on labor and capital income) and let $\tau$ represent the marginal tax rate (on labor or capital income) without accounting for noncompliance.  Then, the marginal tax rate faced by a household, after accounting for compliance decisions, is $\tau^{nc} = (1 - \eta)\tau$.

When computing the role of compliance on the effective tax rate, we take a weighted average over the rates of noncompliance for labor and capital income.  If we let $X$ represent labor income, $Y$ capital income, and $\eta_X$ and $\eta_Y$ represent the noncompliance rates on labor and capital income, respecitvely.  Then the noncompliance rate applied to the effective tax rate is given by $\eta = \frac{\eta_X X + \eta_Y Y}{X + Y}$.  If we use $\tau$ to represent the effective tax rate, then the effective tax rate used in the households budget constraint and to determine tax liability is $\tau^{nc} = (1 - \eta)\tau$, where the $\eta$ represents the average rate of noncompliance across income sources.

(SecTaxCalcFactor)=
###### Factor Transforming Income Units

  The tax functions $\tau^{etr}_{s,t}$, $\tau^{mtrx}_{s,t}$, and $\tau^{mtry}_{s,t}$ are typcically estimated on data with income in current currency units. However, the consumption units of the `OG-Core` model are not in the same units as the real-world income data. For this reason, we have to transform the income by a $factor$ so that it is in the same units as the income data on which the tax functions were estimated.

  The tax rate functions are each functions of capital income and labor income $\tau(x,y)$. In order to make the tax functions return accurate tax rates associated with the correct levels of income, we multiply the model income $x^m$ and $y^m$ by a $factor$ so that they are in the same units as the real-world U.S. income data $\tau(factor\times x^m, factor\times y^m)$. We define the $factor$ such that average steady-state household total income in the model times the $factor$ equals the U.S. data average total income.

  ```{math}
  :label: EqTaxCalcFactor
    factor\Biggl[\sum_{s=E+1}^{E+S}\sum_{j=1}^J\lambda_j\bar{\omega}_s\left(\bar{w}e_{j,s}\bar{n}_{j,s} + \bar{r}_{p}\bar{b}_{j,s}\right)\Biggr] = \text{Avg. household income in data}
  ```

  We do not know the steady-state wage, interest rate, household labor supply, and savings *ex ante*. So the income $factor$ is an endogenous variable in the steady-state equilibrium computational solution. We hold the factor constant throughout the nonsteady-state equilibrium solution.


#### Consumption taxes

Linear consumption taxes, $\tau^c_{i,t}$ can vary over time and by consumption good.


(SecGovWealthTax)=
#### Wealth taxes

  Wealth taxes can be implemented through the $tax^{w}_{j,s,t}(b_{j,s,t})$ function, shown in equations {eq}`EqTaxCalcLiabETR2` through {eq}`EqMTRy_derive`. This functional form allows for zero to flat to progressive wealth taxation and is given by the following,

  ```{math}
  :label: WealthTaxFunc
    tax^{w}_{j,s,t} = p^w\left(\frac{h^{w}b_{j,s,t}}{h^{w}b_{j,s,t} + m^{w}}\right)b_{j,s,t}
  ```

  where $p^w\geq 0$ is a nonnegative scale parameter of the overall tax rate, $h^w> 0$ is a strictly positive scale coefficient parameter on the linear term inside of the parentheses, and $m^w\geq 0$ is a nonnegative constant additive coefficient in the denominator of the rate function in parentheses. This functional form allows us to represent a zero wealth tax rate ($p^w=0$), a flat wealth tax rate ($p^w>0$ and $m_w= 0$), and a progressive wealth tax rate ($p^w$, $h^w$, and $m^w$ > 0).

  The expression for the effective tax rate on wealth is the following.

  ```{math}
  :label: EqETRwealth
    \tau^{etr,w}_{t} = \frac{tax^{w}_{j,s,t}}{b_{j,s,t}} = p^w\left(\frac{h^{w}b_{j,s,t}}{h^{w}b_{j,s,t} + m^{w}}\right) \quad\forall j,s,t
  ```

  The analytical expression for the marginal tax rate on wealth defined in equation {eq}`` is the following.

  ```{math}
  :label: EqMTRwealth
    \tau^{mtrw}_{t} = \frac{\partial tax^{w}_{j,s,t}}{\partial b_{j,s,t}} = \tau^{etr,w}_t\left[2 - \left(\frac{h^w b_{j,s,t}}{h^w b_{j,s,t} + m^w}\right)\right] \quad\forall j,s,t
  ```


#### Corporate income taxes

Businesses face a linear tax rate $\tau^{corp}_{m,t}$, which can vary by industry and over time.  In the case of a single industry, `OG-Core` provides the parameters `c_corp_share_of_assets` to scale the tax rate applied to the representative firm so that it represents a weighted average between the rate on businesses entities taxes at the entity level (e.g., C corporations in the United States) and those with no entity level tax.  The parameter `adjustment_factor_for_cit_receipts` is additionally provided to represent a wedge between marginal and average tax rates (which could otherwise be zero with a linear tax function).

### Spending

  Government spending is comprised of government provided pension benefits, lump sum transfers, universal basic income payments, infrastructure investment, spending on public goods, and interest payments on debt.  Below, we describe the transfer spending amounts.  Spending on infrastructure, public goods, and interest are described in {ref}`SecUnbalGBCbudgConstr`. Because government spending on lump-sum transfers to households $TR_t$, public goods $G_t$, and government infrastructure capital $I_g$ are all functions of nominal GDP, we define GDP here,

  ```{math}
  :label: EqGovtNomGDP
    Y_t \equiv \sum_{m=1}^M p_{m,t} Y_{m,t} \quad\forall t
  ```

  where GDP $Y_t$ is in terms of the numeraire good of industry-$M$ output.[^NomRealGDP]


(SecGovPensions)=
#### Pensions

The `OG-Core` model allows for five different systems for public pensions:

1. U.S.-style social security system
2. Defined benefit system
3. Notional defined contribution system
4. Points system
5. Hybrid defined benefit and notional defined contribution system

These can be selected with the `pension_system` parameter.  Accepted values are `US-Style Social Security`, `Defined Benefits`, `Notional Defined Contribution`, `Points System`, `Hybrid DB-NDC`.  We discuss each of these in turn below.

For all systems, $R$ represents the age at which the individual becomes eligible to receive the government provided retirement benefit. The retirement age is a property of a birth cohort rather than of a calendar year: a household born in period $t-s$ (of age $s$ in period $t$) faces the retirement age $R_{t-s}$, and it faces the same $R_{t-s}$ in every period of its life unless a reform moves its cohort to a new value. The base value is the parameter `retirement_age`. Changes are entered through `pension_regimes`, a list of reforms applied in order, each with a calendar `year` from which it applies, an optional `max_age_at_start` (the oldest age, in that calendar year, to which the reform applies; cohorts not yet born always qualify, and leaving it out applies the reform to everyone including current retirees), and the new parameter values. This is how a phased-in increase in the retirement age that exempts those near retirement is expressed. The model expands these inputs into an array over periods and ages, so the household problem, the aggregates, and the steady state (which uses the final period's value) all read the same cohort-specific retirement age. The parameters of the defined benefit system (`alpha_db`, `avg_earn_num_years`, `yr_contrib`) and of the notional defined contribution system (`tau_p`, `k_ret`, `indR`) can be changed by cohort in the same way, so a reform of the benefit formula can also be phased in by birth cohort. With no regimes, the base `alpha_db` is a path by lifetime income group and the calendar year in which a benefit is paid, and the other parameters are single values. The replacement-rate adjustment `replacement_rate_adjust` is, by contrast, always indexed by the calendar year in which a benefit is paid.

##### US-style social security system

Under the U.S.-style social security system, households over age $R$ received a pension amount that is a function of their earnings history.  The earnings history includes the highest earning `avg_earn_num_years` prior to retirement (which OG-Core assumes happens at age $R$).  This history determines the Average Indexed Monthly Earnings (AIME):

  ```{math}
  :label: eqn:AIME
  AIME_{j,R,t+R} = {\sum_{u=0}^{AIMEper} w_{t+u}e_{j,u,t+u}n_{j,u,t+u}}{ 12 * AIMEper}
  ```

 The AIME in turn, determines a household's (PIA) based on three rates and brackets:

   ```{math}
   :label: eqn:PIA
    PIAbase_{j,R,t+R} =
      \begin{cases}
        PIArate_1 \times AIME_{j,R,t+R}, \text{for } AIME_{j,R,t+R} \leq AIMEbkt_1 \\
        PIArate_1 \times AIMEbkt_1 + PIArate_2 \times (AIME_{j,R,t+R} - AIMEbkt_1), \text{for } AIMEbkt_1 < AIME_{j,R,t+R} \leq AIMEbkt_2 \\
        PIArate_1 \times AIMEbkt_1 + PIArate_2 \times (AIMEbkt_2 - AIMEbkt_1) + PIArate_3 \times (AIME_{j,R,t+R} - AIMEbkt_2), \text{for } AIMEbkt_2 < AIME_{j,R,t+R} \\
      \end{cases}
  ```

 The PIA is then capped at a maximum, set by the parameter `PIA_maxpayment`, and, when the parameter `PIA_minpayment` is nonzero, floored at that minimum: $PIA_{j,R,t+R} = \max\{\min\{PIAbase_{j,R,t+R}, \text{ PIA max payment amount}\}, \text{ PIA min payment amount}\}$.

  The replacement rate, $\theta_j$ is then calculated as annual earnings, with an adjustment for the wage rate.:

  ```{math}
  :label: eqn:theta
    \theta_{j,R,t+R} = \frac{PIA_{j,R,t+R} \times 12}{factor \times w_{t+R}}
  ```

  Note that $AIME_{j,R,t+R}$ is a function of each households' earning history, but their choice of earning may depend on their retirement benefit.  Solving this exactly would introduce and additional fixed point problem in both the steady state solution and in the time path solution.  The latter would be extremely computationally taxing.  Therefore, we make the simplification of determining the AIME for each type $j$ using the steady state solution and earnings for type $j$ in the steady state.  This leads to some approximation error, but because $\theta_j$ is adjusted by the current wage rate, this approximation error is minized.

  The pension amount for households under the US-style social security system is then:

  ```{math}
  :label: eqn:ss_pension
    pension_{j,s,t} = \theta_j \times w_t \quad \forall s \geq R_{t-s}
  ```


##### Defined benefit system

The defined benefit system pension amount is given as:

  ```{math}
  :label: eqn:db_pension
  pension_{j,s,t} = \biggl[\frac{\sum_{u=R-ny}^{R-1}w_{t-s+u}e_{j,u,t-s+u}n_{j,u,t-s+u}}{ny}\biggr]\times Cy \times \alpha^{DB}_{j,s,t} \quad \forall s \geq R
  ```

where the sum runs over the household's own earnings history (the wage, effective labor units, and labor supply at each working age $u$ were those of period $t-s+u$), and:
  - $ny$ are the number of years over which average earnings are calculated (`avg_earn_num_years`)
  - $Cy$ are the number of years of contributions (`yr_contrib`).  In our model, there is no exit from the labor force, so workers will contribute for $R$ years, but $Cy$ could be some number less than $R$ if there is a maximum number of years of contributions one can accrue under the DB system.
  - $\alpha^{DB}_{j,s,t}$ is the replacement rate per year of contribution (`alpha_db`), which may vary across lifetime income groups.
  - $R$, $ny$, $Cy$, and $\alpha^{DB}$ are the values that apply to the household's birth cohort $t-s$ at the age and period in which the benefit is paid. With no `pension_regimes`, $R$, $ny$, and $Cy$ are the same for every cohort and $\alpha^{DB}_{j,t}$ is a path by the calendar year of payment, so a change in it affects the benefits of current retirees as well as those of future retirees. A regime with a `max_age_at_start` instead changes the formula only for the cohorts it covers, so the benefits of those already retired are unchanged.

  Given this pension system and the fact that there is only variation in labor supply along the intensive margin (so we don't need to consider changes in $Cy$), the partial derivatives from the household section are given by:

  ```{math}
  :label: eqn:db_deriv
    \frac{\partial \theta_{j,u,t+u-s}}{\partial n_{j,s,t}} =
      \begin{cases}
        0 , & \text{if}\ s < R - Cy \\
        w_{t}e_{j,s}\alpha^{DB}_{j,t+R-s}\times \frac{Cy}{ny}, & \text{if}\  R - Cy <= s < R  \\
        0, & \text{if}\ s \geq R \\
      \end{cases}
  ```

##### Notional defined contribution system

The pension amount under a notional defined contribution system is given as:

  ```{math}
  :label: eqn:ndc_pension
   pension_{j,s,t} = \biggl[\sum_{u=E}^{R-1}\tau^{p}_{u,t-s+u}w_{t-s+u}e_{j,u,t-s+u}n_{j,u,t-s+u}\prod_{a=u+1}^{R-1}(1 + g_{NDC,t-s+a})\biggr]\delta_{R,s,t} \quad \forall s \geq R
  ```

where the sum runs over the household's own contribution history, quantities are in real, non-stationarized units, and:

  * $\tau^{p}_{u,t-s+u}$ is the notional contribution rate (`tau_p`) that the household's cohort faced at working age $u$, in period $t-s+u$. It sets the share of labor income credited to the household's notional account. It is not a tax: it is not deducted from household income or counted as government revenue. The only payroll tax households pay is `tau_payroll`, which applies to all labor income under every pension system
  * $g_{NDC,t}$ is the rate credited to notional balances in period $t$. A contribution made at age $u$ earns the rates credited in each period from the next one through the period before retirement. The credited rate is the growth realized from period $t-1$ to period $t$ of the measure chosen with `ndc_growth_rate`. Because model aggregates are stationarized by $e^{g_y t}$ and by population, the growth of an aggregate $X$ is recovered from its stationarized value $\hat{X}$ as

    ```{math}
    1 + g_{X,t} = \frac{\hat{X}_{t}}{\hat{X}_{t-1}}e^{g_y}(1 + g_{n,t-1})
    ```

    where $g_{n,t-1}$ is population growth from period $t-1$ to $t$. The options are:
    * `'Curr GDP'`: the growth of real GDP, $X = Y$
    * `'wage bill'`: the growth of total real labor earnings, $X = wL$ (as in Latvia and Poland)
    * `'avg wage'`: the growth of real labor earnings per person, $\frac{\hat{w}_t\hat{L}_t}{\hat{w}_{t-1}\hat{L}_{t-1}}e^{g_y} - 1$ (as in Sweden and Norway)
    * `'r'`: the interest rate on household savings, $r_{p,t}$
    * `'LR GDP'` (the default): the constant long-run growth rate of real GDP, $e^{g_y}(1 + \bar{g}_{n}) - 1$

    All of these equal their long-run values in the steady state, except `'r'`, which equals the steady-state interest rate. Growth into period 0 uses the period-0 level as the level of the period before, and periods before the transition path take the period-0 rate. Some countries credit an average of past growth (for example, Italy credits the 5-year average of GDP growth); the parameter `ndc_growth_avg_years` sets the number of years averaged, with the rate credited in period $t$ being the arithmetic mean of the rates in that period and the preceding periods in the window.
  * $\delta_{R,s,t}$ is the conversion coefficient and its calculation is detailed below. It, like $R$ and $\tau^p$, takes the values (`k_ret`, `indR`) that apply to the household's birth cohort at the age and period in which the benefit is paid, so that a reform entered through `pension_regimes` with a `max_age_at_start` changes the formula only for the cohorts it covers.

  ```{math}
  \delta_{R} = (dir_{R} + ind_{R} - k)^{-1}
  ```

where $k$ is an adjustment that takes into account the number of payments per year.  In particular, $k=0.5 - (6/13n)$, where $n$ is the number of payments per year.  So if the payments are made monthly, $n=12$ and thus $k=0.4615$.

The $dir_{R, t}$ term is an adjustment to make the payments actuarially fair given mortality risk. It is the expected discounted number of payments to a household that is alive at retirement age $R$ and retires in period $t$:

  ```{math}
  dir_{R, t} = \sum_{k=0}^{E+S-R}\left[\prod_{i=0}^{k-1}(1-\hat{\rho}_{R+i, t+i})\right](1+\hat{g}_{y, t})^{-k}
  ```

where the empty product (for $k=0$) equals one. The timing follows the household's problem: $\rho_{s}$ is the probability that a household of age $s$ dies before reaching age $s+1$, so the first payment, at age $R$, is certain and the payment at age $R+k$ is received only if the household survives ages $R$ through $R+k-1$. $\hat{g}_{y, t}$ is the rate used to discount payments, chosen with `dir_growth_rate` from the same options as `ndc_growth_rate` (without averaging). A household that retires in period $t$ uses the value of that rate in period $t$ for all of its payments, as countries typically fix a norm rate in the divisor when a pension is awarded.

$\hat{\rho}_{s,t}$ are the mortality rates used in the pension system, the model's mortality rates averaged over lifetime income groups with the population weights $\lambda_j$. The parameter `ndc_mortality` determines which rates apply. With `'cohort'` (the default), $\hat{\rho}_{R+i, t+i}$ are the rates the retiring cohort faces at each age as it ages through the transition path, as written above. With `'period'`, the rates are those of the period of retirement, $\hat{\rho}_{R+i, t}$ (a period life table). With `'steady state'`, every cohort uses the steady-state rates $\hat{\rho}_{R+i}$. The three coincide in the steady state.

Finally, $ind_{R}$ is an adjustment for survivor benefits.  Since we model households (and not individuals), we set $ind_{R} = 0$ by default.  This can be changed with the parameter `indR` if one would like to account for the fact that households lose members over time.

Given this pension system, the partial derivatives from the household section are given by:

  ```{math}
  :label: eqn:ndc_deriv
  \frac{\partial \theta_{j,u,t+u-s}}{\partial n_{j,s,t}} =
    \begin{cases}
      \tau^{p}_{t}w_{t}e_{j,s}(1+g_{NDC,t})^{u - s}\delta_{R,t}, & \text{if}\ s<R-1 \\
      0, & \text{if}\ s \geq R \\
    \end{cases}
  ```

##### Points system

Under a points system, the pension amount is given as:

  ```{math}
  :label: eqn:ps_pension
  pension_{j,s,t} = \sum_{s=E}^{R-1}w_{t}e_{j,s,t}n_{j,s,t}\times v_{t} \quad \forall s \geq R_{t-s}
  ```

where $v_{t}$ is the value of a point at time $t$

Given this pension system, the partial derivatives from the household section are given by:

  ```{math}
  :label: eqn:ps_deriv
  \frac{\partial \theta_{j,u,t+u-s}}{\partial n_{j,s,t}} =
  \begin{cases}
    0 , & \text{if}\ s < R \\
    w_{t}e_{j,s}v_{t}, & \text{if}\ s \geq R \\
  \end{cases}
 ```

##### Hybrid defined benefit and notional defined contribution system

The hybrid system pays a defined benefit on labor income up to a cutoff and a notional defined contribution benefit on labor income in a band above that cutoff. The two cutoffs are set relative to average labor income in the first model period, $f_{1,t} \bar{y}$ and $f_{2,t} \bar{y}$ with $0 \leq f_{1,t} \leq f_{2,t}$ in every period, where $f_{1,t}$ is the parameter `factor1_db_ndc_hybrid`, $f_{2,t}$ is the parameter `factor2_db_ndc_hybrid`, and $\bar{y}$ is the parameter `avg_labor_inc_t1_model_units`. The value $\bar{y}$ is the average labor income in the current (first) period in model units, taken from an outside model, so the cutoffs do not depend on endogenous variables such as the wage. The user must either supply their own series for each of `factor1_db_ndc_hybrid` and `factor2_db_ndc_hybrid`, or supply a single scalar in the list. A series is entered as a list of two or more values by year, starting in the base year, and every year after the last value entered keeps that value; the two lists need not have the same length. A single scalar is the initial factor $f_{1,0}$ (or $f_{2,0}$) of a path that has the growth trend removed from it, $f_{1,t} = f_{1,0}/e^{g_y t}$, which reflects a type of inflation indexation (equation {eq}`eqn:hybrid_indexed` below). The defaults are the scalars $f_{1,0} = 1.0$ and $f_{2,0} = 2.0$, with $\bar{y} = 1.0$, so by default the cutoffs are indexed in this way. To hold a factor constant instead, enter the same value twice, for example `[1.0, 1.0]`. The cutoffs that apply to any labor income are those of the period in which it was earned, so a change in the cutoffs affects the benefits of current workers only through the income they earn after the change, and leaves the benefits earned in earlier years unchanged.

How the cutoffs are indexed over time is set by the path of the factors. Because $\bar{y}$ is in stationarized model units (labor income divided by $e^{g_y t}$), the cutoffs in levels are $f_{1,t}\bar{y}e^{g_y t}$ and $f_{2,t}\bar{y}e^{g_y t}$:

  * Constant factors, $f_{1,t} = f_{1,0}$, give cutoffs that grow with labor augmenting technological change, at rate $g_y$.
  * An indexed system, in which the cutoffs stay fixed in real terms at their initial level, sets the factors as below. This is the path the model builds when a factor is entered as a single scalar.
    ```{math}
    :label: eqn:hybrid_indexed
      f_{1,t} = \frac{f_{1,0}}{e^{g_y t}} \quad\text{and}\quad f_{2,t} = \frac{f_{2,0}}{e^{g_y t}}
    ```
    so that the cutoffs are $f_{1,t}\bar{y} = f_{1,0}\bar{y}e^{-g_y t}$ and $f_{2,t}\bar{y} = f_{2,0}\bar{y}e^{-g_y t}$ in model units, where $f_{1,0}$ and $f_{2,0}$ are the initial factor values and $g_y$ is the growth rate of labor augmenting technological change per model period. With $g_y > 0$ these factors fall toward zero, and the steady state uses the last value of each path, so in the long run nearly all labor income lies above both cutoffs and earns no benefit.
  * Any other kind of indexation is some other transformation of the path of $f_{1,t}$ and $f_{2,t}$. For example, cutoffs that grow at rate $g_c$ per model period in real terms have $f_{1,t} = f_{1,0}e^{(g_c - g_y)t}$.

Labor income $y_{j,s,t} = w_t e_{j,s,t} n_{j,s,t}$ is split into the following parts:

  * Labor income up to $f_{1,t} \bar{y}$ earns a defined benefit, computed with the formula and parameters ($R$, $ny$, $Cy$, $\alpha^{DB}$) of the [defined benefit system](#defined-benefit-system) applied to this capped income, $\min\{y_{j,s,t}, f_{1,t} \bar{y}\}$.
  * Labor income between $f_{1,t} \bar{y}$ and $f_{2,t} \bar{y}$ earns a notional defined contribution benefit, paid in addition to the defined benefit. The household's notional account is credited with $\tau^{p} \max\{\min\{y_{j,s,t}, f_{2,t} \bar{y}\} - f_{1,t} \bar{y}, 0\}$, and the benefit is computed with the formula and parameters ($\tau^p$, the credited rate $g_{NDC}$ set by `ndc_growth_rate` and `ndc_growth_avg_years`, and the conversion coefficient $\delta_R$ with its discount rate `dir_growth_rate`) of the [notional defined contribution system](#notional-defined-contribution-system).
  * Labor income above $f_{2,t} \bar{y}$ earns no pension benefit.

The cutoffs determine only which income earns which benefit. They do not change what the household pays: as under the other pension systems, the payroll tax `tau_payroll` applies to all labor income $y_{j,s,t}$, with no cap, and the notional rate `tau_p` is used only to credit the notional account, not deducted from household income.

Define the labor income in each band as

  ```{math}
  :label: eqn:hybrid_bands
    y^{DB}_{j,s,t} = \min\{w_t e_{j,s,t}n_{j,s,t}, f_{1,t}\bar{y}\} \quad\text{and}\quad y^{NDC}_{j,s,t} = \max\bigl\{\min\{w_t e_{j,s,t}n_{j,s,t}, f_{2,t}\bar{y}\} - f_{1,t}\bar{y}, 0\bigr\}
  ```

The pension amount under the hybrid system is then the sum of the two benefits:

  ```{math}
  :label: eqn:hybrid_pension
  \begin{split}
   pension_{j,s,t} = &\biggl[\frac{\sum_{u=R-ny}^{R-1}y^{DB}_{j,u,t-s+u}}{ny}\biggr]\times Cy \times \alpha^{DB}_{j,s,t} \\
   &+ \biggl[\sum_{u=E}^{R-1}\tau^{p}_{u,t-s+u}y^{NDC}_{j,u,t-s+u}\prod_{a=u+1}^{R-1}(1 + g_{NDC,t-s+a})\biggr]\delta_{R,s,t} \quad \forall s \geq R
  \end{split}
  ```

where, as in the separate systems, each sum runs over the household's own earnings history (so $y^{DB}_{j,u,t-s+u}$ and $y^{NDC}_{j,u,t-s+u}$ use the cutoffs $f_{1,t-s+u}\bar{y}$ and $f_{2,t-s+u}\bar{y}$ of the period in which the household was age $u$) and the parameters are those that apply to the household's birth cohort. Along the transition path the notional balances are credited with the same time-varying rates as under the notional defined contribution system. Setting $f_{1,t}$ above all labor income gives the defined benefit system, and setting $f_{1,t} = 0$ with $f_{2,t}$ above all labor income gives the notional defined contribution system. As in the other systems, the benefit is scaled by `replacement_rate_adjust`.

Given this pension system, the partial derivative of the benefit with respect to labor supply at a working age $s < R$ is the defined benefit derivative in equation {eq}`eqn:db_deriv` when the household's labor income is below $f_{1,t} \bar{y}$, the notional defined contribution derivative in equation {eq}`eqn:ndc_deriv` when it is between $f_{1,t} \bar{y}$ and $f_{2,t} \bar{y}$, and zero when it is above $f_{2,t} \bar{y}$.

##### Aggregate Pension Spending

Total pension spending is the sum of the pension payments to each household in the model:

  ```{math}
  :label: EqPensionSpending
  Pensions_{t} = \sum_{j=1}^{J}\sum_{s=E+1}^{E+S} pension_{j,s,t}\omega_{s,t}\lambda_j \quad\forall t
  ```

(SecGovLumpSumTfers)=
#### Lump sum transfers:

  Aggregate non-pension transfers to households are assumed to be a fixed fraction $\alpha_{tr}$ of GDP each period:

  ```{math}
  :label: EqUnbalGBCtfer
    TR_t = g_{tr,t}\:\alpha_{tr}\: Y_t \quad\forall t
  ```

  The time dependent multiplier $g_{tr,t}$ in front of the right-hand-side of {eq}`EqUnbalGBCtfer` will equal 1 in most initial periods. It will potentially deviate from 1 in some future periods in order to provide a closure rule that ensures a stable long-run debt-to-GDP ratio. We will discuss the closure rule in Section {ref}`SecUnbalGBCcloseRule`.


(SecUBI)=
#### Universal basic income

[TODO: This section is far along but needs to be updated.]

 Universal basic income (UBI) transfers show up in the household budget constraint {eq}`EqHHBC`. Household amounts of UBI can vary by household age $s$, lifetime income group $j$, and time period $t$.  These transfers are represented by $ubi_{j,s,t}$.


(SecUBIcalc)=
##### Calculating UBI

  Household transfers in model units of the numeraire good $ubi_{j,s,t}$ are a function of five policy parameters described in the [`default_parameters.json`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/default_parameters.json) file (`ubi_growthadj`, `ubi_nom_017`, `ubi_nom_1864`, `ubi_nom_65p`, and `ubi_nom_max`).  Three additional parameters provide information on household structure by age, lifetime income group, and year: [`ubi_num_017_mat`, `ubi_num_1864_mat`, `ubi_num_65p_mat`].

  As a convenience to users, UBI policy parameters `ubi_nom_017`, `ubi_nom_1864`, `ubi_nom_65p`, and `ubi_nom_max` are entered as nominal amounts (e.g., in dollars or pounds). The parameter `ubi_nom_017` represents the nominal value of the UBI transfer to each household per dependent child age 17 and under. The parameter `ubi_nom_1864` represents the nominal value of the UBI transfer to each household per adult between the ages of 18 and 64. And `ubi_nom_65p` is the nominal value of UBI transfer to each household per senior 65 and over. The maximum UBI benefit per household, `ubi_nom_max`, is also a nominal amount.  From these parameters, the model computes nominal UBI payments to each household in the model:

  ```{math}
  :label: EqUBIubi_nom_jst0
    \begin{split}
      ubi^{nom}_{j,s,t=0} = \min\Bigl(&\texttt{ubi\_nom\_max}, \\
      &\texttt{ubi\_nom\_017} * \texttt{ubi\_num\_017\_mat}_{j,s} + \\
      &\texttt{ubi\_nom\_1864} * \texttt{ubi\_num\_1864\_mat}_{j,s} + \\
      &\texttt{ubi\_nom\_65p} * \texttt{ubi\_num\_65p\_mat}_{j,s}\Bigr) \quad\forall j,s
    \end{split}
  ```

  The rest of the time periods of the household UBI transfer and the respective steady-states are determined by whether the UBI is growth adjusted or not as given in the `ubi_growthadj` boolean parameter. The following two sections cover these two cases.


(SecUBI_NonGrowthAdj)=
###### UBI specification not adjusted for economic growth

  A non-growth adjusted UBI (`ubi_growthadj = False`) is one in which the initial nonstationary nominal-valued $t=0$ UBI matrix $ubi^{nom}_{j,s,t=0}$ does not grow, while the economy's long-run growth rate is $g_y$ for the most common parameterization is positive ($g_y>0$).

  ```{math}
  :label: EqUBIubi_nom_NonGrwAdj_jst
    ubi^{nom}_{j,s,t} = ubi^{nom}_{j,s,t=0} \quad\forall j,s,t
  ```

  As described in chapter {ref}`Chap_Stnrz`, the stationarized UBI transfer to each household $\hat{ubi}_{j,s,t}$ is the nonstationary transfer divided by the growth rate since the initial period. When the long-run economic growth rate is positive $g_y>0$ and the UBI specification is not growth-adjusted, the steady-state stationary UBI household transfer is zero $\overline{ubi}_{j,s}=0$ for all lifetime income groups $j$ and ages $s$ as time periods $t$ go to infinity. However, to simplify, we assume in this case that the stationarized steady-state UBI transfer matrix to households is the stationarized value of that matrix in period $T$.[^UBIgrowthadj]

  ```{math}
  :label: EqUBIubi_mod_NonGrwAdj_SS
    \overline{ubi}_{j,s} = ubi_{j,s,t=T} \quad\forall j,s
  ```

  Note that in non-growth-adjusted case, if $g_y<0$, then the stationary value of $\hat{ubi}_{j,s,t}$ is going to infinity as $t$ goes to infinity. Therefore, a UBI specification must be growth adjusted for any assumed negative long run growth $g_y<0$.[^GrowthAdj_note]


(SecUBI_GrowthAdj)=
###### UBI specification adjusted for economic growth

  Put description of growth-adjusted specification here.

#### Spending in the `reform` simulation

While aggregate spending on $G$, $TR$, and $I_g$ in the baseline simulation are set as fractions of GDP, in the `reform` simulation, these spending amounts can be set in two different ways.  The method is controlled by the `baseline_spending` parameter.  If `baseline_spending=False`, the behavior of these spending is analgous to that in the baseline simulation; they are set as fractions of GDP, in this case **GDP in the reform simulation**. Thus, with the default assumption of `baseline_spending` it's assumed that spending levels in these three categories are function of GDP in the reform.  In this case, users will see that, even with the parameters $\alpha_G$, $\alpha_T$, and $\alpha_I$ are unchanged, the *level* of spending will change in the reform *if* GDP in the reform is different.

With the assumption of `baseline_spending=True`, the level of spending in the reform is held to the level of spending in the baseline.  If the user wishes to adjust the level of spending, relative to the baseline level, in the reform, then the parameters `alpha_bs_G`, `alpha_bs_T`, and `alpha_bs_I` can be used to proportionally increase or decrease the levels of spending on $G$, $TR$, and $I_g$ in the reform simulation, relative to the levels in the baseline simulation. Note that the `alpha_bs_*` parameters are time varying, so the proportional change in spending can be different across time.  E.g., for government consumption expenditures, we'd have the reform amount of $G$ determined as:

  ```{math}
    G^{reform}_t = \alpha^{BS}_{G,t}G^{baseline}_{t}
  ```

Foreign aid to the domestic government, $FA_t$, is treated the same way. In the baseline simulation, and in the reform when `baseline_spending=False`, foreign aid is a fraction $\alpha_{FA,t}$ of GDP. When `baseline_spending=True`, foreign aid in the reform is held to its level in the baseline, scaled by the time-varying parameter `alpha_bs_FA`:

  ```{math}
    FA^{reform}_t = \alpha^{BS}_{FA,t}FA^{baseline}_{t}
  ```

This means that changes to `alpha_FA` in the reform have no effect when `baseline_spending=True`; use `alpha_bs_FA` to change the level of foreign aid relative to the baseline instead.

Debt contracted abroad $D^{ext}_t$ is not affected by `baseline_spending`: in the reform simulation it is always the fraction $\alpha_{D,ext,t}$ of GDP in the reform.

Note that the budget closure rule (described in Section ref{`SecUnbalGBCcloseRule`}) still takes effect in the case of `baseline_spending=True`.  What this means is that the relation described above holds until the period in which the closure rules takes effect.  Once the closure rule begins, the path of $G$ (and/or $TR$, depending on the closure rule used) will adjust as determined by the rule to close the government budget in the long run.

(SecUnbalGBCrev)=
## Government Tax Revenue

  We see from the household's budget constraint that net taxes $tax_{j,s,t}$ and government transfers $tr_{j,s,t}$ enter into the household's decision,

  ```{math}
  :label: EqHHBC3
  \begin{aligned}
    p_t c_{j,s,t} + &\sum_{i=1}^I (1 + \tau^{c}_{i,t})p_{i,t}c_{min,i} + b_{j,s+1,t+1} = \\
    &(1 + r_{p,t})b_{j,s,t} + w_t e_{j,s} n_{j,s,t} ... \\
    &\qquad +\: (1-\tau^{bq}_t)bq_{j,s,t} + rm_{j,s,t} + tr_{j,s,t} + ubi_{j,s,t} + pension_{j,s,t} - tax_{j,s,t}  \\
    &\qquad\quad\forall j,t\quad\text{and}\quad E+1\leq s\leq E+S \quad\text{where}\quad b_{j,E+1,t}=0
  \end{aligned}
  ```

  where we defined the tax liability function $tax_{j,s,t}$ in {eq}`EqTaxCalcLiabETR` as effective tax rate times respective total income and wealth. The transfer distribution function $\eta_{j,s,t}$ can vary by lifetime ability group $j$, age $s$, and time period $t$. Households also remit tax on bequests received ($\tau^{bq}$) and on consumption ($\tau^c$). And government revenue from the corporate income tax rate schedule $\tau^{corp}_{m,t}$ and the tax on depreciation expensing schedule $\delta^\tau_{m,t}$ enters the firms' profit function in each industry $m$.

  ```{math}
  :label: EqFirmsProfit2
  \begin{aligned}
    PR_{m,t} &= (1 - \tau^{corp}_{m,t})\Bigl[p_{m,t}F(K_{m,t},K_{g,m,t},L_{m,t}) - w_t L_{m,t}\Bigr] - \\
    &\quad\quad\quad\quad\quad \bigl(r_t + \delta_{M,t}\bigr)K_{m,t} + \left(\tau^{corp}_{m,t}\delta^\tau_{m,t} + \tau^{inv}_{m,t}\delta_{M,t}\right)K_{m,t} \quad\forall m,t
  \end{aligned}
  ```

  We define total government revenue from taxes in terms of the numeraire good as the following,

  ```{math}
  :label: EqUnbalGBCgovRev
  \begin{aligned}
    Rev_t &= \underbrace{\sum_{m=1}^M\Bigl[\tau^{corp}_{m,t}\bigl(p_{m,t}Y_{m,t} - w_t L_{m,t}\bigr) - \bigl(\tau^{corp}_{m,t}\delta^\tau_{m,t} + \tau^{inv}_{m,t}\delta_{M,t}\bigr)K_{m,t}\Bigr]}_{\text{corporate income tax revenue}} \\
    &\quad + \underbrace{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\lambda_j\omega_{s,t}\tau^{etr,xy}_{s,t}\left(x_{j,s,t},y_{j,s,t}\right)\bigl(x_{j,s,t} + y_{j,s,t}\bigr)}_{\text{household income tax revenue}} \\
    &\quad + \underbrace{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\sum_{i=1}^I\lambda_j\omega_{s,t}\tau^{c}_{i,t}p_{i,t}c_{i,j,s,t}}_{\text{consumption tax revenue}} \\
    &\quad + \underbrace{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\lambda_j\omega_{s,t}\tau^{etr,w}_{t}b_{j,s,t}}_{\text{wealth tax revenue}}\\
    &\quad + \underbrace{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\lambda_j\omega_{s,t}\tau^{bq}_{t}bq_{j,s,t}}_{\text{bequest tax revenue}}
    \quad\forall t
  \end{aligned}
  ```

  where household labor income is defined as $x_{j,s,t}\equiv w_t e_{j,s}n_{j,s,t}$ and capital income $y_{j,s,t}\equiv r_{p,t} b_{j,s,t}$.


(SecUnbalGBCbudgConstr)=
## Government Budget Constraint

  Let the level of government debt in period $t$ be given by $D_t$. The government may also owe debt contracted abroad at its own interest rate, $D^{ext}_t$, described below. The government budget constraint requires that government revenue $Rev_t$, external foreign assistance, $FA_t$, plus the budget deficit financed by new borrowing ($D_{t+1} - D_t$ and $D^{ext}_{t+1} - D^{ext}_t$) equal expenditures on interest on the debt, government spending on public goods $G_t$, total infrastructure investments $I_{g,t}$, total pension outlays, total transfer payments to households $TR_t$, and $UBI_t$ every period $t$,

  ```{math}
  :label: EqUnbalGBCbudgConstr
    D_{t+1} + D^{ext}_{t+1} + Rev_t + FA_t = (1 + r_{gov,t})D_t + (1 + r_{ext,t})D^{ext}_t + G_t + I_{g,t} + Pensions_t + TR_t + UBI_t  \quad\forall t
  ```

  where $r_{gov,t}$ is the interest rate paid by the government defined in equation {eq}`EqUnbalGBC_rate_wedge` below, $G_{t}$ is government spending on public goods, $I_{g,t}$ is total government spending on infrastructure investment, $TR_{t}$ are non-pension government transfers, and $UBI_t$ is the total UBI transfer outlays across households in time $t$. All variables in {eq}`EqUnbalGBCbudgConstr` are real variables denominated in units of current-period output in industry $M$ the numeraire ($p_{M,t}=1$ for all $t$).

  We assume that government spending on public goods in terms of the numeraire good is a fixed fraction of GDP each period in the initial periods.

  ```{math}
  :label: EqUnbalGBC_Gt
    G_t = g_{g,t}\:\alpha_{g}\: Y_t \quad\forall t
  ```

  Similar to transfers $TR_t$, the time dependent multiplier $g_{g,t}$ in front of the right-hand-side of {eq}`EqUnbalGBC_Gt` will equal 1 in most initial periods. It will potentially deviate from 1 in some future periods in order to provide a closure rule that ensures a stable long-run debt-to-GDP ratio. We make this more specific in the next section.

  Total government infrastructure investment spending, $I_{g,t}$ is assumed to be a time-dependent fraction of GDP.

  ```{math}
  :label: EqUnbalGBC_Igt
    I_{g,t} = \alpha_{I,t}\: Y_t \quad\forall t
  ```

  The government also chooses what percent of total infrastructure investment goes to each industry $\alpha_{I,m,t}$, although these are exogenously calibrated parameters in the model.

  ```{math}
  :label: EqUnbalGBC_Igmt
    I_{g,m,t} = \alpha_{I,m,t}\: I_{g,t} \quad\forall m,t
  ```

  The stock of public capital (i.e., infrastructure) in each industry $m$ evolves according to the law of motion,

  ```{math}
  :label: EqUnbalGBC_Kgmt
    K_{g,m,t+1} = (1 - \delta_g) K_{g,m,t} + (1 - \phi_g) I_{g,m,t} \quad\forall m,t
  ```

  where $\delta_g$ is the depreciation rate on infrastructure and $\phi_g$ (`infra_investment_leakage_rate`) is the fraction of infrastructure investment lost to leakage (e.g., corruption or other frictions). The stock of public capital in each industry $m$ complements labor and private capital in the production function of the representative firm, in Equation {eq}`EqFirmsCESprodfun`.

  Governments also borrow abroad directly, outside the domestic bond market: from official lenders such as multilateral development banks and bilateral donors, often on concessional terms (at interest rates well below market rates, in amounts the lenders decide), and on commercial terms through bonds and loans issued abroad. `OG-Core` models this debt contracted abroad as a stock $D^{ext}_t$ held entirely by foreigners, set as a time-dependent fraction $\alpha_{D,ext,t}$ (`alpha_D_ext`) of GDP and paying the exogenous real interest rate $r_{ext,t}$ (`r_D_ext_annual`, converted to a model-period rate). The model has no exchange rate, so for debt in foreign currency $r_{ext,t}$ is the foreign-currency rate less foreign inflation, plus any expected real depreciation.

  ```{math}
  :label: EqUnbalGBC_D_ext
    D^{ext}_t = \alpha_{D,ext,t}\: Y_t \quad\forall t
  ```

  The government receives the new external borrowing $D^{ext}_{t+1} - D^{ext}_t$ and pays the interest $r_{ext,t}D^{ext}_t$ each period. Debt contracted abroad is separate from the market debt $D_t$: it is not part of household savings, so it does not enter the household portfolio return $r_{p,t}$ in {eq}`eq_portfolio_return`, and the debt-to-GDP target $\alpha_D$ of the budget closure rule and the share of new market debt bought by foreigners $\zeta_D$ refer to market debt; $D^f_t$ is then the part of market debt held by foreigners. The debt-elastic premium on $r_{gov,t}$ in {eq}`EqUnbalGBC_rate_wedge` is evaluated at all government debt, $D_t + D^{ext}_t$. Its interest is paid abroad at $r_{ext,t}$ in the goods market clearing condition (Chapter {ref}`Chap_MarkClr`). Because the stock is set by $\alpha_{D,ext,t}$, a reform that changes borrowing changes market debt only, unless it also changes $\alpha_{D,ext,t}$; and in the steady state all government debt is $\alpha_D + \alpha_{D,ext}$ of GDP. By default $\alpha_{D,ext,t} = 0$.

  Aggregate spending on UBI at time $t$ is the sum of UBI payments across all households at time $t$:

  ```{math}
  :label: EqUnbalGBC_UBI
    UBI_t \equiv \sum_{s=E+1}^{E+S}\sum_{j=1}^J \lambda_j\omega_{s,t} ubi_{j,s,t} \quad\forall t
  ```


(SecRateWedge)=
## Interest Rate on Government Debt and Household Savings

  Despite the model having no aggregate risk, it may be helpful to build in an interest rate differential between the rate of return on private capital and the interest rate on government debt. Doing so helps to add realism by including a risk premium. `OG-Core` allows users to set an exogenous wedge between these two rates. `OG-Core` also allows for the risk premium on debt to be a quadratic function of the debt-to-GDP ratio. The interest rate on government debt is given by:

  ```{math}
  :label: EqUnbalGBC_rate_wedge
    r_{gov, t} = (1 - \tau_{d, t})r_{t} - \mu_{d} + \beta_{1}\frac{D_t + D^{ext}_t}{Y_t} + \beta_{2}\left(\frac{D_t + D^{ext}_t}{Y_t}\right)^2
  ```

  where $r_t$ is the marginal product of capital faced by firms. The two parameters, $\tau_{d,t}$ and $\mu_{d,t}$ can be used to allow for a government interest rate that is a percentage hair cut from the market rate or a government interest rate with a constant risk premium.  The parameters $\beta_1$ and $\beta_2$ can be used to allow for a quadratic risk premium on government debt that is a function of the debt-to-GDP ratio. The ratio counts all government debt, the market debt $D_t$ and the debt contracted abroad $D^{ext}_t$, since the risk of the government's debt depends on everything it owes; with no debt contracted abroad it is $D_t/Y_t$.


(SecUnbalGBCcloseRule)=
## Budget Closure Rule

  If total government transfers to households $TR_t$ and government spending on public goods $G_t$ are both fixed fractions of GDP, one can imagine corporate and household tax structures that cause the debt level of the government to either tend toward infinity or to negative infinity, depending on whether too little revenue or too much revenue is raised, respectively.

  A virtue of dynamic general equilibrium models is that the model must be stationary in the long-run in order to solve it. That is, no variables can be indefinitely growing as time moves forward. The labor augmenting productivity growth $g_y$ from Chapter {ref}`Chap_Firms` and the potential population growth $\tilde{g}_{n,t}$ from the calibration chapter on demographics in the country-specific repository documentation render the model nonstationary. But we show how to stationarize the model against those two sources of growth in Chapter {ref}`Chap_Stnrz`. However, even after stationarizing the effects of productivity and population growth, the model could be rendered nonstationary and, therefore, not solvable if government debt were becoming too positive or too negative too quickly.

  For the model to be stationary, the debt-to-GDP ratio must be stable in the long run. Because the debt-to-GDP ratio is a quotient of two macroeconomic variables, the non-stationary and stationary versions of this ratio are equivalent. Let $T$ be some time period in the future. The stationarizing assumption is the following,

  ```{math}
  :label: EqUnbalGBC_DY
    D_t = \alpha_D Y_t \quad\Rightarrow\quad \frac{D_t}{Y_t} = \alpha_D \quad\text{for}\quad t\geq T
  ```

  where $\alpha_D$ is a scalar long-run value of the debt-to-GDP ratio. This long-run stability condition on the debt-to-GDP ratio clearly applies to the steady-state as well as any point in the time path for $t>T$.


  We detail three possible closure-rule options here for stabilizing the debt-to-GDP ratio in the long run, although `OG-Core` only has the capability currently to execute the first closure rule that adjusts government spending $G_t$. We expect to have the other two rules implemented as `OG-Core` options soon. Each rule uses some combination of changes in government spending on public goods $G_t$ and government transfers to households $TR_t$ to stabilize the debt-to-GDP ratio in the long-run.

  1. Change only government spending on public goods $G_t$.
  2. Change only government transfers to households $TR_t$.
  3. Change both government spending $G_t$ and transfers $TR_t$ by the same percentage.


(SecUnbalGBC_chgGt)=
### Change government spending only

  We specify a closure rule that is automatically implemented after some period $T_{G1}$ to stabilize government debt as a percent of GDP (debt-to-GDP ratio) by period $T_{G2}$. Let $\alpha_D$ represent the long-run debt-to-GDP ratio at which we want the economy to eventually settle.

  ```{math}
  :label: EqUnbalGBCclosure_Gt
  \begin{split}
    &G_t = g_{g,t}\:\alpha_{g}\: Y_t \\
    &\text{where}\quad g_{g,t} =
      \begin{cases}
        1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad\:\text{if}\quad t < T_{G1} \\
        \frac{\left[\rho_{d}\alpha_{D}Y_{t} + (1-\rho_{d})D_{t}\right] - (1+r_{gov,t})D_{t} - I_{g,t} - TR_{t} - UBI_{t} + Rev_{t} + FA_{t} + D^{ext}_{t+1} - (1+r_{ext,t})D^{ext}_{t}}{\alpha_g Y_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
        \frac{\alpha_{D}Y_{t} - (1+r_{gov,t})D_{t} - I_{g,t} - TR_{t} - UBI_{t} + Rev_{t} + FA_{t} + D^{ext}_{t+1} - (1+r_{ext,t})D^{ext}_{t}}{\alpha_g Y_t} \qquad\qquad\quad\:\:\,\text{if}\quad t \geq T_{G2}
      \end{cases} \\
    &\text{and}\quad g_{tr,t} = 1 \quad\forall t
  \end{split}
  ```

  The first case in {eq}`EqUnbalGBCclosure_Gt` says that government spending $G_t$ will be a fixed fraction $\alpha_g$ of GDP $Y_t$ for every period before $T_{G1}$. The second case specifies that, starting in period $T_{G1}$ and continuing until before period $T_{G2}$, government spending be adjusted to set tomorrow's debt $D_{t+1}$ to be a convex combination between its long-run stable level $\alpha_D Y_t$ and the current debt level $D_t$, where $\alpha_D$ is a target debt-to-GDP ratio and $\rho_d\in(0,1]$ is the percent of the way to jump toward the target $\alpha_D Y_t$ from the current debt level $D_t$. The last case specifies that, for every period after $T_{G2}$, government spending $G_t$ is set such that the next-period debt be a fixed target percentage $\alpha_D$ of GDP.


(SecUnbalGBC_chgTRt)=
### Change government transfers only

  If government transfers to households are specified by {eq}`EqUnbalGBCtfer` and the long-run debt-to-GDP ratio can only be stabilized by changing transfers, then the budget closure rule must be the following.

  ```{math}
  :label: EqUnbalGBCclosure_TRt
  \begin{split}
    &TR_t = g_{tr,t}\:\alpha_{tr}\: Y_t \\
    &\text{where}\quad g_{tr,t} =
      \begin{cases}
        1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\:\:\:\,\text{if}\quad t < T_{G1} \\
        \frac{\left[\rho_{d}\alpha_{D}Y_{t} + (1-\rho_{d})D_{t}\right] - (1+r_{gov,t})D_{t} - G_{t} - I_{g,t} -  UBI_{t} + Rev_{t} + FA_{t} + D^{ext}_{t+1} - (1+r_{ext,t})D^{ext}_{t}}{\alpha_{tr} Y_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
        \frac{\alpha_{D}Y_{t} - (1+r_{gov,t})D_{t} - G_{t} - I_{g,t} - UBI_{t} + Rev_{t} + FA_{t} + D^{ext}_{t+1} - (1+r_{ext,t})D^{ext}_{t}}{\alpha_{tr}Y_t} \qquad\qquad\quad\:\:\:\text{if}\quad t \geq T_{G2}
      \end{cases} \\
    &\text{and}\quad g_{g,t} = 1 \quad\forall t
  \end{split}
  ```

  The first case in {eq}`EqUnbalGBCclosure_TRt` says that government transfers $TR_t$ will be a fixed fraction $\alpha_{tr}$ of GDP $Y_t$ for every period before $T_{G1}$. The second case specifies that, starting in period $T_{G1}$ and continuing until before period $T_{G2}$, government transfers be adjusted to set tomorrow's debt $D_{t+1}$ to be a convex combination between the target debt $\alpha_D Y_t$ and the current debt level $D_t$. The last case specifies that, for every period after $T_{G2}$, government transfers $TR_t$ are set such that the next-period debt be a fixed target percentage $\alpha_D$ of GDP.


(SecUnbalGBC_chgGtTRt)=
### Change both government spending and transfers

  In some cases, changing only government spending $G_t$ or only government transfers $TR_t$ will not be enough. That is, there exist policies for which a decrease in government spending to zero after period $T_{G1}$ will not stabilize the debt-to-GDP ratio. And negative government spending on public goods does not make sense. [^negative_val_note] On the other hand, negative transfers do make sense. Notwithstanding, one might want the added stabilization ability of changing both government spending $G_t$ and transfers $TR_t$ to stabilize the long-run debt-to-GDP ratio.

  In our specific form of this joint option, we assume that the factor by which we scale government spending and transfers is the same $g_{g,t} = g_{tr,t}$ for all $t$. We label this single scaling factor $g_{trg,t}$.

  ```{math}
  :label: EqUnbalGBCclosure_gTRGt
    g_{trg,t}\equiv g_{g,t} = g_{tr,t} \quad\forall t
  ```

  If government spending on public goods is specified by {eq}`EqUnbalGBC_Gt` and government transfers to households are specified by {eq}`EqUnbalGBCtfer` and the long-run debt-to-GDP ratio can only be stabilized by changing both spending and transfers, then the budget closure rule must be the following.

  ```{math}
  :label: EqUnbalGBCclosure_TRGt
  \begin{split}
    &G_t + TR_t = g_{trg,t}\left(\alpha_g + \alpha_{tr}\right)Y_t \quad\Rightarrow\quad G_t = g_{trg,t}\:\alpha_g\: Y_t \quad\text{and}\quad TR_t = g_{trg,t}\:\alpha_{tr}\: Y_t \\
    &\text{where}\quad g_{trg,t} =
    \begin{cases}
      1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad\:\,\text{if}\quad t < T_{G1} \\
      \frac{\left[\rho_{d}\alpha_{D}Y_{t} + (1-\rho_{d})D_{t}\right] - (1+r_{gov,t})D_{t} - I_{g,t} - UBI_{t} + Rev_{t} + FA_{t} + D^{ext}_{t+1} - (1+r_{ext,t})D^{ext}_{t}}{\left(\alpha_g + \alpha_{tr}\right)Y_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
      \frac{\alpha_{D}Y_{t} - (1+r_{gov,t})D_{t} - I_{g,t} - UBI_{t} + Rev_{t} + FA_{t} + D^{ext}_{t+1} - (1+r_{ext,t})D^{ext}_{t}}{\left(\alpha_g + \alpha_{tr}\right)Y_t} \qquad\qquad\quad\:\:\:\text{if}\quad t \geq T_{G2}
    \end{cases}
  \end{split}
  ```

  The first case in {eq}`EqUnbalGBCclosure_TRGt` says that government spending and government transfers $TR_t$ will their respective fixed fractions $\alpha_g$ and $\alpha_{tr}$ of GDP $Y_t$ for every period before $T_{G1}$. The second case specifies that, starting in period $T_{G1}$ and continuing until before period $T_{G2}$, government spending and transfers be adjusted by the same rate to set tomorrow's debt $D_{t+1}$ to be a convex combination between target debt $\alpha_D Y_t$ and the current debt level $D_t$. The last case specifies that, for every period after $T_{G2}$, government spending and transfers are set such that the next-period debt be a fixed target percentage $\alpha_D$ of GDP.

  Each of these budget closure rules {eq}`EqUnbalGBCclosure_Gt`, {eq}`EqUnbalGBCclosure_TRt`, and {eq}`EqUnbalGBCclosure_TRGt` allows the government to run increasing deficits or surpluses in the short run (before period $T_{G1}$). But then the adjustment rule is implemented gradually beginning in period $t=T_{G1}$ to return the debt-to-GDP ratio back to its long-run target of $\alpha_D$. Then the rule is implemented exactly in period $T_{G2}$ by adjusting some combination of government spending $G_t$ and transfers $TR_t$ to set the debt $D_{t+1}$ such that it is exactly $\alpha_D$ proportion of GDP $Y_t$.


(SecUnbalGBCcaveat)=
## Some Caveats and Alternatives

  `OG-Core` adjusts some combination of government spending $G_t$ and government transfers $TR_t$ as its closure rule instrument because of its simplicity and lack of distortionary effects. Since government spending does not enter into the household's utility function, its level does not affect the solution of the household problem. In contrast, government transfers do appear in the household budget constraint. However, household decisions do not individually affect the amount of transfers, thereby rendering government transfers as exogenous from the household's perspective. As an alternative, one could choose to adjust taxes to close the budget (or a combination of all of the government fiscal policy levers).

  There is no guarantee that any of our stated closure rules {eq}`EqUnbalGBCclosure_Gt`, {eq}`EqUnbalGBCclosure_TRt`, or {eq}`EqUnbalGBCclosure_TRGt` is sufficient to stabilize the debt-to-GDP ratio in the long run. For large and growing deficits, the convex combination parameter $\rho_d$ might be too gradual, or the budget closure initial period $T_{G1}$ might be too far in the future, or the target debt-to-GDP ratio $\alpha_D$ might be too high. The existence of any of these problems might be manifest in the steady state computation stage. However, it is possible for the steady-state to exist, but for the time path to never reach it. These problems can be avoided by choosing conservative values for $T_{G1}$, $\rho_d$, and $\alpha_D$ that close the budget quickly.

  And finally, in closure rules {eq}`EqUnbalGBCclosure_Gt` and {eq}`EqUnbalGBCclosure_TRGt` in which government spending is used to stabilize the long-run budget, it is also possible that government spending is forced to be less than zero to make this happen. This would be the case if tax revenues bring in less than is needed to financed transfers and interest payments on the national debt. None of the equations we've specified above preclude that result, but it does raise conceptual difficulties. Namely, what does it mean for government spending to be negative? Is the government selling off public assets? We caution those using this budget closure rule to consider carefully how the budget is closed in the long run given their parameterization. We also note that such difficulties present themselves across all budget closure rules when analyzing tax or spending proposals that induce structural budget deficits. In particular, one probably needs a different closure instrument if government spending must be negative in the steady-state to hit your long-term debt-to-GDP target.


(SecGovfootnotes)=
## Footnotes

  This section contains the footnotes for this chapter.

  [^interpolation_note]: We use two criterion to determine whether the function should be interpolated. First, we require a minimum number of observations of filers of that age and in that tax year. Second, we require that that sum of squared errors meet a predefined threshold.

  [^param_note]: We assume that whatever parameters the tax functions have in the last year of the budget window persist forever.

  [^NomRealGDP]: All quantities in OG-Core are real in the sense that their prices can only be denominated in terms of one of the goods in the model. We have made the assumption that all prices are given in terms of industry-$M$ output as the numeraire good. However, we can convert the real amounts in the model to currency prices using the income adjustment factor described in equation {eq}`EqIncFactor` in Section {ref}`SecHHincFactor` of Chapter {ref}`Chap_House`.

  [^UBIgrowthadj]: The steady-state assumption in equation {eq}`EqUBIubi_mod_NonGrwAdj_SS` implies that the UBI amount is growth adjusted for every period after the steady-state is reached.

  [^GrowthAdj_note]: We impose this requirement of `ubi_growthadj = False` when `g_y_annual < 0` in the [`default_parameters.json`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/default_parameters.json) "validators" specification of the parameter.

  [^negative_val_note]: Negative values for government spending on public goods would mean that revenues are coming into the country from some outside source, which revenues are triggered by government deficits being too high in an arbitrary future period $T_{G2}$.
