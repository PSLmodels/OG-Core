(Chap_Eqm)=
# Equilibrium

The equilibrium for the `OG-Core` model is broadly characterized as the solution to the model for all possible periods from the current period $t=1$ to infinity $t=\infty$. However, the solution algorithm for the equilibrium makes it useful to divide the equilibrium definition into two sub-definitions.

The first equilibrium definition we charactersize is the {ref}`SecSSeqlb`. This is really a long-run equilibrium concept. It is where the economy settles down after a large number of periods into the future. The distributions in the economy (e.g., population, wealth, labor supply) have settled down and remain constant from some future period $t=\bar{T}$ for the rest of time $t=\infty$.

The second equilibrium definition we characterize is the {ref}`SecNSSeqlb`. This equilibrium concept is "non-steady-state" because it characterizes equilibrium in all time periods from the current period $t=1$ to the period in which the economy has reached the steady-state $t=\bar{T}$. It is "non-steady-state" because the distributions in the economy (e.g., population, wealth, labor supply) are changing across time periods.

(SecSSeqlb)=
## Stationary Steady-State Equilibirum

In this section, we define the stationary steady-state equilibrium of the `OG-Core` model. Chapters {ref}`Chap_House` through {ref}`Chap_MarkClr` derive the equations that characterize the equilibrium of the model. However, we cannot solve for any equilibrium of the model in the presence of nonstationarity in the variables. Nonstationarity in `OG-Core` comes from productivity growth $g_y$ in the production function {eq}`EqFirmsCESprodfun`, population growth $\tilde{g}_{n,t}$ as described in the calibration chapter on demographics in the country-specific repository documentation, and the potential for unbounded growth in government debt as described in Chapter {ref}`Chap_UnbalGBC`.

We implemented an automatic government budget closure rule using government spending $G_t$ as the instrument that stabilizes the debt-to-GDP ratio at a long-term rate in {eq}`EqUnbalGBCclosure_Gt`. And we showed in Chapter {ref}`Chap_Stnrz` how to stationarize all the other characterizing equations.

We first give a general definition of the steady-state (long-run) equilibrium of the model. We then detail the computational algorithm for solving for the equilibrium in each distinct case of the model. There are three distinct cases or parameterization permutations of the model that have to do with the following specification choices.

* Baseline or reform
* Balanced budget or allow for government deficits/surplusses
* Small open economy or partially/closed economy
* Fixed baseline spending level or not

In all of the specifications of `OG-Core`, we use a two-stage fixed point algorithm to solve for the equilibrium solution. The solution is mathematically characterized by $2JS$ equations and $2JS$ unknowns. The most straightforward and simple way to solve these equations would be a multidimensional root finder. However, because each of the equations is highly nonlinear and depends on all of the $2JS$ variables (low sparsity) and because the dimensionality $2JS$ is high, standard root finding methods are not tractable.

Our approach is to choose the minimum number of macroeconomic variables in an outer loop in order to be able to solve the household's $2JS$ Euler equations in terms of only the $\bar{n}_{j,s}$ and $\bar{b}_{j,s+1}$ variables directly, holding all other variables constant. The household system of Euler equations has a provable root solution and is orders of magnitude more tractable (less nonlinear) to solve holding these outer loop variables constant.

The steady-state solution method for each of the cases above is associated with a solution method that has a subset of the following outer-loop variables $\{\bar{r}, \overline{TR}, \overline{BQ}, \bar{Y}, factor\}$.


(SecEqlbSSdef)=
### Stationary Steady-State Equilibrium Definition

With the stationarized model, we can now define the stationary steady-state equilibrium. This equilibrium will be long-run values of the endogenous variables that are constant over time. In a perfect foresight model, the steady-state equilibrium is the state of the economy at which the model settles after a finite amount of time, regardless of the initial condition of the model. Once the model arrives at the steady-state, it stays there indefinitely unless it receives some type of shock or stimulus.

These stationary values have all the growth components from productivity growth and population growth removed as defined in {numref}`TabStnrzStatVars`. Because the productivity growth rate $g_y$ and population growth rate series $\tilde{g}_{n,t}$ are exogenous. We can transform the stationary equilibrium values of the variables back to their nonstationary values by reversing the identities in {numref}`TabStnrzStatVars`.

We define a stationary steady-state equilibrium as the following.

```{admonition} **Definition: Stationary steady-state equilibrium**
:class: note
A non-autarkic stationary steady-state equilibrium in the `OG-Core` model is defined as constant allocations of stationary household labor supply $n_{j,s,t}=\bar{n}_{j,s}$ and savings $\hat{b}_{j,s+1,t+1}=\bar{b}_{j,s+1}$ for all $j$, $t$, and $E+1\leq s\leq E+S$, and constant prices $\hat{w}_t=\bar{w}$ and $r_t=\bar{r}$ for all $t$ such that the following conditions hold:
1. The population has reached its stationary steady-state distribution $\hat{\omega}_{s,t} = \bar{\omega}_s$ for all $s$ and $t$ as characterized in Section {ref}`SecDemogPopSSTP`,
2. households optimize according to {eq}`EqStnrzHHeul_n`, {eq}`EqStnrzHHeul_b`, and {eq}`EqStnrzHHeul_bS`,
3. firms optimize according to {eq}`EqStnrzFOC_L` and {eq}`EqStnrzFOC_K`,
4. Government activity behaves according to {eq}`EqUnbalGBC_rate_wedge`, {eq}`EqStnrzGovBC`, {eq}`EqStnrz_rate_hh`, and {eq}`EqStnrzClosureRule_Gt`, and
5. markets clear according to {eq}`EqStnrzMarkClrLab`, {eq}`EqStnrz_DtDdDf`, {eq}`EqStnrz_KtKdKf`, and {eq}`EqStnrzMarkClrBQ`.

```


(SecEqlbSSsoln)=
### Steady-state solution method: default specification

The default specification of the model is the baseline specification (`baseline = True`) in which the government can run deficits and surpluses (`budget_balance = False`), in which the economy is a large partially open economy [$\zeta_D,\zeta_K\in(0,1)$], and in which baseline government spending $G$ and transfers $TR$ are not held at baseline levels (`baseline_spending = False`).

The computational algorithm for solving for the steady-state follows the steps below. We use a notational convention of a subscript $a$ or a subscript $b$ in steps (2) and (3) to differentiate between the same inner-loop variables computed two different ways at two different stages of the algorithm ($\bar{w}_a$, $\bar{w}_b$, $\bar{r}_{gov,a}$, $\bar{r}_{gov,b}$, $\bar{r}_{hh,a}$, $\bar{r}_{hh,b}$, $\bar{Y}_a$, $\bar{Y}_b$, $\bar{K}_a$, $\bar{K}_b$).

1. Use the techniques from Section {ref}`SecDemogPopSSTP` to solve for the steady-state population distribution vector $\boldsymbol{\bar{\omega}}$ and steady-state growth rate $\bar{g}_n$ of the exogenous population process.

2. Choose an initial guess for the values of the steady-state interest rate (marginal product of capital) $\bar{r}^i$, total bequests $\overline{BQ}^{\,i}$, total household transfers $\overline{TR}^{\,i}$, and income multiplier $factor^i$, where superscript $i$ is the index of the iteration number of the guess.

    1. Given guess $\bar{r}^i$:

        1. Use firm's two first order conditions {eq}`EqStnrzFOC_L` and {eq}`EqStnrzFOC_K` to solve for the steady-state capital-labor ratio $\frac{\bar{K}}{\bar{L}}$ and the steady-state wage $\bar{w}_a$.

        2. Solve for the exogenous government interest rate $\bar{r}_{gov,a}$ using equation {eq}`EqUnbalGBC_rate_wedge`.

    2. Given guess $\overline{TR}^{\,i}$:

        1. Solve for $\bar{Y}_a$ from long-run aggregate transfers assumption as a percent of GDP {eq}`EqStnrzTfer`.

        2. Solve for total private capital $\bar{K}_a$ given GDP $\bar{Y}_a$ and the marginal product of capital guess $\bar{r}^i$ using firms first order condition for capital demand {eq}`EqStnrzFOC_K`.

        3. Solve for steady-state government debt $\bar{D}$ using the long-run debt-to-GDP relationship to $\bar{Y}_a$ {eq}`EqStnrz_DY`.

        4. Solve for foreign holdings of government debt $\bar{D}^f$ as a fixed fraction of government debt using the steady-state version of the exogenous foreign government debt rule {eq}`EqStnrz_zetaD` $\bar{D}^f = \zeta_D \bar{D}$.

        5. Use $\bar{D}^f$ and $\bar{D}$ in the government debt market clearing equation {eq}`EqStnrz_DtDdDf` to solve for domestic holdings of government bonds $\bar{D}^d$.

        6. Given $\bar{r}^i$, $\bar{r}_{gov,a}$, $\bar{K}_a$, and $\bar{D}$, solve for $\bar{r}_{hh,a}$ using {eq}`EqStnrz_rate_hh`.

        7. Use $factor^i$ to solve for the tax functions $\tau^{etr}_{s,t}$, $\tau^{mtrx}_{s,t}$, and $\tau^{mtry}_{s,t}$ defined in {eq}`EqTaxCalcLiabETR2`, {eq}`EqTaxCalcMTRx`, and {eq}`EqTaxCalcMTRy`, respectively of Chapter {ref}`Chap_TaxCalc` using the factor to transform model units to dollars in the tax functions as described in Section {ref}`SecTaxCalcFactor`.

    3. Given values $\bar{r}_{hh,a}$, $\bar{w}_a$ $\overline{BQ}^i$, $\overline{TR}^i$, and $factor^i$, solve for the steady-state household labor supply $\bar{n}_{j,s}$ and savings $\bar{b}_{j,s+1}$ decisions for all $j$ and $E+1\leq s\leq E+S$.

        1. Each of the $j\in 1,2,...J$ sets of $2S$ steady-state Euler equations can be solved separately. `OG-Core` parallelizes this process using the maximum number of processors possible (up to $J$ processors). Solve each system of Euler equations using a multivariate root-finder to solve the $2S$ necessary conditions of the household given by the following steady-state versions of stationarized household Euler equations {eq}`EqStnrzHHeul_n`, {eq}`EqStnrzHHeul_b`, and {eq}`EqStnrzHHeul_bS` simultaneously for each $j$.

        ```{math}
        :label: EqSS_HHBC
          \bar{c}_{j,s} &= (1 + \bar{r}_{hh,a})\bar{b}_{j,s} + \bar{w}_a e_{j,s}\bar{n}_{j,s} - e^{g_y}\bar{b}_{j,s+1} + \zeta_{j,s}\frac{\overline{BQ}^i}{\lambda_j\bar{\omega}_{s}} + \eta_{j,s}\frac{\overline{TR}^i}{\lambda_j\bar{\omega}_{s}} + \hat{ubi}_{j,s} - \bar{T}_{s}  \\
          &\qquad\qquad\forall j\quad\text{and}\quad E+1\leq s\leq E+S \quad\text{where}\quad \bar{b}_{j,E+1}=0
        ```

        ```{math}
        :label: EqSS_HHeul_n
          \bar{w}_a e_{j,s}\bigl(1 - \tau^{mtrx}_{s}\bigr)(\bar{c}_{j,s})^{-\sigma} = \chi^n_{s}\biggl(\frac{b}{\tilde{l}}\biggr)\biggl(\frac{\bar{n}_{j,s}}{\tilde{l}}\biggr)^{\upsilon-1}\Biggl[1 - \biggl(\frac{\bar{n}_{j,s}}{\tilde{l}}\biggr)^\upsilon\Biggr]^{\frac{1-\upsilon}{\upsilon}} \\
          \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\forall j \quad\text{and}\quad E+1\leq s\leq E+S \\
        ```

        ```{math}
        :label: EqSS_HHeul_b
          (\bar{c}_{j,s})^{-\sigma} = e^{-\sigma g_y}\biggl[\chi^b_j\rho_s(\bar{b}_{j,s+1})^{-\sigma} + \beta_j\bigl(1 - \rho_s\bigr)\Bigl(1 + \bar{r}_{hh,a}\bigl[1 - \tau^{mtry}_{s+1}\bigr]\Bigr)(\bar{c}_{j,s+1})^{-\sigma}\biggr] \\
          \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\forall j \quad\text{and}\quad E+1\leq s\leq E+S-1 \\
        ```

        ```{math}
        :label: EqSS_HHeul_bS
          (\bar{c}_{j,E+S})^{-\sigma} = e^{-\sigma g_y}\chi^b_j(\bar{b}_{j,E+S+1})^{-\sigma} \quad\forall j
        ```

    4. Given values for $\bar{n}_{j,s}$ and $\bar{b}_{j,s+1}$ for all $j$ and $s$, solve for steady-state $\bar{L}$, $\bar{B}$, $\bar{K}_b$, $\bar{K}^d$, $\bar{K}^f$, and $\bar{Y}_b$.

        1. Use $\bar{n}_{j,s}$ and the steady-state version of the stationarized labor market clearing equation {eq}`EqStnrzMarkClrLab` to get a value for $\bar{L}$.

           ```{math}
           :label: EqSS_MarkClrLab
             \bar{L} = \sum_{s=E+1}^{E+S}\sum_{j=1}^{J} \bar{\omega}_{s}\lambda_j e_{j,s}\bar{n}_{j,s}
           ```
        2. Use $\bar{b}_{j,s+1}$ and the steady-state version of the stationarized expresssion for total savings by domestic households {eq}`EqStnrz_Bt`to solve for $\bar{B}$.

           ```{math}
           :label: EqSS_Bt
             \bar{B} \equiv \frac{1}{1 + \bar{g}_{n}}\sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\bar{\omega}_{s-1}\lambda_j\bar{b}_{j,s} + i_s\bar{\omega}_{s}\lambda_j\bar{b}_{j,s}\Bigr)
           ```

        3. Use the steady-state world interest rate $\bar{r}^*$ and aggregate labor $\bar{L}$ to solve for total private capital demand at the world interest rate $\bar{K}^{r^*}$ using the steady-state version of {eq}`EqStnrzFOC_K2`

           ```{math}
           :label: EqSS_FOC_K2
             \bar{K}^{r^*} = \bar{L}\left(\frac{(1-\gamma)^\frac{1}{\varepsilon}}{\left[\frac{\bar{r}^* + \delta - \tau^{corp}\delta^\tau}{(1-\tau^{corp})(Z)\gamma^\frac{1}{\varepsilon}}\right]^{\varepsilon-1} - \gamma^\frac{1}{\varepsilon}}\right)^\frac{\varepsilon}{\varepsilon-1}
           ```

        4. Given $\bar{B}$ and $\bar{D}^d$, use {eq}`EqStnrz_DomCapCnstr` to solve for $\bar{K}^d$. Given $\bar{K}^{r*}$ and $\bar{K}^d$, use {eq}`EqStnrz_ExDemK` to solve for $\overline{ED}^{K,r^*}$. Use $\overline{ED}^{K,r^*}$ in {eq}`EqStnrz_zetaK` to get $\bar{K}^f$. And use $\bar{K}^d$ and $\bar{K}^f$ in {eq}`EqStnrz_KtKdKf` to solve for $\bar{K}_b$.

        5. Use $\bar{K}_b$ and $\bar{L}$ in the production function {eq}`EqStnrzCESprodfun` to get a new $\bar{Y}_b$.

3. Given updated inner-loop values based on initial guesses for outer-loop variables $\{\bar{r}^i, \overline{BQ}^i, \overline{TR}^i, factor^i\}$, solve for updated values of outer-loop variables $\{\bar{r}^{i'}, \overline{BQ}^{i'}, \overline{TR}^{i'}, factor^{i'}\}$ using remaining equations.

    1. Use $\bar{Y}_b$ and $\bar{K}_b$ in {eq}`EqStnrzFOC_K` to solve for updated value of the rental rate on private capital $\bar{r}^{i'}$.

       ```{math}
       :label: EqSS_FOC_K
         \bar{r}^{i'} = (1 - \tau^{corp})(Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{\bar{Y}_b}{\bar{K}_b}\right]^\frac{1}{\varepsilon} - \delta + \tau^{corp}\delta^\tau
       ```

    2. Use $\bar{r}^{i'}$, $\bar{K}_b$, $\bar{D}$ and $\bar{b}_{j,s}$ in equations {eq}`EqUnbalGBC_rate_wedge`, {eq}`EqStnrz_rate_hh`, and {eq}`EqStnrzMarkClrBQ` to solve for updated aggregate bequests $\overline{BQ}^{i'}$.

        1. Use $\bar{r}^{i'}$ in equation {eq}`EqUnbalGBC_rate_wedge` to get an updated $\bar{r}_{gov,b}$, and then use $\bar{r}^{i'}$, the new $\bar{r}_{gov,b}$, $\bar{K}_b$, and $\bar{D}$ in {eq}`EqStnrz_rate_hh` to get a new value for $\bar{r}_{hh,b}$.

        2. Use new $\bar{r}_{hh,b}$ and $\bar{b}_{j,s}$ in {eq}`EqStnrzMarkClrBQ` to get an updated value for aggregate steady-state aggregate bequests $\overline{BQ}^{i'}$.

           ```{math}
           :label: EqSS_MarkClrBQ
             \overline{BQ}^{i'} = \left(\frac{1+\bar{r}_{hh,b}}{1 + \bar{g}_{n}}\right)\left(\sum_{s=E+2}^{E+S+1}\sum_{j=1}^J\rho_{s-1}\lambda_j\bar{\omega}_{s-1}\bar{b}_{j,s}\right)
           ```

    3. Use $\bar{Y}_b$ in long-run aggregate transfers assumption {eq}`EqStnrzTfer` to get an updated value for total transfers to households $\overline{TR}^{i'}$.

       ```{math}
       :label: EqSS_Tfer
         \overline{TR}^{i'} = \alpha_{tr}\:\bar{Y}_b
       ```

    4. Use updated $\bar{r}^{i'}$, the new $\bar{r}_{hh,b}$, $\bar{n}_{j,s}$, and $\bar{b}_{j,s+1}$ in equation {eq}`EqTaxCalcFactor` to get an updated value for the income factor $factor^{i'}$.

        1. Use updated $\bar{r}^{i'}$ to get a new value for $\bar{w}_b$ using {eq}`EqStnrz_w_of_r` to get an update value of the steady-state wage $\bar{w}_b$.

           ```{math}
           :label: EqSS_w_of_r
             \bar{w}_b = (1 - \gamma)^\frac{1}{\varepsilon}Z\left[\gamma^\frac{1}{\varepsilon}\left(\frac{(1-\gamma)^\frac{1}{\varepsilon}}{\left[\frac{\bar{r}^{i'} + \delta - \tau^{corp}\delta^\tau}{(1-\tau^{corp})Z\gamma^\frac{1}{\varepsilon}}\right]^{\varepsilon - 1} - \gamma^\frac{1}{\varepsilon}}\right) + (1-\gamma)^\frac{1}{\varepsilon}\right]^\frac{1}{\varepsilon - 1}
           ```

        2. Use new $\bar{r}_{hh,b}$, $\bar{w}_b$, $\bar{n}_{j,s}$, and $\bar{b}_{j,s+1}$ in equation {eq}`EqTaxCalcFactor` to get an updated value for the income factor $factor^{i'}$.

           ```{math}
           :label: EqSS_factor
             factor^{i'} = \frac{\text{Avg. household income in data}}{\text{Avg. household income in model}} = \frac{\text{Avg. household income in data}}{\sum_{s=E+1}^{E+S}\sum_{j=1}^J \lambda_j\bar{\omega}_s\left(\bar{r}_{hh,b}\bar{b}_{j,s} + \bar{w}_b e_{j,s}\bar{n}_{j,s}\right)} \quad\forall t
           ```

4. If the updated values of the outer-loop variables $\{\bar{r}^{i'}, \overline{BQ}^{i'}, \overline{TR}^{i'}, factor^{i'}\}$ are close enough to the initial guess for the outer-loop variables $\{\bar{r}^i, \overline{BQ}^i, \overline{TR}^i, factor^i\}$ then the fixed point is found and the steady-state equilibrium is the fixed point solution. If the outer-loop variables are not close enough to the initial guess for the outer-loop variables, then update the initial guess of the outer-loop variables $\{\bar{r}^{i+1}, \overline{BQ}^{i+1}, \overline{TR}^{i+1}, factor^{i+1}\}$ as a convex combination of the first initial guess $\{\bar{r}^{i}, \overline{BQ}^{i}, \overline{TR}^{i}, factor^{i}\}$ and the updated values $\{\bar{r}^{i'}, \overline{BQ}^{i'}, \overline{TR}^{i'}, factor^{i'}\}$ and repeat steps (2) through (4).

    1. Define a tolerance $toler_{ss,out}$ and a distance metric $\left\lVert\,\cdot\,\right\rVert$ on the space of 4-tuples of outer-loop variables $\{\bar{r}^{i}, \overline{BQ}^{i}, \overline{TR}^{i}, factor^{i}\}$. If the distance between the original guess for the outer-loop variables and the updated values for the outer-loop variables is less-than-or-equal-to the tolerance value, then the steady-state equilibrium has been found and it is the fixed point values of the variables at this point in the iteration.

       ```{math}
       :label: EqSS_toldistdone
         \left\lVert\left(\bar{r}^{i'}, \overline{BQ}^{i'}, \overline{TR}^{i'}, factor^{i'}\right) - \left(\bar{r}^{i}, \overline{BQ}^{i}, \overline{TR}^{i}, factor^{i}\right)\right\rVert \leq toler_{ss,out}
       ```

        1. Make sure that steady-state government spending is nonnegative $\bar{G}\geq 0$. If steady-state government spending is negative, that means the government is getting resources to supply the debt from outside the economy each period to stabilize the debt-to-GDP ratio. $\bar{G}<0$ is a good indicator of unsustainable policies.
	      2. Make sure that the resource constraint (goods market clearing) {eq}`EqStnrzMarkClrGoods` is satisfied. It is redundant, but this is a good check as to whether everything worked correctly.
	      3. Make sure that the government budget constraint {eq}`EqStnrzGovBC` binds.
	      4. Make sure that all the $2JS$ household Euler equations are solved to a satisfactory tolerance.

    2. If the distance metric of the original value of the outer-loop variables and the updated values is greater than the tolerance $toler_{ss,out}$, then an updated initial guess for the outer-loop variables is made as a convex combination of the first guess and the updated guess and steps (2) through (4) are repeated.

        1. The distance metric not being satisfied is the following condition.

           ```{math}
           :label: EqSS_toldistrepeat
             \left\lVert\left(\bar{r}^{i'}, \overline{BQ}^{i'}, \overline{TR}^{i'}, factor^{i'}\right) - \left(\bar{r}^{i}, \overline{BQ}^{i}, \overline{TR}^{i}, factor^{i}\right)\right\rVert > toler_{ss,out}
           ```

        2. If the distance metric is not satisfied {eq}`EqSS_toldistrepeat`, then an updated initial guess for the outer-loop variables $\{\bar{r}^{i+1}, \overline{BQ}^{i+1}, \overline{TR}^{i+1}, factor^{i+1}\}$ is made as a convex combination of the previous initial guess $\{\bar{r}^{i}, \overline{BQ}^{i}, \overline{TR}^{i}, factor^{i}\}$ and the updated values based on the previous initial guess $\{\bar{r}^{i'}, \overline{BQ}^{i'}, \overline{TR}^{i'}, factor^{i'}\}$ and repeats steps (2) through (4) with this new initial guess. The parameter $\xi_{ss}\in(0,1]$ governs the degree to which the new guess $i+1$ is close to the updated guess $i'$.

           ```{math}
           :label: EqSS_updateguess
             \left(\bar{r}^{i+1}, \overline{BQ}^{i+1}, \overline{TR}^{i+1}, factor^{i+1}\right) &= \xi_{ss}\left(\bar{r}^{i'}, \overline{BQ}^{i'}, \overline{TR}^{i'}, factor^{i'}\right) + ... \\
             &\qquad(1-\xi_{ss})\left(\bar{r}^{i}, \overline{BQ}^{i}, \overline{TR}^{i}, factor^{i}\right)
           ```

        3. Because the outer loop of the steady-state solution only has four variables, there are only four error functions to minimize or set to zero. We use a root-finder and its corresponding Newton method for the updating the guesses of the outer-loop variables because it works well and is faster than the bisection method described in the previous step. The `OG-Core` code has the option to use either the bisection method or the root fining method to updated the outer-loop variables. The root finding algorithm is generally faster but is less robust than the bisection method in the previous step.


(SecSSeqlbResults)=
### Steady-state results: default specification

  [TODO: Update the results in this section.] In this section, we use the baseline calibration described in Chapter {ref}`Chap_Calibr`, which includes the baseline tax law from `Tax-Calculator`, to show some steady-state results from `OG-Core`. Figures {numref}`FigSSeqlbHHcons`, {numref}`FigSSeqlbHHlab`, and {numref}`FigSSeqlbHHsave` show the household steady-state variables by age $s$ and lifetime income group $j$.

  ```{figure} ./images/HHcons_SS.png
  ---
  height: 500px
  name: FigSSeqlbHHcons
  ---
  Consumption $c_{j,s}$ by age $s$ and lifetime income group $j$
  ```

  ```{figure} ./images/HHlab_SS.png
  ---
  height: 500px
  name: FigSSeqlbHHlab
  ---
  Labor supply $n_{j,s}$ by age $s$ and lifetime income group $j$
  ```

  ```{figure} ./images/HHsav_SS.png
  ---
  height: 500px
  name: FigSSeqlbHHsave
  ---
  Savings $b_{j,s}$ by age $s$ and lifetime income group $j$
  ```

  {numref}`TabSSeqlbAggrVars` lists the steady-state prices and aggregate variable values along with some of the maximum error values from the characterizing equations.

  ```{list-table} **Steady-state prices, aggregate variables, and maximum errors**
  :header-rows: 1
  :name: TabSSeqlbAggrVars
  * - Variable
    - Value
    - Variable
    - Value
  * - $\bar{r}$
    - 0.630
    - $\bar{w}$
    - 1.148
  * - $\bar{Y}$
    - 0.630
    - $\bar{C}$
    - 0.462
  * - $\bar{I}$
    - 0.144
    - $\bar{K}$
    - 1.810
  * - $\bar{L}$
    - 0.357
    - $\bar{B}$
    - 2.440
  * - $\overline{BQ}$
    - 0.106
    - $factor$
    - 141,580
  * - $\overline{Rev}$
    - 0.096
    - $\overline{TR}$
    - 0.057
  * - $\bar{G}$
    - 0.023
    - $\bar{D}$
    - 0.630
  * - Max. abs. labor supply Euler error
    - 4.57e-13
    - Max. abs. savings Euler error
    - 8.52e-13
  * - Resource constraint error
    - -4.39e-15
    - Serial computation time
    - 1 hr. 25.9 sec.
  ```

  The steady-state computation time does not include any of the exogenous parameter computation processes, the longest of which is the estimation of the baseline tax functions which computation takes 1 hour and 15 minutes.


(SecNSSeqlb)=
## Stationary Non-steady-State Equilibrium

  In this section, we define the stationary non-steady-state equilibrium of the `OG-Core` model. Chapters {ref}`Chap_House` through {ref}`Chap_MarkClr` derive the equations that characterize the equilibrium of the model in their non-stationarized form. And chapter {ref}`Chap_Stnrz` derives the stationarized versions of the characterizing equations. The steady-state equilibrium definition in Section {ref}`SecEqlbSSdef` defines the long-run equilibrium where the economy settles down after many periods. The non-steady-state equilibrium in this section describes the equilibrium in all periods from the current period to the steady-state. We will need the steady-state solution from Section {ref}`SecSSeqlb` to solve for the non-steady-state equilibrium transition path.


(SecEqlbNSSdef)=
### Stationary Non-steady-State Equilibrium Definition

  We define a stationary non-steady-state equilibrium as the following.

  ```{admonition} **Definition: Stationary Non-steady-state functional equilibrium**
  :class: note
  A non-autarkic non-steady-state functional equilibrium in the `OG-Core` model is defined as stationary allocation functions of the state $\bigl\{n_{j,s,t} = \phi_{j,s}\bigl(\boldsymbol{\hat{\Gamma}}_t\bigr)\bigr\}_{s=E+1}^{E+S}$ and $\bigl\{\hat{b}_{j,s+1,t+1}=\psi_{j,s}\bigl(\boldsymbol{\hat{\Gamma}}_t\bigr)\bigr\}_{s=E+1}^{E+S}$ for all $j$ and $t$ and stationary price functions $\hat{w}(\boldsymbol{\hat{\Gamma}}_t)$ and $r(\boldsymbol{\hat{\Gamma}}_t)$ for all $t$ such that:

  1. Households have symmetric beliefs $\Omega(\cdot)$ about the evolution of the distribution of savings as characterized in {eq}`EqBeliefs`, and those beliefs about the future distribution of savings equal the realized outcome (rational expectations),

  $$
    \boldsymbol{\hat{\Gamma}}_{t+u} = \boldsymbol{\hat{\Gamma}}^e_{t+u} = \Omega^u\left(\boldsymbol{\hat{\Gamma}}_t\right) \quad\forall t,\quad u\geq 1
  $$

  2. Households optimize according to {eq}`EqStnrzHHeul_n`, {eq}`EqStnrzHHeul_b`, and {eq}`EqStnrzHHeul_bS`,
  3. Firms optimize according to {eq}`EqStnrzFOC_L` and {eq}`EqStnrzFOC_K`,
  4. Government activity behaves according to {eq}`EqUnbalGBC_rate_wedge`, {eq}`EqStnrzGovBC`, {eq}`EqStnrz_rate_hh`, and {eq}`EqStnrzClosureRule_Gt`, and
  5. Markets clear according to {eq}`EqStnrzMarkClrLab`, {eq}`EqStnrz_DtDdDf`, {eq}`EqStnrz_KtKdKf`, and {eq}`EqStnrzMarkClrBQ`.

  ```


(SecEqlbNSSsoln)=
### Stationary non-steady-state solution method: default specification

[TODO: Need to update and finish this section.]

This section describes the computational algorithm for the solution method for the stationary non-steady-state equilibrium described in the {ref}`SecEqlbSSdef`. The default specification of the model is the baseline specification (`baseline = True`) in which the government can run deficits and surplusses (`budget_balance = False`), in which the economy is a large partially open economy [$\zeta_D,\zeta_K\in(0,1)$], and in which baseline government spending $G_t$ and transfers $TR_t$ are not held constant until the closure rule (`baseline_spending = False`).

The computational algorithm for the -steady-state solution follows similar steps to the steady-state solution described in Section {ref}`SecEqlbSSsoln`. There is an outer-loop of guessed values of macroeconomic variables $\{r_t, BQ_t, TR_t\}$, but in this case, we guess the entire transition path of those variables. Then we solve the inner loop of mostly microeconomic variables for the whole transition path (many generations of households), given the outer-loop guesses. We iterate between these steps until we find a fixed point.

We call this solution algorithm the time path iteration (TPI) method or transition path iteration. This method was originally outlined in a series of papers between 1981 and 1985 [^citation_note] and in the seminal book {cite}`AuerbachKotlikoff:1987` [Chapter 4] for the perfect foresight case and in {cite}`NishiyamaSmetters:2007` Appendix II and {cite}`EvansPhillips:2014`[Sec. 3.1] for the stochastic case. The intuition for the TPI solution method is that the economy is infinitely lived, even though the agents that make up the economy are not. Rather than recursively solving for equilibrium policy functions by iterating on individual value functions, one must recursively solve for the policy functions by iterating on the entire transition path of the endogenous objects in the economy (see {cite}`StokeyLucas1989` [Chapter 17]).

The key assumption is that the economy will reach the steady-state equilibrium $\boldsymbol{\bar{\Gamma}}$ described in {ref}`SecEqlbSSdef` in a finite number of periods $T<\infty$ regardless of the initial state $\boldsymbol{\hat{\Gamma}}_1$. The first step in solving for the non-steady-state equilibrium transition path is to solve for the steady-state using the method described in Section {ref}`SecEqlbSSsoln`. After solving for the steady-state, one must then find a fixed point over the entire time path or transition path of endogenous objects that satisfies the characterizing equilibrium equations in every period.

The non-steady-state solution algorithm follows the steps below. We again use a notational convention of a subscript $a$ or a subscript $b$ in steps (?) and (?) to differentiate between the same inner-loop variables computed two different ways at two different stages of the algorithm ($\hat{w}_{t,a}$, $\hat{w}_{t,b}$, $r_{gov,t,a}$, $r_{gov,t,b}$, $r_{hh,t,a}$, $r_{hh,t,b}$, $\hat{Y}_{t,a}$, $\hat{Y}_{t,b}$, $\hat{K}_{t,a}$, $\hat{K}_{t,b}$).

We outline the stationary non-steady state (transition path) solution algorithm in the following steps.

1. Use the techniques from Section {ref}`SecDemogPopSSTP` to solve for the transition path of the stationarized population distribution matrix $\{\hat{\omega}_{s,t}\}_{s,t=E+1,1}^{E+S,T}$ and population growth rate vector $\{\tilde{g}_{n,t}\}_{t=1}^T$ of the exogenous population process.

2. Compute the steady-state solution $\{\bar{n}_{j,s},\bar{b}_{j,s+1}\}_{s=E+1}^{E+S}$ corresponding to {ref}`SecEqlbSSdef` with the {ref}`SecEqlbSSsoln`.

3. Given initial state of the economy $\boldsymbol{\hat{\Gamma}}_1$ and steady-state solutions $\{\bar{n}_{j,s},\bar{b}_{j,s+1}\}_{s=E+1}^{E+S}$, guess transition paths of outer-loop macroeconomic variables $\{r^i,\hat{BQ}^i,\hat{TR}^i\}$ such that $\hat{BQ}_1^i$ is consistent with $\boldsymbol{\hat{\Gamma}}_1$ and $\{r_t^i, \hat{BQ}_t^i, \hat{TR}_t^i\} = \{\bar{r}, \overline{BQ}, \overline{TR}\}$ for all $t\geq T$.

    1. If the economy is assumed to reach the steady state by period $T$, then we must be able to solve for every cohort's decisions in period $T$ including the decisions of agents in their first period of economically relevant life $s=E+S$. This means we need to guess time paths for the outer-loop variables that extend to period $t=T+S$. However, the values of the time path of outer-loop variables for every period $t\geq T$ are simply equal to the steady-state values.

    2. Given guess of time path for $r^i=\{r_1^i,r_2^i,...r_T^i\}

        1. Solve for the transition path of $r_{gov,t}$ using equation {eq}`EqUnbalGBC_rate_wedge`.

        2. Solve for the transition path of $r_{hh,t}$ using equation {eq}`EqStnrz_rate_hh`.

3. Given initial condition $\boldsymbol{\hat{\Gamma}}_1$, guesses for the aggregate time paths $\{\boldsymbol{r}^i,\boldsymbol{\hat{BQ}}^i, \boldsymbol{\hat{TR}}^i\}$, we solve for the inner loop lifetime decisions of every household that will be alive across the time path $\{n_{j,s,t},\hat{b}_{j,s+1,t+1}\}_{s=E+1}^{E+S}$ for all $j$ and $1\leq t\leq T$.

	- Given time path guesses $\{\boldsymbol{r}^i,\boldsymbol{\hat{BQ}}^i, \boldsymbol{\hat{TR}}^i\}$, we can compute the path of wages, $\boldsymbol{w}^i$ and then solve for each household's lifetime decisions $\{n_{j,s,t},\hat{b}_{j,s+1,t+1}\}_{s=E+1}^{E+S}$ for all $j$, $E+1\leq s \leq E+S$, and $1\leq t\leq T_2+S-1$.
        2. The household problem can be solved with a multivariate root finder solving the $2S$ equations and unknowns at once for all $j$ and $1\leq t\leq T+S-1$. The root finder uses $2S$ household Euler equations {eq}`EqStnrzHHeul_n`, {eq}`EqStnrzHHeul_b`, and {eq}`EqStnrzHHeul_bS` to solve for each household's $2S$ lifetime decisions.

    	- After solving the first iteration of time path iteration, subsequent initial values for the $J$, $2S$ root finding problems are based on the solution in the prior iteration. This speeds up computation further and makes the initial guess for the highly nonlinear system of equations start closer to the solution value.

4. Given partial equilibrium household nonsteady-state solutions $\{n_{j,s,t},\hat{b}_{j,s+1,t+1}\}_{s=E+1}^{E+S}$ for all $j$ and $1\leq t\leq T$ based on macroeconomic variable time path guesses $\{\boldsymbol{r}^i,\boldsymbol{\hat{BQ}}^i, \boldsymbol{\hat{TR}}^i\}$, compute new values for these aggregates implied by the households' solutions, $\{\boldsymbol{r}^{i'},\boldsymbol{\hat{BQ}}^{i'}, \boldsymbol{\hat{TR}}^{i'}\}$.

	1. We solve for the updated interest rate as follows:
		1. Use the guess at the path of total transfers, $\hat{TR}_{t}^{i}$ and the transfer spending rule given in Equation {eq}`EqUnbalGBCtfer` to find the implied path of GDP: $\hat{Y}_{t}^{i} = \frac{\hat{TR}_{t}^{i}}{\alpha_{tr}}$.

        2. Using the path of GDP and the household savings and labor supply decisions, $\{n_{j,s,t},\hat{b}_{j,s+1,t+1}\}_{s=E+1}^{E+S}$, compute the path of stationarizaed total tax revenue, $\hat{Revenue}_{t}^{i}$.

        3. Using the long-run debt-to-GDP ratio, the path of GDP, the path of total tax revenue, and Equation {eq}`EqUnbalGBCclosure_Gt`, find the path of stationarized government debt, $\hat{D}_{t}^{i}$.

        4. Use the capital market clearing condition from Equation {eq}`EqStnrzMarkClrCap` and $D_{t}^{i}$ to find aggregate capital in each period,

    		$$
  		    	\hat{K}_{t}^{i}=\frac{1}{1 + g_{n,t}}\sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\omega_{s-1,t-1}\lambda_j   \hat{b}_{j,s,t} + i_s\omega_{s,t}\lambda_j \hat{b}_{j,s,t}\Bigr) - D_{t}^{i}
  		  	$$

        5. Use the labor market clearing condition from Equation {eq}`EqStnrzMarkClrLab` to find the path of aggregate labor supply:

		    $$
		    	\hat{L}_{t}^{i}=\sum_{s=E+1}^{E+S}\sum_{j=1}^{J} \omega_{s,t}\lambda_j e_{j,s}n_{j,s,t}
		    $$

        6. Use the firm's production function from Equation {eq}`EqStnrzCESprodfun` to compute an updated value of $\hat{Y}_{t}$ given the values for the factors of production:

		    $$
		    	\hat{Y}_{t}^{i'} = Z_{t}\biggl[(\gamma)^\frac{1}{\varepsilon}(\hat{K}_{t}^{i})^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma)^\frac{1}{\varepsilon}(\hat{L}_{t}^{i})^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1}
		    $$

        7. Use the firm's first order condition for its choice of capital to find the updated path of interest rates,
        	$$
  		    	r_{t}^{i'} = (1 - \tau_{t}^{corp})(Z_{t})^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{\hat{Y}_{t}^{i'}}{\hat{K}_{t}^{i}}\right]^\frac{1}{\varepsilon} - \delta + \tau_{t}^{corp}\delta_{t}^\tau
  		  	$$

    2. The stationarized law of motion for total bequests {eq}`EqStnrzMarkClrBQ` provides the expression in which household savings decisions $\{b_{j,s+1,t+1}\}_{s=E+1}^{E+S}$ imply a value for aggregate bequests, $BQ_{t}^{\,i'}$. When computing aggregate bequests, we use the updated path of interest rates found above.

	    $$
		    \hat{BQ}_{t}^{\,i'} = \left(\frac{1+r_{t}^{i'}}{1 + g_{n,t}}\right)\left(\sum_{s=E+2}^{E+S+1}\sum_{j=1}^J\rho_{s-1}\lambda_j\omega_{s-1,t-1}\hat{b}_{j,s,t}\right)
	    $$

    3. In equation {eq}`EqStnrzTfer`, we defined total household transfers as a fixed percentage of GDP ($\hat{TR}_t=\alpha_{tr}\hat{Y}_t$).  To find the updated value for transfers, we find the amount of transfers implied by the most updated value of GDP, $\hat{TR}_{t}^{i'}=\alpha_{tr}\hat{Y}_{t}^{i'}$.

5. The updated values for the outer loop variables are then used to compute the percentage differences between the initial and implied values:

    1. $error_r = max\left\{\frac{r_{t}^{i'} - r_{t}^i}{r_{t}^i}\right\}_{t=0}^{T}$
    2. $error_{bq} =  max\left\{\frac{\hat{BQ}_{t}^{\,i'} - \hat{BQ}_{t}^{\,i}}{\hat{BQ}_{t}^{\,i}}\right\}_{t=0}^{T}$
    3. $error_{tr} = \left\{\frac{\hat{TR}_{t}^{\,i'} - \hat{TR}_{t}^{\,i}}{\hat{TR}_{t}^{\,i}}\right\}_{t=0}^{T}$

6. If the maximum absolute error among the three outer loop error terms is greater than some small positive tolerance $toler_{tpi,out}$, $\max\big|\left(error_r,error_{bq},error_{tr},error_f\right)\bigr| > toler_{tpi,out}$, then update the guesses for the outer loop variables as a convex combination governed by $\xi_{tpi}\in(0,1]$ of the respective initial guesses and the new implied values and repeat steps (3) through (5).

	$$
		[\boldsymbol{r}^{i+1},\boldsymbol{\hat{BQ}}^{i+1},\boldsymbol{\hat{TR}}^{i+1} ] = \xi_{tpi}[\boldsymbol{r}^{i'},\boldsymbol{\hat{BQ}}^{i'},\boldsymbol{\hat{TR}}^{i'}] + (1-\xi_{tpi})[\boldsymbol{r}^{i},\boldsymbol{\hat{BQ}}^{i},\boldsymbol{\hat{TR}}^{i}]
	$$

7. If the maximum absolute error among the five outer loop error terms is less-than-or-equal-to some small positive tolerance $toler_{tpi,out}$ in each period along the transition path, $\max\big|\left(error_r,error_{bq},error_{tr},error_f\right)\bigr| \leq toler_{tpi,out}$ then the non-steady-state equilibrium has been found.

	1. Make sure that the resource constraint (goods market clearing) {eq}`EqStnrzMarkClrGoods` is satisfied in each period along the time path. It is redundant, but this is a good check as to whether everything worked correctly.
	2. Make sure that the government budget constraint {eq}`EqStnrzGovBC` binds.
	3. Make sure that all the $(T+S)\times2JS$ household Euler equations are solved to a satisfactory tolerance.


(SecNSSeqlbResults)=
### Baseline Non-steady-state Results

[TODO: Add baseline non-steady-state results here.]


(SecEqlbFootnotes)=
## Footnotes

[^citation_note]: See {cite}`AuerbachEtAl:1981,AuerbachEtAl:1983`, {cite}`AuerbachKotlikoff:1983a,AuerbachKotlikoff:1983b,AuerbachKotlikoff:1983c`, and {cite}`AuerbachKotlikoff:1985`.
