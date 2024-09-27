(Chap_Stnrz)=
# Stationarization

The previous chapters derive all the equations necessary to solve for the steady-state and nonsteady-state equilibria of this model. However, because labor productivity is growing at rate $g_y$ as can be seen in the firms' production function {eq}`EqFirmsCESprodfun` and the population is growing at rate $\tilde{g}_{n,t}$ as defined in {eq}`EqPopGrowthTil`, the model is not stationary. Different endogenous variables of the model are growing at different rates. We have already specified three potential budget closure rules {eq}`EqUnbalGBCclosure_Gt`, {eq}`EqUnbalGBCclosure_TRt`, and {eq}`EqUnbalGBCclosure_TRGt` using some combination of government spending $G_t$ and transfers $TR_t$ that stationarize the debt-to-GDP ratio.

{numref}`TabStnrzStatVars` lists the definitions of stationary versions of all the endogenous variables. Variables with a ``$\:\,\hat{}\,\:$'' signify stationary variables. The first column of variables are growing at the productivity growth rate $g_y$. These variables are most closely associated with individual variables. The second column of variables are growing at the population growth rate $\tilde{g}_{n,t}$. These variables are most closely associated with population values. The third column of variables are growing at both the productivity growth rate $g_y$ and the population growth rate $\tilde{g}_{n,t}$. These variables are most closely associated with aggregate variables. The last column shows that the interest rates $r_t$, $r_{p,t}$ and $r_{gov,t}$, and household labor supply $n_{j,s,t}$ are already stationary.


```{list-table} **Stationary variable definitions.** Note: The interest rate $r_t$ in firm first order condition is already stationary because $Y_{m,t}$ and $K_{m,t}$ grow at the same rate and $p_{m,t}$ is stationary. Household labor supply $n_{j,s,t}\in[0,\tilde{l}]$ is stationary.
:header-rows: 2
:name: TabStnrzStatVars
* - **Sources of growth**
  -
  -
  -
* - $e^{g_y t}$
  - $\tilde{N}_t$
  - $e^{g_y t}\tilde{N}_t$
  - Not growing
* - $\hat{b}_{j,s,t}\equiv \frac{b_{j,s,t}}{e^{g_y t}}$
  - $\hat{\omega}_{s,t}\equiv\frac{\omega_{s,t}}{\tilde{N}_t}$
  - $\hat{Y}_{m,t}\equiv\frac{Y_{m,t}}{e^{g_y t}\tilde{N}_t}$
  - $n_{j,s,t}$
* - $\hat{bq}_{j,s,t}\equiv \frac{bq_{j,s,t}}{e^{g_y t}}$
  - $\hat{L}_{m,t}\equiv\frac{L_{m,t}}{\tilde{N}_t}$
  - $\hat{K}_{m,t}\equiv\frac{K_{m,t}}{e^{g_y t}\tilde{N}_t}$
  - $r_t$
* - $\hat{c}_{j,s,t}\equiv \frac{c_{j,s,t}}{e^{g_y t}}$
  -
  - $\hat{BQ}_{j,t}\equiv\frac{BQ_{j,t}}{e^{g_y t}\tilde{N}_t}$
  - $r_{p,t}$
* - $\hat{c}_{i,j,s,t}\equiv \frac{c_{i,j,s,t}}{e^{g_y t}}$
  -
  - $\hat{C}_{i,t}\equiv\frac{C_{i,t}}{e^{g_y t}\tilde{N}_t}$
  - $r_{gov,t}$
* - $\hat{tr}_{j,s,t}\equiv \frac{tr_{j,s,t}}{e^{g_y t}}$
  -
  - $\hat{K}_{g,m,t}\equiv\frac{K_{g,m,t}}{e^{g_y t}\tilde{N}_t}$
  - $r_{K,t}$
* - $\hat{ubi}_{j,s,t}\equiv\frac{ubi_{j,s,t}}{e^{g_y t}}$
  -
  - $\hat{TR}_t\equiv\frac{TR_t}{e^{g_y t}\tilde{N}_t}$
  - $p_{i,t} \equiv \frac{\tilde{p}_{i,t}}{\tilde{p}_{M,t}}$
* - $\hat{T}_{j,s,t}\equiv \frac{T_{j,s,t}}{e^{g_y t}}$
  -
  - $\hat{UBI}_t\equiv\frac{UBI_t}{e^{g_y t}\tilde{N}_t}$
  - $p_t \equiv \frac{\tilde{p}_t}{\tilde{p}_{M,t}}$
* - $\hat{w}_t\equiv \frac{w_t}{e^{g_y t}}$
  -
  - $\hat{D}_t\equiv\frac{D_t}{e^{g_y t}\tilde{N}_t}$
  - $p_{m,t} \equiv \frac{\tilde{p}_{m,t}}{\tilde{p}_{M,t}}$
* - $\hat{rm}_{j,s,t} \equiv \frac{rm_{j,s,t}}{e^{g_y t}}$
  -
  - $\hat{RM}_t \equiv \frac{RM_t}{e^{g_y t}\tilde{N}_t}$
  -
* - $\hat{pensions}_{j,s,t} \equiv \frac{pensions_{j,s,t}}{e^{g_y t}}$
  -
  -
  -
```

The usual definition of equilibrium would be allocations and prices such that households optimize {eq}`EqHH_ciDem2`, {eq}`EqHHeul_n`, {eq}`EqHHeul_b`, and {eq}`EqHHeul_bS`, firms optimize {eq}`EqFirmFOC_L` and {eq}`EqFirmFOC_K`, and markets clear {eq}`EqMarkClrLab`, {eq}`EqMarkClr_DtDdDf`, {eq}`EqMarkClr_KtKdKf`, {eq}`EqMarkClrGoods_Mm1`, {eq}`EqMarkClrGoods_M`, and {eq}`EqMarkClrBQ`. In this chapter, we show how to stationarize each of these characterizing equations so that we can use our fixed point methods described in Sections {ref}`SecEqlbSSsoln` and {ref}`SecEqlbNSSsoln` of Chapter {ref}`Chap_Eqm` to solve for the equilibria in the steady-state and transition path equilibrium definitions.


(SecStnrzHH)=
## Stationarized Household Equations

  The stationary versions of the household industry-specific goods preferences and demand equations are obtained by dividing both sides of the equations by the productivity growth rate $e^{g_y t}$,

  ```{math}
  :label: EqStnrzCompCons
    \hat{c}_{j,s,t} \equiv \prod_{i=1}^I \left(\hat{c}_{i,j,s,t} - \hat{c}_{min,i,t}\right)^{\alpha_i} \quad\forall j,s,t \quad\text{with}\quad \sum_{i=1}^I\alpha_i=1
  ```
  ```{math}
  :label: EqStnrz_cmDem2
    \hat{c}_{i,j,s,t} = \alpha_i\left(\frac{[1+\tau^c_{i,t}]p_{i,t}}{p_t}\right)^{-1}\hat{c}_{j,s,t} + \hat{c}_{min,i,t} \quad\forall i,j,s,t
  ```
  ```{math}
  :label: EqStnrz_cmin
    \hat{c}_{min,i,t} \equiv
    \begin{cases}
      \frac{c_{min,i}}{e^{g_y t}} \quad\text{for}\quad t < T \\
      \frac{c_{min,i}}{e^{g_y T}} \quad\text{for}\quad t \geq T
    \end{cases} \quad\forall i
  ```

  where {eq}`EqStnrzCompCons` is the stationarized Stone-Geary consumption aggregator for composite consumption and  {eq}`EqStnrz_cmDem2` is the stationarized household demand for the composite consumption good. The composite price aggregation equation {eq}`EqCompPnorm2` is already stationary.

  Note that the only way to stationarize the consumption aggregator {eq}`EqStnrzCompCons` and consumption demand {eq}`EqStnrz_cmDem2` is to divide $c_{min,i}$ by the growth rate $e^{g_y t}$. However, $c_{min,i}$ is already stationary. It is constant for each $m$. Therefore, the version of $\hat{c}_{min,i,t}$ divided by $e^{g_y t}$ would be changing over time (nonstationary) for $g_y\neq 0$. For this reason, we define $\hat{c}_{min,i,t}$ in {eq}`EqStnrz_cmin` as being constant after the steady-state period $T$ at whatever value it reaches at that period. In most cases with $g_y>0$, that value will be close to zero. But we use $\bar{c}_{min,i} = c_{min,i}/e^{g_y T}$ from {eq}`EqStnrz_cmin` as the steady-state value of $c_{min,i}$.

  The stationary version of the household budget constraint {eq}`EqHHBC` is found by dividing both sides of the equation by $e^{g_y t}$. For the savings term $b_{j,s+1,t+1}$, we must also multiply by $e^{g_y(t+1)}$, which leaves an $e^{g_y} = \frac{e^{g_y(t+1)}}{e^{g_y t}}$ in front of the stationarized variable.

  ```{math}
  :label: EqStnrzHHBC
    p_t\hat{c}_{j,s,t} + &\sum_{i=1}^I (1 + \tau^{c}_{i,t})p_{i,t}\hat{c}_{min,i} + e^{g_y}\hat{b}_{j,s+1,t+1} = \\
    &(1 + r_{p,t})\hat{b}_{j,s,t} + \hat{w}_t e_{j,s} n_{j,s,t} ... \\
    &\qquad +\: \hat{bq}_{j,s,t} + \hat{rm}_{j,s,t} + \hat{tr}_{j,s,t} + \hat{ubi}_{j,s,t} + \hat{pension}_{j,s,t} - \hat{tax}_{j,s,t}  \\
    &\quad\forall j,t\quad\text{and}\quad E+1\leq s\leq E+S \quad\text{where}\quad \hat{b}_{j,E+1,t}=0
  ```

  Because total bequests $BQ_t$ in $bq_{j,s,t}$ and total government transfers $TR_t$ in $tr_{j,s,t}$ grow at both the labor productivity growth rate and the population growth rate, we have to multiply and divide each of those terms by the economically relevant population $\tilde{N}_t$. This stationarizes total bequests $\hat{BQ}_t$, total transfers $\hat{TR}_t$, and the respective population level in the denominator $\hat{\omega}_{s,t}$.

  We stationarize the Euler equations for labor supply {eq}`EqHHeul_n` by dividing both sides by $e^{g_y(1-\sigma)}$. On the left-hand-side, $e^{g_y}$ stationarizes the wage $\hat{w}_t$ and $e^{-\sigma g_y}$ goes inside the parentheses and stationarizes consumption $\hat{c}_{j,s,t}$. On the right-and-side, the $e^{g_y(1-\sigma)}$ terms cancel out.

  ```{math}
  :label: EqStnrz_eul_n
    &\frac{\hat{w}_t e_{j,s}}{p_t}\bigl(1 - \tau^{mtrx}_{s,t}\bigr)(\hat{c}_{j,s,t})^{-\sigma} = \chi^n_{s}\biggl(\frac{b}{\tilde{l}}\biggr)\biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^{\upsilon-1}\Biggl[1 - \biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^\upsilon\Biggr]^{\frac{1-\upsilon}{\upsilon}} \\
    &\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\forall j,t, \quad\text{and}\quad E+1\leq s\leq E+S \\
  ```

  We stationarize the Euler equations for savings {eq}`EqHHeul_b` and {eq}`EqHHeul_bS` by dividing both sides of the respective equations by $e^{-\sigma g_y t}$. On the right-hand-side of the equation, we then need to multiply and divide both terms by $e^{-\sigma g_y(t+1)}$, which leaves a multiplicative coefficient $e^{-\sigma g_y}$,

  ```{math}
  :label: EqStnrz_eul_b
    \frac{(\hat{c}_{j,s,t})^{-\sigma}}{p_t} &= e^{-\sigma g_y}\Biggl[\chi^b_j\rho_s(\hat{b}_{j,s+1,t+1})^{-\sigma} + \\
    &\qquad\qquad\quad \beta_j\bigl(1 - \rho_s\bigr)\left(\frac{1 + r_{p,t+1}\bigl[1 - \tau^{mtry}_{s+1,t+1}\bigr] - \tau^{mtrw}_{t+1}}{p_{t+1}}\right)(\hat{c}_{j,s+1,t+1})^{-\sigma}\Biggr] \\
    &\qquad\qquad\qquad\qquad\qquad\qquad\qquad\forall j,t, \quad\text{and}\quad E+1\leq s\leq E+S-1 \\
  ```

  ```{math}
  :label: EqStnrz_eul_bS
    \frac{(\hat{c}_{j,E+S,t})^{-\sigma}}{p_t} = e^{-\sigma g_y}\chi^b_j(\hat{b}_{j,E+S+1,t+1})^{-\sigma} \quad\forall j,t
  ```

  where $\tau^{mtrw}_{t+1}$ is defined in {eq}`EqMTRwealth` in Section {ref}`SecGovWealthTax` of Chapter {ref}`Chap_UnbalGBC`.

  The stationarized versions of the remittance equations {eq}`EqHH_AggrRemit` and {eq}`EqHH_IndRemitRec` from Section {ref}`SecHHremit` in Chapter {ref}`Chap_House` are the following.

  ```{math}
  :label: EqHH_AggrRemitStnrz
  \hat{RM}_t = \begin{cases}
    &\alpha_{RM,1}\hat{Y}_t \quad\text{for}\quad t=1, \\
    &\frac{\left(1 + g_{RM,t}\right)}{e^{g_y}\left(1 + \tilde{g}_{n,t}\right)}\hat{RM}_{t-1} \quad\text{for}\quad 2\leq t \leq T_{G1}, \\
    &\left(\frac{t - T_{G1}}{T_{G2} - T_{G1}}\right)\alpha_{RM,T}\hat{Y}_t + \left(1 - \frac{t - T_{G1}}{T_{G2} - T_{G1}}\right)\frac{\left(1 + g_{RM,t}\right)}{e^{g_y}\left(1 + \tilde{g}_{n,t}\right)}\hat{RM}_{t-1} \:\:\text{for}\:\: T_{G1} < t < T_{G2}, \\
    &\alpha_{RM,T}\hat{Y}_t \quad\forall t\geq T_{G2}
  \end{cases}
  ```

  ```{math}
  :label: EqHH_IndRemitRecStnrz
  \hat{rm}_{j,s,t} = \eta_{RM,j,s,t}\frac{\hat{RM}_t}{\lambda_j\hat{\omega}_{s,t}} \quad\forall \quad j,s,t
  ```


(SecStnrzFirms)=
## Stationarized Firms Equations

  The nonstationary production function {eq}`EqFirmsCESprodfun` for each industry can be stationarized by dividing both sides by $e^{g_y t}\tilde{N}$. This stationarizes output $\hat{Y}_{m,t}$ on the left-hand-side. Because the general CES production function is homogeneous of degree one, $F(xK,xK_g,xL) = xF(K,K_g,L)$, the right-hand-side of the production function is also stationarized by dividing by $e^{g_y t}\tilde{N}_t$.

  ```{math}
  :label: EqStnrzCESprodfun
    \begin{split}
      \hat{Y}_{m,t} &= F(\hat{K}_{m,t}, \hat{K}_{g,m,t}, \hat{L}_{m,t}) \\
      &\equiv Z_{m,t}\biggl[(\gamma_m)^\frac{1}{\varepsilon_m}(\hat{K}_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m} + (\gamma_{g,m})^\frac{1}{\varepsilon_m}(\hat{K}_{g,m,t})^\frac{\varepsilon_m-1}{\varepsilon_m} + ... \\
      &\qquad\qquad\qquad (1-\gamma_m-\gamma_{g,m})^\frac{1}{\varepsilon_m}(\hat{L}_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m}\biggr]^\frac{\varepsilon_m}{\varepsilon_m-1} \quad\forall m,t
    \end{split}
  ```

  Notice that the growth term multiplied by the labor input drops out in this stationarized version of the production function. We stationarize the nonstationary profit function {eq}`EqFirmsProfit` in the same way, by dividing both sides by $e^{g_y t}\tilde{N}_t$.

  ```{math}
  :label: EqStnrzProfit
    \hat{PR}_{m,t} &= (1 - \tau^{corp}_{m,t})\Bigl[F(\hat{K}_{m,t},\hat{K}_{g,m,t},\hat{L}_{m,t}) - \hat{w}_t \hat{L}_{m,t}\Bigr] - ... \\
    &\qquad\qquad\quad \bigl(r_t + \delta_{M,t}\bigr)\hat{K}_{m,t} + \tau^{corp}_{m,t}\delta^\tau_{m,t}\hat{K}_{m,t} + \tau^{inv}_{m,t}\delta_{M,t}\hat{K}_{m,t} \quad\forall m,t
  ```

  The firms' first order equation for labor demand {eq}`EqFirmFOC_L` is stationarized by dividing both sides by $e^{g_y t}$. This stationarizes the wage $\hat{w}_t$ on the left-hand-side and cancels out the $e^{g_y t}$ term in front of the right-hand-side. To complete the stationarization, we multiply and divide the $\frac{Y_{m,t}}{e^{g_y t}L_{m,t}}$ term on the right-hand-side by $\tilde{N}_t$.

  ```{math}
  :label: EqStnrzFOC_L
    \hat{w}_t = p_{m,t}(Z_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m}\left[(1-\gamma_m-\gamma_{g,m})\frac{\hat{Y}_{m,t}}{\hat{L}_{m,t}}\right]^\frac{1}{\varepsilon_m} \quad\forall m,t
  ```

  It can be seen from the firms' first order equation for capital demand {eq}`EqFirmFOC_K` that the interest rate is already stationary. If we multiply and divide the $\frac{Y_{m,t}}{K_{m,t}}$ term on the right-hand-side by $e^{g_y t}\tilde{N}_t$, those two aggregate variables become stationary. In other words, $Y_{m,t}$ and $K_{m,t}$ grow at the same rate and $\frac{Y_{m,t}}{K_{m,t}} = \frac{\hat{Y}_{m,t}}{\hat{K}_{m,t}}$.

  ```{math}
  :label: EqStnrzFOC_K
    r_t = (1 - \tau^{corp}_{m,t})p_{m,t}(Z_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m}\left[\gamma_m\frac{\hat{Y}_{m,t}}{\hat{K}_{m,t}}\right]^\frac{1}{\varepsilon_m} - \delta_{M,t} + \tau^{corp}_{m,t}\delta^\tau_{m,t} + \tau^{inv}_{m,t}\delta_{M,t} \quad\forall m,t
  ```

  A stationary version of the firms' gross revenue attributed to each factor of production {eq}`EqFirmsMargRevEq` is found by dividing both sides of the equation by $e^{g_y t}\tilde{N}_t$.

  ```{math}
  :label: EqStnrzMargRevEq
    \hat{Y}_{m,t} = MPK_{m,t}\hat{K}_{m,t} + MPK_{g,m,t}\hat{K}_{g,m,t} + \hat{MPL}_{m,t}\hat{L}_{m,t} \quad\forall m,t
  ```

  Note that this implies that both the marginal product of private capital $MPK_{m,t}$ and the marginal product of public capital $MPK_{g,m,t}$ are already stationary, as seen in {eq}`EqFirmsMPK_opt` and {eq}`EqFirmsMPKg_opt`. However, we see in {eq}`EqFirmsMPL_opt` that the marginal product of labor is growing at rate $e^{g_y t}$ because of its relationship to the wage $w_t$. The division of both sides of {eq}`EqFirmsMargRevEq` by $e^{g_y t}\tilde{N}_t$ gives us a stationarized marginal product of labor $\hat{MPL}_{m,t}$ and a stationarized labor demand $\hat{L}_{m,t}$.

  Using the derivation of firm profits when firms are optimizing in {eq}`EqFirmsProfit_Kg` and the expressions for optimized stationary revenue {eq}`EqStnrzMargRevEq`, we can show the stationary equation for firm profits when firms are optimizing. As before, stationary profits are positive when stationary public capital is positive $\hat{K}_{g,m,t}>0$.
  ```{math}
  :label: EqStnrzProfit_Kg
    \hat{PR}_{m,t} = (1 - \tau^{corp}_{m,t})p_{m,t}MPK_{g,m,t}\hat{K}_{g,m,t} \quad\forall m,t
  ```

  Using the derivation from {eq}`EqFirmsPayout` and {eq}`EqFirms_rKt` in Chapter {ref}`Chap_Firms`, we can stationarize the terms in the right-hand-side of the expression for $r_{K,t}$ by multiplying and dividing the quotient in the last term by $e^{g_y t}\tilde{N}_t$. This implies that the interest rate paid out by the financial intermediary on private capital $r_{K,t}$ is stationary, whether the variables on the right-hand-side are non-stationary in {eq}`EqFirms_rKt` or stationarized as in {eq}`EqStnrz_rKt`.

  ```{math}
  :label: EqStnrz_rKt
    r_{K,t} =  r_t + \frac{\sum_{m=1}^M(1 - \tau^{corp}_{m,t})p_{m,t}MPK_{g,m,t}\hat{K}_{g,m,t}}{\sum_{m=1}^M\hat{K}_{m,t}} \quad\forall t
  ```

(SecStnrzGovt)=
## Stationarized Government Equations

  Each of the tax rate functions $\tau^{etr,xy}_{s,t}$, $\tau^{etr,2}_{t}$ $\tau^{mtrx}_{s,t}$, $\tau^{mtry}_{s,t}$, and $\tau^{mtrw}_{t}$ is stationary. The total tax liability function $tax_{j,s,t}$ is growing at the rate of labor productivity growth $g_y$ This can be see by looking at the decomposition of the total tax liability function into the effective tax rate times total income {eq}`EqTaxCalcLiabETR`. The effective tax rate function is stationary, and household income is growing at rate $g_y$. So household total tax liability is stationarized by dividing both sides of the equation by $e^{g_y t}$.

  ```{math}
  :label: EqStnrzLiabETR
    \hat{tax}_{j,s,t} = \tau^{etr,xy}_{s,t}\left(\hat{w}_t e_{j,s}n_{j,s,t} + r_{p,t}\hat{b}_{j,s,t}\right) + \tau^{etr,w}_t\hat{b}_{j,s,t} \quad\forall j,t \quad\text{and}\quad E+1\leq s\leq E+S
  ```

  We can stationarize the simple expressions for total government spending on household transfers $TR_t$ in {eq}`EqUnbalGBCtfer` and on public goods $G_t$ in {eq}`EqUnbalGBC_Gt` by dividing both sides by $e^{g_y t}\tilde{N}_t$,

  ```{math}
  :label: EqStnrzNomGDP
    \hat{Y}_t \equiv \sum_{m=1}^M p_{m,t} \hat{Y}_{m,t} \quad\forall t
  ```

  ```{math}
  :label: EqStnrzTfer
    \hat{TR}_t = g_{tr,t}\:\alpha_{tr}\: \hat{Y}_t \quad\forall t
  ```

  ```{math}
  :label: EqStnrz_Gt
    \hat{G}_t = g_{g,t}\:\alpha_{g}\: \hat{Y}_t \quad\forall t
  ```

  where the time varying multipliers $g_{g,t}$ and $g_{tr,t}$, respectively, are defined in {eq}`EqStnrzClosureRule_Gt` and {eq}`EqStnrzClosureRule_TRt` below. These multipliers $g_{g,t}$ and $g_{tr,t}$ do not have a ``$\:\,\hat{}\,\:$'' on them because their specifications {eq}`EqUnbalGBCclosure_Gt` and {eq}`EqUnbalGBCclosure_TRt` that are functions of nonstationary variables are equivalent to {eq}`EqStnrzClosureRule_Gt` and {eq}`EqStnrzClosureRule_TRt` specified in stationary variables.

  We can stationarize the expression for total government revenue $Rev_t$ in {eq}`EqUnbalGBCgovRev` by dividing both sides of the equation by $e^{g_y t}\tilde{N}_t$.
  ```{math}
  :label: EqStnrzGovRev
    \hat{Rev}_t &= \underbrace{\sum_{m=1}^M\Bigl[\tau^{corp}_{m,t}\bigl(p_{m,t}\hat{Y}_{m,t} - \hat{w}_t\hat{L}_{m,t}\bigr) - \tau^{corp}_{m,t}\delta^\tau_{m,t}\hat{K}_{m,t} - \tau^{inv}_{m,t}\hat{I}_{m,t}\Bigr]}_{\text{corporate tax revenue}} \\
    &\qquad + \underbrace{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\lambda_j\hat{\omega}_{s,t}\tau^{etr,xy}_{s,t}\left(\hat{x}_{j,s,t},\hat{y}_{j,s,t}\right)\bigl(\hat{x}_{j,s,t} + \hat{y}_{j,s,t}\bigr)}_{\text{household tax revenue}} \\
    &\quad + \underbrace{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\sum_{i=1}^I\lambda_j\omega_{s,t}\tau^{c}_{i,t}p_{i,t}\hat{c}_{i,j,s,t}}_{\text{consumption tax revenue}} \\
    &\quad + \underbrace{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\lambda_j\omega_{s,t}\tau^{etr,w}_{t}\hat{b}_{j,s,t}}_{\text{wealth tax revenue}} \quad\forall t
  ```

  Every term in the government budget constraint {eq}`EqUnbalGBCbudgConstr` is growing at both the productivity growth rate and the population growth rate, so we stationarize it by dividing both sides by $e^{g_y t}\tilde{N}_t$. We also have to multiply and divide the next period debt term $D_{t+1}$ by $e^{g_y(t+1)}\tilde{N}_{t+1}$, leaving the term $e^{g_y}(1 + \tilde{g}_{n,t+1})$.

  ```{math}
  :label: EqStnrzGovBC
    e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\hat{D}_{t+1} + \hat{Rev}_t = (1 + r_{gov,t})\hat{D}_t + \hat{G}_t + \hat{I}_{g,t} + \hat{Pensions}_t + \hat{TR}_t + \hat{UBI}_t \quad\forall t
  ```

  The stationarized versions of the rule for total government infrastructure investment spending $I_{g,t}$ in {eq}`EqUnbalGBC_Igt` and the rule for government investment spending in each industry in {eq}`EqUnbalGBC_Igt` are found by dividing both sides of the respective equations by $e^{g_y t}\tilde{N}_t$.
  ```{math}
  :label: EqStnrz_Igt
    \hat{I}_{g,t} = \alpha_{I,t}\: \hat{Y}_t \quad\forall t
  ```
  ```{math}
  :label: EqStnrz_Igmt
    \hat{I}_{g,m,t} = \alpha_{I,m,t}\: \hat{I}_{g,t} \quad\forall m,t
  ```

  The stationarized version of the law of motion for the public capital stock in each industry $K_{g,m,t}$ in {eq}`EqUnbalGBC_Kgmt` is found by dividing both sides of the equation by $e^{g_y t}\tilde{N}_t$ then multiply and divide the $K_{g,m,t+1}$ term on the left-hand-side by $e^{g_y(t+1)}\tilde{N}_{t+1}$, leaving the term $e^{g_y}(1 + \tilde{g}_{n,t+1})$ in the denominator of the right-hand-side.

  ```{math}
  :label: EqStnrz_Kgmt
    \hat{K}_{g,m,t+1} = \frac{(1 - \delta_g)\hat{K}_{g,m,t} + \hat{I}_{g,m,t}}{e^{g_y}(1 + \tilde{g}_{n,t+1})}  \quad\forall m,t
  ```

  Stationary aggregate universal basic income expenditure is found in one of two ways depending on how the individual UBI payments $ubi_{j,s,t}$ are modeled. In Section {ref}`SecUBI` of Chapter {ref}`Chap_UnbalGBC`, we discuss how UBI payments to households $ubi_{j,s,t}$ can be growth adjusted so that they grow over time at the rate of productivity growth or non-growth adjusted such that they are constant overtime. In the first case, when UBI benefits are growth adjusted and growing over time, the stationary aggregate government UBI payout $\hat{UBI}_t$ is found by dividing {eq}`EqUnbalGBC_UBI` by $e^{g_y t}\tilde{N}_t$. In the second case, when UBI benefits are constant over time and not growing with productivity, the stationary aggregate government UBI payout $\hat{UBI}_t$ is found by dividing {eq}`EqUnbalGBC_UBI` by only $\tilde{N}_t$.

  ```{math}
  :label: EqStnrzGBC_UBI
    \hat{UBI}_t =
      \begin{cases}
        \sum_{s=E+1}^{E+S}\sum_{j=1}^J \lambda_j\hat{\omega}_{s,t} \hat{ubi}_{j,s,t} \quad\forall t \quad\text{if}\quad ubi_{j,s,t} \:\:\text{is growth adjusted} \\
        \sum_{s=E+1}^{E+S}\sum_{j=1}^J \lambda_j\hat{\omega}_{s,t} ubi_{j,s,t} \quad\forall t \quad\text{if}\quad ubi_{j,s,t} \:\:\text{is not growth adjusted}
      \end{cases}
  ```

  The expression for the interest rate on government debt $r_{gov,t}$ in {eq}`EqUnbalGBC_rate_wedge` is already stationary because every term on the right-hand-side is already stationary. The net return on capital, $r_{K,t}$ is also stationary as shown in {eq}`EqStnrz_rKt`. The expression for the return to household savings $r_{p,t}$ in {eq}`eq_portfolio_return` is equivalent to its stationary representation because the same macroeconomic variables occur linearly in both the numerator and denominator.

  ```{math}
    :label: EqStnrz_rate_p
    r_{p,t} = \frac{r_{gov,t}\hat{D}_{t} + r_{K,t}\hat{K}_{t}}{\hat{D}_{t} + \hat{K}_{t}} \quad\forall t \quad\text{where}\quad \hat{K}_t \equiv \sum_{m=1}^M \hat{K}_{m,t}
  ```

  The long-run debt-to-GDP ratio condition is also the same in both the nonstationary version in {eq}`EqUnbalGBC_DY` as well as the stationary version below because the endogenous side is a ratio of macroeconomic variables that are growing at the same rate, with the exception of already stationary $p_t$.

  ```{math}
  :label: EqStnrz_DY
    \hat{D}_t = \alpha_D\hat{Y}_t \quad\Rightarrow\quad \frac{\hat{D}_t}{\hat{Y}_t} = \alpha_D \quad\text{for}\quad t\geq T
  ```

  The three potential budget closure rules {eq}`EqUnbalGBCclosure_Gt`, {eq}`EqUnbalGBCclosure_TRt`, and {eq}`EqUnbalGBCclosure_TRGt` are the last government equations to stationarize. In each of the cases, we simply divide both sides by $e^{g_y t}\tilde{N}_t$.

  ```{math}
  :label: EqStnrzClosureRule_Gt
  \begin{split}
    &\hat{G}_t = g_{g,t}\:\alpha_{g}\: \hat{Y}_t \\
    &\text{where}\quad g_{g,t} =
    \begin{cases}
      1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad\text{if}\quad t < T_{G1} \\
      \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\left[\rho_{d}\alpha_{D}\hat{Y}_{t} + (1-\rho_{d})\hat{D}_{t}\right] - (1+r_{gov,t})\hat{D}_{t} - \hat{TR}_{t} - \hat{I}_{g,t} - \hat{UBI}_t + \hat{Rev}_{t}}{\alpha_g \hat{Y}_t} \:\text{if}\: T_{G1}\leq t<T_{G2} \\
      \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\alpha_{D}\hat{Y}_{t} - (1+r_{gov,t})\hat{D}_{t} - \hat{TR}_{t} - \hat{I}_{g,t} - \hat{UBI}_t + \hat{Rev}_{t}}{\alpha_g \hat{Y}_t} \qquad\qquad\quad\,\text{if}\quad t \geq T_{G2}
    \end{cases} \\
    &\text{and}\quad g_{tr,t} = 1 \quad\forall t
  \end{split}
  ```
  or

  ```{math}
  :label: EqStnrzClosureRule_TRt
  \begin{split}
    &\hat{TR}_t = g_{tr,t}\:\alpha_{tr}\: \hat{Y}_t \\
    &\text{where}\quad g_{tr,t} =
    \begin{cases}
      1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\:\:\:\text{if}\quad t < T_{G1} \\
      \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\left[\rho_{d}\alpha_{D}\hat{Y}_{t} + (1-\rho_{d})\hat{D}_{t}\right] - (1+r_{gov,t})\hat{D}_{t} - \hat{G}_{t} - \hat{I}_{g,t} - \hat{UBI}_t + \hat{Rev}_{t}}{\alpha_{tr} \hat{Y}_t} \:\text{if}\: T_{G1}\leq t<T_{G2} \\
      \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\alpha_{D}\hat{Y}_{t} - (1+r_{gov,t})\hat{D}_{t} - \hat{G}_{t} - \hat{I}_{g,t} - \hat{UBI}_t + \hat{Rev}_{t}}{\alpha_{tr} \hat{Y}_t} \qquad\qquad\quad\,\text{if}\quad t \geq T_{G2}
    \end{cases} \\
    &\text{and}\quad g_{g,t} = 1 \quad\forall t
  \end{split}
  ```
  or

  ```{math}
  :label: EqStnrzClosureRule_TRGt
  \begin{split}
    &\hat{G}_t + \hat{TR}_t = g_{trg,t}\left(\alpha_g + \alpha_{tr}\right)\hat{Y}_t \quad\Rightarrow\quad \hat{G}_t = g_{trg,t}\:\alpha_g\: \hat{Y}_t \quad\text{and}\quad \hat{TR}_t = g_{trg,t}\:\alpha_{tr}\: \hat{Y}_t \\
    &\text{where}\quad g_{trg,t} =
    \begin{cases}
      1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\:\quad \text{if}\quad t < T_{G1} \\
      \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\left[\rho_{d}\alpha_{D}\hat{Y}_{t} + (1-\rho_{d})\hat{D}_{t}\right] - (1+r_{gov,t})\hat{D}_{t} - \hat{I}_{g,t} - \hat{UBI}_t + \hat{Rev}_{t}}{\left(\alpha_g + \alpha_{tr}\right)\hat{Y}_t} \:\text{if}\: T_{G1}\leq t<T_{G2} \\
      \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\alpha_{D}\hat{Y}_{t} - (1+r_{gov,t})\hat{D}_{t} - \hat{I}_{g,t} - \hat{UBI}_t + \hat{Rev}_{t}}{\left(\alpha_g + \alpha_{tr}\right)\hat{Y}_t} \qquad\qquad\quad\,\text{if}\quad t \geq T_{G2}
    \end{cases}
  \end{split}
  ```


### Stationarized Pension System Equations

#### Stiationarized Notional Defined Contributions Equations

The stationarized NDC pension amount is given by:

  ```{math}
  :label: eqn:ndc_amount_stationarized
  \hat{\theta}_{j,u,t+u-s}=
    \begin{cases}
      0, & \text{if}\ u < R \\
      \biggl[\sum_{s=E}^{R-1}\tau^{p}_{t}\frac{\hat{w}_{t}}{e^{g_{y}(u-s)}}e_{j,s}n_{j,s,t}(1 + g_{NDC,t})^{R-s-1}\biggr]\delta_{R, t}, & \text{otherwise}
    \end{cases}
  ```


The stationarized derivative of the pension amount it slightly simpler since it involved only current period wages.  We give the derivation first.

The FOC for the choice of labor supply is given by:

  ```{math}
  \begin{split}
    \biggl(\frac{1}{1 + \tau^{c}_{j,s,t}}\biggr)&\biggl(w_{t}e_{j,s} - \frac{\partial T_{j,s,t}}{\partial n_{j,s,t}}\biggr)c^{-\sigma}_{j,s,t} + \sum_{u=R}^{E+S}\beta^{u-s}\prod_{v=s}^{u}(1-\rho_{v})\frac{\partial \theta_{j,u,t+u-s}}{\partial n_{j,s,t}}c^{-\sigma}_{j,u,t+u-s}\biggl(\frac{1}{1+\tau^{c}_{j,u,t+u-s}}\biggr) \\
    & = MDU_l(n_{j,s,t})e^{g_{y}t(1-\sigma)}
  \end{split}
  ```

where we now pull the growth factor out of the marginal disutility of labor term to aid in the exposition of the stationarization.  To stationarize this equation, we divide both sides through by $e^{g_{y}t(1-\sigma)}$.

  ```{math}
  \begin{split}
    \biggl(\frac{1}{1 + \tau^{c}_{j,s,t}}\biggr)&\biggl(\frac{w_{t}}{e^{g_{y}t}}e_{j,s} - \frac{\partial T_{j,s,t}}{\partial n_{j,s,t}}\biggr)\frac{c^{-\sigma}_{j,s,t}}{e^{g_{y}t(-\sigma)}} + \sum_{u=R}^{E+S}\beta^{u-s}\prod_{v=s}^{u}(1-\rho_{v})\frac{\partial \theta_{j,u,t+u-s}}{\partial n_{j,s,t}e^{g_{y}t}}\frac{c^{-\sigma}_{j,u,t+u-s}}{e^{g_{y}t(-\sigma)}}\biggl(\frac{1}{1+\tau^{c}_{j,u,t+u-s}}\biggr) \\
    & = MDU_l(n_{j,s,t})\frac{e^{g_{y}t(1-\sigma)}}{e^{g_{y}t(1-\sigma)}}
  \end{split}
  ```

Which we can write as:

  ```{math}
  \begin{split}
    \biggl(\frac{1}{1 + \tau^{c}_{j,s,t}}\biggr)&\biggl(\hat{w}_{t}e_{j,s} - \frac{\partial T_{j,s,t}}{\partial n_{j,s,t}}\biggr)\hat{c}^{-\sigma}_{j,s,t} + \sum_{u=R}^{E+S}\beta^{u-s}\prod_{v=s}^{u}(1-\rho_{v})\frac{\partial \hat{\theta}_{j,u,t+u-s}}{\partial n_{j,s,t}}\hat{c}^{-\sigma}_{j,u,t+u-s}e^{g_{y}(u-s)(-\sigma)}\biggl(\frac{1}{1+\tau^{c}_{j,u,t+u-s}}\biggr) \\
    & = MDU_l(n_{j,s,t})
  \end{split}
  ```

where $\frac{\partial \hat{\theta}_{j,u,t+u-s}}{\partial n_{j,s,t}}$ is given by:

  ```{math}
  :label: eqn:ndc_deriv_stationarized
  \frac{\partial \theta_{j,u,t+u-s}}{\partial n_{j,s,t}} =
    \begin{cases}
      \tau^{p}_{t}\hat{w}_{t}e_{j,s}(1+g_{NDC,t})^{u - s}\delta_{R,t}, & \text{if}\ s<R-1 \\
      0, & \text{if}\ s \geq R \\
    \end{cases}
  ```


#### Stationarized Defined Benefits Equations

Stationarized pension amount:

  ```{math}
  :label: eqn:db_amount_staionarized
  \hat{\theta}_{j,u,t+u-s} = \biggl[\frac{\sum_{s=R-ny}^{R-1}\frac{\hat{w}_{t}}{e^{g_{y}(u-s)}}e_{j,s}n_{j,s,t}}{ny}\biggr]\times Cy \times \alpha_{DB}, \ \ \forall u \geq R
  ```

Stationarized pension amount derivative:

  ```{math}
  :label: eqn:db_deriv_stationarized
    \frac{\partial \hat{\theta}_{j,u,t+u-s}}{\partial n_{j,s,t}} =
      \begin{cases}
        0 , & \text{if}\ s < R - Cy \\
        \hat{w}_{t}e_{j,s}\alpha_{DB}\times \frac{Cy}{ny}, & \text{if}\  R - Cy <= s < R  \\
        0, & \text{if}\ s \geq R \\
      \end{cases}
  ```

#### Stationarized Points System Equations

Stationarized pension amount:

  ```{math}
  :label: eqn:ps_amount_stationarized
  \hat{\theta}_{j,u,t+u-s} =\sum_{s=E}^{R-1}\frac{\hat{w}_{t}}{e^{g_{y}(u-s)}}e_{j,s}n_{j,s,t}v_{t}, \ \ \forall u \geq R
  ```

Stationarized pension amount derivative:

  ```{math}
  :label: eqn:ps_deriv_stationarized
  \frac{\partial \hat{\theta}_{j,u,t+u-s}}{\partial n_{j,s,t}} =
    \begin{cases}
      \hat{w}_{t}e_{j,s} v_{t}, & \text{if}\ s < R  \\
      0, & \text{if}\ s \geq R \\
    \end{cases}
  ```

(SecStnrzMC)=
## Stationarized Market Clearing Equations

  The labor market clearing equation {eq}`EqMarkClrLab` is stationarized by dividing both sides by $\tilde{N}_t$.

  ```{math}
  :label: EqStnrzMarkClrLab
    \sum_{m=1}^M \hat{L}_{m,t} = \sum_{s=E+1}^{E+S}\sum_{j=1}^{J} \hat{\omega}_{s,t}\lambda_j e_{j,s}n_{j,s,t} \quad \forall t
  ```

  Total savings by domestic households $B_t$ from {eq}`EqMarkClr_Bt` is stationarized by dividing both sides by $e^{g_y t}\tilde{N}_t$. The $\omega_{s,t-1}$ terms on the right-hand_side require multiplying and dividing by $\tilde{N}_{t-1}$, which leads to the division of $1 + \tilde{g}_{n,t}$.

  ```{math}
  :label: EqStnrz_Bt
    \hat{B}_t \equiv \frac{1}{1 + \tilde{g}_{n,t}}\sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\hat{\omega}_{s-1,t-1}\lambda_j b_{j,s,t} + i_s\hat{\omega}_{s,t-1}\lambda_j \hat{b}_{j,s,t}\Bigr) \quad \forall t
  ```

  And the total domestic savings constraint {eq}`EqMarkClr_DomCapCnstr` is stationarized by dividing both sides by $e^{g_y t}\tilde{N}_t$.

  ```{math}
  :label: EqStnrz_DomCapCnstr
    \hat{K}^d_t + \hat{D}^d_t = \hat{B}_t \quad \forall t
  ```

  The stationarized law of motion for foreign holdings of government debt {eq}`EqMarkClr_zetaD` and the government debt market clearing condition {eq}`EqMarkClr_DtDdDf`, respectively, are solved for by dividing both sides by $e^{g_y t}\tilde{N}_t$.

  ```{math}
  :label: EqStnrz_zetaD
    e^{g_y}\bigl[1 + \tilde{g}_{n,t+1}\bigr]\hat{D}^{f}_{t+1} = \hat{D}^{f}_{t} + \zeta_{D}\Bigl(e^{g_y}\bigl[1 + \tilde{g}_{n,t+1}\bigr]\hat{D}_{t+1} - \hat{D}_{t}\Bigr) \quad\forall t
  ```

  ```{math}
  :label: EqStnrz_DtDdDf
    \hat{D}_t = \hat{D}^d_t + \hat{D}^f_t \quad\forall t
  ```

  The private capital market clearing equation {eq}`EqMarkClr_KtKdKf` is stationarized by dividing both sides by $e^{g_y t}\tilde{N}_t$, as is the expression for excess demand at the world interest rate {eq}`EqMarkClr_ExDemK` and the exogenous expression for foreign private capital flows {eq}`EqMarkClr_zetaK`.

  ```{math}
  :label: EqStnrz_KtKdKf
    \hat{K}_t = \hat{K}^d_t + \hat{K}^f_t \quad\forall t \quad\text{where}\quad \hat{K_t} \equiv \sum_{m=1}^M \hat{K}_{m,t}
  ```

  ```{math}
  :label: EqStnrz_ExDemK
    \hat{ED}^{K,r^*}_t \equiv \hat{K}^{r^*}_t - \hat{K}^d_t \quad\forall t \quad\text{where}\quad \hat{K}^{r^*}_t \equiv \sum_{m=1}^M \hat{K}^{r^*}_{m,t}
  ```

  ```{math}
  :label: EqStnrz_zetaK
    \hat{K}^{f}_t = \zeta_{K}\hat{ED}^{K,r^*}_t \quad\forall t
  ```

  We stationarize the goods market clearing equations for the first $M-1$ industries {eq}`EqMarkClrGoods_Mm1` and for the $M$th industry {eq}`EqMarkClrGoods_M` by dividing both sides by $e^{g_y t}\tilde{N}_t$. On the right-hand-side, we must multiply and divide the $K^d_{t+1}$ term and the $D^f_{t+1}$ term, respectively, by $e^{g_y(t+1)}\tilde{N}_{t+1}$ leaving the coefficient $e^{g_y}(1+\tilde{g}_{n,t+1})$.
  ```{math}
  :label: EqStnrzMarkClrGoods_Mm1
    \hat{Y}_{m,t} = \hat{C}_{m,t} \quad\forall t \quad\text{and}\quad m=1,2,...M-1
  ```
  ```{math}
  :label: EqStnrzMarkClrGoods_M
    \hat{Y}_{M,t} &= \hat{C}_{M,t} + \hat{I}_{M,t} + \hat{I}_{g,t} + \hat{G}_t + r_{p,t} \hat{K}^f_t + r_{p,t}\hat{D}^f_t ... \\
    &\quad - \Bigl(e^{g_y}\bigl[1 + \tilde{g}_{n,t+1}\bigr]\hat{K}^f_{t+1} - \hat{K}^f_t\Bigr) - \Bigl(e^{g_y}\bigl[1 + \tilde{g}_{n,t+1}\bigr]\hat{D}^f_{t+1} - \hat{D}^f_t\Bigr) - \hat{RM}_t \quad\forall t
  ```
  where
  ```{math}
  :label: EqStnrzEqCmt
    \hat{C}_{m,t} \equiv \sum_{i=1}^{I}\sum_{s=E+1}^{E+S}\sum_{j=1}^{J}\hat{\omega}_{s,t}\lambda_j \pi_{i,m} \hat{c}_{i,j,s,t} \quad\forall m,t
  ```
  and
  ```{math}
  :label: EqStnrzMarkClrGoods_IMt
    \hat{I}_{M,t} &\equiv e^{g_y}\bigl(1 + \tilde{g}_{n,t+1}\bigr)\sum_{m=1}^M \hat{K}_{m,t+1} - (1 - \delta_{M,t})\sum_{m=1}^M \hat{K}_{m,t} \quad\forall t \\
    &= e^{g_y}\bigl(1 + \tilde{g}_{n,t+1}\bigr)\hat{K}_{t+1} - (1 - \delta_{M,t})\hat{K}_t \\
    &= e^{g_y}\bigl(1 + \tilde{g}_{n,t+1}\bigr)(\hat{K}^d_{t+1} + \hat{K}^f_{t+1}) - (1 - \delta_{M,t})(\hat{K}^d_t + \hat{K}^f_t)
  ```

  We stationarize the law of motion for total bequests $BQ_t$ in {eq}`EqMarkClrBQ` by dividing both sides by $e^{g_y t}\tilde{N}_t$. Because the population levels in the summation are from period $t-1$, we must multiply and divide the summed term by $\tilde{N}_{t-1}$ leaving the term in the denominator of $1+\tilde{g}_{n,t}$.

  ```{math}
  :label: EqStnrzMarkClrBQ
    \hat{BQ}_{t} = \left(\frac{1+r_{p,t}}{1 + \tilde{g}_{n,t}}\right)\left(\sum_{s=E+2}^{E+S+1}\sum_{j=1}^J\rho_{s-1}\lambda_j\hat{\omega}_{s-1,t-1}\hat{b}_{j,s,t}\right) \quad\forall t
  ```

  The demand side of aggregate consumption $\hat{C}_{m,t}$, aggregate investment $\hat{I}_{M,t}$, and aggregate bequests $\hat{BQ}_t$ is each indirectly affected by the size of remittances, described equations {eq}`EqHH_AggrRemit` and {eq}`EqHH_IndRemitRec` in Section {ref}`SecHHremit` of Chapter {ref}`Chap_House` and in the stationarized versions of those equations {eq}`EqHH_AggrRemitStnrz` and {eq}`EqHH_IndRemitRecStnrz` in Section {ref}`SecStnrzHH` in this Chapter.
