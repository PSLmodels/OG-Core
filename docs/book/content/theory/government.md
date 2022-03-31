
(Chap_UnbalGBC)=
# Government

In `OG-Core`, the government enters by levying taxes on households, providing transfers to households, levying taxes on firms, spending resources on public goods and infrastructure, and making rule-based adjustments to stabilize the economy in the long-run. It is this last activity that is the focus of this chapter.


(SecUnbalGBC_policy)=
## Government Tax and Transfer Policy

Government levies taxes on households and firms, funds public pensions, and makes other transfers to households.

### Taxes

#### Individual income taxes

#### Consumption taxes

#### Wealth taxes

#### Corporate income taxes

### Spending

Government spending is comprised of government provided pension benefits, lump sum transfers, universal basic income payments, infrastructure investment, spending on public goods, and interest payments on debt.  Below, we describe the transfer spending amounts.  Spending on infrastructure, public goods, and interest are described in {ref}`SecUnbalGBCbudgConstr`.
#### Pensions

[TODO: Add description of government pensions and the relevant parameters]

#### Lump sum transfers:

 Aggregate non-pension transfers to households are assumed to be a fixed fraction $\alpha_{tr}$ of GDP each period:

  ```{math}
  :label: EqUnbalGBCtfer
    TR_t = g_{tr,t}\:\alpha_{tr}\: Y_t \quad\forall t
  ```
  The time dependent multiplier $g_{tr,t}$ in front of the right-hand-side of {eq}`EqUnbalGBCtfer` will equal 1 in most initial periods. It will potentially deviate from 1 in some future periods in order to provide a closure rule that ensures a stable long-run debt-to-GDP ratio. We will discuss the closure rule in Section {ref}`SecUnbalGBCcloseRule`.

  We assume that total non-pension transfers are distributed in a lump sum manner to households.  The distribution across households by age and lifetime income group is parameterized by the the parameters $\eta_{j,s,t}$, which are in the time specific $\boldsymbol{\eta}_{t}$ matrix. Thus, transfers to households of lifetime income group $j$, age $s$, at time $t$ are given as:

   ```{math}
  :label: Eq_tr
    tr_{j,s,t} = \boldsymbol{\eta}_{t} TR_{t}
  ```

#### Universal basic income

[TODO: This section is far along but needs to be updated.]

 Universal basic income (UBI) transfers show up in the household budget constraint {eq}`EqHHBC`. Household amounts of UBI can vary by household age $s$, lifetime income group $j$, and time period $t$.  These transfers are represented by $ubi_{j,s,t}$.


(SecUBIcalc)=
##### Calculating UBI

  Household transfers in model units $ubi_{j,s,t)}$ are a function of five policy parameters described in the [`default_parameters.json`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/default_parameters.json) file (`ubi_growthadj`, `ubi_nom_017`, `ubi_nom_1864`, `ubi_nom_65p`, and `ubi_nom_max`).  Three additional parameters provide information on household structure by age, lifetime income group, and year: [`ubi_num_017_mat`, `ubi_num_1864_mat`, `ubi_num_65p_mat`].

  As a convenience to users, UBI policy parameters `ubi_nom_017`, `ubi_nom_1864`, `ubi_nom_65p`, and `ubi_nom_max` are entered as nominal amounts (e.g., in dollars or pounds). The parameter `ubi_nom_017` represents the nominal value of the UBI transfer to each household per dependent child age 17 and under. The parameter `ubi_nom_1864` represents the nominal value of the UBI transfer to each household per adult between the ages of 18 and 64. And `ubi_nom_65p` is the nominal value of UBI transfer to each household per senior 65 and over. The maximum UBI benefit per household, `ubi_nom_max`, is also a nominal amount.  From these parameters, the model computes nominal UBI payments to each household in the model:

  ```{math}
  :label: EqUBIubi_nom_jst0
    \begin{split}
      ubi^{nom}_{j,s,t=0} = \min\Bigl(&\texttt{ubi_nom_max}, \\
      &\texttt{ubi_nom_017} * \texttt{ubi_num_017_mat}_{j,s} + \\
      &\texttt{ubi_nom_1864} * \texttt{ubi_num_1864_mat}_{j,s} + \\
      &\texttt{ubi_nom_65p} * \texttt{ubi_num_65p_mat}_{j,s}\Bigr) \quad\forall j,s
    \end{split}
  ```

  The rest of the time periods of the household UBI transfer and the respective steady-states are determined by whether the UBI is growth adjusted or not as given in the `ubi_growthadj` boolean parameter. The following two sections cover these two cases.


(SecUBI_NonGrowthAdj)=
###### UBI specification not adjusted for economic growth

  A non-growth adjusted UBI (`ubi_growthadj = False`) is one in which the initial nonstationary nominal-valued $t=0$ UBI matrix $ubi^{\$}_{j,s,t=0}$ does not grow, while the economy's long-run growth rate is $g_y$ for the most common parameterization is positive ($g_y>0$).

  ```{math}
  :label: EqUBIubi_nom_NonGrwAdj_jst
    ubi^{nom}_{j,s,t} = ubi^{nom}_{j,s,t=0} \quad\forall j,s,t
  ```

  As described in the [OG-Core chapter on stationarization](https://pslmodels.github.io/OG-Core/content/theory/stationarization.html), the stationarized UBI transfer to each household $\hat{ubi}_{j,s,t}$ is the nonstationary transfer divided by the growth rate since the initial period. When the long-run economic growth rate is positive $g_y>0$ and the UBI specification is not growth-adjusted the steady-state stationary UBI household transfer is zero $\overline{ubi}_{j,s}=0$ for all lifetime income groups $j$ and ages $s$ as time periods $t$ go to infinity. However, to simplify, we assume in this case that the stationarized steady-state UBI transfer matrix to households is the stationarized value of that matrix in period $T$.

  ```{math}
  :label: EqUBIubi_mod_NonGrwAdj_SS
    \overline{ubi}_{j,s} = ubi_{j,s,t=T} \quad\forall j,s
  ```

  Note that in non-growth-adjusted case, if $g_y<0$, then the stationary value of $\hat{ubi}_{j,s,t}$ is going to infinity as $t$ goes to infinity. Therefore, a UBI specification must be growth adjusted for any assumed negative long run growth $g_y<0$.[^GrowthAdj_note]


(SecUBI_GrowthAdj)=
###### UBI specification adjusted for economic growth

  Put description of growth-adjusted specification here.


(SecUnbalGBCrev)=
## Government Tax Revenue

  We see from the household's budget constraint that taxes $T_{s,t}$ and transfers $TR_{t}$ enter into the household's decision,

  ```{math}
  :label: EqHHBC
    c_{j,s,t} + b_{j,s+1,t+1} &= (1 + r_{p,t})b_{j,s,t} + w_t e_{j,s} n_{j,s,t} + \zeta_{j,s}\frac{BQ_t}{\lambda_j\omega_{s,t}} + \eta_{j,s,t}\frac{TR_{t}}{\lambda_j\omega_{s,t}} + ubi_{j,s,t} - T_{s,t}  \\
    &\quad\forall j,t\quad\text{and}\quad s\geq E+1 \quad\text{where}\quad b_{j,E+1,t}=0\quad\forall j,t
  ```

  where we defined the tax liability function $T_{s,t}$ in {eq}`EqTaxCalcLiabETR` as an effective tax rate times total income and the transfer distribution function $\eta_{j,s,t}$ is uniform across all households. And government revenue from the corporate income tax rate $\tau^{corp}_t$ and the tax on depreciation expensing $\tau^\delta$ enters the firms' profit function.

  ```{math}
  :label: EqFirmsProfit2
    PR_t = (1 - \tau^{corp}_t)\bigl(Y_t - w_t L_t\bigr) - \bigl(r_t + \delta\bigr)K_t + \tau^{corp}_t\delta^\tau_t K_t \quad\forall t
  ```

  We define total government revenue from taxes as the following,

  ```{math}
  :label: EqUnbalGBCgovRev
    Rev_t = \underbrace{\tau^{corp}_t\bigl[Y_t - w_t L_t\bigr] - \tau^{corp}_t\delta^\tau_t K_t}_{\text{corporate tax revenue}} + \underbrace{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\lambda_j\omega_{s,t}\tau^{etr}_{s,t}\left(x_{j,s,t},y_{j,s,t}\right)\bigl(x_{j,s,t} + y_{j,s,t}\bigr)}_{\text{household tax revenue}} \quad\forall t
  ```

  where household labor income is defined as $x_{j,s,t}\equiv w_t e_{j,s}n_{j,s,t}$ and capital income $y_{j,s,t}\equiv r_{p,t} b_{j,s,t}$.

(SecUnbalGBCbudgConstr)=
## Government Budget Constraint

  Let the level of government debt in period $t$ be given by $D_t$. The government budget constraint requires that government revenue $Rev_t$ plus the budget deficit ($D_{t+1} - D_t$) equal expenditures on interest of the debt, government spending on public goods $G_t$, infrastructure investments $I_{gov,t}$, and total transfer payments to households $TR_t$ every period $t$,

  ```{math}
  :label: EqUnbalGBCbudgConstr
    D_{t+1} + Rev_t = (1 + r_{gov,t})D_t + G_t + I_{g,t} + TR_t + UBI_t  \quad\forall t
  ```

  where $r_{gov,t}$ is the interest rate paid by the government, $G_{t}$ is government spending on public goods, $I_{gov,t}$ is government spending on infrastructure investment, $TR_{t}$ are non-pension government transfers, and $UBI_t$ is the total UBI transfer outlays across households in time $t$.



  We assume that government spending on public goods is a fixed fraction of GDP each period in the initial periods.

  ```{math}
  :label: EqUnbalGBC_Gt
    G_t = g_{g,t}\:\alpha_{g}\: Y_t
  ```

  Similar to transfers $TR_t$, the time dependent multiplier $g_{g,t}$ in front of the right-hand-side of {eq}`EqUnbalGBC_Gt` will equal 1 in most initial periods. It will potentially deviate from 1 in some future periods in order to provide a closure rule that ensures a stable long-run debt-to-GDP ratio. We make this more specific in the next section.

  Government infrastructure investment spending, $I_{g,t}$ is assumed to be a time-dependent fraction of GDP.

  ```{math}
  :label: EqUnbalGBC_Igt
    I_{g,t} = \alpha_{I,t}\: Y_t \quad\forall t
  ```
  The stock of public capital (i.e., infrastructure) evolves according to the law of motion,

  ```{math}
  :label: EqUnbalGBC_Kgt
    K_{g,t+1} = (1 - \delta^{g}) K_{g,t} + I_{g,t} \quad\forall t,
  ```

  where $\delta^g$ is the depreciation rate on infrastructure.  The stock of public capital complements labor and private capital in the production function of the representative firm, in Equation {eq}`EqFirmsCESprodfun`.

  Aggregate spending on UBI at time $t$ is the sum of UBI payments across all households at time $t$:

  ```{math}
  :label: EqUnbalGBC_UBI
    UBI_t \equiv \sum_{s=E+1}^{E+S}\sum_{j=1}^J \lambda_j\omega_{s,t} ubi_{j,s,t} \quad\forall t
  ```


(SecRateWedge)=
## Interest Rate on Government Debt and Household Savings

  Despite the model having no aggregate risk, it may be helpful to build in an interest rate differential between the rate of return on private capital and the interest rate on government debt. Doing so helps to add realism by including a risk premium. `OG-Core` allows users to set an exogenous wedge between these two rates. The interest rate on government debt,

  ```{math}
    :label: EqUnbalGBC_rate_wedge
    r_{gov, t} = (1 - \tau_{d, t})r_{t} - \mu_{d}
  ```

  where $r_t$ is the marginal product of capital faced by firms. The two parameters, $\tau_{d,t}$ and $\mu_{d,t}$ can be used to allow for a government interest rate that is a percentage hair cut from the market rate or a government interest rate with a constant risk premium.


(SecUnbalGBCcloseRule)=
## Budget Closure Rule

  If total government transfers to households $TR_t$ and government spending on public goods $G_t$ are both fixed fractions of GDP, one can imagine corporate and household tax structures that cause the debt level of the government to either tend toward infinity or to negative infinity, depending on whether too little revenue or too much revenue is raised, respectively.

  A virtue of dynamic general equilibrium models is that the model must be stationary in the long-run in order to solve it. That is, no variables can be indefinitely growing as time moves forward. The labor augmenting productivity growth $g_y$ from Chapter {ref}`Chap_Firms` and the potential population growth $\tilde{g}_{n,t}$ from the calibration chapter on demographics in the country-specific repository documentation render the model nonstationary. But we show how to stationarize the model against those two sources of growth in Chapter {ref}`Chap_Stnrz`. However, even after stationarizing the effects of productivity and population growth, the model could be rendered nonstationary and, therefore, not solvable if government debt were becoming too positive or too negative too quickly.

  For the model to be stationary, the debt-to-GDP ratio must be stable in the long run. Because the debt-to-GDP ratio is a quotient of two macroeconomic variables, the non-stationary and stationary versions of this ratio are equivalent. Let $T$ be some time period in the future. The stationarizing assumption is the following,

  ```{math}
  :label: EqUnbalGBC_DY
    \frac{D_t}{Y_t} = \alpha_D \quad\text{for}\quad t\geq T
  ```

  where $\alpha_D$ is a scalar long-run value of the debt-to-GDP ratio. This long-run stability condition on the debt-to-GDP ratio clearly applies to the steady-state as well as any point in the time path for $t>T$.


  We detail three possible closure-rule options here for stabilizing the debt-to-GDP ratio in the long run, although `OG-Core` only has the capability currently to execute the first closure rule that adjusts government spending $G_t$. We expect to have the other two rules implemented as `OG-Core` options soon. Each rule uses some combination of changes in government spending on public goods $G_t$ and government transfers to households $TR_t$ to stabilize the debt-to-GDP ratio in the long-run.

  1. Change only government spending on public goods $G_t$.
  2. Change only government transfers to households $TR_t$.
  3. Change both government spending $G_t$ and transfers $TR_t$ by the same percentage.


(SecUnbalGBC_chgGt)=
### Change government spending only

  We specify a closure rule that is automatically implemented after some period $T_{G1}$ to stabilize government debt as a percent of GDP (debt-to-GDP ratio). Let $\alpha_D$ represent the long-run debt-to-GDP ratio at which we want the economy to eventually settle.

  ```{math}
  :label: EqUnbalGBCclosure_Gt
  \begin{split}
    &G_t = g_{g,t}\:\alpha_{g}\: Y_t \\
    &\text{where}\quad g_{g,t} =
      \begin{cases}
        1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\:\:\:\,\text{if}\quad t < T_{G1} \\
        \frac{\left[\rho_{d}\alpha_{D}Y_{t} + (1-\rho_{d})D_{t}\right] - (1+r_{gov,t})D_{t} - I_{g,t} - TR_{t} - UBI_{t} + Rev_{t}}{\alpha_g Y_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
        \frac{\alpha_{D}Y_{t} - (1+r_{gov,t})D_{t} - I_{g,t} - TR_{t} - UBI_{t} + Rev_{t}}{\alpha_g Y_t} \qquad\qquad\quad\text{if}\quad t \geq T_{G2}
      \end{cases} \\
    &\text{and}\quad g_{tr,t} = 1 \quad\forall t
  \end{split}
  ```

  The first case in {eq}`EqUnbalGBCclosure_Gt` says that government spending $G_t$ will be a fixed fraction $\alpha_g$ of GDP $Y_t$ for every period before $T_{G1}$. The second case specifies that, starting in period $T_{G1}$ and continuing until before period $T_{G2}$, government spending be adjusted to set tomorrow's debt $D_{t+1}$ to be a convex combination between $\alpha_D Y_t$ and the current debt level $D_t$, where $\alpha_D$ is a target debt-to-GDP ratio and $\rho_d\in(0,1]$ is the percent of the way to jump toward the target $\alpha_D Y_t$ from the current debt level $D_t$. The last case specifies that, for every period after $T_{G2}$, government spending $G_t$ is set such that the next-period debt be a fixed target percentage $\alpha_D$ of GDP.

(SecUnbalGBC_chgTRt)=
### Change government transfers only

  If government transfers to households are specified by {eq}`EqUnbalGBCtfer` and the long-run debt-to-GDP ratio can only be stabilized by changing transfers, then the budget closure rule must be the following.

  ```{math}
  :label: EqUnbalGBCclosure_TRt
  \begin{split}
    &TR_t = g_{tr,t}\:\alpha_{tr}\: Y_t \\
    &\text{where}\quad g_{tr,t} =
      \begin{cases}
        1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\:\,\text{if}\quad t < T_{G1} \\
        \frac{\left[\rho_{d}\alpha_{D}Y_{t} + (1-\rho_{d})D_{t}\right] - (1+r_{gov,t})D_{t} - G_{t} - I_{g,t} -  UBI_{t} + Rev_{t}}{\alpha_{tr} Y_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
        \frac{\alpha_{D}Y_{t} - (1+r_{gov,t})D_{t} - G_{t} - I_{g,t} - UBI_{t} + Rev_{t}}{\alpha_{tr} Y_t} \qquad\qquad\quad\text{if}\quad t \geq T_{G2}
      \end{cases} \\
    &\text{and}\quad g_{g,t} = 1 \quad\forall t
  \end{split}
  ```

  The first case in {eq}`EqUnbalGBCclosure_TRt` says that government transfers $TR_t$ will be a fixed fraction $\alpha_{tr}$ of GDP $Y_t$ for every period before $T_{G1}$. The second case specifies that, starting in period $T_{G1}$ and continuing until before period $T_{G2}$, government transfers be adjusted to set tomorrow's debt $D_{t+1}$ to be a convex combination between $\alpha_D Y_t$ and the current debt level $D_t$. The last case specifies that, for every period after $T_{G2}$, government transfers $TR_t$ are set such that the next-period debt be a fixed target percentage $\alpha_D$ of GDP.


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
    &G_t + TR_t = g_{trg,t}\left(\alpha_g + \alpha_{tr}\right)Y_t \quad\Rightarrow\quad G_t = g_{trg,t}\:\alpha_g\: Y_t \quad\text{and}\quad TR_t = g_{trg,t}\:\alpha_{tr}\:Y_t \\
    &\text{where}\quad g_{trg,t} =
    \begin{cases}
      1 \qquad\qquad\qquad\qquad\qquad\qquad\quad\:\text{if}\quad t < T_{G1} \\
      \frac{\left[\rho_{d}\alpha_{D}Y_{t} + (1-\rho_{d})D_{t}\right] - (1+r_{gov,t})D_{t} - I_{g,t} - UBI_{t} + Rev_{t}}{\left(\alpha_g + \alpha_{tr}\right)Y_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
      \frac{\alpha_{D}Y_{t} - (1+r_{gov,t})D_{t} - I_{g,t} - UBI_{t} + Rev_{t}}{\left(\alpha_g + \alpha_{tr}\right)Y_t} \qquad\qquad\quad\text{if}\quad t \geq T_{G2}
    \end{cases}
  \end{split}
  ```

 The first case in {eq}`EqUnbalGBCclosure_TRGt` says that government spending and government transfers $Tr_t$ will their respective fixed fractions $\alpha_g$ and $\alpha_{tr}$ of GDP $Y_t$ for every period before $T_{G1}$. The second case specifies that, starting in period $T_{G1}$ and continuing until before period $T_{G2}$, government spending and transfers be adjusted by the same rate to set tomorrow's debt $D_{t+1}$ to be a convex combination between $\alpha_D Y_t$ and the current debt level $D_t$. The last case specifies that, for every period after $T_{G2}$, government spending and transfers are set such that the next-period debt be a fixed target percentage $\alpha_D$ of GDP.

 Each of these budget closure rules {eq}`EqUnbalGBCclosure_Gt`, {eq}`EqUnbalGBCclosure_TRt`, and {eq}`EqUnbalGBCclosure_TRGt` allows the government to run increasing deficits or surpluses in the short run (before period $T_{G1}$). But then the adjustment rule is implemented gradually beginning in period $t=T_{G1}$ to return the debt-to-GDP ratio back to its long-run target of $\alpha_D$. Then the rule is implemented exactly in period $T_{G2}$ by adjusting some combination of government spending $G_t$ and transfers $TR_t$ to set the debt $D_{t+1}$ such that it is exactly $\alpha_D$ proportion of GDP $Y_t$.

(SecUnbalGBCcaveat)=
## Some Caveats and Alternatives

`OG-Core` adjusts some combination of government spending $G_t$ and government transfers $TR_t$ as its closure rule instrument because of its simplicity and lack of distortionary effects. Since government spending does not enter into the household's utility function, its level does not affect the solution of the household problem. In contrast, government transfers do appear in the household budget constraint. However, household decisions do not individually affect the amount of transfers, thereby rendering government transfers as exogenous from the household's perspective. As an alternative, one could choose to adjust taxes to close the budget (or a combination of all of the government fiscal policy levers).

There is no guarantee that any of our stated closure rules {eq}`EqUnbalGBCclosure_Gt`, {eq}`EqUnbalGBCclosure_TRt`, or {eq}`EqUnbalGBCclosure_TRGt` is sufficient to stabilize the debt-to-GDP ratio in the long run. For large and growing deficits, the convex combination parameter $\rho_d$ might be too gradual, or the budget closure initial period $T_{G1}$ might be too far in the future, or the target debt-to-GDP ratio $\alpha_D$ might be too high. The existence of any of these problems might be manifest in the steady state computation stage. However, it is possible for the steady-state to exist, but for the time path to never reach it. These problems can be avoided by choosing conservative values for $T_{G1}$, $\rho_d$, and $\alpha_D$ that close the budget quickly.

And finally, in closure rules {eq}`EqUnbalGBCclosure_Gt` and {eq}`EqUnbalGBCclosure_TRGt` in which government spending is used to stabilize the long-run budget, it is also possible that government spending is forced to be less than zero to make this happen. This would be the case if tax revenues bring in less than is needed to financed transfers and interest payments on the national debt. None of the equations we've specified above preclude that result, but it does raise conceptual difficulties. Namely, what does it mean for government spending to be negative? Is the government selling off public assets? We caution those using this budget closure rule to consider carefully how the budget is closed in the long run given their parameterization. We also note that such difficulties present themselves across all budget closure rules when analyzing tax or spending proposals that induce structural budget deficits. In particular, one probably needs a different closure instrument if government spending must be negative in the steady-state to hit your long-term debt-to-GDP target.


[^negative_val_note]: Negative values for government spending on public goods would mean that revenues are coming into the country from some outside source, which revenues are triggered by government deficits being too high in an arbitrary future period $T_{G2}$.

(SecUBIfootnotes)=
## Footnotes

[^GrowthAdj_note]: We impose this requirement of `ubi_growthadj = False` when `g_y_annual < 0` in the [`default_parameters.json`](https://github.com/PSLmodels/OG-Core/blob/master/ogcore/default_parameters.json) "validators" specification of the parameter.
