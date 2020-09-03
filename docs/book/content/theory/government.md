
(Chap_UnbalGBC)=
# Government

In `OG-USA`, the government enters by levying taxes on households, providing transfers to households, levying taxes on firms, spending resources on public goods, and making rule-based adjustments to stabilize the economy in the long-run. It is this last activity that is the focus of this chapter.


(SecUnbalGBCrev)=
## Government Tax Revenue

  We see from the household's budget constraint that taxes $T_{s,t}$ and transfers $TR_{t}$ enter into the household's decision,
  
  ```{math}
  :label: EqHHBC
      c_{j,s,t} + b_{j,s+1,t+1} &= (1 + r_{t})b_{j,s,t} + w_t e_{j,s} n_{j,s,t} + \zeta_{j,s}\frac{BQ_t}{\lambda_j\omega_{s,t}} + \eta_{j,s,t}\frac{TR_{t}}{\lambda_j\omega_{s,t}} - T_{s,t}  \\
      &\quad\forall j,t\quad\text{and}\quad s\geq E+1 \quad\text{where}\quad b_{j,E+1,t}=0\quad\forall j,t
  ```

  where we defined the tax liability function $T_{s,t}$ in {eq}`EqTaxCalcLiabETR` as an effective tax rate times total income and the transfer distribution function $\eta_{j,s,t}$ is uniform across all households as in {eq}`EqTaxCalcEtajs`. And government revenue from the corporate income tax rate $\tau^{corp}$ and the tax on depreciation expensing $\tau^\delta$ enters the firms' profit function.
  
  ```{math}
  :label: EqFirmsProfit
    PR_t = (1 - \tau^{corp})\bigl(Y_t - w_t L_t\bigr) - \bigl(r_t + \delta\bigr)K_t + \tau^{corp}\delta^\tau K_t \quad\forall t
  ```

  We define total government revenue from taxes as the following.
  
  ```{math}
  :label: EqUnbalGBCgovRev
    Rev_t = \underbrace{\tau^{corp}\bigl[Y_t - w_t L_t\bigr] - \tau^{corp}\delta^\tau K_t}_{\text{corporate tax revenue}} + \underbrace{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\lambda_j\omega_{s,t}\tau^{etr}_{s,t}\left(x_{j,s,t},y_{j,s,t}\right)\bigl(x_{j,s,t} + y_{j,s,t}\bigr)}_{\text{household tax revenue}} \quad\forall t
  ```

(SecUnbalGBCbudgConstr)=
## Government Budget Constraint

  Let the level of government debt in period $t$ be given by $D_t$. The government budget constraint requires that government revenue $Rev_t$ plus the budget deficit ($D_{t+1} - D_t$) equal expenditures on interest of the debt, government spending on public goods $G_t$, and total transfer payments to households $TR_t$ every period $t$.

  ```{math}
  :label: EqUnbalGBCbudgConstr
    D_{t+1} + Rev_t = (1 + r_t)D_t + G_t + TR_t \quad\forall t
  ```

  We assume that total government transfers to households are a fixed fraction $\alpha_{tr}$ of GDP each period.
  
  ```{math}
  :label: EqUnbalGBCtfer
    TR_t = g_{tr,t}\:\alpha_{tr}\: Y_t \quad\forall t
  ```

  The time dependent multiplier $g_{tr,t}$ in front of the right-hand-side of {eq}`EqUnbalGBCtfer` will equal 1 in most initial periods. It will potentially deviate from 1 in some future periods in order to provide a closure rule that ensures a stable long-run debt-to-GDP ratio. We will discuss the closure rule in Section {ref}`SecUnbalGBCcloseRule`.

  We also assume that government spending on public goods is a fixed fraction of GDP each period in the initial periods.
  
  ```{math}
  :label: EqUnbalGBC_Gt
    G_t = g_{g,t}\:\alpha_{g}\: Y_t
  ```

  Similar to transfers $TR_t$, the time dependent multiplier $g_{g,t}$ in front of the right-hand-side of {eq}`EqUnbalGBC_Gt` will equal 1 in most initial periods. It will potentially deviate from 1 in some future periods in order to provide a closure rule that ensures a stable long-run debt-to-GDP ratio. We make this more specific in the next section.

Government spending on goods and services is comprised on spending on public infrastructure, $I_{g,t}$ and non-capital expenditures, $G_{g,t}$ such that $G_{t} = I_{g,t} + G_{g,t}$.  We assume that infrastructure spending is a fraction fo total government spending, $I_{g,t} = \alpha_{i,t} * G_{g,t}$.  The stock of public capital (i.e., infrastructure) evolves according to the law of motion, $K_{g,t+1} = (1 - \delta^{g}) K_{g,t} + I_{g,t}$.  The stock of public capital complements labor and private capital in the production function of the representative firm, in Equation {eq}`EqFirmsCESprodfun`.

(SecRateWedge)=
## Interest Rate on Government Debt

Despite the model having no aggregate risk, it may be helpful to build in an interest rate differential between the rate of return on private capital and the interest rate on government debt.  Doing so helps to add realism by including a risk premium.  `OG-USA` allows users to set an exogenous wedge between these two rates.  The interest rate on government debt,

```{math}
  :label: EqUnbalGBC_rate_wedge
  r_{gov, t} = (1 - \tau_{d, t})r_{t} - \mu_{d}
```

The two parameters, $\tau_{d,t}$ and $\mu_{d,t}$ can be used to allow for a government interest rate that is a percentage hair cut from the market rate or a government interest rate with a constant risk premia.

In the cases where there is a differential ($\tau_{d,t}$ or $\mu_{d,t} \neq 0$), then we need to be careful to specify how the household chooses government debt and private capital in its portfolio of asset holdings.  We make the assumption that under the exogenous interest rate wedge, the household is indifferent between holding its assets as debt and private capital.  This amounts to an assumption that these two assets are perfect substitutes given the exogenous wedge in interest rates.  Given the indifference between government debt and private capital at these two interest rates, we assume that the household holds debt and capital in the same ratio that debt and capital are demanded by the government and private firms, respectively. The interest rate on the household portfolio of asset is thus given by:

```{math}
  :label: EqUnbalGBC_rate_wedge
  r_{hh,t} = \frac{r_{gov,t}D_{t} + r_{t}K_{t}}{D_{t} + K_{t}}
```


(SecUnbalGBCcloseRule)=
## Budget Closure Rule

  If total government transfers to households $TR_t$ and government spending on public goods $G_t$ are both fixed fractions of GDP, one can imagine corporate and household tax structures that cause the debt level of the government to either tend toward infinity or to negative infinity, depending on whether too little revenue or too much revenue is raised, respectively.

  A virtue of dynamic general equilibrium models is that the model must be stationary in order to solve it. That is, no variables can be indefinitely growing as time moves forward. The labor augmenting productivity growth $g_y$ from Chapter {ref}`Chap_Firms` and the potential population growth $\tilde{g}_{n,t}$ from Chapter {ref}`Chap_Demog` render the model nonstationary. But we show how to stationarize the model against those two sources of growth in Chapter {ref}`Chap_Stnrz`. However, even after stationarizing the effects of productivity and population growth, the model could be rendered nonstationary and, therefore, not solvable if government debt were becoming too positive or too negative too quickly.

  The `OG-USA` model offers three different options for budget closure rules. Each rule uses some combination of changes in government spending on public goods $G_t$ and government transfers to households $TR_t$ to stabilize the debt-to-GDP ratio in the long-run.
  
1. Change only government spending on public goods $G_t$.
2. Change only government transfers to households $TR_t$.
3. Change both government spending $G_t$ and transfers $TR_t$ by the same percentage.

(SecUnbalGBC_chgGt)=
### Change government spending only

 We specify a closure rule that is automatically implemented after some period $T_{G1}$ to stabilize government debt as a percent of GDP (debt-to-GDP ratio). Let $\alpha_D$ represent the long-run debt-to-GDP ratio at which we want the economy to eventually settle.
  
```{math}
:label: EqUnbalGBCclosure_Gt
     G_t = g_{g,t}\:\alpha_{g}\: Y_t \\
     \text{where}\quad g_{g,t} =
       \begin{cases}
         1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\:\:\:\,\text{if}\quad t < T_{G1} \\
         \frac{\left[\rho_{d}\alpha_{D}Y_{t} + (1-\rho_{d})D_{t}\right] - (1+r_{t})D_{t} - TR_{t} + Rev_{t}}{\alpha_g Y_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
         \frac{\alpha_{D}Y_{t} - (1+r_{t})D_{t} - TR_{t} + Rev_{t}}{\alpha_g Y_t} \qquad\qquad\quad\:\:\:\,\text{if}\quad t \geq T_{G2}
       \end{cases} \\
     \quad\text{and}\quad g_{tr,t} = 1 \quad\forall t
 ```

 The first case in {eq}`EqUnbalGBCclosure_Gt` says that government spending $G_t$ will be a fixed fraction $\alpha_g$ of GDP $Y_t$ for every period before $T_{G1}$. The second case specifies that, starting in period $T_{G1}$ and continuing until before period $T_{G2}$, government spending be adjusted to set tomorrow's debt $D_{t+1}$ to be a convex combination between $\alpha_D Y_t$ and the current debt level $D_t$, where $\alpha_D$ is a target debt-to-GDP ratio and $\rho_d\in(0,1]$ is the percent of the way to jump toward the target $\alpha_D Y_t$ from the current debt level $D_t$. The last case specifies that, for every period after $T_{G2}$, government spending $G_t$ is set such that the next-period debt be a fixed target percentage $\alpha_D$ of GDP.

(SecUnbalGBC_chgTRt)=
### Change government transfers only

 If government transfers to households are specified by {eq}`EqUnbalGBCtfer` and the long-run debt-to-GDP ratio can only be stabilized by changing transfers, then the budget closure rule must be the following.
    
```{math}
:label: EqUnbalGBCclosure_TRt
     TR_t = g_{tr,t}\:\alpha_{tr}\: Y_t \\
     \text{where}\quad g_{tr,t} =
       \begin{cases}
         1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\:\text{if}\quad t < T_{G1} \\
         \frac{\left[\rho_{d}\alpha_{D}Y_{t} + (1-\rho_{d})D_{t}\right] - (1+r_{t})D_{t} - G_{t} + Rev_{t}}{\alpha_{tr} Y_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
         \frac{\alpha_{D}Y_{t} - (1+r_{t})D_{t} - G_{t} + Rev_{t}}{\alpha_{tr} Y_t} \qquad\qquad\quad\:\:\:\:\text{if}\quad t \geq T_{G2}
       \end{cases} \\
     \quad\text{and}\quad g_{g,t} = 1 \quad\forall t
 ```

 The first case in {eq}`EqUnbalGBCclosure_TRt` says that government transfers $TR_t$ will be a fixed fraction $\alpha_{tr}$ of GDP $Y_t$ for every period before $T_{G1}$. The second case specifies that, starting in period $T_{G1}$ and continuing until before period $T_{G2}$, government transfers be adjusted to set tomorrow's debt $D_{t+1}$ to be a convex combination between $\alpha_D Y_t$ and the current debt level $D_t$. The last case specifies that, for every period after $T_{G2}$, government transfers $TR_t$ are set such that the next-period debt be a fixed target percentage $\alpha_D$ of GDP.


(SecUnbalGBC_chgGtTRt)=
### Change both government spending and transfers

 In some cases, changing only government spending $G_t$ or only government transfers $TR_t$ will not be enough. That is, there exist policies for which a decrease in government spending to zero after period $T_{G1}$ will not stabilize the debt-to-GDP ratio. And negative government spending on public goods does not make sense.\footnote{Negative values for government spending on public goods would mean that revenues are coming into the country from some outside source, which revenues are triggered by government deficits being too high in an arbitrary future period $T_{G2}$.} On the other hand, negative transfers do make sense. Notwithstanding, one might want the added stabilization ability of changing both government spending $G_t$ and transfers $TR_t$ to stabilize the long-run debt-to-GDP ratio.

 In our specific form of this joint option, we assume that the factor by which we scale government spending and transfers is the same $g_{g,t} = g_{tr,t}$ for all $t$. We label this single scaling factor $g_{trg,t}$.
 
 ```{math}
 :label: EqUnbalGBCclosure_gTRGt
   g_{trg,t}\equiv g_{g,t} = g_{tr,t} \quad\forall t
 ```

 If government spending on public goods is specified by {eq}`EqUnbalGBC_Gt` and government transfers to households are specified by {eq}`EqUnbalGBCtfer` and the long-run debt-to-GDP ratio can only be stabilized by changing both spending and transfers, then the budget closure rule must be the following.
 
 ```{math}
 :label: EqUnbalGBCclosure_TRGt
     &G_t + TR_t = g_{trg,t}\left(\alpha_g + \alpha_{tr}\right)Y_t \quad\Rightarrow\quad G_t = g_{trg,t}\:\alpha_g\: Y_t \quad\text{and}\quad TR_t = g_{trg,t}\:\alpha_{tr}\:Y_t \\
     &\text{where}\quad g_{trg,t} =
       \begin{cases}
         1 \qquad\qquad\qquad\qquad\qquad\qquad\:\:\:\,\text{if}\quad t < T_{G1} \\
         \frac{\left[\rho_{d}\alpha_{D}Y_{t} + (1-\rho_{d})D_{t}\right] - (1+r_{t})D_{t} + Rev_{t}}{\left(\alpha_g + \alpha_{tr}\right)Y_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
         \frac{\alpha_{D}Y_{t} - (1+r_{t})D_{t} + Rev_{t}}{\left(\alpha_g + \alpha_{tr}\right)Y_t} \qquad\qquad\quad\:\:\:\:\text{if}\quad t \geq T_{G2}
       \end{cases}
  ```

 The first case in {eq}`EqUnbalGBCclosure_TRGt` says that government spending and government transfers $Tr_t$ will their respective fixed fractions $\alpha_g$ and $\alpha_{tr}$ of GDP $Y_t$ for every period before $T_{G1}$. The second case specifies that, starting in period $T_{G1}$ and continuing until before period $T_{G2}$, government spending and transfers be adjusted by the same rate to set tomorrow's debt $D_{t+1}$ to be a convex combination between $\alpha_D Y_t$ and the current debt level $D_t$. The last case specifies that, for every period after $T_{G2}$, government spending and transfers are set such that the next-period debt be a fixed target percentage $\alpha_D$ of GDP.

 Each of these budget closure rules {eq}`EqUnbalGBCclosure_Gt`, {eq}`EqUnbalGBCclosure_TRt`, and {eq}`EqUnbalGBCclosure_TRGt` allows the government to run increasing deficits or surpluses in the short run (before period $T_{G1}$). But then the adjustment rule is implemented gradually beginning in period $t=T_{G1}$ to return the debt-to-GDP ratio back to its long-run target of $\alpha_D$. Then the rule is implemented exactly in period $T_{G2}$ by adjusting some combination of government spending $G_t$ and transfers $TR_t$ to set the debt $D_{t+1}$ such that it is exactly $\alpha_D$ proportion of GDP $Y_t$.

(SecUnbalGBCcaveat)=
## Some Caveats and Alternatives

`OG-USA` adjusts some combination of government spending $G_t$ and government transfers $TR_t$ as its closure rule instrument because of its simplicity and lack of distortionary effects. Since government spending does not enter into the household's utility function, its level does not affect the solution of the household problem. In contrast, government transfers do appear in the household budget constraint. However, household decisions do not individually affect the amount of transfers, thereby rendering government transfers as exogenous from the household's perspective. As an alternative, one could choose to adjust taxes to close the budget (or a combination of all of the government fiscal policy levers).

There is no guarantee that any of our stated closure rules {eq}`EqUnbalGBCclosure_Gt`, {eq}`EqUnbalGBCclosure_TRt`, or {eq}`EqUnbalGBCclosure_TRGt` is sufficient to stabilize the debt-to-GDP ratio in the long run. For large and growing deficits, the convex combination parameter $\rho_d$ might be too gradual, or the budget closure initial period $T_{G1}$ might be too far in the future, or the target debt-to-GDP ratio $\alpha_D$ might be too high. The existence of any of these problems might be manifest in the steady state computation stage. However, it is possible for the steady-state to exist, but for the time path to never reach it. These problems can be avoided by choosing conservative values for $T_{G1}$, $\rho_d$, and $\alpha_D$ that close the budget quickly.

And finally, in closure rules {eq}`EqUnbalGBCclosure_Gt` and {eq}`EqUnbalGBCclosure_TRGt` in which government spending is used to stabilize the long-run budget, it is also possible that government spending is forced to be less than zero to make this happen. This would be the case if tax revenues bring in less than is needed to financed transfers and interest payments on the national debt. None of the equations we've specified above preclude that result, but it does raise conceptual difficulties. Namely, what does it mean for government spending to be negative? Is the government selling off pubic assets? We caution those using this budget closure rule to consider carefully how the budget is closed in the long run given their parameterization. We also note that such difficulties present themselves across all budget closure rules when analyzing tax or spending proposals that induce structural budget deficits. In particular, one probably needs a different closure instrument if government spending must be negative in the steady-state to hit your long-term debt-to-GDP target.
