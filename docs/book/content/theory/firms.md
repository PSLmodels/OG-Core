(Chap_Firms)=
# Firms

The production side of the `OG-Core` model is populated by $M$ industries indexed by $m=1,2,...M$, each of which industry has a unit measure of identical perfectly competitive firms that rent private capital $K_{m,t}$ and public capital $K_{g,m,t}$ and hire labor $L_{m,t}$ to produce output $Y_{m,t}$. Firms face a flat corporate income tax $\tau^{corp}_{m,t}$ and can deduct capital expenses for tax purposes at a rate $\delta^\tau_{m,t}$. Tax parameters can vary by industry $m$ and over time, $t$.

(EqFirmsProdFunc)=
## Production Function

  Firms in each industry produce output $Y_{m,t}$ using inputs of private capital $K_{m,t}$, public capital $K_{g,m,t}$, and labor $L_{m,t}$ according to a general constant elasticity (CES) of substitution production function,

  ```{math}
  :label: EqFirmsCESprodfun
    \begin{split}
      Y_{m,t} &= F(K_{m,t}, K_{g,m,t}, L_{m,t}) \\
      &\equiv Z_{m,t}\biggl[(\gamma_m)^\frac{1}{\varepsilon_m}(K_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m} + (\gamma_{g,m})^\frac{1}{\varepsilon_m}(K_{g,m,t})^\frac{\varepsilon_m-1}{\varepsilon_m} + \\
      &\quad\quad\quad\quad\quad(1-\gamma_m-\gamma_{g,m})^\frac{1}{\varepsilon_m}(e^{g_y t}L_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m}\biggr]^\frac{\varepsilon_m}{\varepsilon_m-1} \quad\forall m,t
    \end{split}
  ```
  where $Z_{m,t}$ is an exogenous scale parameter (total factor productivity) that can be time dependent, $\gamma_m$ represents private capital's share of income, $\gamma_{g,m}$ is public capital's share of income, and $\varepsilon_m$ is the constant elasticity of substitution among the two types of capital and labor. We have included constant productivity growth rate $g_y$ as the rate of labor augmenting technological progress.

  A nice feature of the CES production function is that the Cobb-Douglas production function is a nested case for $\varepsilon_m=1$.[^Kg0_case]
  ```{math}
  :label: EqFirmsCDprodfun
    Y_{m,t} = Z_{m,t} (K_{m,t})^{\gamma_m} (K_{g,m,t})^{\gamma_{g,m}}(e^{g_y t}L_{m,t})^{1-\gamma_m-\gamma_{g,m}} \quad\forall m,t \quad\text{for}\quad \varepsilon_m=1
  ```

Industry $M$ in the model is unique in two respects.  First, we will define industry $M$ goods as the numeraire in OG_Core.  Therefore, all quantities are in terms of industry $M$ goods and all prices are relative to the price of a unit of industry $M$ goods.  Second, the model solution is greatly simplified if just one production industry produces capital goods.  The assumption in OG-Core is that industry $M$ is the only industry producing capital goods (though industry $M$ goods can also be used for consumption).

(EqFirmsFOC)=
## Optimality Conditions

  The static per-period profit function of the representative firm in each industry $m$ is the following.

  ```{math}
  :label: EqFirmsProfit
    PR_{m,t} &= (1 - \tau^{corp}_{m,t})\Bigl[p_{m,t}F(K_{m,t},K_{g,m,t},L_{m,t}) - w_t L_{m,t}\Bigr] - \\
    &\qquad\qquad\quad \bigl(r_t + \delta_{M,t}\bigr)K_{m,t} + \tau^{corp}_{m,t}\delta^\tau_{m,t} K_{m,t} \quad\forall m,t
  ```

  Gross income for the firms is $p_{m,t}F(K_{m,t},K_{g,m,t},L_{m,t})$. Labor costs to the firm are $w_t L_{m,t}$, and capital costs are $(r_t +\delta_{M,t})K_{m,t}$. The government supplies public capital $K_{g,m,t}$ to the firms at no cost. The per-period interest rate (rental rate) of capital for firms is $r_t$. The per-period economic depreciation rate for private capital is $\delta_{M,t}\in[0,1]$.[^delta_M] The $\delta^\tau_{m,t}$ parameter in the last term of the profit function governs how much of capital depreciation can be deducted from the corporate income tax.

  Taxes enter the firm's profit function {eq}`EqFirmsProfit` in two places. The first is the corporate income tax rate $\tau^{corp}_{m,t}$, which is a flat tax on corporate income that can vary by industry $m$. Corporate income is defined as gross income minus labor costs. This will cause the corporate tax to only have a direct effect on the firms' capital demand decision.

  The tax policy also enters the profit function {eq}`EqFirmsProfit` through depreciation deductions at rate $\delta^\tau_{m,t}$, which then lower corporate tax liability. When $\delta^\tau_{m,t}=0$, no depreciation expense is deducted from the firm's tax liability. When $\delta^\tau_{m,t}=\delta_{M,t}$, all economic depreciation is deducted from corporate income.

  Firms take as given prices $p_{m,t}$, $w_t$, and $r_t$ and the level of public capital supply $K_{g,m,t}$. Taking the derivative of the profit function {eq}`EqFirmsProfit` with respect to labor $L_{m,t}$ and setting it equal to zero (using the general CES form of the production function {eq}`EqFirmsCESprodfun`) and taking the derivative of the profit function with respect to private capital $K_{m,t}$ and setting it equal to zero, respectively, characterizes the optimal labor and capital demands.

  ```{math}
  :label: EqFirmFOC_L
    w_t = e^{g_y t}p_{m,t}(Z_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m}\left[(1-\gamma_m-\gamma_{g,m})\frac{Y_{m,t}}{e^{g_y t}L_{m,t}}\right]^\frac{1}{\varepsilon_m} \quad\forall m,t
  ```

  ```{math}
  :label: EqFirmFOC_K
    r_t = (1 - \tau^{corp}_{m,t})p_{m,t}(Z_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m}\left[\gamma_m\frac{Y_{m,t}}{K_{m,t}}\right]^\frac{1}{\varepsilon_m} - \delta_{M,t} + \tau^{corp}_{m,t}\delta^\tau_{m,t} \quad\forall m,t
  ```

  Note that the presence of the public capital good creates economic rents. These rents will accrue to the owners of capital via the financial intermediary. See Section Chapter {ref}`Chap_FinInt` for more details on the determination of the return to the household's portfolio. Because public capital is exogenous to the firm's decisions, the optimality condition for capital demand {eq}`EqFirmFOC_K` is only affected by public capital $K_{g,m,t}$ through the $Y_{m,t}$ term.

(EqFirmsPosProfits)=
## Positive Profits from Government Infrastructure Investment

  The CES production function in {eq}`EqFirmsCESprodfun` exhibits constant returns to scale (CRS). A feature of CRS production functions is that gross revenue $Y_{m,t}$ is a sum of the gross revenue attributed to each factor of production,

  ```{math}
  :label: EqFirmsMargRevEq
    Y_{m,t} = MPK_{m,t} K_{m,t} + MPK_{g,m,t} K_{g,m,t} + MPL_{m,t} L_{m,t} \quad\forall m,t
  ```

  where $MPK_{m,t}$ is the marginal product of private capital in industry $m$, $MPK_{g,m,t}$ is the marginal product of public capital, and $MPL_{m,t}$ is the marginal product of labor.[^MPfactors] Each of the terms in {eq}`EqFirmsMargRevEq` is growing at the macroeconomic variable rate of $e^{g_y t}\tilde{N_t}$ (see the third column of {numref}`TabStnrzStatVars`). Firm profit maximization for private capital demand from equation {eq}`EqFirmFOC_K` implies that the marginal product of private capital is equal to the real cost of capital:

  ```{math}
  :label: EqFirmsMPK_opt
    MPK_{m,t} =  \frac{r_t + \delta_{M,t} - \tau^{corp}_{m,t}\delta^\tau_{m,t}}{p_{m,t}(1 - \tau^{corp}_{m,t})} \quad\forall m,t
  ```

  Firm profit maximization for labor demand from equation {eq}`EqFirmFOC_L` implies that the marginal product of labor is equal to the real wage rate:

  ```{math}
  :label: EqFirmsMPL_opt
    MPL_{m,t} =  \frac{w_t}{p_{m,t}} \quad\forall m,t
  ```

  Even though firms take the stock of public capital $K_{g,t}$ from government infrastructure investment as given, we can still calculate the marginal product of public capital from the production function {eq}`EqFirmsCESprodfun`.

  ```{math}
  :label: EqFirmsMPKg_opt
    MPK_{g,m,t} =  \left(Z_{m,t}\right)^{\frac{\varepsilon_m - 1}{\varepsilon_m}}\left(\gamma_{g,m}\frac{Y_{m,t}}{K_{g,m,t}}\right)^{\frac{1}{\varepsilon_m}} \quad\forall m,t
  ```

  If we plug the expressions for $MPK_{m,t}$, $MPK_{g,m,t}$, and $MPL_{m,t}$ from {eq}`EqFirmsMPK_opt`, {eq}`EqFirmsMPKg_opt`, and {eq}`EqFirmsMPL_opt`, respectively, into the total revenue $Y_{m,t}$ decomposition in {eq}`EqFirmsMargRevEq` and then substitute that into the profit function {eq}`EqFirmsProfit`, we see that positive economic rents arise when public capital is positive $K_{g,m,t}>0$.

  ```{math}
  :label: EqFirmsProfit_Kg
    \begin{split}
      PR_{m,t} &= (1 - \tau^{corp}_{m,t})\Bigl[p_{m,t}Y_{m,t} - w_t L_{m,t}\Bigr] - \bigl(r_t + \delta_{M,t}\bigr)K_{m,t} + \tau^{corp}_{m,t}\delta^\tau_{m,t} K_{m,t} \\
      &= (1 - \tau^{corp}_{m,t})\Biggl[\biggl(\frac{r_t + \delta_{M,t} - \tau^{corp}_{m,t}\delta^{\tau}_{m,t}}{1 - \tau^{corp}_{m,t}}\biggr)K_{m,t} + p_{m,t}MPK_{g,m,t}K_{g,m,t} + w_t L_{m,t}\Biggr] ... \\
      &\quad\quad - (1 - \tau^{corp}_{m,t})w_t L_{m,t} - (r_t + \delta_{M,t})K_{m,t} + \tau^{corp}_{m,t}\delta^{\tau}_{m,t} K_{m,t} \\
      &= (1 - \tau^{corp}_{m,t})p_{m,t}MPK_{g,m,t}K_{g,m,t} \quad\forall m,t
    \end{split}
  ```

  We assume these positive economic profits resulting from government infrastructure investment are passed on to the owners of private capital through an adjusted interest rate $r_{K,t}$ provided by the financial intermediary (see Chapter {ref}`Chap_FinInt`) that zeroes out profits among the perfectly competitive firms and is a function of $p_{m,t}$, $MPK_{g,m,t}$ and $K_{g,m,t}$ in each industry $m$. Total payouts from the financial intermediary $r_{K,t}\sum_{m=1}^M K_{m,t}$ are a function of the perfectly competitive payout to owners of private capital $r_t \sum_{m=1}^M K_{m,t}$ plus any positive profits when $K_{g,m,t}>0$ from {eq}`EqFirmsProfit_Kg`.

  ```{math}
  :label: EqFirmsPayout
    r_{K,t}\sum_{m=1}^M K_{m,t} =  r_t \sum_{m=1}^M K_{m,t} + \sum_{m=1}^M(1 - \tau^{corp}_{m,t})p_{m,t}MPK_{g,m,t}K_{g,m,t} \quad\forall t
  ```

  This implies that the rate of return paid from the financial intermediary to the households $r_{K,t}$ is the interest rate on private capital $r_t$ plus the ratio of total positive profits across industries (a function of $K_{g,m,t}$ in each industry) divided by total private capital from {eq}`EqFirmsProfit_Kg`, in which the units are put in terms of $K_{m,t}$ (which is in terms of the $M$th industry output, see equation {eq}`eq_rK` in Chapter {ref}`Chap_FinInt`).

  ```{math}
  :label: EqFirms_rKt
    r_{K,t} =  r_t + \frac{\sum_{m=1}^M(1 - \tau^{corp}_{m,t})p_{m,t}MPK_{g,m,t}K_{g,m,t}}{\sum_{m=1}^M K_{m,t}} \quad\forall t
  ```

(SecFirmsfootnotes)=
## Footnotes

  [^Kg0_case]: It is important to note a special case of the Cobb-Douglas ($\varepsilon_m=1$) production function that we have to manually restrict. The inputs of production of private capital $K_{m,t}$ and labor $L_{m,t}$ are endogenous and have characteristics of the model that naturally bound them away from zero. But public capital $K_{g,m,t}$, although it is a function of endogenous variables in {eq}`EqUnbalGBC_Igt` and {eq}`EqUnbalGBC_Igmt`, can be exogenously set to zero as a policy parameter choice by setting $\alpha_{I,t}=0$ or $\alpha_{I,m,t}=0$. In the Cobb-Douglas case of the production function $\varepsilon_m=1$ {eq}`EqFirmsCDprodfun`, $K_{g,m,t}=0$ would zero out production and break the model. In the case when $\varepsilon_m=1$ and $K_{g,m,t}=0$, we set $\gamma_{g,m}=0$, thereby restricting the production function to only depend on private capital $K_{m,t}$ and labor $L_{m,t}$. This necessary restriction limits us from performing experiments in the model of the effect of changing $K_{g,m,t}=0$ to $K_{g,m,t}>0$ or vice versa in the $\varepsilon_m=1$ case.

  [^delta_M]: Because we are assuming that only the output of the $M$th industry can be used for investment, government spending, or government debt, and because that industry's output is the numeraire, the only depreciation rate that matters or can be nonzero is that of the $M$th industry $\delta_{M,t}$.

  [^MPfactors]: See Section {ref}`SecAppDerivCES` of the {ref}`Chap_Deriv` Chapter for the derivations of the marginal product of private capital $MPK_{m,t}$, marginal product of public capital $MPk_{g,m,t}$, and marginal product of labor $MPL_{m,t}$.
