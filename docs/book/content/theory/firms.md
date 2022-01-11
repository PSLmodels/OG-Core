(Chap_Firms)=
# Firms

The production side of the `OG-Core` model is populated by a unit measure of identical perfectly competitive firms that rent private capital $K_t$ and public capital $K_{g,t}$ and hire labor $L_t$ to produce output $Y_t$. Firms also face a flat corporate income tax $\tau^{corp}$ as well as a tax on the amount of capital they depreciate $\tau^\delta$.

(EqFirmsProdFunc)=
## Production Function

  Firms produce output $Y_t$ using inputs of private capital $K_t$, public capital $K_{g,t}$, and labor $L_t$ according to a general constant elasticity (CES) of substitution production function,

  ```{math}
  :label: EqFirmsCESprodfun
    \begin{split}
      Y_t &= F(K_t, K_{g,t}, L_t) \\
      &\equiv \begin{cases}
        &Z_t\biggl[(\gamma)^\frac{1}{\varepsilon}(K_t)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}(K_{g,t})^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(e^{g_y t}L_t)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \:\text{if}\: K_{g,t}>0 \\
        &Z_t\biggl[(\gamma)^\frac{1}{\varepsilon}(K_t)^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma)^\frac{1}{\varepsilon}(e^{g_y t}L_t)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \quad\text{if}\quad K_{g,t}=0 \quad\text{and}\quad \varepsilon=1
      \end{cases}\quad\forall t
    \end{split}
  ```
  where $Z_t$ is an exogenous scale parameter (total factor productivity) that can be time dependent, $\gamma$ represents private capital's share of income, $\gamma_{g}$ is public capital's share of income, and $\varepsilon$ is the constant elasticity of substitution among the two types of capital and labor. We have included constant productivity growth rate $g_y$ as the rate of labor augmenting technological progress.

  A nice feature of the CES production function is that the Cobb-Douglas production function is a nested case for $\varepsilon=1$.
  ```{math}
  :label: EqFirmsCDprodfun
    Y_t =
    \begin{cases}
      &Z_t K_t^\gamma K_{g,t}^{\gamma_{g}}(e^{g_y t}L_t)^{1-\gamma-\gamma_{g}} \quad\text{for}\quad K_{g,t}>0 \\
      &Z_t K_t^\gamma (e^{g_y t}L_t)^{1-\gamma} \quad\text{for}\quad K_{g,t}=0
    \end{cases} \quad\forall t \quad\text{for}\quad \varepsilon=1
  ```

(EqFirmsFOC)=
## Optimality Conditions

  The profit function of the representative firm is the following.

  ```{math}
  :label: EqFirmsProfit
    PR_t = (1 - \tau^{corp}_t)\Bigl[F(K_t,K_{g,t},L_t) - w_t L_t\Bigr] - \bigl(r_t + \delta\bigr)K_t + \tau^{corp}\delta^\tau K_t \quad\forall t
  ```

  Gross income for the firms is given by the production function $F(K,K_g,L)$ because we have normalized the price of the consumption good to 1. Labor costs to the firm are $w_t L_t$, and capital costs are $(r_t +\delta)K_t$. The government supplies public capital to the firms at no cost. The per-period interest rate (rental rate) of capital for firms is $r_t$. The per-period economic depreciation rate for private capital is $\delta$. The $\delta^\tau$ parameter in the last term of the profit function governs how much of capital depreciation can be deducted from the corporate income tax.

  Taxes enter the firm's profit function {eq}`EqFirmsProfit` in two places. The first is the corporate income tax rate $\tau^{corp}_t$, which is a flat tax on corporate income. Corporate income is defined as gross income minus labor costs. This will cause the corporate tax to only distort the firms' capital demand decision.

  The next place where tax policy enters the profit function {eq}`EqFirmsProfit` is through a refund of a percent of depreciation costs $\delta^\tau$ refunded at the corporate income tax rate $\tau^{corp}_t$. When $\delta^\tau=0$, no depreciation expense is deducted from the firm's tax liability. When $\delta^\tau=\delta$, all economic depreciation is deducted from corporate income.

  Firms take as given prices $w_t$ and $r_t$ and the level of public capital supply $K_{g,t}$. Taking the derivative of the profit function {eq}`EqFirmsProfit` with respect to labor $L_t$ and setting it equal to zero (using the general CES form of the production function {eq}`EqFirmsCESprodfun`) and taking the derivative of the profit function with respect to capital $K_t$ and setting it equal to zero, respectively, characterizes the optimal labor and capital demands.

  ```{math}
  :label: EqFirmFOC_L
    w_t = e^{g_y t}(Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[(1-\gamma-\gamma_{g})\frac{Y_t}{e^{g_y t}L_t}\right]^\frac{1}{\varepsilon} \quad\forall t
  ```

  ```{math}
  :label: EqFirmFOC_K
    r_t = (1 - \tau^{corp}_t)(Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{Y_t}{K_t}\right]^\frac{1}{\varepsilon} - \delta + \tau^{corp}\delta^\tau \quad\forall t
  ```

  Note that the presence of the public capital good creates economic rents. However, given perfect competition, any economic profits will be competed away. For this reason, the optimality condition for capital demand {eq}`EqFirmFOC_K` is only affected by public capital $K_{g,t}$ through the $Y_t$ term.

(EqFirmsPosProfits)=
## Positive Profits from Government Infrastructure Investment

  The CES production function in {eq}`EqFirmsCESprodfun` exhibits constant returns to scale (CRS). A feature of CRS production functions is that gross revenue $Y_t$ is a sum of the gross revenue attributed to each factor of production,

  ```{math}
  :label: EqFirmsMargRevEq
    Y_t = MPK_t K_t + MPK_{g,t} K_{g,t} + MPL_t L_t \quad\forall t
  ```

  where $MPK_t$ is the marginal product of private capital, $MPK_{g,t}$ is the marginal product of public capital, and $MPL_t$ is the marginal product of labor. Each of the terms in {eq}`EqFirmsMargRevEq` is growing at the macroeconomic variable rate of $e^{g_y t}\tilde{N_t}$ (see the third column of {numref}`TabStnrzStatVars`). Firm profit maximization for private capital demand from equation {eq}`EqFirmFOC_K` implies that the marginal product of private capital is the following.

  ```{math}
  :label: EqFirmsMPK_opt
    MPK_t =  \frac{r_t + \delta - \tau^{corp}_t\delta^{\tau}}{1 - \tau^{corp}_t} \quad\forall t
  ```

  Firm profit maximization for labor demand from equation {eq}`EqFirmFOC_L` implies that the marginal product of labor is the following.

  ```{math}
  :label: EqFirmsMPL_opt
    MPL_t =  w_t \quad\forall t
  ```

  Even though firms take the stock of public capital $K_{g,t}$ from government infrastructure investment as given, we can still calculate the marginal product of public capital from the production function {eq}`EqFirmsCESprodfun`.

  ```{math}
  :label: EqFirmsMPKg_opt
    MPK_{g,t} =  Z_t^{\frac{\varepsilon - 1}{\varepsilon}}\left(\frac{\gamma_g Y_t}{K_{g,t}}\right)^{\frac{1}{\varepsilon}} \quad\forall t
  ```

  If we plug the expressions for $MPK_t$, $MPK_{g,t}$, and $MPL_t$ from {eq}`EqFirmsMPK_opt`, {eq}`EqFirmsMPKg_opt`, and {eq}`EqFirmsMPL_opt`, respectively, into the total revenue $Y_t$ decomposition in {eq}`EqFirmsMargRevEq` and then substitute that into the profit function {eq}`EqFirmsProfit`, we see that positive economic rents arise when public capital is positive $K_{g,t}>0$.

  ```{math}
  :label: EqFirmsProfit_Kg
    \begin{split}
      PR_t &= (1 - \tau^{corp}_t)\Bigl[Y_t - w_t L_t\Bigr] - \bigl(r_t + \delta\bigr)K_t + \tau^{corp}_t\delta^\tau K_t \\
      &= (1 - \tau^{corp}_t)\Biggl[\biggl(\frac{r_t + \delta - \tau^{corp}_t\delta^{\tau}}{1 - \tau^{corp}_t}\biggr)K_t + MPK_{g,t}K_{g,t} + w_t L_t\Biggr] ... \\
      &\quad\quad - (1 - \tau^{corp}_t)w_t L_t - (r_t + \delta)K_t + \tau^{corp}_t\delta^{\tau}K_t \\
      &= (1 - \tau^{corp}_t)MPK_{g,t}K_{g,t} \\
    \end{split}
  ```

  We assume these positive economic profits resulting from government infrastructure investment are passed on to the owners of private capital through an adjusted interest rate $r_{K,t}$ provided by the financial intermediary (see Chapter {ref}`Chap_FinInt`) that zeroes out profits among the perfectly competitive firms and is a function of $MPK_{g,t}$ and $K_{g,t}$. Total payouts from the financial intermediary $r_{K,t}K_t$ are a function of the perfectly competitive payout to owners of private capital $r_t K_t$ plus any positive profits when $K_{g,t}>0$ from {eq}`EqFirmsProfit_Kg`.

  ```{math}
  :label: EqFirmsPayout
    r_{K,t}K_t =  r_tK_t + (1 - \tau^{corp}_t)MPK_{g,t}K_{g,t} \quad\forall t
  ```

  This implies that the rate of return paid from the financial intermediary to the households $r_{K,t}$ is the interest rate on private capital $r_t$ plus the positive profits from {eq}`EqFirmsProfit_Kg`, in which the units are put in terms of $K_t$ (see equation {eq}`eq_rK` in Chapter {ref}`Chap_FinInt`).

  ```{math}
  :label: EqFirms_rKt
    r_{K,t} =  r_t + (1 - \tau^{corp}_t)MPK_{g,t}\left(\frac{K_{g,t}}{K_t}\right) \quad\forall t
  ```
