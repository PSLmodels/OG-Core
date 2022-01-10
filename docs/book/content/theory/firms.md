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
      &\equiv Z_t\biggl[(\gamma)^\frac{1}{\varepsilon}(K_t)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}(K_{g,t})^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(e^{g_y t}L_t)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \quad\forall t
    \end{split}
  ```
  where $Z_t$ is an exogenous scale parameter (total factor productivity) that can be time dependent, $\gamma$ represents private capital's share of income, $\gamma_{g}$ is public capital's share of income, and $\varepsilon$ is the constant elasticity of substitution among the two types of capital and labor. We have included constant productivity growth rate $g_y$ as the rate of labor augmenting technological progress.

  A nice feature of the CES production function is that the Cobb-Douglas production function is a nested case for $\varepsilon=1$.
  ```{math}
  :label: EqFirmsCDprodfun
    Y_t = Z_t K_t^\gamma K_{g,t}^{\gamma_{g}}(e^{g_y t}L_t)^{1-\gamma-\gamma_{g}} \quad\text{for}\quad \varepsilon=1 \quad\forall t
  ```

(EqFirmsFOC)=
## Optimality Conditions

  The profit function of the representative firm is the following.

  ```{math}
  :label: EqFirmsProfit
    PR_t = (1 - \tau^{corp})\Bigl[F(K_t,K_{g,t},L_t) - w_t L_t\Bigr] - \bigl(r_t + \delta\bigr)K_t + \tau^{corp}\delta^\tau K_t \quad\forall t
  ```

  Gross income for the firms is given by the production function $F(K,K_g,L)$ because we have normalized the price of the consumption good to 1. Labor costs to the firm are $w_t L_t$, and capital costs are $(r_t +\delta)K_t$. The government supplies public capital to the firms at no cost. The per-period interest rate (rental rate) of capital for firms is $r_t$. The per-period economic depreciation rate for private capital is $\delta$. The $\delta^\tau$ parameter in the last term of the profit function governs how much of capital depreciation can be deducted from the corporate income tax.

  Taxes enter the firm's profit function {eq}`EqFirmsProfit` in two places. The first is the corporate income tax rate $\tau^{corp}$, which is a flat tax on corporate income. Corporate income is defined as gross income minus labor costs. This will cause the corporate tax to only distort the firms' capital demand decision.

  The next place where tax policy enters the profit function {eq}`EqFirmsProfit` is through a refund of a percent of depreciation costs $\delta^\tau$ refunded at the corporate income tax rate $\tau^{corp}$. When $\delta^\tau=0$, no depreciation expense is deducted from the firm's tax liability. When $\delta^\tau=\delta$, all economic depreciation is deducted from corporate income.

  Firms take as given prices $w_t$ and $r_t$ and the level of public capital supply $K_{g,t}$. Taking the derivative of the profit function {eq}`EqFirmsProfit` with respect to labor $L_t$ and setting it equal to zero (using the general CES form of the production function {eq}`EqFirmsCESprodfun`) and taking the derivative of the profit function with respect to capital $K_t$ and setting it equal to zero, respectively, characterizes the optimal labor and capital demands.

  ```{math}
  :label: EqFirmFOC_L
    w_t = e^{g_y t}(Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[(1-\gamma-\gamma_{g})\frac{Y_t}{e^{g_y t}L_t}\right]^\frac{1}{\varepsilon} \quad\forall t
  ```

  ```{math}
  :label: EqFirmFOC_K
    r_t = (1 - \tau^{corp})(Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{Y_t}{K_t}\right]^\frac{1}{\varepsilon} - \delta + \tau^{corp}\delta^\tau \quad\forall t
  ```

  Note that the presence of the public capital good creates economic rents. However, given perfect competition, any economic profits will be competed away. For this reason, the optimality condition for capital demand {eq}`EqFirmFOC_K` is only affected by public capital $K_{g,t}$ through the $Y_t$ term.
