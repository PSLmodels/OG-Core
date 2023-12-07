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
    \pi(K_{m,t}, K^{\tau}_{m,t}, L_{m,t}) &= (1 - \tau^{corp}_{m,t})\Bigl[p_{m,t}F(K_{m,t},K_{g,m,t},L_{m,t}) - w_t L_{m,t} - \Psi(I_{m,t}, K_{m,t})\Bigr] -  \\
    &\qquad\qquad\quad I_{m,t} + \tau^{corp}_{m,t}\delta^\tau_{m,t}K^{\tau}_{m,t} + \tau^{inv}_{m,t}I_{m,t} \quad\forall m,t
  ```

  Gross revenue for the firms is $p_{m,t}F(K_{m,t},K_{g,m,t},L_{m,t})$. Labor costs to the firm are $w_t L_{m,t}$, and investment, $I_{m,t}$ contributes to the capital stock through the following law of motion:

  ```{math}
  :label: EqFirmsInvest
    K_{m,t+1} = (1-\delta_M) K_{m,t} + I_{m,t} \quad\forall m,t
  ```

   The government supplies public capital $K_{g,m,t}$ to the firms at no cost. The per-period economic depreciation rate for private capital is $\delta_{M,t}\in[0,1]$.[^delta_M] The $\delta^\tau_{m,t}$ parameter in the second-to-last term of the profit function governs how much of capital depreciation can be deducted from the corporate income tax. Note that the last term above represents the benefits from any investment tax credit $\tau^{inv}_{m,t}$.  We assume that capital adjustment costs, $\Psi(I_{m,t}, K_{m,t})$, are quadratic in the investment rate $I_{m,t}$ and the capital stock $K_{m,t}$ and are deductible from corporate income when determining taxable income.  The quadratic adjustment costs are given by the following function:

  ```{math}
    :label: EqFirmsAdjCosts
    \Psi(I_{m,t}, K_{m,t}) = \frac{\psi}{2}\frac{\left(\frac{I_{m,t}}{K_{m,t}} - \mu_m \right)^2}{\frac{I_{m,t}}{K_{m,t}}} \quad\forall m,t
  ```

  The parameter $\mu_m = \delta_M + \bar{g}_n + g_y$, is the steady state investment rate.  Thus adjustment cost are zero in the steady state, but affect firms along the transition path.  The parameter $\psi$ governs the strength of the adjustment costs.

  Taxes enter the firm's profit function {eq}`EqFirmsProfit` in two places. The first is the corporate income tax rate $\tau^{corp}_{m,t}$, which is a flat tax on corporate income that can vary by industry $m$. Corporate income is defined as gross income minus labor costs. This will cause the corporate tax to only have a direct effect on the firms' capital demand decision.

  The tax policy also enters the profit function {eq}`EqFirmsProfit` through depreciation deductions at rate $\delta^\tau_{m,t}$, which then lower corporate tax liability. When $\delta^\tau_{m,t}=0$, no depreciation expense is deducted from the firm's tax liability. When $\delta^\tau_{m,t}=\delta_{M}$, all economic depreciation is deducted from corporate income. The investment tax credit is characterized by the parameter $\tau^{inv}_{m,t}$ multiplied by the amount of investment.  The tax basis for the capital stock, $K^\tau_{m,t}$ is determined through the following law of motion:

  ```{math}
  :label: EqFirmsKtau
    K^{\tau}_{m,t} = (1-\delta^{\tau}_{m,t}) K^\tau_{m,t-1} + (1-\tau^{inv}_{m,t})I_{m,t} \quad\forall m,t
  ```

  Note the timing difference in the laws of motion for the tax basis for the capital stock and the capital stock. Investment is immediately deducted from the tax basis for the capital stock, where are there is a time to build in the physical capital stock (new investment doesn't become productive until the next period).

  The expected (pre-tax) return to a shareholder investing in firm $m$ is given by:

  ```{math}
  :label: EqFirmsExpectedReturn
  \begin{split}
    E_{t}(R_{m,t}) = \frac{\pi_{m,t} + E_{t}V_{m,t+1} - V{m,t}}{V_{m,t}}
  \end{split}
  ```

  where $V_t$ is the value of the firm, $E_tV_{t+1}$ is the expected value of the firm next period, and $\pi_t$ is the profit of the firm in period $t$. Without aggregate uncertainty, it must be the case that the expected return to a shareholder for an investment in any industry is the same: $E_{t}(R_{m,t}) = E_{t}(R_{t})$. Futhermore, the rate of return on equity must be the same as the rate of return on government debt, adjusted for the risk premium.  That is, $E_{t}(R_{t}) = 1 + r_{t} = 1 + \left(\frac{r_{gov,t} + \mu_d}{(1-\tau_{d,t})}\right)$ where $r_{gov,t}$ is the risk-free rate of return on government debt and the parameters $\mu_d$  and $\tau_{d,t}$ define the risk premium of equity over government debt that we've built into this model with no uncertainty.

  Firms take as given prices $p_{m,t}$, $w_t$, and $r_t$ and the level of public capital supply $K_{g,m,t}$. These firms are forward looking and maximize the present discounted value of profits, where firms discount by the real interest rate $r_t$. The firm's problem is the following:

  ```{math}
  :label: EqFirmsValue
  \begin{split}
    V(K_{m,t}, K^\tau_{m,t-1}) &=  \max_{\{I_{m,u}, K_{m,u+1},K^\tau_{m,u}, L_{m,u}\}_{u=t}^{\infty}}  \sum_{u=t}^\infty \left(\prod_{v=t}^u \frac{1}{1+r_v}\right) \pi(K_{m,u}, L_{m,u}) \quad\forall m,t,\\
     \quad \text{subject to}: & \\
      & \\
      I_{m,t} &=  K_{m,t+1} - (1-\delta_M) K_{m,t} + I_{m,t} \quad\forall m,t \\
       K^{\tau}_{m,t} &= (1-\delta^{\tau}_{m,t}) K^\tau_{m,t-1} + (1-\tau^{inv}_{m,t})I_{m,t} \quad\forall m,t

  \end{split}
  ```


  Note that one can use the two laws of motion (for the physical capital stock and the tax basis of the capital stock) to eliminate to choice variables: choosing $K_{m,u+1}$ will determine $I_{m,t}$ and $K^\tau_{m,t}$. Taking the derivative of the profit function {eq}`EqFirmsProfit` with respect to labor $L_{m,t}$ and setting it equal to zero (using the general CES form of the production function {eq}`EqFirmsCESprodfun`) and taking the derivative of the profit function with respect to private capital $K_{m,t+1}$ and setting it equal to zero, respectively, characterizes the optimal labor and capital demands.

  ```{math}
  :label: EqFirmFOC_L
    w_t = e^{g_y t}p_{m,t}(Z_{m,t})^\frac{\varepsilon_m-1}{\varepsilon_m}\left[(1-\gamma_m-\gamma_{g,m})\frac{Y_{m,t}}{e^{g_y t}L_{m,t}}\right]^\frac{1}{\varepsilon_m} \quad\forall m,t
  ```

  ```{math}
  :label: EqFirmFOC_K
    r_{t+1} = \frac{(1 - \tau^{corp}_{m,t+1})\left(p_{m,t+1}(Z_{m,t+1})^\frac{\varepsilon_m-1}{\varepsilon_m}\left[\gamma_m\frac{Y_{m,t+1}}{K_{m,t+1}}\right]^\frac{1}{\varepsilon_m} - \frac{\partial \Psi(I_{m,t+1},K_{m,t+1})}{\partial K_{m,t+1}}\right) + 1 - \delta_{m} + \tau^{corp}_{m,t+1}\delta^\tau_{m,t+1}\left[(1-\tau^{inv}_{m,t})(1-\delta^\tau_{m,t})-(1-\delta_m)(1-\tau^{inv}_{m,t+1})\right] - \tau^{inv}_{m,t+1}(1-\delta_{m})}{(1-\tau^{corp}_{m,t})\frac{\partial \Psi(I_{m,t},K_{m,t})}{\partial K_{m,t+1}}+1 -\tau^{inv}_{m,t}-\tau^{corp}_{m,t}\delta^{\tau}_{m,t}(1-\tau^{inv}_{m,t})} - 1 \quad\forall m,t
  ```

  The derivatives of the quadratic adjustment cost function are given by:

  ```{math}
  :label: EqFirmAdjCost_dKp1
    \frac{\partial \Psi(I_{m,t},K_{m,t})}{\partial K_{m,t+1}} = \frac{\psi}{2}\frac{\left(\frac{I_{m,t}}{K_{m,t}}- \mu_{m}\right)}{I_{t}}\left[1 -\mu\frac{K_{m,t}}{I_{m,t}}\right] \forall m,t
  ```

  and

  ```{math}
  :label: EqFirmAdjCost_dK
    \frac{\partial \Psi(I_{m,t},K_{m,t})}{\partial K_{m,t}} = \frac{-\psi}{2}\left(\frac{K_{m,t+1}}{I^2_{m,t}}\right)\left(\frac{I_{m,t}}{K_{m,t}}-\mu\right)\left(\frac{I_{m,t}}{K_{m,t}}+\mu\right) \forall m,t
  ```

  Note that the presence of the public capital good creates economic rents. These rents will accrue to the owners of capital via the financial intermediary. See Section Chapter {ref}`Chap_FinInt` for more details on the determination of the return to the household's portfolio. Because public capital is exogenous to the firm's decisions, the optimality condition for capital demand {eq}`EqFirmFOC_K` is only affected by public capital $K_{g,m,t}$ through the $Y_{m,t}$ term.

(EqFirmsPosProfits)=
## Positive Profits from Government Infrastructure Investment

Note that when $K_{g,t}$ is greater than zero, the public capital factor of production generates above normal profits for the firm. That is, the actual rate of return on equity, $E_t(R_t)$, will exceed the discount factor used by the firm, $1+r_t$. This is because the firm does not pay for the public capital that it uses in production. The above normal return will be distributed to shareholders as part of the firms profits and thus the return on equity.  We detail this further in the {ref}`Chap_FinInt` Chapter.

(SecFirmsfootnotes)=
## Footnotes

  [^Kg0_case]: It is important to note a special case of the Cobb-Douglas ($\varepsilon_m=1$) production function that we have to manually restrict. The inputs of production of private capital $K_{m,t}$ and labor $L_{m,t}$ are endogenous and have characteristics of the model that naturally bound them away from zero. But public capital $K_{g,m,t}$, although it is a function of endogenous variables in {eq}`EqUnbalGBC_Igt` and {eq}`EqUnbalGBC_Igmt`, can be exogenously set to zero as a policy parameter choice by setting $\alpha_{I,t}=0$ or $\alpha_{I,m,t}=0$. In the Cobb-Douglas case of the production function $\varepsilon_m=1$ {eq}`EqFirmsCDprodfun`, $K_{g,m,t}=0$ would zero out production and break the model. In the case when $\varepsilon_m=1$ and $K_{g,m,t}=0$, we set $\gamma_{g,m}=0$, thereby restricting the production function to only depend on private capital $K_{m,t}$ and labor $L_{m,t}$. This necessary restriction limits us from performing experiments in the model of the effect of changing $K_{g,m,t}=0$ to $K_{g,m,t}>0$ or vice versa in the $\varepsilon_m=1$ case.

  [^delta_M]: Because we are assuming that only the output of the $M$th industry can be used for investment, government spending, or government debt, and because that industry's output is the numeraire, the only depreciation rate that matters or can be nonzero is that of the $M$th industry $\delta_{M,t}$.

  [^MPfactors]: See Section {ref}`SecAppDerivCES` of the {ref}`Chap_Deriv` Chapter for the derivations of the marginal product of private capital $MPK_{m,t}$, marginal product of public capital $MPk_{g,m,t}$, and marginal product of labor $MPL_{m,t}$.
