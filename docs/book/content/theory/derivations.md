(Chap_Deriv)=
# Derivations


This appendix contains derivations from the theory in the body of this book.


(SecAppDerivIndSpecCons)=
## Household first order condition for industry-specific consumption demand

  The derivation for the household first order condition for industry-specific consumption demand {eq}`EqHHFOCcm` is the following:
  ```{math}
  :label: EqAppDerivHHIndSpecConsFOC
    \tilde{p}_{m,t} = \tilde{p}_{j,s,t}\alpha_m(c_{j,m,s,t} - c_{min,m})^{\alpha_m-1}\prod_{u\neq m}^M\left(c_{j,u,s,t} - c_{min,u}\right)^{\alpha_u} \\
    \tilde{p}_{m,t}(c_{j,m,s,t} - c_{min,m}) = \tilde{p}_{j,s,t}\alpha_m(c_{j,m,s,t} - c_{min,m})^{\alpha_m}\prod_{u\neq m}^M\left(c_{j,u,s,t} - c_{min,u}\right)^{\alpha_u} \\
    \tilde{p}_{m,t}(c_{j,m,s,t} - c_{min,m}) = \tilde{p}_{j,s,t}\alpha_m\prod_{m=1}^M\left(c_{j,m,s,t} - c_{min,m}\right)^{\alpha_m} = \alpha_m \tilde{p}_{j,s,t}c_{j,s,t}
  ```


(SecAppDerivCES)=
## Properties of the CES Production Function

  The constant elasticity of substitution (CES) production function of capital and labor was introduced by {cite}`Solow:1956` and further extended to a consumption aggregator by {cite}`Armington:1969`. The CES production function of private capital $K$, public capital $K_g$ and labor $L$ we use in Chapter {ref}`Chap_Firms` is the following,

  ```{math}
  :label: EqAppDerivCESprodfun
    Y &= F(K, K_g, L) \\
    &\equiv Z\biggl[(\gamma)^\frac{1}{\varepsilon}(K)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_g)^\frac{1}{\varepsilon}(K_g)^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_g)^\frac{1}{\varepsilon}(L)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1}
  ```

  where $Y$ is aggregate output (GDP), $Z$ is total factor productivity, $\gamma$ is a share parameter that represents private capital's share of income in the Cobb-Douglas case ($\varepsilon=1$), $\gamma_g$ is public capital's share of income, and $\varepsilon$ is the elasticity of substitution between capital and labor. The stationary version of this production function is given in Chapter {ref}`Chap_Stnrz`. We drop the $m$ and $t$ subscripts, the ``$\:\,\hat{}\,\:$'' stationary notation, and use the stationarized version of the production function for simplicity.

  The Cobb-Douglas production function is a nested case of the general CES production function with unit elasticity $\varepsilon=1$.
  ```{math}
  :label: EqAppDerivCES_CobbDoug
    Y = Z(K)^\gamma(K_{g})^{\gamma_{g}}(L)^{1-\gamma-\gamma_{g}}
  ```

  The marginal productivity of private capital $MPK$ is the derivative of the production function with respect to private capital $K$. Let the variable $\Omega$ represent the expression inside the square brackets in the production function {eq}`EqAppDerivCESprodfun`.
  ```{math}
  :label: EqAppDerivCES_MPK
    MPK &\equiv \frac{\partial F}{\partial K} = \left(\frac{\varepsilon}{\varepsilon-1}\right)Z\left[\Omega\right]^\frac{1}{\varepsilon-1}\gamma^\frac{1}{\varepsilon}\left(\frac{\varepsilon-1}{\varepsilon}\right)(K)^{-\frac{1}{\varepsilon}} \\
    &= Z\left[\Omega\right]^\frac{1}{\varepsilon-1}\left(\frac{\gamma}{K}\right)^\frac{1}{\varepsilon} = \frac{Z\left[\Omega\right]^\frac{1}{\varepsilon-1}}{Z^\frac{1}{\varepsilon-1}\left[\Omega\right]^\frac{1}{\varepsilon-1}}\left(\frac{\gamma}{K}\right)^\frac{1}{\varepsilon}Y^\frac{1}{\varepsilon} \\
    &= (Z)^\frac{\varepsilon-1}{\varepsilon}\left(\gamma\frac{Y}{K}\right)^\frac{1}{\varepsilon}
  ```

  The marginal productivity of public capital $MPK_g$ is the derivative of the production function with respect to public capital $K_g$.
  ```{math}
  :label: EqAppDerivCES_MPKg
    MPK_g &\equiv \frac{\partial F}{\partial K_g} = \left(\frac{\varepsilon}{\varepsilon-1}\right)Z\left[\Omega\right]^\frac{1}{\varepsilon-1}\gamma_g^\frac{1}{\varepsilon}\left(\frac{\varepsilon-1}{\varepsilon}\right)(K_g)^{-\frac{1}{\varepsilon}} \\
    &= Z\left[\Omega\right]^\frac{1}{\varepsilon-1}\left(\frac{\gamma_g}{K_g}\right)^\frac{1}{\varepsilon} = \frac{Z\left[\Omega\right]^\frac{1}{\varepsilon-1}}{Z^\frac{1}{\varepsilon-1}\left[\Omega\right]^\frac{1}{\varepsilon-1}}\left(\frac{\gamma_g}{K_g}\right)^\frac{1}{\varepsilon}Y^\frac{1}{\varepsilon} \\
    &= (Z)^\frac{\varepsilon-1}{\varepsilon}\left(\gamma_g\frac{Y}{K_g}\right)^\frac{1}{\varepsilon}
  ```

  The marginal productivity of labor $MPL$ is the derivative of the production function with respect to labor $L$.
  ```{math}
  :label: EqAppDerivCES_MPL
    MPL &\equiv \frac{\partial F}{\partial L} = \left(\frac{\varepsilon}{\varepsilon-1}\right)Z\left[\Omega\right]^\frac{1}{\varepsilon-1}(1-\gamma-\gamma_g)^\frac{1}{\varepsilon}\left(\frac{\varepsilon-1}{\varepsilon}\right)(L)^{-\frac{1}{\varepsilon}} \\
    &= Z\left[\Omega\right]^\frac{1}{\varepsilon-1}\left(\frac{1-\gamma-\gamma_g}{L}\right)^\frac{1}{\varepsilon} = \frac{Z\left[\Omega\right]^\frac{1}{\varepsilon-1}}{Z^\frac{1}{\varepsilon-1}\left[\Omega\right]^\frac{1}{\varepsilon-1}}\left(\frac{1-\gamma-\gamma_g}{L}\right)^\frac{1}{\varepsilon}Y^\frac{1}{\varepsilon} \\
    &= (Z)^\frac{\varepsilon-1}{\varepsilon}\left([1-\gamma-\gamma_g]\frac{Y}{L}\right)^\frac{1}{\varepsilon}
  ```


(SecAppDerivCESwr)=
### Wages as a function of interest rates

The below shows that with the addition of public capital as a third factor of production, wages and interest rates are more than a function of the capital labor ratio.  This means that in the solution method for `OG-Core` we will need to guess both the interest rate $r_t$ and wage $w_t$.

```{math}
:label: EqAppDerivCES_YL
\begin{split}
    Y &= Z\biggl[(\gamma)^\frac{1}{\varepsilon}(K)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}(K_{g})^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(L)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \\
    &= Z\biggl[(\gamma)^\frac{1}{\varepsilon}(K)^\frac{\varepsilon-1}{\varepsilon}\left(\frac{L^\frac{\varepsilon-1}{\varepsilon}}{L^\frac{\varepsilon-1}{\varepsilon}}\right) + (\gamma_{g})^\frac{1}{\varepsilon}(K_{g})^\frac{\varepsilon-1}{\varepsilon}\left(\frac{L^\frac{\varepsilon-1}{\varepsilon}}{L^\frac{\varepsilon-1}{\varepsilon}}\right) + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(L)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1}
    \\
    &= ZL\biggl[(\gamma)^\frac{1}{\varepsilon}\left(\frac{K}{L}\right)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}\left(\frac{K_{g}}{L}\right)^\frac{\varepsilon-1}{\varepsilon}+ (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1}\\
    \Rightarrow\quad \frac{Y}{L} &= Z\biggl[(\gamma)^\frac{1}{\varepsilon}\left(\frac{K}{L}\right)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}\left(\frac{K_{g}}{L}\right)^\frac{\varepsilon-1}{\varepsilon}+ (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1}
\end{split}
```


```{math}
:label: EqAppDerivCES_YK
\begin{split}
    Y &= Z\biggl[(\gamma)^\frac{1}{\varepsilon}(K)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}(K_{g})^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(L)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \\
    &= Z\biggl[(\gamma)^\frac{1}{\varepsilon}(K)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}(K_{g})^\frac{\varepsilon-1}{\varepsilon}\left(\frac{K^\frac{\varepsilon-1}{\varepsilon}}{K^\frac{\varepsilon-1}{\varepsilon}}\right) + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(L)^\frac{\varepsilon-1}{\varepsilon}\left(\frac{K^\frac{\varepsilon-1}{\varepsilon}}{K^\frac{\varepsilon-1}{\varepsilon}}\right)\biggr]^\frac{\varepsilon}{\varepsilon-1}\\
    &= ZK\biggl[(\gamma)^\frac{1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}\left(\frac{K_{g}}{K}\right)^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}\left(\frac{L}{K}\right)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \\
    \Rightarrow\quad \frac{Y}{K} &= Z\biggl[(\gamma)^\frac{1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}\left(\frac{K_{g}}{K}\right)^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}\left(\frac{L}{K}\right)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1}
\end{split}
```

Solving for the firm's first order conditions for capital and labor demand from profit maximization {eq}`EqStnrzProfit` gives the following equations in their respective stationarized forms from Chapter {ref}`Chap_Stnrz`.

```{math}
:label: EqFirmFOC_L_der
    w = (Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[(1-\gamma-\gamma_{g})\frac{Y}{L}\right]^\frac{1}{\varepsilon}
```

```{math}
:label: EqFirmFOC_K_der
    r = (1 - \tau^{corp})(Z)^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{Y}{K}\right]^\frac{1}{\varepsilon} - \delta + \tau^{corp}\delta^\tau
```

As can be seen from {eq}`EqFirmFOC_L_der` and {eq}`EqFirmFOC_K_der`, the wage $w$ and interest rate $r$ are functions of $Y/L$ and $Y/K$, respectively. Equations {eq}`EqAppDerivCES_YL` and {eq}`EqAppDerivCES_YK` show that both $Y/L$ and $Y/K$ are functions of the capital-labor ratio $K/L$, the public-capital-labor ratio, $K_{g}/L$, and the public-private capital ratio, $K/K_{g}$. We cannot solve these equations for $r$ and $w$ solely as functions of the same ratios.


In the Cobb-Douglas unit elasticity case ($\varepsilon=1$) of the CES production function, the first order conditions are:
```{math}
:label: EqAppDerivCES_CDFOCL
  \text{if}\:\:\,\varepsilon=1:\quad w = (1-\gamma-\gamma_g)Z\left(\frac{K}{L}\right)^\gamma \left(\frac{K_{g}}{L}\right)^{\gamma_{g}}
```
```{math}
:label: EqAppDerivCES_CDFOCK
 \text{if}\:\:\:\varepsilon=1:\quad r = (1 - \tau^{corp})\gamma Z\left(\frac{K_{g}}{K}\right)^{\gamma_{g}}\left(\frac{L}{K}\right)^{1-\gamma-\gamma_{g}} - \delta + \tau^{corp}\delta^\tau
```

Again, even if this simple case, we cannot solve for $r$ as a function of $w$ for the reasons above.
