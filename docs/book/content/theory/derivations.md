(Chap_Deriv)=
# Derivations


This appendix contains derivations from the theory in the body of this book.

(SecAppDerivCES)=
## Properties of the CES Production Function

  The constant elasticity of substitution (CES) production function of capital and labor was introduced by {cite}`Solow:1956` and further extended to a consumption aggregator by {cite}`Armington:1969`. The CES production function of aggregate capital $K_t$ and aggregate labor $L_t$ we use in Chapter {ref}`Chap_Firms` is the following,

  ```{math}
  :label: EqFirmsCESprodfun
    Y_t = F(K_t, K_{g,t}, L_t) \equiv Z_t\biggl[(\gamma)^\frac{1}{\varepsilon}(K_t)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}(K_{g,t})^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(e^{g_y t}L_t)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \quad\forall t
  ```

  where $Y_t$ is aggregate output (GDP), $Z_t$ is total factor productivity, $\gamma$ is a share parameter that represents private capital's share of income in the Cobb-Douglas case ($\varepsilon=1$), $\gamma_{g}$ is public capita's share of income, and $\varepsilon$ is the elasticity of substitution between capital and labor. The stationary version of this production function is given in Chapter {ref}`Chap_Stnrz`. We drop the $t$ subscripts, the ``$\:\,\hat{}\,\:$'' stationary notation, and use the stationarized version of the production function {eq}`EqStnrzCESprodfun` for simplicity.

  ```{math}
  :label: EqStnrzCESprodfun
    Y= Z\biggl[(\gamma)^\frac{1}{\varepsilon}(K)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}(K_{g})^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(L)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \quad\forall t
  ````

  The Cobb-Douglas production function is a nested case of the general CES production function with unit elasticity $\varepsilon=1$.
  ```{math}
  :label: EqAppDerivCES_CobbDoug
    Y = Z(K)^\gamma(K_{g})^{\gamma_{g}}(L)^{1-\gamma-\gamma_{g}}
  ```

(SecAppDerivCESwr)=
### Wages as a function of interest rates

The below shows that with the addition of public capital as a third factor of production, wages and interest rates are more than a function of the capital labor ratio.  This means that in the solution method for `OG-USA` we will need to guess both the interest rate $r_t$ and wage $w_t$.

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
    Y &= Z\biggl[(\gamma)^\frac{1}{\varepsilon}(K)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}(K_{g})^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(L)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \\
    &= Z\biggl[(\gamma)^\frac{1}{\varepsilon}(K)^\frac{\varepsilon-1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}(K_{g})^\frac{\varepsilon-1}{\varepsilon}\left(\frac{K^\frac{\varepsilon-1}{\varepsilon}}{K^\frac{\varepsilon-1}{\varepsilon}}\right) + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}(L)^\frac{\varepsilon-1}{\varepsilon}\left(\frac{K^\frac{\varepsilon-1}{\varepsilon}}{K^\frac{\varepsilon-1}{\varepsilon}}\right)\biggr]^\frac{\varepsilon}{\varepsilon-1}\\
    &= ZK\biggl[(\gamma)^\frac{1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}\left(\frac{K_{g}}{K}\right)^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}\left(\frac{L}{K}\right)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \\
    \Rightarrow\quad \frac{Y}{K} &= Z\biggl[(\gamma)^\frac{1}{\varepsilon} + (\gamma_{g})^\frac{1}{\varepsilon}\left(\frac{K_{g}}{K}\right)^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma-\gamma_{g})^\frac{1}{\varepsilon}\left(\frac{L}{K}\right)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1}
```

Solving for the firm's first order conditions for capital and labor demand from profit maximization {eq}`EqStnrzProfit` gives the following equations in their respective stationarized forms from Chapter {ref}`Chap_Stnrz`.

```{math}
:label: EqFirmFOC_L
    w = (Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[(1-\gamma-\gamma_{g})\frac{Y}{L}\right]^\frac{1}{\varepsilon}
```

```{math}
:label: EqFirmFOC_K
    r = (1 - \tau^{corp})(Z)^\frac{\varepsilon-1}{\varepsilon}\left[\left(\gamma\frac{Y}{K}\right)^\frac{1}{\varepsilon}+ \left(\gamma\frac{Y}{K_{g}}\right)^\frac{1}{\varepsilon}\frac{K_{g}}{K}\right] - \delta + \tau^{corp}\delta^\tau
```

As can be seen from {eq}`EqStnrzFOC_L` and {eq}`EqFirmFOC_K`, the wage $w$ and interest rate $r$ are functions of $Y/L$ and $Y/K$ and $Y/K_{g}$, respectively. Equations {eq}`EqAppDerivCES_YL` and {eq}`EqAppDerivCES_YK` show that both $Y/L$ and $Y/K$ are functions of the capital-labor ratio $K/L$, the public-capital labor ratio, $K_{g}/L$, and the public-private capital ratio, $K/K_{g}$. We cannot solve these equations for $r$ and $w$ solely as functions of the same ratios.


In the Cobb-Douglas unit elasticity case ($\varepsilon=1$) of the CES production function, the first order conditions are:
```{math}
:label: EqAppDerivCES_CDFOCL
    \text{if}\:\:\,\varepsilon=1:\quad w &= (1-\gamma)Z\left(\frac{K}{L}\right)^\gamma \left(\frac{K_{g}}{L}\right)^{\gamma_{g}} \\
    \text{if}\:\:\:\varepsilon=1:\quad r &= (1 - \tau^{corp})\gamma 
```
```{math}
:label: EqAppDerivCES_CDFOCK   
Z\left(\frac{L}{K}\right)^{1-\gamma-\gamma_{g}}\left(\frac{K_{g}}{K}\right)^{\gamma_{g}} - \delta + \tau^{corp}\delta^\tau
```

Again, even if this simple case, we cannot solve for $r$ as a function of $w$... there are 3 ratios here, not one as in the case without public capital in the production function.
