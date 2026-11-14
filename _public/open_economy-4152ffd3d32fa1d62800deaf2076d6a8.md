(Chap_SmOpEcn)=
# Open Economy Options

`OG-Core` offers a wide range of specifications regarding the type and degree of openness assumed in the economy. In none of our specifications do we fully model foreign economies as is done by {cite}`BenzellEtAl:2017` and others. However, one of the findings of {cite}`BenzellEtAl:2017` is that a full multi-country model is closely approximated by the types of large partial open economy specifications we use in `OG-Core`. Our specifications range from fully closed, to partially closed, to small open economy, to large open economy. We discuss some of these specifications in Chapter {ref}`Chap_MarkClr`. But the open economy assumptions only refer to how foreign capital can flow into the private capital market $K_t\equiv\sum_{m=1}^M K_{m,t}$ and into the government bond market $D_t$. The labor market and goods markets are closed.

(SecSmallOpen)=
## Small Open Economy
In the small open economy version of `OG-Core`, the economy faces an exogenous world interest rate on capital $r^{*}_{t}$. The parameterization for this setting is $\zeta_K=1$. This implies that foreign capital flows freely into (out of) the country to take up all the excess demand (excess supply) and that firms face the world interest rate $r^{*}_{t}$ as the competitive, zero-profit rate of return on capital as the interest rate analogous to $r_t$ that goes into the profit function {eq}`EqFirmsProfit` of each firm in each industry, $r_{K,t}$ in {eq}`EqFirmsPayout` and {eq}`eq_rK`, $r_{gov,t}$ in {eq}`EqUnbalGBC_rate_wedge`, and $r_{p,t}$ in {eq}`eq_portfolio_return`. In this case, the rate of return on capital inside the country is exogenously fixed at $r^{*}_{t}$ {eq}`EqSmOpen_rstar_r`, and foreign private capital inflows $K^f_t$ are just the difference between total private capital demand $K_t\equiv\sum_{m=1}^M K_{m,t}$ by firms at the world interest rate and total domestic private capital supply by domestic households $K^d_t$ at the world interest rate.

```{math}
:label: EqSmOpen_rstar_r
  r_t = r^*_t \quad\forall t
```

```{math}
:label: EqSmOpen_Kft
  K^f_t = K_t - K^d_t \quad\forall t \quad\text{where}\quad K_t\equiv\sum_{m=1}^M K_{m,t}
```


(SecPartialOpen)=
## Partially Open Economy

The partially open economy is the default specification of `OG-Core` in which $0<\zeta_K,\zeta_D<1$. In this case, foreign flows of private capital $K^f_t$ and foreign holdings of government bonds $D^f_t$ partially supply the respective domestic demands for these two types of capital. The equations for this partially open specification are described in Sections {ref}`SecMarkClrMktClr_G` and {ref}`SecMarkClrMktClr_K` of Chapter {ref}`Chap_MarkClr`.

The partially open economy specification can also be referred to as a large open economy because changes in underlying policy or parameter assumptions will influence the equilibrium interest rate on private capital $r_t$, while the world interest rate $r^*_t$ remains fixed. The degree to which the domestic rental rate on private capital $r_t$ responds to policy parameter changes depends on the degree to which the economy is open. In the most open case, the small open economy specification described in Section {ref}`SecSmallOpen` with $\zeta_K=1$, foreign flows of private capital $K^f_t$ are the most flexible and the domestic interest rate is exogenously fixed at the world interest rate $r^*_t$. As $\zeta_K$ goes to 0, foreign private capital flows $K^f_t$ become less flexible and the domestic interest rate $r_t$ has to adjust more to make domestic private capital demand $K_t$ equal total private capital supply $K^d_t + K^f_t$.

Note that in our partially open economy specification, the world interest rate $r^*_t$ is necessary for determining equilibrium because the foreign supply of private capital $K^f_t$ depends on a concept of excess demand that is based on domestic demand at the world interest rate {eq}`EqMarkClr_ExDemK`. It is also worth noting that our partially open economy specification is not a multi-country model in which the rest of the world or multiple other countries are explicitly modeled. In our specification, the rest of the world is simply modeled as the relationship between the domestic rental rate on private capital $r_t$ and the world interest rate $r^*_t$ as influenced by the supply of foreign private capital $\zeta_K$ and by foreign purchases of new issues of government bonds $\zeta_D$.


(SecClosed)=
## Closed Economy

The closed economy specification in `OG-Core` is parameterized by $\zeta_D=0$ and $\zeta_K=0$ and is characterized as no foreign inflows of private capital. The government debt market clearing condition and the private capital market clearing condition are the following.

```{math}
:label: EqClosed_D
  D_t = D^d_t \quad\forall t
```

```{math}
:label: EqClosed_K
  K_t = K^d_t \quad\forall t
```

In the closed economy setting, the world interest rate $r^*_t$ is not relevant.
