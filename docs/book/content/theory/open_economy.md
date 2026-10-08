(Chap_SmOpEcn)=
# Open Economy Options

`OG-Core` offers a wide range of specifications regarding the type and degree of openness assumed in the economy. In none of our specifications do we fully model foreign economies as is done by {cite}`BenzellEtAl:2017` and others. However, one of the findings of {cite}`BenzellEtAl:2017` is that a full multi-country model is closely approximated by the types of large partial open economy specifications we use in `OG-Core`. Our specifications range from fully closed, to partially closed, to small open economy, to large open economy. We discuss some of these specifications in Chapter {ref}`Chap_MarkClr`. But the open economy assumptions only refer to how foreign investment can flow into the equity market $V_t\equiv\sum_{m=1}^M V_{m,t}$ and into the government bond market $D_t$. The labor market and goods markets are closed.

(SecSmallOpen)=
## Small Open Economy
In the small open economy version of `OG-Core`, the economy faces an exogenous world interest rate on capital $r^{*}_{t}$. The parameterization for this setting is $\zeta_K=1$. This implies that foreign investment flows freely into (out of) the country to take up all the excess demand (supply) and that the equilibrium rate of return on equity in the domestic market is equilibrated with the world interest rate $r^{*}_{t}$. To solve this parameterization of the model, we can assume the rate of return on capital inside the country is exogenously fixed at $r^{*}_{t}$ {eq}`EqSmOpen_rstar_r`, and foreign investor inflows, $V^f_t$, are just the difference between total value of firms when returning $r^{*}_{t}$, $V_t\equiv\sum_{m=1}^M V_{m,t}$, and total domestic holdings of equity by domestic households, $V^d_t$, at the world interest rate, $r^{*}_{t}$.

```{math}
:label: EqSmOpen_rstar_r
  r_t = r^*_t \quad\forall t
```

```{math}
:label: EqSmOpen_Kft
  V^f_t = V_t - V^d_t \quad\forall t \quad\text{where}\quad V_t\equiv\sum_{m=1}^M V_{m,t}
```


(SecPartialOpen)=
## Partially Open Economy

The partially open economy is the default specification of `OG-Core` in which $0<\zeta_K,\zeta_D<1$. In this case, foreign equity holdings, $V^f_t$, and foreign holdings of government bonds, $D^f_t$, partially satisfy the respective domestic demands for these two types of assets. The equations for this partially open specification are described in Sections {ref}`SecMarkClrMktClr_G` and {ref}`SecMarkClrMktClr_K` of Chapter {ref}`Chap_MarkClr`.

The partially open economy specification can also be referred to as a large open economy because changes in underlying policy or parameter assumptions will influence the equilibrium interest rate on private capital $r_t$, while the world interest rate $r^*_t$ remains fixed. The degree to which the domestic interest rate, $r_t$, responds to policy parameter changes depends on the degree to which the economy is open. In the most open case, the small open economy specification described in Section {ref}`SecSmallOpen` with $\zeta_K=1$, foreign holdings of domestic equity, $V^f_t$, are the most responsive and the domestic interest rate is equilibrated to the world interest rate $r^*_t$. As $\zeta_K$ goes to 0, foreign equity holdings $V^f_t$ become less responsive to interest rate differentials and the domestic interest rate $r_t$ has to adjust more to make domestic equity supply, $V_t$, equal total demand to hold equities, $V^d_t + V^f_t$.

Note that in our partially open economy specification, the world interest rate $r^*_t$ is necessary for determining equilibrium because the foreign demand to hold domestic equities, $V^f_t$, depends on a concept of excess demand that is based on domestic demand at the world interest rate {eq}`EqMarkClr_ExDemK`. It is also worth noting that our partially open economy specification is not a multi-country model in which the rest of the world or multiple other countries are explicitly modeled. In our specification, the rest of the world is simply modeled as the relationship between the domestic interest rate, $r_t$, and the world interest rate, $r^*_t$, as influenced by foreign demand for domestic equity via $\zeta_K$ and by foreign purchases of new issues of government bonds via $\zeta_D$.


(SecClosed)=
## Closed Economy

The closed economy specification in `OG-Core` is parameterized by $\zeta_D=0$ and $\zeta_K=0$ and is characterized as no foreign investment inflows. The government debt market clearing condition and the equity market clearing condition are the following.

```{math}
:label: EqClosed_D
  D_t = D^d_t \quad\forall t
```

```{math}
:label: EqClosed_K
  V_t = V^d_t \quad\forall t
```

In the closed economy setting, the world interest rate $r^*_t$ is not relevant.
