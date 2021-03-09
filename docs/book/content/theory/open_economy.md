(Chap_SmOpEcn)=
# Open Economy Options

`OG-USA` offers a wide range of specifications regarding the type and degree of openness assumed in the economy. In none of our specifications do we fully model foreign economies as is done by (Kotlikoff cites) and others. However, one of the findings of (Kotlikoff) is that a full multi-country model is closely approximated by the types of large partial open economy specifications we use in `OG-USA`. Our specifications range from fully closed, to partially closed, to small open economy, to large open economy. We discussed some of these specifications in the previous chapter {ref}`Chap_MarkClr`. But the open economy assumptions only refer to how foreign capital can flow into the private capital market $K_t$ and into the government bond market $D_t$. The labor market and goods market are closed.

(SecSmallOpen)=
## Small Open Economy
In the small open economy version of `OG-USA`, the economy faces an exogenous world interest rate on capital $r^{*}_{t}$. The parameterization for this setting is $\zeta_K=1$. This implies that foreign capital flows into (out of) the country to take up all the excess demand (excess supply) and that households face the world interest rate $r^{*}_{t}$ on their private savings and that firms pay the world interest rate to rent capital. The world interest rate then determines the interest rate paid by the government $r_{gov,t}$ through equation {eq}`EqUnbalGBC_rate_wedge`. In this case, the rate of return on capital inside the country is exogenously fixed at $r^{*}_{t}$ {eq}`EqSmOpen_rstar_r`, and foreign private capital inflows $K^f_t$ are just the difference between total private capital demand $K_t$ by firms at the world interest rate and total domestic private capital supply by domestic households $K^d_t$ at the world interest rate.

```{math}
:label: EqSmOpen_rstar_r
  r_t = r^*_t \quad\forall t
```

```{math}
:label: EqSmOpen_Kft
  K^{f}_{t} = K_{t} - K^{d}_{t} \quad\forall t
```


(SecPartialOpen)=
## Partially Open Economy

The partially open economy is the default specification of `OG-USA` in which $0<\zeta_K,\zeta_D<1$. In this case, foreign flows of private capital $K^f_t$ and foreign holdings of government bonds $D^f_t$ partially supply the respective domestic demands for these two types of capital. The equations for this partially open specification are described in Sections {ref}`SecMarkClrMktClr_G` and {ref}`SecMarkClrMktClr_K` of Chapter {ref}`Chap_MarkClr`.

The partially open economy specification can also be referred to as a large open economy because changes in underlying policy or parameter assumptions will influence the equilibrium interest rate on private capital $r_t$, while the world interest rate $r^*_t$ remains fixed. The degree to which the domestic rental rate on private capital $r_t$ responds to policy parameter changes depends on the degree to which the economy is open. In the most open case, the small open economy specification described in Section {ref}`SecSmallOpen` with $\zeta_K=1$, foreign flows of private capital $K^f_t$ are the most flexible and the domestic interest rate is exogenously fixed at the world interest rate $r^*_t$. As $\zeta_K$ goes to 0, foreign private capital flows $K^f_t$ become less flexible and the domestic interest rate $r_t$ has to adjust more to make domestic private capital demand $K_t$ equal total private capital supply $K^d_t + K^f_t$.

Note that in our partially open economy specification, the world interest rate $r^*_t$ is necessary for determining equilibrium because the foreign supply of private capital $K^f_t$ depends on a concept of excess demand that is based on domestic demand at the world interest rate {eq}`EqMarkClr_ExDemK`. It is also worth noting that our partially open economy specification is not a multi-country model in which the rest of the world or multiple other countries are explicitly modeled. In our specification, the rest of the world is simply modeled as the relationship between the domestic rental rate on private capital $r_t$ and the world interest rate $r^*_t$ as influenced by the supply of foreign private capital $\zeta_K$ and by foreign purchases of new issues of government bonds $\zeta_D$.


(SecClosed)=
## Closed Economy

The closed economy specification in `OG-USA` is parameterized by $\zeta_D=0$ and $\zeta_K=0$ and is characterized as no foreign inflows of private capital. The government debt market clearing condition and the private capital market clearing condition are the following.

```{math}
:label: EqClosed_D
  D_t = D^d_t \quad\forall t
```

```{math}
:label: EqClosed_K
  K_t = K^d_t \quad\forall t
```

In the closed economy setting, the world interest rate $r^*_t$ is not relevant.
