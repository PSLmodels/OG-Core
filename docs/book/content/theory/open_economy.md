(Chap_SmOpEcn)=
# Open Economy Options

(SecSmallOpen)=
## Small Open Economy
In the small open economy version of `OG-USA`, the county faces an exogenous world interest rate, $r^{*}_{t}$ that determines the amount of savings and investment.  If the supply of savings from households does not meet the demand for private capital and private borrowing, foreign capital will flow in to make excess demand zero at the world interest rate.  Let the total capital stock be given by the quantity of domestically supplied capital and foreign supplied capital, i.e., $K_{t}= K^{d}_{t}+K^{f}_{t}$.  Then foreign capital is given by:

```{math}
  K^{f}_{t} = K^{demand}_{t} - B_{t} - D_{t},
```

where $B_{t}$ is aggregate household savings and $D_{t}$ is government borrowing.  Capital demand is determined from the firm's first order condition for its choice of capital.

(SecPartialOpen)=
## Partially Open Economy

In the partially open economy version of `OG-USA`, the openness of the economy is modeled through two parameters that capture the extent of foreign lending to the domestic government and the amount of foreign lending of private capital to firms.


The parameter $\zeta_{D}$ gives the share of new debt issues that are purchased by foreigners.  The law of motion for foreign-held debt is therefore given by:

```{math}
  D^{f}_{t+1} = D^{f}_{t} + \zeta_{D}(D_{t+1} - D_{t})
```

Domestic debt holdings as then the remaining debt holdings needed to meet government demand for debt:

```{math}
  D^{d}_{t} = D_{t} - D^{f}_{t}
```


The parameters $\zeta_{K}$ helps to determine the share of domestic capital held by foreigners.  In particular, $\zeta_{D}$ is the share of foreign capital held by foreigners in the small open economy specification:

```{math}
  K^{f}_{t} = \zeta_{K}K^{open}_{t}
```

$K^{open}_{t}$ is the amount of capital that would need to flow into the country to meet firm demand for capital at the exogenous world interest rate from the small open economy specification, net of what domestic households can supply:

```{math}
  K^{open}_{t} = K^{demand, open}_{t} - (B_{t} - D^{d}_{t})
```

where, $K^{demand, open}_{t}$ is total capital demand by domestic firms at $r^
{*}_{t}$, $B_{t}$ are total asset holdings of domestic households, and $D^{d}_{t}$ are holdings of government debt by domestic households.  Total asset holdings from households result from solving the household problem at the endogenous home country interest rate.  Note that there is a disconnect between the interest rates that determine firm capital demand and domestic household savings and the interest rate used to determine $K^{demand, open}_{t}$.  This assumption is useful in that it nests the small open economy case into the partial open economy model.  However, it does leave out the realistic responses of foreign capital supply to differentials in the home country interest rate and the world interest rate.

Given the two equations above, we can find the total supply of capital as:

```{math}
  K^{supply}_{t} & = K^{d}_{t} + K^{f}_{t} \\
   & = B_{t} - D^{d}_{t} + \zeta_{K}K^{open}_{t} \\
```

(SecOpenStationary)=
### Stationarization

(SecForeignDebt)=
#### Foreign debt purchases

The amount of government debt is growing by the rate of productivity growth and the rate of population growth.  Thus, stationarized government debt is given by:

```{math}
  \hat{D}_{t} = \frac{D_{t}}{e^{g_{y}t}N_{t}}
```

The stationarized form of the foreign and domestic capital holdings thus become:

```{math}
    \hat{D}^{f}_{t+1} & = \frac{D^{f}_{t+1}}{e^{g_{y}t+1}N_{t+1}} = \frac{D^{f}_{t}}{e^{g_{y}t+1}N_{t+1}} + \zeta_{D}(\frac{D_{t+1}}{e^{g_{y}t+1}N_{t+1}} - \frac{D_{t}}{e^{g_{y}t+1}N_{t+1}}) \\
    & = \frac{\hat{D}^{f}_{t}N_{t}}{e^{g_{y}}N_{t+1}} + \zeta_{D}(\hat{D}_{t+1} - \frac{\hat{D}_{t}N_{t}}{e^{g_{y}}N_{t+1}}) = \frac{\hat{D}^{f}_{t}}{e^{g_{y}}g_{n,t+1}} + \zeta_{D}(\hat{D}_{t+1} - \frac{\hat{D}_{t}}{e^{g_{y}}g_{n,t+1}})
```

and

```{math}
  \hat{D}^{d}_{t} = \frac{D^{d}_{t}}{e^{g_{y}t}N_{t}} = \frac{D_{t}}{e^{g_{y}t}N_{t}} - \frac{D^{f}_{t}}{e^{g_{y}t}N_{t}} = \hat{D}_{t} - \hat{D}^{f}_{t}
```


Note that in the steady-state, we still have $\hat{D}^{f} = \zeta_{D}\hat{D}$

(SecSForeignCapital)=
#### Foreign capital purchases

In the equation for foreign capital purchases, all quantities are growing at the rate of technological change and population growth.  Thus, to stationarize this equation, we find:

```{math}
  \hat{K}^{f}_{t} = \frac{K^{f}_{t}}{e^{g_{y}t}N_{t}}= \zeta_{K}\frac{K^{open}_{t}}{e^{g_{y}t}N_{t}} = \zeta_{K}\hat{K}^{open}_{t}
```

and

```{math}
  \hat{K}^{open}_{t} = \frac{K^{open}_{t}}{e^{g_{y}t}N_{t}}= \frac{K^{demand, open}_{t}}{e^{g_{y}t}N_{t}} - \left(\frac{B_{t}}{e^{g_{y}t}N_{t}} - \frac{D^{d}_{t}}{e^{g_{y}t}N_{t}}\right) = \hat{K}^{demand, open}_{t} - (\hat{B}_{t}-\hat{D}_{t})
```

and

```{math}
  \hat{K}^{supply}_{t} &= \frac{K^{supply}_{t}}{e^{g_{y}t}N_{t}} = \frac{K^{d}_{t}}{e^{g_{y}t}N_{t}} + \frac{K^{f}_{t}}{e^{g_{y}t}N_{t}} = \hat{K}^{d}_{t} + \hat{K}^{f}_{t} \\
   & = \frac{B_{t}}{e^{g_{y}t}N_{t}} - \frac{D^{d}_{t}}{e^{g_{y}t}N_{t}} + \zeta_{K}\frac{K^{open}_{t}}{e^{g_{y}t}N_{t}} = \hat{B}_{t} - \hat{D}^{d}_{t} + \zeta_{K}\hat{K}^{open}_{t} \\
```

(SecOpenRC)=
#### Resource Constraint

As a result of the foreign ownership of capital, the resource constraint is modified.  In a closed economy, the resource constraint is given by:

```{math}
  Y_{t} = C_{t} + I_{t} + G_{t}
```

In the partially open economy, some of the output is paid to the foreign owners of capital.  This amount is given by $r_{t}K^{f}_{t}$.  In addition, foreign lending to the home country's government relaxes the resource constraint.  In the case , the resource constraint is given by:

```{math}
  Y_{t} = C_{t} + (K^{d}_{t+1} - K^{d}_{t}) + \delta K_{t} +  G_{t} + r_{t}K^{f}_{t} - (D^{f}_{t+1}-D^{f}_{t}) + rD^{f}_{t}
```

The stationarized version of this becomes:

```{math}
  \hat{Y}_{t} = \hat{C}_{t} + (\hat{K}^{d}_{t+1}e^{g_{y}}(1+g_{n,t+1}) - \hat{K}^{d}_{t}) + \delta \hat{K}_{t} +  \hat{G}_{t} + r_{t}\hat{K}^{f}_{t} - (\hat{D}^{f}_{t+1}e^{g_{y}}(1+g_{n,t+1})- \hat{D}^{f}_{t}) + r_{t}\hat{D}^{f}_{t}
```

Note that with a wedge between the interest rate on government debt and private capital as outlined in Chapter {ref}`SecRateWedge` we need to be careful about the interest rates paid and the amount of capital and debt held.  In the case of the partially open economy with an interest rate wedge, we assume that domestic and foreign investors earn a rate of return on their portfolio of:

```{math}
  r_{hh,t} = \frac{r_{t}K_{t} + r_{gov,t}D_{t}}{K_{t} + D_{t}}
```

In the partially open economy, the ratio of private capital to debt held by domestic households might differ from the ratio held by foreign households, but we assume they still earn the same rate of return on their portfolio.\footnote{One reason for this assumption is that it simplifies our solution since we do not need to know the domestic versus foreign holdings of capital before solving the households' problems.}  With this assumption, we modify the resource constraint to be a function of this portfolio interest rate:

```{math}
  \hat{Y}_{t} = \hat{C}_{t} + (\hat{K}^{d}_{t+1}e^{g_{y}}(1+g_{n,t+1}) - \hat{K}^{d}_{t}) + \delta \hat{K}_{t} +  \hat{G}_{t} + r_{hh, t}\hat{K}^{f}_{t} - (\hat{D}^{f}_{t+1}e^{g_{y}}(1+g_{n,t+1})- \hat{D}^{f}_{t}) + r_{hh,t}\hat{D}^{f}_{t}
```
