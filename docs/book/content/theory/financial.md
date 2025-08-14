
(Chap_FinInt)=
# Financial Intermediary

<<<<<<< HEAD
Domestic household wealth $B_{t}$ is invested in a financial intermediary. This intermediary purchases a portfolio of government bonds and private capital in accordance with the domestic demand for these assets and then returns a single portfolio rate of return to all investors.
=======
  Domestic household wealth $W^d_{t}=B_{t}$ and foreign ownership of domestic assets $W^f_{t}$, both in terms of the numeraire good, are invested in a financial intermediary. This intermediary purchases a portfolio of government bonds and private capital in accordance with the domestic and foreign investor demand for these assets and then returns a single portfolio rate of return to all investors.
>>>>>>> upstream/master

  Foreign demand for government bonds is specified in section {ref}`SecMarkClrMktClr_G` of the {ref}`Chap_MarkClr` chapter:

  ```{math}
  :label: EqMarkClr_zetaD2
    D^{f}_{t+1} = D^{f}_{t} + \zeta_{D, t}(D_{t+1} - D_{t}) \quad\forall t
  ```

  This leaves domestic investors to buy up the residual amount of government debt:

  ```{math}
  :label: EqMarkClr_DomesticDebt
    D^{d}_{t} = D_{t} - D^{f}_{t} \quad\forall t
  ```

<<<<<<< HEAD
We assume that debt dominates the capital markets, such that domestic investor demand for equity, $V^{d}_{t}$ is given as:
=======
  We assume that debt dominates the capital markets, such that domestic investor demand for capital, $K^{d}_{t}$ is given as:
>>>>>>> upstream/master

  ```{math}
  :label: eq_domestic_cap_demand
    V^{d}_{t} = B_{t} - D^{d}_{t} \quad\forall t
  ```

<<<<<<< HEAD
Foreign demand for equity is given in {ref}`SecMarkClrMktClr_K`, where $V^{f}_{t}$ is an exogenous fraction of excess equity demand:
=======
  Foreign demand for capital is given in {ref}`SecMarkClrMktClr_K`, where $K^{f}_{t}$ is an exogenous fraction of excess capital demand at the world interest rate:
>>>>>>> upstream/master

  ```{math}
  :label: eq_foreign_cap_demand
    V^{f}_t = \zeta_{K, t}ED^{V,r^*}_t \quad\forall t
  ```

  The total amount invested in the financial intermediary is thus:

<<<<<<< HEAD
```{math}
    B_t & = D^d_t + V^d_t \\
```

The return on the portfolio of assets held in the financial intermediary is the weighted average of the return on equity and government debt. As derived in {eq}`EqFirms_rKt` of Section {ref}`EqFirmsPosProfits`, the presence of public infrastructure in the production function means that the returns to private factors of production ($r_t$ and $w_t$) exhibit decreasing returns to scale.[^MoorePecoraro]. These excess profits are returned to shareholders through the financial intermediary.

The return on the portfolio of assets held by the financial intermediary is thus a weighted average of the return to government debt $r_{gov,t}$ from {eq}`EqUnbalGBC_rate_wedge` and the adjusted return on equity through the distribution of profits and capital gains:

```{math}
:label: eq_portfolio_return
  r_{p,t} = \frac{r_{gov,t}D_{t} + \Pi^d_{m,t}  + (V^d_{t+1} - V^d_{t}) }{D_{t} + V^d_{t}} \quad\forall t \quad\text{where}\quad \Pi^d_t \equiv \sum_{m=1}^M \pi_{m,t} * \frac{V^d_{t}}{V_{t}}
```
=======
  ```{math}
  W_{t} & = W^d_{t} + W^f_{t} \\
      & = D^d_t + K^d_t  + D^f_t + K^f_t \\
      & = D_t + K_t
  ```

  Interest rates on private capital through the financial intermediary and on government bonds differ. The return on the portfolio of assets held in the financial intermediary is the weighted average of these two rates of return. As derived in {eq}`EqFirms_rKt` of Section {ref}`EqFirmsPosProfits`, the presence of public infrastructure in the production function means that the returns to private factors of production ($r_t$ and $w_t$) exhibit decreasing returns to scale.[^MoorePecoraro] It is assumed that competition ensures a zero profit condition among firms and the returns to public infrastructure through the returns of firms are captured by the financial intermediary and returned to share holders. The return on capital is therefore the sum of the (after-tax) returns to private and public capital.

  ```{math}
  :label: eq_rK
    r_{K,t}  = r_{t} + \frac{\sum_{m=1}^M(1 - \tau^{corp}_{m,t})p_{m,t}MPK_{g,m,t}K_{g,m,t}}{\sum_{m=1}^M K_{m,t}} \quad\forall t
  ```

  The return on the portfolio of assets held by the financial intermediary is thus a weighted average of the return to government debt $r_{gov,t}$ from {eq}`EqUnbalGBC_rate_wedge` and the adjusted return on private capital $r_{K,t}$ from {eq}`eq_rK`.

  ```{math}
  :label: eq_portfolio_return
    r_{p,t} = \frac{r_{gov,t}D_{t} + r_{K,t}K_{t}}{D_{t} + K_{t}} \quad\forall t \quad\text{where}\quad K_t \equiv \sum_{m=1}^M K_{m,t}
  ```

>>>>>>> upstream/master

(SecFinfootnotes)=
## Footnotes

  This section contains the footnotes for this chapter.

  [^MoorePecoraro]: See also {cite}`MoorePecoraro:2021` for a similar treatment of government infrastructure investment, positive profits, and returns to the owners of capital.
