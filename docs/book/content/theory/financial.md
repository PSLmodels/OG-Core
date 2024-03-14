
(Chap_FinInt)=

# Financial Intermediary

Domestic household wealth $B_{t}$ is invested in a financial intermediary. This intermediary purchases a portfolio of government bonds and private capital in accordance with the domestic demand for these assets and then returns a single portfolio rate of return to all investors.

Foreign demand for government bonds is specified in section {ref}`SecMarkClrMktClr_G` of the {ref}`Chap_MarkClr` chapter:

  ```{math}
  :label: EqMarkClr_zetaD2
    D^{f}_{t+1} = D^{f}_{t} + \zeta_{D, t}(D_{t+1} - D_{t}) \quad\forall t
  ```

This leaves domestic investors to buy up the residual amount of government debt:

  ```{math}
  :label: EqMarkClr_zetaD2
    D^{d}_{t} = D_{t} - D^{f}_{t} \quad\forall t
  ```

We assume that debt dominates the capital markets, such that domestic investor demand for equity, $V^{d}_{t}$ is given as:

  ```{math}
  :label: eq_domestic_cap_demand
    V^{d}_{t} = B_{t} - D^{d}_{t} \quad\forall t
  ```

Foreign demand for equity is given in {ref}`SecMarkClrMktClr_K`, where $V^{f}_{t}$ is an exogenous fraction of excess equity demand:

  ```{math}
  :label: eq_foreign_cap_demand
    V^{f}_t = \zeta_{K, t}ED^{V,r^*}_t \quad\forall t
  ```

The total amount invested in the financial intermediary is thus:

```{math}
    B_t & = D^d_t + V^d_t \\
```

The return on the portfolio of assets held in the financial intermediary is the weighted average of the return on equity and government debt. As derived in {eq}`EqFirms_rKt` of Section {ref}`EqFirmsPosProfits`, the presence of public infrastructure in the production function means that the returns to private factors of production ($r_t$ and $w_t$) exhibit decreasing returns to scale.[^MoorePecoraro]. These excess profits are returned to shareholders through the financial intermediary.

The return on the portfolio of assets held by the financial intermediary is thus a weighted average of the return to government debt $r_{gov,t}$ from {eq}`EqUnbalGBC_rate_wedge` and the adjusted return on equity through the distribution of profits and capital gains:

```{math}
:label: eq_portfolio_return
  r_{p,t} = \frac{r_{gov,t}D_{t} + \Pi^d_{m,t}  + (V^d_{t+1} - V^d_{t}) }{D_{t} + V^d_{t}} \quad\forall t \quad\text{where}\quad \Pi^d_t \equiv \sum_{m=1}^M \pi_{m,t} * \frac{V^d_{t}}{V_{t}}
```

(SecFinfootnotes)=
## Footnotes

  [^MoorePecoraro]: See also {cite}`MoorePecoraro:2021` for a similar treatment of government infrastructure investment, positive profits, and returns to the owners of capital.
