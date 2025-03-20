(Chap_MarkClr)=
# Market Clearing

  $M+3$ markets must clear in `OG-Core`---the labor market, the private capital market, the government bond market, and $M$ goods markets. By Walras' Law, we only need to use $M+2$ of those market clearing conditions because the remaining one is redundant. In the model, we choose to use the labor, private capital, government bond market, and the first $M-1$ goods market clearing conditions and to ignore the $M$th goods market clearing condition. But we present all $M+3$ market clearing conditions here. Further, the redundant $M$th goods market clearing condition---sometimes referred to as the resource constraint---makes for a nice check on the solution method to see if everything worked.

  We also characterize here the law of motion for total bequests $BQ_t$. Although it is not technically a market clearing condition, one could think of the bequests law of motion as the bequests market clearing condition.


(SecMarkClrMktClr)=
## Market Clearing Conditions

  The sections below detail the labor, government debt, private capital, and $M$ goods market clearing conditions of the model in the baseline case of a large partially open economy.


  (SecMarkClrMktClr_L)=
  ### Labor market clearing

  Labor market clearing {eq}`EqMarkClrLab` requires that aggregate labor demand $\sum_{m=1}^M L_{m,t}$ measured in efficiency units equal the sum of household efficiency labor supplied $e_{j,s}n_{j,s,t}$.

  ```{math}
  :label: EqMarkClrLab
    \sum_{m=1}^M L_{m,t} = \sum_{s=E+1}^{E+S}\sum_{j=1}^{J} \omega_{s,t}\lambda_j e_{j,s}n_{j,s,t} \quad \forall t
  ```


  (SecMarkClrMktClr_CapGen)=
  ### Capital markets generalities

  Before describing the government bond market and private capital market clearing conditions, respectively, we define some general capital market conditions relative to both markets. Both the government bond market and private capital market are characterized by interest rates that differ exogenously. As described in {eq}`EqUnbalGBC_rate_wedge`, the interest rate at which the government repays debt or earns on surpluses $r_{gov,t}$ differs from the marginal product of capital $r_t$ by an exogenous wedge. And the marginal product of capital is determined in equilibrium $r_t$ (except in the closed economy case). But we make a simplifying assumption that households are indifferent regarding the allocation of their savings between holding government debt or investing in private capital.[^indif_KD_note] And we assume that this indifference exists in spite of the difference in relative returns between $r_{gov,t}$ and $r_{K,t}$. We define total domestic household savings in a given period as $B_t$.

  ```{math}
  :label: EqMarkClr_Bt
    B_t \equiv \sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\omega_{s-1,t-1}\lambda_j b_{j,s,t} + i_s\omega_{s,t-1}\lambda_j b_{j,s,t}\Bigr) \quad \forall t
  ```

  And total domestic household savings is constrained to be allocated between domestic private capital ownership $K^d_t$ and domestic holdings of government bonds $D^d_t$.

  ```{math}
  :label: EqMarkClr_DomCapCnstr
    K^d_t + D^d_t = B_t \quad \forall t
  ```


  (SecMarkClrMktClr_G)=
  ### Government bond market clearing

  The government in `OG-Core` can run deficits or surpluses each period, as shown in equation {eq}`EqUnbalGBCbudgConstr` in Section {ref}`SecUnbalGBCbudgConstr`. Because the government can borrow or save on net each period $D_t$, someone must lend or borrow those assets on the supply side.

  We assume that foreigners hold a fixed percentage of new domestic government debt issuance. Let $D_{t+1} - D_t$ be the total new issuance government debt, and let $D^f_{t+1} - D^f_t$ be the amount of those new issuances held by foreigners. We assume that foreign holdings of new government issuances of debt $D^f_{t+1}-D^f_t$ are an exogenous percentage $\zeta_D\in[0,1]$ of total new government debt issuances. This percentage $\zeta_D$ is something we calibrate.

  ```{math}
  :label: EqMarkClr_zetaD
    D^{f}_{t+1} = D^{f}_{t} + \zeta_{D}(D_{t+1} - D_{t}) \quad\forall t
  ```

  The government debt market clearing condition is the following, where total domestic government debt $D_t$ equals the amount of that debt held by domestic households $D^d_t$ plus the amount of that debt held by foreign households $D^f_t$.

  ```{math}
  :label: EqMarkClr_DtDdDf
    D_t = D^d_t + D^f_t \quad\forall t
  ```

  We discuss the meaning of different permutations of the $\zeta_D$ parameter corner solutions $\zeta_D=0$ and $\zeta_D=1$ in Chapter {ref}`Chap_SmOpEcn`. If $\zeta_D=0$, we are assuming a closed economy government bond market in which all new government debt is held by domestic households and in which all government debt in the long-run (steady-state) is held by domestic households $\bar{D}^f=0$.


  (SecMarkClrMktClr_K)=
  ### Equity market clearing

  Equity in domestic firms, $V_t\equiv\sum_{m=1}^M K_{m,t}$, is held by domestic households, $V^d_t$ and foreign investors $V^f_t$. The equity market clearing condition is thus:

  ```{math}
  :label: EqMarkClr_KtKdKf
<<<<<<< HEAD
   V_t = V^d_t + V^f_t \quad\forall t \quad\text{where}\quad V_t \equiv  \sum_{m=1}^M V_{m,t}
=======
  K_t = K^d_t + K^f_t \quad\forall t \quad\text{where}\quad K_t \equiv  \sum_{m=1}^M K_{m,t}
>>>>>>> upstream/master
  ```

  Assume that there exists some exogenous world interest rate $r^*_t$. We assume that foreign equity holdings, $V^f_t$, is an exogenous percentage $\zeta_K\in[0,1]$ of the excess total supply of domestic equity, $ES^{K,r^*}_t$ that would exist if domestic firm values were determined by investment decisions made under the exogenous world interest rate $r^*_t$ and domestic equity demand were determined by the model consistent return on household savings $r_{p,t}$. This percentage $\zeta_K$ is something we calibrate. Define excess supply of equity at the exogenous world interest rate $r^*_t$ as $ES^{V,r^*}_t$, where $V^{r^*}_t\equiv\sum_{m=1}^M V^{r^*}_{m,t}$ is value of domestic firms at the world interest rate $r^*_t$, and $V^{d}_t$ is the domestic demand for equity, which is modeled as being a function of the actual rate of return faced by households $r_{p,t}$. Then our measure of excess supply at the world interest rate is the following.

  ```{math}
  :label: EqMarkClr_ExDemK
    ES^{V,r^*}_t \equiv V^{r^*}_t - V^d_t \quad\forall t \quad\text{where}\quad V^{r^*}_t\equiv \sum_{m=1}^M V^{r^*}_{m,t}
  ```

  Then we assume that total foreign private capital supply $V^f_t$ is a fixed fraction of this equity supply at the world interest rate $r^*$.

  ```{math}
  :label: EqMarkClr_zetaK
    V^{f}_t = \zeta_{K}ES^{K,r^*}_t \quad\forall t
  ```

  This approach nicely nests the small open economy specification discussed in Section {ref}`SecSmallOpen` of Chapter {ref}`Chap_SmOpEcn` in which $\zeta_K=1$, foreigners flexibly demand the excess supply of domestic equity, the domestic interest rate is fixed at the exogenous world interest rate $r^*$, and domestic households face the least amount of crowd out by government debt. The opposite extreme is the closed private capital market assumption of $\zeta_K=0$ in which $V^f_t=0$ and households must hold all the domestic firms' equity. In this specification, the interest rate is the most flexible and adjusts to equilibrate domestic equity demand, $V^d_t$, with the supply of equity, $V_t$.

  For the intermediate specifications of $\zeta_K\in(0,1)$, foreigners provide a fraction of the excess supply of equity in {eq}`EqMarkClr_ExDemK`. This allows for partial inflows of foreign equity holdings, partial crowd-out of government spending on private investment, and partial adjustment of the domestic interest rate $r_t$. This latter set of model specifications could be characterized as large-open economy or partial capital mobility.


  (SecMarkClrMktClr_goods)=
  ### Goods market clearing

  All $M$ industry goods markets must clear. Total demand of production good $m$ for consumption can be written as a function of total household demand for each consumption good $i$

  ```{math}
  :label: EqMarkConsDemand
    C_{m,t} = \sum_{i=1}^{I} \pi_{i,m} C_{i,t} \quad\forall t \quad\text{and}\quad m=1,2,...M
  ```
  where
  ```{math}
  :label: EqCmt
    C_{i,t} \equiv \sum_{s=E+1}^{E+S}\sum_{j=1}^{J}\omega_{s,t}\lambda_j c_{i,j,s,t} \quad\forall i,t
  ```

  Because we make a simplifying assumption that only the $M$th industry output can be used as investment, government spending, or government debt, consumption demand equals total output of good $m$ in the first $M-1$ industries.
  ```{math}
  :label: EqMarkClrGoods_Mm1
    Y_{m,t} = C_{m,t} \quad\forall t \quad\text{and}\quad m=1,2,...M-1
  ```

  The output of the $M$th industry can be used for private investment, infrastructure investment, government spending, and government debt.[^M_ind] As such, the market clearing condition in the $M$th industry will look more like the traditional $Y=C+I+G+NX$ expression.[^RCrates_note] Note also that adjustment costs are paid in units of capital, which is the same units as the output of the $M$th industry. Therefore we must include the adjustment costs in the market clearing condition for the $M$th industry.

  ```{math}
  :label: EqMarkClrGoods_M
<<<<<<< HEAD
    Y_{M,t} = C_{M,t} + I_{M,t} + I_{g,t} + G_t + r_{g,t}D^f_t - \Pi^f_{t} + (V^f_{t+1} - V^f_t) - \bigl(D^f_{t+1} - D^f_t\bigr)  + \Psi_{M,t} \quad\forall t
=======
    Y_{M,t} = C_{M,t} + I_{M,t} + I_{g,t} + G_t + r_{p,t} K^f_t + r_{p,t}D^f_t - (K^f_{t+1} - K^f_t) - \bigl(D^f_{t+1} - D^f_t\bigr) - RM_t \quad\forall t
>>>>>>> upstream/master
  ```
  where
  ```{math}
  :label: EqMarkClrGoods_IMt
    I_{M,t} &\equiv \sum_{m=1}^M K_{m,t+1} - (1 - \delta_{M,t})\sum_{m=1}^M K_{m,t} \quad\forall t \\
    &= K_{t+1} - (1 - \delta_{M,t})K_t \\
    &= (K^d_{t+1} + K^f_{t+1}) - (1 - \delta_{M,t})(K^d_t + K^f_t)
  ```
and
  ```{math}
  :label: EqMarkClrGoods_IMt
    \Psi_{M,t} &\equiv \sum_{m=1}^M \Psi(I_{m,t},K_{m,t}) \quad\forall t \\
  ```
  and
  ```{math}
  :label: EqMarkClrGoods_Pi
    \Pi^f_{t} &\equiv \sum_{m=1}^M \pi_{m,t} * \frac{V^f}{V} \quad\forall t \\
  ```

  In the partially open economy, we must add to the right-hand-side of {eq}`EqMarkClrGoods_M` the output paid to the foreign owners of capital $r_{p,t} K^f_t$ and to the foreign holders of government debt $r_{p,t}D^f_t$. And we must subtract off the foreign inflow component $K^f_{t+1} - K^f_t$ from private capital investment as shown in the first term in parentheses on the right-hand-side of {eq}`EqMarkClrGoods_M`. You can see in the definition of private investment {eq}`EqMarkClrGoods_IMt` where this amount of foreign capital is part of $I_{M,t}$.

  Similarly, we must subtract off the foreign purchases of new government debt $D^f_{t+1} - D^f_t$ and aggregate remittances $RM_t$ as shown in the second term in parentheses and the last term, respectively, on the right-hand-side of {eq}`EqMarkClrGoods_M`. The new foreign purchases of government debt are part of $I_{g,t}$ and $G_t$, as they are functions of GDP $Y_t$, as shown in {eq}`EqUnbalGBC_Gt`, {eq}`EqUnbalGBC_Igt`, and the government budget constraint {eq}`EqUnbalGBCbudgConstr`. Foreign lending relaxes the resource constraint. And aggregate remittances are part of aggregate consumption $C_t$ and also relax the resource constraint.

  Net exports (imports) of capital in the form of foreign private capital inflows $K^f_t$, foreign holdings of government debt $D^f_t$, and remittances $RM_t$ are clearly accounted for in {eq}`EqMarkClrGoods_M`. Foreign remittances received by domestic households are described in Section {ref}`SecHHremit` of Chapter {ref}`Chap_House`.


(SecMarkClrBQ)=
## Total Bequests Law of Motion

  Total bequests $BQ_t$ are the collection of savings of household from the previous period who died at the end of the period. These savings are augmented by the interest rate because they are returned after being invested in the production process.

  ```{math}
  :label: EqMarkClrBQ
    BQ_{t} = (1+r_{p,t})\left(\sum_{s=E+2}^{E+S+1}\sum_{j=1}^J\rho_{s-1}\lambda_j\omega_{s-1,t-1}b_{j,s,t}\right) \quad\forall t
  ```

  Because the form of the period utility function in {eq}`EqHHPerUtil` ensures that $b_{j,s,t}>0$ for all $j$, $s$, and $t$, total bequests will always be positive $BQ_{j,t}>0$ for all $j$ and $t$.


(SecMarkClr_footnotes)=
## Footnotes

  This section contains the footnotes for this chapter.

  [^indif_KD_note]: By assuming that households are indifferent between the savings allocation to private capital $K^d_t$ and government bonds $D^d_t$, we avoid the need for another state variable in the solution method. In our approach the allocation between the two types of capital is simply a residual of the exogenous proportion $\zeta_K$ of total private captial $K_t$ allocated to foreigners implied by equations {eq}`EqMarkClr_zetaK` and {eq}`EqMarkClr_KtKdKf` and a residual of the exogenous proportion $\zeta_D$ of total government bonds $D_t$ allocated to foreigners implied by equations {eq}`EqMarkClr_zetaD` and {eq}`EqMarkClr_DtDdDf`.

  [^M_ind]: Our assumption that only the $M$th industry output can be used as investment, government spending, and government debt is a strong one. However, it greatly simplifies our equilibrium solution method in the transition path. Intuitively, think of an economy that has two industries---delivery services and trucks. The delivery services industry uses trucks and labor to produce its output. The trucks industry uses trucks and labor to produce its output. Both industries face depreciation of their capital (trucks). But only in the trucks industry can the output be used for both consumption and investment.

  [^RCrates_note]: Because we treat household return $r_{p,t}$ as an average between the return on private capital $r_{K,t}$ and the return on government bonds $r_{gov,t}$ in {eq}`eq_portfolio_return`, and because this return is actually given to households in the budget constraint {eq}`EqHHBC`, it is required for market clearing that the return paid to foreign suppliers of private capital $K^f_t$ and foreign holders of government bonds $D^f_t$ be paid that same average return $r_{p,t}$.
