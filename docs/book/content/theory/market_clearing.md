(Chap_MarkClr)=
# Market Clearing

Four markets must clear in `OG-USA`---the labor market, the private capital market, the government bonds market, and the goods market. By Walras' Law, we only need to use three of those market clearing conditions because the fourth one is redundant. In the model, we choose to use the labor, private capital, and government bonds market clearing conditions and to ignore the goods market clearing condition. But we present all four market clearing conditions here. Further, the redundant goods market clearing condition---sometimes referred to as the resource constraint---makes for a nice check on the solution method to see if everything worked.

We also characterize here the law of motion for total bequests $BQ_t$. Although it is not technically a market clearing condition, one could think of the bequests law of motion as the bequests market clearing condition.

(SecMarkClrMktClr)=
## Market Clearing Conditions

  The sections below detail the labor, government debt, private capital, and gooods market clearing conditions of the model in the baseline case of a large partially open economy.


  (SecMarkClrMktClr_L)=
  ### Labor market clearing

  Labor market clearing {eq}`EqMarkClrLab` requires that aggregate labor demand $L_t$ measured in efficiency units equal the sum of household efficiency labor supplied $e_{j,s}n_{j,s,t}$.

  ```{math}
  :label: EqMarkClrLab
    L_t = \sum_{s=E+1}^{E+S}\sum_{j=1}^{J} \omega_{s,t}\lambda_j e_{j,s}n_{j,s,t} \quad \forall t
  ```

  (SecMarkClrMktClr_CapGen)=
  ### Capital markets generalities

  Before describing the government bond market and private capital market clearing conditions, respectively, we define some general capital market conditions relative to both markets. Both the government bond market and private capital market are characterized by interest rates that differ exogenously. As described in {eq}`EqUnbalGBC_rate_wedge`, the interest rate at which the government repays debt or earns on surplusses $r_{gov,t}$ differs from the marginal product of capital $r_t$ by an exogenous wedge. And the marginal product of capital is determined in equilibrium $r_t$. But we make a simplifying assumption that households are indifferent regarding the allocation of their savings between holding government debt or investing in private capital.[^indif_KD_note] And we assume that this indifference exists in spite of the difference in relative returns between $r_{gov,t}$ and $r_t$. We define total domestic household savings in a given period as $B_t$.

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

  The government in `OG-USA` can run deficits or surplusses each period, as shown in equation {eq}`EqUnbalGBCbudgConstr` in Section {ref}`SecUnbalGBCbudgConstr`. Because the government can borrow or save on net each period $D_t$, someone must lend or borrow those assets on the supply side.

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
  ### Private capital market clearing

  Domestic firms rent private capital $K_t$ from domestic households $K^d_t$ and from foreign investors $K^f_t$.

  ```{math}
  :label: EqMarkClr_KtKdKf
    K_t = K^d_t + K^f_t \quad\forall t
  ```

  Assume that there exists some exogenous world interest rate $r^*_t$. We assume that foreign capital supply $K^f_t$ is an exogenous percentage $\zeta_K\in[0,1]$ of the excess total domestic private capital demand $ED^{K,r^*}_t$ that would exist if domestic private capital demand were determined by the exogenous world interest rate $r^*_t$ and domestic private capital supply were determined by the model consistent return on household savings $r_{hh,t}$. This percentage $\zeta_K$ is something we calibrate. Define excess total domestic capital demand at the exogenous world interest rate $r^*_t$ as $ED^{K,r^*}_t$. Define $K^{r^*}_t$ as the capital demand by domestic firms at the world interest rate $r^*_t$, and define $K^{d}_t$ as the domestic supply of private capital to firms, which is modeled as being a function of the actual rate faced by households $r_{hh,t}$. Then our measure of excess demand at the world interest rate is the following.

  ```{math}
  :label: EqMarkClr_ExDemK
    ED^{K,r^*}_t \equiv K^{r^*}_t - K^d_t \quad\forall t
  ```

  Then we assume that total foreign private capital supply $K^f_t$ is a fixed fraction of this excess capital demand at the world interest rate $r^*$.

  ```{math}
  :label: EqMarkClr_zetaK
    K^{f}_t = \zeta_{K}ED^{K,r^*}_t \quad\forall t
  ```

  This approach nicely nests the small open economy specification discussed in Section {ref}`SecSmallOpen` of Chapter {ref}`Chap_SmOpEcn` in which $\zeta_K=1$, foreigners flexibly supply all the excess demand for private capital, the marginal product of capital is fixed at the exogenous world interest rate $r^*$, and domestic households face the least amount of crowd out by government debt. The opposite extreme is the closed private capital market assumption of $zeta_K=0$ in which $K^f_t=0$ and households must supply all the capital demanded in the domestic market. In this specification, the interest rate is the most flexible and adjusts to equilibrate domestic private capital supply $K^d_t$ with private capital demand $K_t$.

  For the interemediate specifications of $\zeta_K\in(0,1)$, foreigners provide a fraction of the excess demand defined in {eq}`EqMarkClr_ExDemK`. This allows for partial inflows of foreign private capital, partial crowd-out of government spending on private investment, and partial adjustment of the domestic interest rate $r_t$. This latter set of model specifications could be characterized as large-open economy or partial capital mobility.


  (SecMarkClrMktClr_goods)=
  ### Goods market clearing

  The fourth and final market that must clear is the goods market. It is redundant by Walras' Law and is not needed for computing the equilibrium solution. But it is an equation that must be satisfied and is a good check of the solution accuracy after the solution is obtained.

  In the partially open economy, some of the output is paid to the foreign owners of capital $r_t K^f_t$ and to foreign holders of government debt $r_{gov,t}D^f_t$. In addition, foreign lending to the home country’s government relaxes the resource constraint. The goods market clearing condition or resource constraint is given by the following.

  ```{math}
  :label: EqMarkClrGoods
    \begin{split}
      Y_t &= C_t + \bigl(K^d_{t+1} - K^d_t\bigr) + \delta K_t + G_t + r_t K^f_t - \bigl(D^f_{t+1} - D^f_t\bigr) + r_{gov,t}D^f_t \quad\forall t \\
      &\quad\text{where}\quad C_t \equiv \sum_{s=E+1}^{E+S}\sum_{j=1}^{J}\omega_{s,t}\lambda_j c_{j,s,t}
    \end{split}
  ```

  Net exports (imports) of capital in the form of foreign private capital inflows $K^f_t$ and foreign holdings of government debt $D^f_t$ are clearly accounted for in {eq}`EqMarkClrGoods`.


(SecMarkClrBQ)=
## Total Bequests Law of Motion

  Total bequests $BQ_t$ are the collection of savings of household from the previous period who died at the end of the period. These savings are augmented by the interest rate because they are returned after being invested in the production process.

  ```{math}
  :label: EqMarkClrBQ
    BQ_{t} = (1+r_{hh,t})\left(\sum_{s=E+2}^{E+S+1}\sum_{j=1}^J\rho_{s-1}\lambda_j\omega_{s-1,t-1}b_{j,s,t}\right) \quad\forall t
  ```

  Because the form of the period utility function in {eq}`EqHHPerUtil` ensures that $b_{j,s,t}>0$ for all $j$, $s$, and $t$, total bequests will always be positive $BQ_{j,t}>0$ for all $j$ and $t$.


(SecMarkClr_footnotes)=
## Footnotes

[^indif_KD_note]: By assuming that households are indifferent between the savings allocation to private capital $K^d_t$ and government bonds $D^d_t$, we avoid the need for another state variable in the solution method. In our approach the allocation between the two types of capital is simply a residual of the exogenous proportion $\zeta_K$ of total private captial $K_t$ allocated to foreigners implied by equations {eq}`EqMarkClr_zetaK` and {eq}`EqMarkClr_KtKdKf` and a residual of the exogenous proportion $\zeta_D$ of total government bonds $D_t$ allocated to foreigners implied by equations {eq}`EqMarkClr_zetaD` and {eq}`EqMarkClr_DtDdDf`
