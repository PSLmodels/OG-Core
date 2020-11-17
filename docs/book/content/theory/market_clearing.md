(Chap_MarkClr)=
# Market Clearing

Three markets must clear in `OG-USA`---the labor market, the capital market, and the goods market. By Walras' Law, we only need to use two of those market clearing conditions because the third one is redundant. In the model, we choose to use the labor market clearing condition and the capital market clearing condition, and to ignore the goods market clearing condition. But we present all three market clearing conditions here. Further, the redundant goods market clearing condition---sometimes referred to as the resource constraint---makes for a nice check on the solution method to see if everything worked.

We also characterize here the law of motion for total bequests $BQ_t$. Although it is not technically a market clearing condition, one could think of the bequests law of motion as the bequests market clearing condition.

(SecMarkClrMktClr)=
## Market Clearing Conditions

  The sections below detail the labor, capital, and gooods market clearing conditions of the model in the baseline case of a large open economy.


  (SecMarkClrMktClr_L)=
  ### Labor market clearing

  Labor market clearing {eq}`EqMarkClrLab` requires that aggregate labor demand $L_t$ measured in efficiency units equal the sum of household efficiency labor supplied $e_{j,s}n_{j,s,t}$.

  ```{math}
  :label: EqMarkClrLab
    L_t = \sum_{s=E+1}^{E+S}\sum_{j=1}^{J} \omega_{s,t}\lambda_j e_{j,s}n_{j,s,t} \quad \forall t
  ```


  (SecMarkClrMktClr_K)=
  ### Capital market clearing

  The capital market clearing assumptions of the model are complicated because we are assuming in our baseline specification that the economy is open and that domestic households and foreign investors can invest savings in private capital $K_t$ used by domestic firms and in domestic government debt $D_t$. See chapter {ref}`Chap_SmOpEcn` for a full discussion of the various open and closed economy specifications of the model.

  Define the following variables. Let $B_t$ represent total domestic household savings by all households of lifetime income group $j$ and age $s$ in period $t$.

  ```{math}
  :label: EqMarkClr_Bt
    B_t \equiv \sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\omega_{s-1,t-1}\lambda_j b_{j,s,t} + i_s\omega_{s,t-1}\lambda_j b_{j,s,t}\Bigr) \quad \forall t
  ```

  We assume that total domestic household savings $B_t$ is divided up into total domestic capital investment $K^d_t$ and total domestic government bond holdings $D^d_t$. We make the simplifying assumption that households are indifferent between the savings allocation to private capital $K^d_t$ or government bonds $D^d_t$.[^indif_KD_note]

  ```{math}
  :label: EqMarkClr_DomCapCnstr
    K^d_t + D^d_t = B_t \equiv \sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\omega_{s-1,t-1}\lambda_j b_{j,s,t} + i_s\omega_{s,t-1}\lambda_j b_{j,s,t}\Bigr) \quad \forall t
  ```

  Let $K_t$ be the total capital demand by domestic firms, and let $K^f_t$ be the amount of that capital demand supplied by foreigners. We assume that foreign capital supply $K^f_t$ is an exogenous percentage $\zeta_K\in[0,1]$ of total domestic capital demand. This percentage $\zeta_K$ is something we calibrate.

  ```{math}
  :label: EqMarkClr_zetaK
    K^{f}_t = \zeta_{K}K_t \quad\forall t
  ```

  The private capital market clearing condition is the following.

  ```{math}
  :label: EqMarkClr_KtKdKf
    K_t = K^d_t + K^f_t \quad\forall t
  ```

  Let $D_{t+1} - D_t$ be the total new issuance government debt, and let $D^f_{t+1} - D^f_t$ be the amount of those new issuances held by foreigners. We assume that foreign holdings of new government issuances of debt $D^f_{t+1}-D^f_t$ are an exogenous percentage $\zeta_D\in[0,1]$ of total new government debt issuances. This percentage $\zeta_D$ is something we calibrate.

  ```{math}
  :label: EqMarkClr_zetaD
    D^{f}_{t+1} = D^{f}_{t} + \zeta_{D}(D_{t+1} - D_{t}) \quad\forall t
  ```

  The government debt market clearing condition is the following.

  ```{math}
  :label: EqMarkClr_DtDdDf
    D_t = D^d_t + D^f_t \quad\forall t
  ```

  Capital market clearing {eq}`EqMarkClrCap` requires that aggregate capital demand from firms $K_t$ and from the government $D_t$ equal the sum of capital savings and investment by households $b_{j,s,t}$.

  ```{math}
  :label: EqMarkClrCap
    K_t + D_t = \sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\omega_{s-1,t-1}\lambda_j b_{j,s,t} + i_s\omega_{s,t-1}\lambda_j b_{j,s,t}\Bigr) \quad \forall t
  ```

  Note that the capital demand side of the capital market clearing equation {eq}`EqMarkClrCap` includes both capital demand by firms $K_t$ and capital demand by government $D_t$. It is here that we can see the potential of government deficits to crowd out investment.

  Aggregate consumption $C_t$ is defined as the sum of all household consumptions, and aggregate investment is defined by the resource constraint $Y_t = C_t + I_t + G_t$ as shown in {eq}`EqMarkClrGoods`.

  ```{math}
  :label: EqMarkClrGoods
    Y_t &= C_t + K_{t+1} - \biggl(\sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}i_s\omega_{s,t}\lambda_j b_{j,s,t+1}\biggr) - (1-\delta)K_t + G_t \quad\forall t \\
    &\quad\text{where}\quad C_t \equiv \sum_{s=E+1}^{E+S}\sum_{j=1}^{J}\omega_{s,t}\lambda_j c_{j,s,t}
  ```

  Note that the extra terms with the immigration rate $i_s$ in the capital market clearing equation {eq}`EqMarkClrCap` and the resource constraint {eq}`EqMarkClrGoods` accounts for the assumption that age-$s$ immigrants in period $t$ bring with them (or take with them in the case of out-migration) the same amount of capital as their domestic counterparts of the same age. Note also that the term in parentheses with immigration rates $i_s$ in the sum acts is equivalent to a net exports term in the standard equation $Y=C+I+G+NX$. That is, if immigration rates are positive, then immigrants are bringing capital into the country and the term in parentheses has a negative sign in front of it. Negative exports are imports.

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
