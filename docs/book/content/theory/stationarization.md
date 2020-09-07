(Chap_Stnrz)=
# Stationarization

The previous chapters derive all the equations necessary to solve for the steady-state and nonsteady-state equilibria of this model. However, because labor productivity is growing at rate $g_y$ as can be seen in the firms' production function {eq}`EqFirmsCESprodfun` and the population is growing at rate $\tilde{g}_{n,t}$ as defined in {eq}`EqPopGrowthTil`, the model is not stationary. Different endogenous variables of the model are growing at different rates. We have already specified three potential budget closure rules {eq}`EqUnbalGBCclosure_Gt`, {eq}`EqUnbalGBCclosure_TRt`, and {eq}`EqUnbalGBCclosure_TRGt` using some combination of government spending $G_t$ and transfers $TR_t$ that stationarize the debt-to-GDP ratio.

Table {ref}`TabStnrzStatVars` lists the definitions of stationary versions of these endogenous variables. Variables with a ``$\:\,\hat{}\,\:$'' signify stationary variables. The first column of variables are growing at the productivity growth rate $g_y$. These variables are most closely associated with individual variables. The second column of variables are growing at the population growth rate $\tilde{g}_{n,t}$. These variables are most closely associated with population values. The third column of variables are growing at both the productivity growth rate $g_y$ and the population growth rate $\tilde{g}_{n,t}$. These variables are most closely associated with aggregate variables. The last column shows that the interest rate $r_t$ and household labor supply $n_{j,s,t}$ are already stationary.

<!-- \begin{table}[htbp] \centering \captionsetup{width=3.5in}
\caption{\label{TabStnrzStatVars}\textbf{Stationary variable definitions}}
  \begin{threeparttable}
  \begin{tabular}{>{\small}c >{\small}c >{\small}c |>{\small}c}
    \hline\hline
    \multicolumn{3}{c}{Sources of growth} & Not \\
    & & & \\[-4mm]
    $e^{g_y t}$ & $\tilde{N}_t$ & $e^{g_y t}\tilde{N}_t$ & growing\tnote{a} \\
    \hline
    & & \\[-4mm]
    $\hat{c}_{j,s,t}\equiv\frac{c_{j,s,t}}{e^{g_y t}}$ & $\hat{\omega}_{s,t}\equiv\frac{\omega_{s,t}}{\tilde{N}_t}$ & $\hat{Y}_t\equiv\frac{Y_t}{e^{g_y t}\tilde{N}_t}$ & $n_{j,s,t}$ \\[2mm]
    $\hat{b}_{j,s,t}\equiv\frac{b_{j,s,t}}{e^{g_y t}}$ & $\hat{L}_t\equiv\frac{L_t}{\tilde{N}_t}$ & $\hat{K}_t\equiv\frac{K_t}{e^{g_y t}\tilde{N}_t}$ & $r_t$ \\[2mm]
    $\hat{w}_t\equiv\frac{w_t}{e^{g_y t}}$ &  & $\hat{BQ}_{j,t}\equiv\frac{BQ_{j,t}}{e^{g_y t}\tilde{N}_t}$ &  \\[2mm]
    $\hat{y}_{j,s,t}\equiv\frac{y_{j,s,t}}{e^{g_y t}}$ &  & $\hat{C}_t\equiv \frac{C_t}{e^{g_y t}\tilde{N}_t}$  &  \\[2mm]
    $\hat{T}_{s,t}\equiv\frac{T_{j,s,t}}{e^{g_y t}}$ &  & $\hat{TR}_t\equiv\frac{TR_t}{e^{g_y t}\tilde{N}_t}$ &  \\[2mm]
    \hline\hline
  \end{tabular}
  \begin{tablenotes}
    \scriptsize{\item[a]The interest rate $r_t$ in {eq}`EqFirmFOC_K} is already stationary because $Y_t$ and $K_t$ grow at the same rate. Household labor supply $n_{j,s,t}\in[0,\tilde{l}]$ is stationary.}
  \end{tablenotes}
  \end{threeparttable}
\end{table} -->

<div id="TabStnrzStatVars">

|                                                          |                                                                     |                                                                     |                             |
|:--------------------------------------------------------:|:-------------------------------------------------------------------:|:-------------------------------------------------------------------:|:---------------------------:|
|                                                          |                                 Not                                 |                                                                     |                             |
|                                                          |                                                                     |                                                                     |                             |
|            *e*<sup>*g*<sub>*y*</sub>*t*</sup>            |                          *Ñ*<sub>*t*</sub>                          |         *e*<sup>*g*<sub>*y*</sub>*t*</sup>*Ñ*<sub>*t*</sub>         |           growing           |
|                                                          |                                                                     |                                                                     |                             |
| $\\hat{c}\_{j,s,t}\\equiv\\frac{c\_{j,s,t}}{e^{g\_y t}}$ | $\\hat{\\omega}\_{s,t}\\equiv\\frac{\\omega\_{s,t}}{\\tilde{N}\_t}$ |      $\\hat{Y}\_t\\equiv\\frac{Y\_t}{e^{g\_y t}\\tilde{N}\_t}$      | *n*<sub>*j*, *s*, *t*</sub> |
| $\\hat{b}\_{j,s,t}\\equiv\\frac{b\_{j,s,t}}{e^{g\_y t}}$ |           $\\hat{L}\_t\\equiv\\frac{L\_t}{\\tilde{N}\_t}$           |      $\\hat{K}\_t\\equiv\\frac{K\_t}{e^{g\_y t}\\tilde{N}\_t}$      |      *r*<sub>*t*</sub>      |
|       $\\hat{w}\_t\\equiv\\frac{w\_t}{e^{g\_y t}}$       |                                                                     | $\\hat{BQ}\_{j,t}\\equiv\\frac{BQ\_{j,t}}{e^{g\_y t}\\tilde{N}\_t}$ |                             |
| $\\hat{y}\_{j,s,t}\\equiv\\frac{y\_{j,s,t}}{e^{g\_y t}}$ |                                                                     |     $\\hat{C}\_t\\equiv \\frac{C\_t}{e^{g\_y t}\\tilde{N}\_t}$      |                             |
|  $\\hat{T}\_{s,t}\\equiv\\frac{T\_{j,s,t}}{e^{g\_y t}}$  |                                                                     |     $\\hat{TR}\_t\\equiv\\frac{TR\_t}{e^{g\_y t}\\tilde{N}\_t}$     |                             |

<span id="TabStnrzStatVars"
label="TabStnrzStatVars" ></span>

</div>

(TabStnrzStatVars)=
## Stationary variable definitions
The usual definition of equilibrium would be allocations and prices such that households optimize {eq}`EqHHeul_n`, {eq}`EqHHeul_b`, and {eq}`EqHHeul_bS`, firms optimize {eq}`EqFirmFOC_L` and {eq}`EqFirmFOC_K`, and markets clear {eq}`EqMarkClrLab` and {eq}`EqMarkClrCap`, and {eq}`EqMarkClrBQ`. In this chapter, we show how to stationarize each of these characterizing equations so that we can use our fixed point methods described in Sections {ref}`SecEqlbSSsoln` and {ref}`SecEqlbNSSsoln` to solve for the equilibria in Definitions {ref}`DefSSEql` and {ref}`DefNSSEql`.

(SecStnrzHH)=
## Stationarized Household Equations

  The stationary version of the household budget constraint {eq}`EqHHBC` is found by dividing both sides of the equation by $e^{g_y t}$. For the savings term $b_{j,s+1,t+1}$, we must multiply and divide by $e^{g_y(t+1)}$, which leaves an $e^{g_y} = \frac{e^{g_y(t+1)}}{e^{g_y t}}$ in front of the stationarized variable.
  
  ```{math}
  :label: EqStnrzHHBCstat
      \hat{c}_{j,s,t} + e^{g_y}\hat{b}_{j,s+1,t+1} &= (1 + r_{t})\hat{b}_{j,s,t} + \hat{w}_t e_{j,s} n_{j,s,t} + \zeta_{j,s}\frac{\hat{BQ}_t}{\lambda_j\hat{\omega}_{s,t}} + \eta_{j,s,t}\frac{\hat{TR}_{t}}{\lambda_j\hat{\omega}_{s,t}} - \hat{T}_{s,t}  \\
      \quad\forall j,t\quad\text{and}\quad s\geq E+1 \quad\text{where}\quad b_{j,E+1,t}=0\quad\forall j,t
  ```

  Because total bequests $BQ_t$ and total government transfers $TR_t$ grow at both the labor productivity growth rate and the population growth rate, we have to multiply and divide each of those terms by the economically relevant population $\tilde{N}_t$. This stationarizes total bequests $\hat{BQ}_t$, total transfers $\hat{TR}_t$, and the respective population level in the denominator $\hat{\omega}_{s,t}$.

  We stationarize the Euler equations for labor supply {eq}`EqHHeul_n` by dividing both sides by $e^{g_y(1-\sigma)}$. On the left-hand-side, $e^{g_y}$ stationarizes the wage $\hat{w}_t$ and $e^{-\sigma g_y}$ goes inside the parentheses and stationarizes consumption $\hat{c}_{j,s,t}$. On the right-and-side, the $e^{g_y(1-\sigma)}$ terms cancel out.
  
  ```{math}
  :label: EqStnrzHHeul_n
      \hat{w}_t e_{j,s}\bigl(1 - \tau^{mtrx}_{s,t}\bigr)(\hat{c}_{j,s,t})^{-\sigma} = \chi^n_{s}\biggl(\frac{b}{\tilde{l}}\biggr)\biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^{\upsilon-1}\Biggl[1 - \biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^\upsilon\Biggr]^{\frac{1-\upsilon}{\upsilon}} \\
      \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\forall j,t, \quad\text{and}\quad E+1\leq s\leq E+S \\
  ```

  We stationarize the Euler equations for savings {eq}`EqHHeul_b` and {eq}`EqHHeul_bS` by dividing both sides of the respective equations by $e^{-\sigma g_y t}$. On the right-hand-side of the equation, we then need to multiply and divide both terms by $e^{-\sigma g_y(t+1)}$, which leaves a multiplicative coefficient $e^{-\sigma g_y}$.

  ```{math}
  :label: EqStnrzHHeul_b
      (\hat{c}_{j,s,t})^{-\sigma} = e^{-\sigma g_y}\biggl[\chi^b_j\rho_s(\hat{b}_{j,s+1,t+1})^{-\sigma} + \beta\bigl(1 - \rho_s\bigr)\Bigl(1 + r_{t+1}\bigl[1 - \tau^{mtry}_{s+1,t+1}\bigr]\Bigr)(\hat{c}_{j,s+1,t+1})^{-\sigma}\biggr] \\
      \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\forall j,t, \quad\text{and}\quad E+1\leq s\leq E+S-1 \\
  ```

  ```{math}
  :label: EqStnrzHHeul_bS
    (\hat{c}_{j,E+S,t})^{-\sigma} = e^{-\sigma g_y}\chi^b_j(\hat{b}_{j,E+S+1,t+1})^{-\sigma} \quad\forall j,t \quad\text{and}\quad s = E+S
  ```

(SecStnrzFirms)=
## Stationarized Firms Equations

  The nonstationary production function {eq}`EqFirmsCESprodfun` can be stationarized by dividing both sides by $e^{g_y t}\tilde{N}$. This stationarizes output $\hat{Y}_t$ on the left-hand-side. Because the general CES production function is homogeneous of degree 1, $F(xK,xL) = xF(K,L)$, which means the right-hand-side of the production function is stationarized by dividing by $e^{g_y t}\tilde{N}_t$.
  
  ```{math}
  :label: EqStnrzCESprodfun
    \hat{Y}_t = F(\hat{K}_t, \hat{L}_t) \equiv Z_t\biggl[(\gamma)^\frac{1}{\varepsilon}(\hat{K}_t)^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma)^\frac{1}{\varepsilon}(\hat{L}_t)^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} \quad\forall t
  ```
  Notice that the growth term multiplied by the labor input drops out in this stationarized version of the production function. We stationarize the nonstationary profit function {eq}`EqFirmsProfit` in the same way, by dividing both sides by $e^{g_y t}\tilde{N}_t$.
  
  ```{math}
  :label: EqStnrzProfit
    \hat{PR}_t = (1 - \tau^{corp})\Bigl[F(\hat{K}_t,\hat{L}_t) - \hat{w}_t \hat{L}_t\Bigr] - \bigl(r_t + \delta\bigr)\hat{K}_t + \tau^{corp}\delta^\tau \hat{K}_t \quad\forall t
  ```

  The firms' first order equation for labor demand {eq}`EqFirmFOC_L` is stationarized by dividing both sides by $e^{g_y t}$. This stationarizes the wage $\hat{w}_t$ on the left-hand-side and cancels out the $e^{g_y t}$ term in front of the right-hand-side. To complete the stationarization, we multiply and divide the $\frac{Y_t}{e^{g_y t}L_t}$ term on the right-hand-side by $\tilde{N}_t$.
  
  ```{math}
  :label: EqStnrzFOC_L
    \hat{w}_t = (Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[(1-\gamma)\frac{\hat{Y}_t}{\hat{L}_t}\right]^\frac{1}{\varepsilon} \quad\forall t
  ```

  It can be seen from the firms' first order equation for capital demand {eq}`EqFirmFOC_K` that the interest rate is already stationary. If we multiply and divide the $\frac{Y_t}{K_t}$ term on the right-hand-side by $e^{t_y t}\tilde{N}_t$, those two aggregate variables become stationary. In other words, $Y_t$ and $K_t$ grow at the same rate and $\frac{Y_t}{K_t} = \frac{\hat{Y}_t}{\hat{K}_t}$.
  
  ```{math}
  :label: EqFirmFOC_K
      r_t &= (1 - \tau^{corp})(Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{\hat{Y}_t}{\hat{K}_t}\right]^\frac{1}{\varepsilon} - \delta + \tau^{corp}\delta^\tau \quad\forall t \\
      &= (1 - \tau^{corp})(Z_t)^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{Y_t}{K_t}\right]^\frac{1}{\varepsilon} - \delta + \tau^{corp}\delta^\tau \quad\forall t
 ```

(SecStnrzGovt)=
## Stationarized Government Equations

  Each of the tax rate functions $\tau^{etr}_{s,t}$, $\tau^{mtrx}_{s,t}$, and $\tau^{mtry}_{s,t}$ is stationary. The total tax liability function $T_{s,t}$ is growing at the rate of labor productivity growth $g_y$ This can be see by looking at the decomposition of the total tax liability function into the effective tax rate times total income {eq}`EqTaxCalcLiabETR`. The effective tax rate function is stationary, and household income is growing at rate $g_y$. So household total tax liability is stationarized by dividing both sides of the equation by $e^{g_y t}$.
  
  ```{math}
  :label: EqStnrzLiabETR
      \hat{T}_{s,t} &= \tau^{etr}_{s,t}(\hat{x}_{j,s,t}, \hat{y}_{j,s,t})\left(\hat{x}_{j,s,t} + \hat{y}_{j,s,t}\right) \qquad\qquad\qquad\quad\:\:\forall t \quad\text{and}\quad E+1\leq s\leq E+S \\
      &= \tau^{etr}_{s,t}(\hat{w}_t e_{j,s}n_{j,s,t}, r_t\hat{b}_{j,s,t})\left(\hat{w}_t e_{j,s}n_{j,s,t} + r_t\hat{b}_{j,s,t}\right) \quad\forall t \quad\text{and}\quad E+1\leq s\leq E+S
  ```

  We can stationarize the simple expressions for total government spending on public goods $G_t$ in {eq}`EqUnbalGBC_Gt` and on household transfers $TR_t$ in {eq}`EqUnbalGBCtfer` by dividing both sides by $e^{g_y t}\tilde{N}_t$,
  
  ```{math}
  :label: EqStnrz_Gt
    \hat{G}_t = g_{g,t}\:\alpha_{g}\:\hat{Y}_t \quad\forall t
  ```
  ```{math}
  :label: EqStnrzTfer
    \hat{TR}_t = g_{tr,t}\:\alpha_{tr}\:\hat{Y}_t \quad\forall t
  ```

  where the time varying multipliers $g_{g,t}$ and $g_{tr,t}$, respectively, are defined in {eq}`EqStnrzClosureRule_Gt` and {eq}`EqStnrzClosureRule_TRt` below. These multipliers $g_{g,t}$ and $g_{tr,t}$ do not have a ``$\:\,\hat{}\,\:$'' on them because their specifications {eq}`EqUnbalGBCclosure_Gt` and {eq}`EqUnbalGBCclosure_TRt` that are functions of nonstationary variables are equivalent to {eq}`EqStnrzClosureRule_Gt` and {eq}`EqStnrzClosureRule_TRt` specified in stationary variables.

  We can stationarize the expression for total government revenue $Rev_t$ in {eq}`EqUnbalGBCgovRev` by dividing both sides of the equation by $e^{g_y t}\tilde{N}_t$.
  ```{math}
  :label: EqStnrzGovRev
    \hat{Rev}_t = \underbrace{\tau^{corp}\bigl[\hat{Y}_t - \hat{w}_t\hat{L}_t\bigr] - \tau^{corp}\delta^\tau \hat{K}_t}_{\text{corporate tax revenue}} + \underbrace{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\lambda_j\hat{\omega}_{s,t}\tau^{etr}_{s,t}\left(\hat{x}_{j,s,t},\hat{y}_{j,s,t}\right)\bigl(\hat{x}_{j,s,t} + \hat{y}_{j,s,t}\bigr)}_{\text{household tax revenue}} \quad\forall t
  ```
  
  Every term in the government budget constraint {eq}`EqUnbalGBCbudgConstr` is growing at both the productivity growth rate and the population growth rate, so we stationarize it by dividing both sides by $e^{g_y t}\tilde{N}_t$. We also have to multiply and divide the next period debt term $D_{t+1}$ by $e^{g_y(t+1)}\tilde{N}_{t+1}$, leaving the term $e^{g_y}(1 + \tilde{g}_{n,t+1})$.
  
  ```{math}
  :label: EqStnrzGovBC
    e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\hat{D}_{t+1} + \hat{Rev}_t = (1 + r_t)\hat{D}_t + \hat{G}_t + \hat{TR}_t \quad\forall t
  ```

  The three potential budget closure rules {eq}`EqUnbalGBCclosure_Gt`, {eq}`EqUnbalGBCclosure_TRt`, and {eq}`EqUnbalGBCclosure_TRGt` are the last government equations to stationarize. In each of the cases, we simply divide both sides by $e^{g_y t}\tilde{N}_t$.
  
  ```{math}
  :label: EqStnrzClosureRule_Gt
        &\hat{G}_t = g_{g,t}\:\alpha_{g}\: \hat{Y}_t \\
        &\text{where}\quad g_{g,t} =
          \begin{cases}
            1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad\:\:\text{if}\quad t < T_{G1} \\
            \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\left[\rho_{d}\alpha_{D}\hat{Y}_{t} + (1-\rho_{d})\hat{D}_{t}\right] - (1+r_{t})\hat{D}_{t} - \hat{TR}_{t} + \hat{Rev}_{t}}{\alpha_g \hat{Y}_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
            \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\alpha_{D}\hat{Y}_{t} - (1+r_{t})\hat{D}_{t} - \hat{TR}_{t} + \hat{Rev}_{t}}{\alpha_g \hat{Y}_t} \qquad\qquad\qquad\text{if}\quad t \geq T_{G2}
          \end{cases} \\
        &\quad\text{and}\quad g_{tr,t} = 1 \quad\forall t
  ```
  or

  ```{math}
  :label: EqStnrzClosureRule_TRt
        &\hat{TR}_t = g_{tr,t}\:\alpha_{tr}\: \hat{Y}_t \\
        &\text{where}\quad g_{tr,t} =
          \begin{cases}
            1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad\text{if}\quad t < T_{G1} \\
            \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\left[\rho_{d}\alpha_{D}\hat{Y}_{t} + (1-\rho_{d})\hat{D}_{t}\right] - (1+r_{t})\hat{D}_{t} - \hat{G}_{t} + \hat{Rev}_{t}}{\alpha_{tr} \hat{Y}_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
            \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\alpha_{D}\hat{Y}_{t} - (1+r_{t})\hat{D}_{t} - \hat{G}_{t} + \hat{Rev}_{t}}{\alpha_{tr} \hat{Y}_t} \qquad\qquad\qquad\text{if}\quad t \geq T_{G2}
          \end{cases} \\
      &\quad\text{and}\quad g_{g,t} = 1 \quad\forall t
  ```
  or

  ```{math}
  :label: EqStnrzClosureRule_TRGt
        &\hat{G}_t + \hat{TR}_t = g_{trg,t}\left(\alpha_g + \alpha_{tr}\right)\hat{Y}_t \quad\Rightarrow\quad \hat{G}_t = g_{trg,t}\:\alpha_g\:\hat{Y}_t \quad\text{and}\quad \hat{TR}_t = g_{trg,t}\:\alpha_{tr}\:\hat{Y}_t \\
        &\text{where}\quad g_{trg,t} =
          \begin{cases}
            1 \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad\:\:\,\text{if}\quad t < T_{G1} \\
            \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\left[\rho_{d}\alpha_{D}\hat{Y}_{t} + (1-\rho_{d})\hat{D}_{t}\right] - (1+r_{t})\hat{D}_{t} + \hat{Rev}_{t}}{\left(\alpha_g + \alpha_{tr}\right)\hat{Y}_t} \quad\text{if}\quad T_{G1}\leq t<T_{G2} \\
            \frac{e^{g_y}\left(1 + \tilde{g}_{n,t+1}\right)\alpha_{D}\hat{Y}_{t} - (1+r_{t})\hat{D}_{t} + \hat{Rev}_{t}}{\left(\alpha_g + \alpha_{tr}\right)\hat{Y}_t} \qquad\qquad\quad\:\:\:\:\,\text{if}\quad t \geq T_{G2}
          \end{cases}
  ```

(SecStnrzMC)=
## Stationarized Market Clearing Equations

  The labor market clearing equation {eq}`EqMarkClrLab` is stationarized by dividing both sides by $\tilde{N}_t$.
  
  ```{math}
  :label: EqStnrzMarkClrLab
    \hat{L}_t = \sum_{s=E+1}^{E+S}\sum_{j=1}^{J} \hat{\omega}_{s,t}\lambda_j e_{j,s}n_{j,s,t} \quad \forall t
  ```
  
  The capital market clearing equation {eq}`EqMarkClrCap` is stationarized by dividing both sides by $e^{g_y t}\tilde{N}_t$. Because the right-hand-side has population levels from the previous period $\omega_{s,t-1}$, we have to multiply and divide both terms inside the parentheses by $\tilde{N}_{t-1}$ which leaves us with the term in front of $\frac{1}{1+\tilde{g}_{n,t}}$.
  
  ```{math}
  :label: EqStnrzMarkClrCap
    \hat{K}_t + \hat{D}_t = \frac{1}{1 + \tilde{g}_{n,t}}\sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\hat{\omega}_{s-1,t-1}\lambda_j \hat{b}_{j,s,t} + i_s\hat{\omega}_{s,t-1}\lambda_j \hat{b}_{j,s,t}\Bigr) \quad \forall t
  ```

  We stationarize the goods market clearing {eq}`EqMarkClrGoods` condition by dividing both sides by $e^{g_y t}\tilde{N}_t$. On the right-hand-side, we must multiply and divide the $K_{t+1}$ term by $e^{g_y(t+1)}\tilde{N}_{t+1}$ leaving the coefficient $e^{g_y}(1+\tilde{g}_{n,t+1})$. And the term that subtracts the sum of imports of next period's immigrant savings we must multiply and divide by $e^{g_(t+1)}$, which leaves the term $e^{g_y}$.
  
  ```{math}
  :label: EqStnrzMarkClrGoods
      \hat{Y}_t &= \hat{C}_t + e^{g_y}(1 + \tilde{g}_{n,t+1})\hat{K}_{t+1} - e^{g_y}\biggl(\sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}i_s\hat{\omega}_{s,t}\lambda_j \hat{b}_{j,s,t+1}\biggr) - (1-\delta)\hat{K}_t + \hat{G}_t \quad\forall t \\
      &\quad\text{where}\quad \hat{C}_t \equiv \sum_{s=E+1}^{E+S}\sum_{j=1}^{J}\hat{\omega}_{s,t}\lambda_j\hat{c}_{j,s,t}
  ```

  We stationarize the law of motion for total bequests $BQ_t$ in {eq}`EqMarkClrBQ` by dividing both sides by $e^{g_y t}\tilde{N}_t$. Because the population levels in the summation are from period $t-1$, we must multiply and divide the summed term by $\tilde{N}_{t-1}$ leaving the term in the denominator of $1+\tilde{g}_{n,t}$.
  
  ```{math}
  :label: EqStnrzMarkClrBQ
    \hat{BQ}_{t} = \left(\frac{1+r_{t}}{1 + \tilde{g}_{n,t}}\right)\left(\sum_{s=E+2}^{E+S+1}\sum_{j=1}^J\rho_{s-1}\lambda_j\hat{\omega}_{s-1,t-1}\hat{b}_{j,s,t}\right) \quad\forall t
  ```

