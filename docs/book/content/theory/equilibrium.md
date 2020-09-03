(Chap_Eqm)=
# Equilibrium

(Chap_SSeqlb)=
## Steady-State Equilibirum

In this chapter, we define the stationary steady-state equilibrium of the `OG-USA` model. Chapters {ref}`Chap_Demog` through {ref}`Chap_MarkClr` derive the equations that characterize the equilibrium of the model. However, we cannot solve for any equilibrium of the model in the presence of nonstationarity in the variables. Nonstationarity in `OG-USA` comes from productivity growth $g_y$ in the production function {eq}`EqFirmsCESprodfun`, population growth $\tilde{g}_{n,t}$ as described in Chapter {ref}`Chap_Demog`, and the potential for unbounded growth in government debt as described in Chapter {ref}`Chap_UnbalGBC`.

We implemented an automatic government budget closure rule using government spending $G_t$ as the instrument that stabilizes the debt-to-GDP ratio at a long-term rate in {eq}`EqUnbalGBCclosure`. And we showed in Chapter {ref}`Chap_Stnrz` how to stationarize all the other characterizing equations.

(SecEqlbSSdef)=
### Stationary Steady-State Equilibrium Definition

  With the stationarized model, we can now define the stationary steady-state equilibrium. This equilibrium will be long-run values of the endogenous variables that are constant over time. In a perfect foresight model, the steady-state equilibrium is the state of the economy at which the model settles after a finite amount of time, regardless of the initial condition of the model. Once the model arrives at the steady-state, it stays there indefinitely unless it receives some type of shock or stimulus.

  These stationary values have all the growth components from productivity growth and population growth removed as defined in Table {ref}`TabStnrzStatVars`. Because the productivity growth rate $g_y$ and population growth rate series $\tilde{g}_{n,t}$ are exogenous. We can transform the stationary equilibrium values of the variables back to their nonstationary values by reversing the identities in Table {ref}`TabStnrzStatVars`.

  \vspace{5mm}
  \hrule
  \vspace{-1mm}
  \begin{definition}[\textbf{Stationary steady-state equilibrium}]\label{DefSSEql}
    A non-autarkic stationary steady-state equilibrium in the `OG-USA` model is defined as constant allocations of stationary household labor supply $n_{j,s,t}=\bar{n}_{j,s}$ and savings $\hat{b}_{j,s+1,t+1}=\bar{b}_{j,s+1}$ for all $j$, $t$, and $E+1\leq s\leq E+S$, and constant prices $\hat{w}_t=\bar{w}$ and $r_t=\bar{r}$ for all $t$ such that the following conditions hold:
    \begin{enumerate}
      \item the population has reached its stationary steady-state distribution $\hat{\omega}_{s,t} = \bar{\omega}_s$ for all $s$ and $t$ as characterized in Section {ref}`SecDemogPopSSTP`,
      \item households optimize according to {eq}`EqStnrzHHeul_n`, {eq}`EqStnrzHHeul_b`, and {eq}`EqStnrzHHeul_b`,
      \item firms optimize according to {eq}`EqStnrzFOC_L` and {eq}`EqFirmFOC_K`,
      \item Government activity behaves according to {eq}`EqStnrzGovBC` and {eq}`EqStnrzClosureRule`, and
      \item markets clear according to {eq}`EqStnrzMarkClrLab`, {eq}`EqStnrzMarkClrCap`, and {eq}`EqStnrzMarkClrBQ`.
    \end{enumerate}
  \end{definition}
  \vspace{-2mm}
  \hrule
  \vspace{5mm}

(SecEqlbSSsoln)=
### Stationary Steady-state Solution Method

  \renewcommand\theenumi{\arabic{enumi}}
  \renewcommand\theenumii{\alph{enumii}}
  \renewcommand\theenumiii{\roman{enumiii}}

  This section describes the solution method for the stationary steady-state equilibrium described in Definition {ref}`DefSSEql`. The steady-state is characterized by $2JS$ equations and $2JS$ unknowns. However, because some of the other equations cannot be solved for analytically and substituted into the Euler equations, we use a fixed point algorithm to solve for the steady-state. We begin by making a guess at steady-state interest rate $\bar{r}$, total bequests $\overline{BQ}$, total household transfers $\overline{TR}$, and income multiplier $factor$. We call these four steady-state variables the "outer loop" variables in our steady-state solution method and the determination of a fixed point over these variables the ``outer loop'' of the steady-state solution method. The outer loop variables are the macroeconomic variables necessary to solve the household's problem.  

  For each iteration over these outer-loop variables, we solve the household problem given the values of these macroeconomic variables.  We call this solution the ``inner loop'' of the steady-state solution method.  In the inner loop, we solve for the steady-state household decisions $\bar{b}_{j,s}$ and labor supply $\bar{n}_{j,s}$ for all $j$ and $E+1\leq s\leq E+S$, and then use the household decisions to compute updated values for macroeconomics variables. Because the lifetime optimization problem of each household of type $j$ is a highly nonlinear system of $2S$ equations and $2S$ unknowns, we solve for the $2S$ household's decisions simultaneously for a given type $j$ household.  We then use the solutions for type-$j$ households as the initial guesses a the solutions for type-$j+1$ households.
  
  The macroeconomic variables computed from the solutions to the household problem are used to update the values of those macroeconomic variables in the outer-loop.  This process continues until a fixed point is found.  That is, until the macroeconomic variables in the outer loop result in household decisions that are consistent with those macroeconomic variables' values.

 We outline this algorithm in the following steps.


1. Use the techniques from Section {ref}`SecDemogPopSSTP` to solve for the steady-state population distribution vector $\boldsymbol{\bar{\omega}}$ and steady-state growth rate $\bar{g}_n$ of the exogenous population process.
2. Choose an initial guess for the values of the steady-state interest rate $\bar{r}^i$, total bequests $\overline{BQ}^{\,i}$, total household transfers $\overline{TR}^{\,i}$, and income multiplier $factor^i$, where superscript $i$ is the index of the iteration number of the guess.
3. Use $\bar{r}^i$ together with the firm's first order condition for its choice of capital, Equation {eq}`EqFirmFOC_K`, to solve for the capital labor ratio, $\frac{\overline{K}}{\overline{L}}$.  Then use $\frac{\bar{K}}{\bar{L}}$ in the firm's first order condition for its choice of labor to find the implied wage rate $\bar{w}^i$.
4. Given guesses for $\bar{r}^i$, $\bar{w}^i$, $\overline{BQ}^{\,i}$, $\overline{TR}^{\,i}$, and $factor^i$, solve for the steady-state household labor supply $\bar{n}_{j,s}$ and savings $\bar{b}_{j,s}$ decisions for all $j$ and $E+1\leq s\leq E+S$.
    1.  Do this by using a multivariate root-finder to solve the $2S$ necessary conditions of the household given by Equations {eq}`EqHHeul_n`, {eq}`EqHHeul_b`, and {eq}`EqHHeul_bS` simultaneously for $j=1$.
    2. Repeat this root-finding process for each household $j\in{2,...,J}$, using the solutions for households of type $j-1$ as the initial guesses in the root-finder.  
5. Given partial equilibrium household steady-state solutions $\{\bar{c}_{j,s},\bar{n}_{j,s},\bar{b}_{j,s+1}\}_{s=E+1}^{E+S}$ based on macroeconomic variables $\bar{r}^i$, $\overline{BQ}^{\,i}$, $\overline{TR}^{\,i}$, and $factor^i$, compute updated values for the outer loop macroeconomic variables, $\bar{r}^{i'}$, $\bar{w}^{i'}$, $\overline{BQ}^{\,i'}$, $\overline{TR}^{\,i'}$, and $factor^{i'}$ .
    1. We solve for the updated interest rate as follows: 
        1. Use the guess at total transfers, $\overline{TR}^{i}$ and the transfer spending rule given in Equation {eq}`EqUnbalGBCtfer` to find the implied GDP: $\bar{Y}^{i} = \frac{\overline{TR}^{i}}{\alpha_{tr}}$.
        2. Use the long-run debt-to-GDP ratio and $\bar{Y}^{i}$ to find total government debt in the steady-state, $\bar{D}^{i} = \alpha_{D}\bar{Y}^{i}$.
        3. Use the capital market clearing condition from Equation {eq}`EqStnrzMarkClrCap` and $\bar{D}^{i}$ to find aggregate capital,
      		
      		$$
      			\bar{K}^{i}=\frac{1}{1 + \bar{g}_{n}}\sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\bar{\omega}_{s-1}\lambda_j \bar{b}_{j,s} + i_s\bar{\omega}_{s}\lambda_j \bar{b}_{j,s}\Bigr) - \bar{D}^{i}
      		$$
      		
      	4. Use the labor market clearing condition from Equation {eq}`EqStnrzMarkClrLab` to find aggregate labor supply:
      		
      		$$
      			\bar{L}^{i}=\sum_{s=E+1}^{E+S}\sum_{j=1}^{J} \bar{\omega}_{s}\lambda_j e_{j,s}\bar{n}_{j,s}
      		$$
      		
      	5. Use the firm's production function from Equation {eq}`EqStnrzCESprodfun` to compute an updated value of $\bar{Y}$ given the values for the factors of production:
      		
      		$$
      			\bar{Y}^{i'} = \bar{Z}\biggl[(\gamma)^\frac{1}{\varepsilon}(\bar{K}^{i})^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma)^\frac{1}{\varepsilon}(\bar{L}^{i})^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} 
      		$$
      		
      	6. Use the firm's first order condition for its choice of capital to find the updated interest rate, 
      		
      		$$
      		\bar{r}^{i'} = (1 - \tau^{corp})(\bar{Z})^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{\bar{Y}^{i'}}{\bar{K}^{i}}\right]^\frac{1}{\varepsilon} - \delta + \tau^{corp}\delta^\tau
      		$$
      		
      by first using the market clearing conditions {eq}`EqStnrzMarkClrLab` and {eq}`EqStnrzMarkClrCap` together with the the long-run debt
      

    2. The stationarized law of motion for total bequests {eq}`EqStnrzMarkClrBQ` provides the expression in which household savings decisions $\{\bar{b}_{j,s+1}\}_{s=E+1}^{E+S}$ imply a value for aggregate bequests, $\overline{BQ}^{\,i'}$. When computing aggregate bequests, we use the updated interest rate found above.
      
      $$
        \overline{BQ}^{\,i'} = \left(\frac{1+\bar{r}^{i'}}{1 + \bar{g}_{n}}\right)\left(\sum_{s=E+2}^{E+S+1}\sum_{j=1}^J\rho_{s-1}\lambda_j\bar{\omega}_{s-1}\bar{b}_{j,s}\right)
      $$
      
    3. In equation {eq}`EqStnrzTfer`, we defined total household transfers as a fixed percentage of GDP ($\overline{TR}=\alpha_{tr}\bar{Y}$).  To find the updated value for transfers, we find the amount of transfers implied by the most updated value of GDP, $\overline{TR}^{i'}=\alpha_{tr}\bar{Y}^{i'}$.
      
    4. The $factor$ that transforms the model units to U.S. dollar units for the tax functions {eq}`EqTaxCalcFactor` is already defined in terms of steady-state variables. The following is an expression in which household decisions $\{\bar{n}_{j,s},\bar{b}_{j,s+1}\}_{s=E+1}^{E+S}$ imply a value for the steady-state $factor^{i'}$. Note, as with the equation for $\overline{BQ}^{\,i'}$, that we include the updated values of $\bar{w}^{i'}$ and $\bar{r}^{i'}$ on the right-hand-side of the equation.[^step3_note]
      
      $$
        factor^{i'} = \frac{\text{Avg. household income in data}}{\sum_{s=E+1}^{E+S}\sum_{j=1}^J\lambda_j\bar{\omega}_s\left(\bar{w}^{i'}e_{j,s}\bar{n}_{j,s} + \bar{r}^{i'}\bar{b}_{j,s}\right)}
      $$$
      
6. The updated values for the outer loop variables are then used to compute the percentage differences between the initial and implied values:
  	a. $error_r = \frac{\bar{r}^{i'} - \bar{r}^i}{\bar{r}^i}$
  	b. $error_{bq} = \frac{\overline{BQ}^{\,i'} - \overline{BQ}^{\,i}}{\overline{BQ}^{\,i}}$
  	c. $error_{tr} = \frac{\overline{TR}^{\,i'} - \overline{TR}^{\,i}}{\overline{TR}^{\,i}}$
  	d. $error_f = \frac{factor^{i'} - factor^i}{factor^i}$

7. If the maximum absolute error among the four outer loop error terms is greater than some small positive tolerance $toler_{ss,out}$,
    $$
      \max\big|\left(error_r,error_{bq},error_{tr},error_f\right)\bigr| > toler_{ss,out}
    $$
    then update the guesses for the outer loop variables as a convex combination governed by $\xi_{ss}\in(0,1]$ of the respective initial guesses and the new implied values and repeat steps (3) through (5).[^rootfinder_note]
    
    $$
        \left[\bar{r}^{i+1},\overline{BQ}^{\,i+1},\overline{TR}^{\,i+1},factor^{i+1}\right] &= \xi_{ss}\left[\bar{r}^{i'},\overline{BQ}^{\,i'},\overline{TR}^{\,i'},factor^{i'}\right] + \\
        \qquad(1-\xi_{ss})\left[\bar{r}^{i},\overline{BQ}^{\,i},\overline{TR}^{\,i},factor^{i}\right]
    $$

8. If the maximum absolute error among the four outer loop error terms is less-than-or-equal-to some small positive tolerance $toler_{ss,out}$,
    
    $$
      \max\big|\left(error_r,error_{bq},error_{tr},error_f\right)\bigr| \leq toler_{ss,out}
    $$

    then the steady-state has been found.
    1. Make sure that steady-state government spending is nonnegative $\bar{G}\geq 0$. If steady-state government spending is negative, that means the government is getting resources to supply the debt from outside the economy each period to stabilize the debt-to-GDP ratio. $\bar{G}<0$ is a good indicator of unsustainable policies.
    2. Make sure that the resource constraint (goods market clearing) {eq}`EqStnrzMarkClrGoods` is satisfied. It is redundant, but this is a good check as to whether everything worked correctly.
    3. Make sure that the government budget constraint {eq}`EqStnrzGovBC` binds.
    4. Make sure that all the $2JS$ household Euler equations are solved to a satisfactory tolerance.

(SecSSeqlbResults)=
## Baseline Steady-state Results

  In this section, we use the baseline calibration described in Chapter {ref}`Chap_Calibr`, which includes the baseline tax law from \taxcalc, to show some steady-state results from `OG-USA`. Figure {ref}`FigSSeqlbHHvars` shows the household steady-state variables by age $s$ and lifetime income group $j$.


```{figure} ./images/HHcons_SS.png
---
height: 500px
name: FigSSeqlbHHcons
---

```

```{figure} ./images/HHlab_SS.png
---
height: 500px
name: FigSSeqlbHHlab
---

```

```{figure} ./images/HHsav_SS.png
---
height: 500px
name: FigSSeqlbHHsave
---

```


  <!-- \begin{figure}[htb]\centering \captionsetup{width=6.0in}
    \caption{\label{FigSSeqlbHHvars}\textbf{Steady-state distributions of household consumption $\bar{c}_{j,s}$, labor supply $\bar{n}_{j,s}$, and savings $\bar{b}_{j,s+1}$}}
    \begin{subfigure}[b]{0.48\textwidth}
      \includegraphics[width=\textwidth]{images/HHcons_SS.png}
      \caption{Consumption $\bar{c}_{j,s}$}
      \label{FigSSeqlbHHcons}
    \end{subfigure}
    \begin{subfigure}[b]{0.48\textwidth}
      \includegraphics[width=\textwidth]{images/HHlab_SS.png}
      \caption{Labor supply $\bar{n}_{j,s}$}
      \label{FigSSeqlbHHlab}
    \end{subfigure}
    \begin{subfigure}[b]{0.48\textwidth}
      \includegraphics[width=\textwidth]{images/HHsav_SS.png}
      \caption{Savings $\bar{b}_{j,s+1}$}
      \label{FigSSeqlbHHsav}
    \end{subfigure}
  \end{figure} -->

  Table {ref}`TabSSeqlbAggrVars` lists the steady-state prices and aggregate variable values along with some of the maximum error values from the characterizing equations.

  <!-- \begin{table}[htbp] \centering \captionsetup{width=4.1in}
  \caption{\label{TabSSeqlbAggrVars}\textbf{Steady-state prices, aggregate variables, and maximum errors}}
    \begin{threeparttable}
    \begin{tabular}{>{\small}l >{\small}r |>{\small}l >{\small}r}
      \hline\hline
      \multicolumn{1}{c}{\small{Variable}} & \multicolumn{1}{c}{\small{Value}} & \multicolumn{1}{c}{\small{Variable}} & \multicolumn{1}{c}{Value} \\
      \hline
      $\bar{r}$ & 0.058 & $\bar{w}$ & 1.148 \\
      \hline
      $\bar{Y}$ & 0.630 & $\bar{C}$ & 0.462 \\
      $\bar{I}$ & 0.144 & $\bar{K}$ & 1.810 \\
      $\bar{L}$ & 0.357 & $\bar{B}$ & 2.440 \\
      $\overline{BQ}$ & 0.106 & $factor$  & 141,580 \\
      \hline
      $\overline{Rev}$ & 0.096 & $\overline{TR}$ & 0.057 \\
      $\bar{G}$ & 0.023 & $\bar{D}$ & 0.630 \\
      \hline
      Max. abs.         & 4.57e-13 & Max. abs.  & 8.52e-13 \\[-2mm]
      \:\: labor supply &   & \:\: savings &     \\[-2mm]
      \:\: Euler error  &   & \:\: Euler error & \\
      Resource        & -4.39e-15 & Serial & 1 hr. 25.9 sec.\tnote{*} \\[-2mm]
      \:\: constraint & & \:\: computation & \\[-2mm]
      \:\: error      & & \:\: time &  \\
      \hline\hline
    \end{tabular}
    \begin{tablenotes}
      \scriptsize{\item[*]The steady-state computation time does not include any of the exogenous parameter computation processes, the longest of which is the estimation of the baseline tax functions which computation takes 1 hour and 15 minutes.}
    \end{tablenotes}
    \end{threeparttable}
  \end{table} -->

<div id="TabSSeqlbAggrVars">

|                   |           |                    |                 |
|:------------------|----------:|:-------------------|----------------:|
| *r̄*               |     0.058 | *w̄*                |           1.148 |
| *Ȳ*               |     0.630 | *C̄*                |           0.462 |
| *Ī*               |     0.144 | *K̄*                |           1.810 |
| *L̄*               |     0.357 | *B̄*                |           2.440 |
| $\\overline{BQ}$  |     0.106 | *f**a**c**t**o**r* |         141,580 |
| $\\overline{Rev}$ |     0.096 | $\\overline{TR}$   |           0.057 |
| *Ḡ*               |     0.023 | *D̄*                |           0.630 |
| Max. abs.         |  4.57e-13 | Max. abs.          |        8.52e-13 |
| labor supply      |           | savings            |                 |
| Euler error       |           | Euler error        |                 |
| Resource          | -4.39e-15 | Serial             | 1 hr. 25.9 sec. |
| constraint        |           | computation        |                 |
| error             |           | time               |                 |

<span id="TabSSeqlbAggrVars"
label="TabSSeqlbAggrVars">\[TabSSeqlbAggrVars\]</span>**Steady-state
prices, aggregate variables, and maximum errors**

</div>

The steady-state computation time does not include any of the exogenous
parameter computation processes, the longest of which is the estimation
of the baseline tax functions which computation takes 1 hour and 15
minutes.

(Chap_NSSeqlb)=
## Stationary Nonsteady-State Equilibrium

In this chapter, we define the stationary nonsteady-state equilibrium of the `OG-USA` model. Chapters {ref}`Chap_Demog` through {ref}`Chap_MarkClr` derive the equations that characterize the equilibrium of the model. We also need the steady-state solution from Chapter {ref}`Chap_SSeqlb` to solve for the nonsteady-state equilibrium transition path. As with the steady-state equilibrium, we must use the stationarized version of the characterizing equations from Chapter {ref}`Chap_Stnrz`.

(SecEqlbNSSdef)=
### Stationary Nonsteady-State Equilibrium Definition

  We define a stationary nonsteady-state equilibrium as the following.

  \vspace{5mm}
  \hrule
  \vspace{-1mm}
  \begin{definition}[\textbf{Stationary Nonsteady-state functional equilibrium}]\label{DefNSSEql}
    A non autarkic nonsteady-state functional equilibrium in the `OG-USA` model is defined as stationary allocation functions of the state $\bigl\{n_{j,s,t} = \phi_s\bigl(\boldsymbol{\hat{\Gamma}}_t\bigr)\bigr\}_{s=E+1}^{E+S}$ and $\bigl\{\hat{b}_{j,s+1,t+1}=\psi_{s}\bigl(\boldsymbol{\hat{\Gamma}}_t\bigr)\bigr\}_{s=E+1}^{E+S}$ for all $j$ and $t$ and stationary price functions $\hat{w}(\boldsymbol{\hat{\Gamma}}_t)$ and $r(\boldsymbol{\hat{\Gamma}}_t)$ for all $t$ such that:
    \begin{enumerate}
      \item households have symmetric beliefs $\Omega(\cdot)$ about the evolution of the distribution of savings as characterized in {eq}`EqBeliefs`, and those beliefs about the future distribution of savings equal the realized outcome (rational expectations),
      \begin{equation*}
        \boldsymbol{\hat{\Gamma}}_{t+u} = \boldsymbol{\hat{\Gamma}}^e_{t+u} = \Omega^u\left(\boldsymbol{\hat{\Gamma}}_t\right) \quad\forall t,\quad u\geq 1
      \end{equation*}
      \item households optimize according to {eq}`EqStnrzHHeul_n`, {eq}`EqStnrzHHeul_b`, and {eq}`EqStnrzHHeul_b`,
      \item firms optimize according to {eq}`EqStnrzFOC_L` and {eq}`EqFirmFOC_K`,
      \item Government activity behaves according to {eq}`EqStnrzGovBC` and {eq}`EqStnrzClosureRule`, and
      \item markets clear according to {eq}`EqStnrzMarkClrLab`, {eq}`EqStnrzMarkClrCap`, and {eq}`EqStnrzMarkClrBQ`.
    \end{enumerate}
  \end{definition}
  \vspace{-2mm}
  \hrule
  \vspace{5mm}

(SecEqlbNSSsoln)=
### Stationary Nonsteady-state Solution Method


  This section describes the solution method for the stationary nonsteady-state equilibrium described in Definition {ref}`DefNSSEql`. We use the time path iteration (TPI) method. This method was originally outlined in a series of papers between 1981 and 1985\footnote{See {cite}`AuerbachEtAl:1981,AuerbachEtAl:1983`, {cite}`AuerbachKotlikoff:1983a,AuerbachKotlikoff:1983b,AuerbachKotlikoff:1983c`, and {cite}`AuerbachKotlikoff:1985`.} and in the seminal book \citet[ch. 4]{AuerbachKotlikoff:1987} for the perfect foresight case and in \citet[Appendix II]{NishiyamaSmetters:2007} and \citet[Sec. 3.1]{EvansPhillips:2014} for the stochastic case. The intuition for the TPI solution method is that the economy is infinitely lived, even though the agents that make up the economy are not. Rather than recursively solving for equilibrium policy functions by iterating on individual value functions, one must recursively solve for the policy functions by iterating on the entire transition path of the endogenous objects in the economy (see \citet[ch. 17]{StokeyLucas1989}).

  The key assumption is that the economy will reach the steady-state equilibrium $\boldsymbol{\bar{\Gamma}}$ described in Definition {ref}`DefSSEql` in a finite number of periods $T<\infty$ regardless of the initial state $\boldsymbol{\hat{\Gamma}}_1$. The first step in solving for the nonsteady-state equilibrium transition path is to solve for the steady-state using the method described in Section {ref}`SecEqlbSSsoln`. After solving for the steady-state, one must then find a fixed point over the entire path of endogenous objects.  We do this by first making an initial guess at these objects in a the general equilibrium ``outer loop'' step, analogous to the outer loop described in the steady-state solution method. The time path iteration method then uses functional iteration to converge on a fixed point for the path of these objects.  The paths of aggregate variables that must be guessed in this outer loop are $\{\boldsymbol{r}^i,\boldsymbol{\hat{w}}^i,\boldsymbol{\hat{BQ}}^i, \boldsymbol{\hat{TR}}^i\}$, where $\boldsymbol{r}^i = \left\{r_1^i,r_2^i,...r_T^i\right\}$, $\boldsymbol{\hat{BQ}}^i = \left\{\hat{BQ}_1^i,\hat{BQ}_2^i,...\hat{BQ}_T^i\right\}$, and $\boldsymbol{\hat{TR}}^i = \left\{\hat{TR}_1^i,\hat{TR}_2^i,...\hat{TR}_T^i\right\}$. The only requirement on these transition paths is that the initial total bequests $\hat{BQ}_1^i$ conform to the initial state of the economy $\boldsymbol{\hat{\Gamma}}_1$, and that the economy has reached the steady-state by period $t=T$ $\{r_T^i, \hat{BQ}_T^i, \hat{TR}_T^i\} = \{\bar{r}, \bar{w}, \overline{BQ}, \overline{TR}\}$.

  The "inner loop" of the nonsteady-state transition path solution method is to solve for the full set of lifetime savings decisions $\bar{b}_{j,s+1,t+1}$ and labor supply decisions $\bar{n}_{j,s,t}$ for every household alive between periods $t=1$ and $t=T$.  To solve for the $2JS$ equations and unknowns for each household's lifetime decisions we use a multivariate root finder.  
  
  We outline the stationary non-steady state solution algorithm in the following steps.

1. Compute the steady-state solution $\{\bar{n}_{j,s},\bar{b}_{j,s}\}_{s=E+1}^{E+S}$ corresponding to Definition {ref}`DefSSEql`.
2. Given initial state of the economy $\boldsymbol{\hat{\Gamma}}_1$ and steady-state solutions $\{\bar{n}_{j,s},\bar{b}_{j,s+1}\}_{s=E+1}^{E+S}$, guess transition paths of outer loop macroeconomic variables $\{\boldsymbol{r}^i,\boldsymbol{\hat{BQ}}^i, \boldsymbol{\hat{TR}}^i\}$ such that $\hat{BQ}_1^i$ is consistent with $\boldsymbol{\hat{\Gamma}}_1$ and $\{r_t^i, \hat{BQ}_t^i, \hat{TR}_t^i\} = \{\bar{r}, \overline{BQ}, \overline{TR}\}$ for all $t\geq T$.
3. Given initial condition $\boldsymbol{\hat{\Gamma}}_1$, guesses for the aggregate time paths $\{\boldsymbol{r}^i,\boldsymbol{\hat{BQ}}^i, \boldsymbol{\hat{TR}}^i\}$, we solve for the inner loop lifetime decisions of every household that will be alive across the time path $\{n_{j,s,t},\hat{b}_{j,s+1,t+1}\}_{s=E+1}^{E+S}$ for all $j$ and $1\leq t\leq T$.
    a. Given time path guesses $\{\boldsymbol{r}^i,\boldsymbol{\hat{BQ}}^i, \boldsymbol{\hat{TR}}^i\}$, we can compute the path of wages, $\boldsymbol{w}^i$ and then solve for each household's lifetime decisions $\{n_{j,s,t},\hat{b}_{j,s+1,t+1}\}_{s=E+1}^{E+S}$ for all $j$, $E+1\leq s \leq E+S$, and $1\leq t\leq T_2+S-1$.
        i. The household problem can be solved with a multivariate root finder solving the $2S$ equations and unknowns at once for all $j$ and $1\leq t\leq T+S-1$. The root finder uses $2S$ household Euler equations {eq}`EqStnrzHHeul_n`, {eq}`EqStnrzHHeul_b`, and {eq}`EqStnrzHHeul_bS` to solve for each household's $2S$ lifetime decisions.
        ii. After solving the first iteration of time path iteration, subsequent initial values for the $J$, $2S$ root finding problems are based on the solution in the prior iteration. This speeds up computation further and makes the initial guess for the highly nonlinear system of equations start closer to the solution value.
4. Given partial equilibrium household nonsteady-state solutions $\{n_{j,s,t},\hat{b}_{j,s+1,t+1}\}_{s=E+1}^{E+S}$ for all $j$ and $1\leq t\leq T$ based on macroeconomic variable time path guesses $\{\boldsymbol{r}^i,\boldsymbol{\hat{BQ}}^i, \boldsymbol{\hat{TR}}^i\}$, compute new values for these aggregates implied by the households' solutions, $\{\boldsymbol{r}^{i'},\boldsymbol{\hat{BQ}}^{i'}, \boldsymbol{\hat{TR}}^{i'}\}$.
   a. We solve for the updated interest rate as follows: 
    	i. Use the guess at the path of total transfers, $\hat{TR}_{t}^{i}$ and the transfer spending rule given in Equation {eq}`EqUnbalGBCtfer` to find the implied path of GDP: $\hat{Y}_{t}^{i} = \frac{\hat{TR}_{t}^{i}}{\alpha_{tr}}$.
    	ii. Using the path of GDP and the household savings and labor supply decisions, $\{n_{j,s,t},\hat{b}_{j,s+1,t+1}\}_{s=E+1}^{E+S}$, compute the path of stationarizaed total tax revenue, $\hat{Revenue}_{t}^{i}$. 
    	iii. Using the long-run debt-to-GDP ratio, the path of GDP, the path of total tax revenue, and Equation {eq}`EqUnbalGBCclosure_Gt`, find the path of stationarized government debt, $\hat{D}_{t}^{i}$.
    	iv. se the capital market clearing condition from Equation {eq}`EqStnrzMarkClrCap` and $D_{t}^{i}$ to find aggregate capital in each period,
    	
    			$$
    				\hat{K}_{t}^{i}=\frac{1}{1 + g_{n,t}}\sum_{s=E+2}^{E+S+1}\sum_{j=1}^{J}\Bigl(\omega_{s-1,t-1}\lambda_j \hat{b}_{j,s,t} + i_s\omega_{s,t}\lambda_j \hat{b}_{j,s,t}\Bigr) - D_{t}^{i}
    			$$
    	
    	v. Use the labor market clearing condition from Equation {eq}`EqStnrzMarkClrLab` to find the path of aggregate labor supply:
    	
    			$$
    				\hat{L}_{t}^{i}=\sum_{s=E+1}^{E+S}\sum_{j=1}^{J} \omega_{s,t}\lambda_j e_{j,s}n_{j,s,t}
    			$$
    	
    	vi. Use the firm's production function from Equation {eq}`EqStnrzCESprodfun` to compute an updated value of $\hat{Y}_{t}$ given the values for the factors of production:
    	
    		$$
    			\hat{Y}_{t}^{i'} = Z_{t}\biggl[(\gamma)^\frac{1}{\varepsilon}(\hat{K}_{t}^{i})^\frac{\varepsilon-1}{\varepsilon} + (1-\gamma)^\frac{1}{\varepsilon}(\hat{L}_{t}^{i})^\frac{\varepsilon-1}{\varepsilon}\biggr]^\frac{\varepsilon}{\varepsilon-1} 
    		$$
    	
    	vii. Use the firm's first order condition for its choice of capital to find the updated path of interest rates, 
    	
    		$$
    			r_{t}^{i'} = (1 - \tau_{t}^{corp})(Z_{t})^\frac{\varepsilon-1}{\varepsilon}\left[\gamma\frac{\hat{Y}_{t}^{i'}}{\hat{K}_{t}^{i}}\right]^\frac{1}{\varepsilon} - \delta + \tau_{t}^{corp}\delta_{t}^\tau
    		$$
    	
    
    b. The stationarized law of motion for total bequests {eq}`EqStnrzMarkClrBQ` provides the expression in which household savings decisions $\{b_{j,s+1,t+1}\}_{s=E+1}^{E+S}$ imply a value for aggregate bequests, $BQ_{t}^{\,i'}$. When computing aggregate bequests, we use the updated path of interest rates found above.
    		$$
    			\hat{BQ}_{t}^{\,i'} = \left(\frac{1+r_{t}^{i'}}{1 + g_{n,t}}\right)\left(\sum_{s=E+2}^{E+S+1}\sum_{j=1}^J\rho_{s-1}\lambda_j\omega_{s-1,t-1}\hat{b}_{j,s,t}\right)
    		$$
    
    c. In equation {eq}`EqStnrzTfer`, we defined total household transfers as a fixed percentage of GDP ($\hat{TR}_t=\alpha_{tr}\hat{Y}_t$).  To find the updated value for transfers, we find the amount of transfers implied by the most updated value of GDP, $\hat{TR}_{t}^{i'}=\alpha_{tr}\hat{Y}_{t}^{i'}$.
    
    
5. The updated values for the outer loop variables are then used to compute the percentage differences between the initial and implied values:
	a. $error_r = max\left\{\frac{r_{t}^{i'} - r_{t}^i}{r_{t}^i}\right\}_{t=0}^{T}$
	b. $error_{bq} =  max\left\{\frac{\hat{BQ}_{t}^{\,i'} - \hat{BQ}_{t}^{\,i}}{\hat{BQ}_{t}^{\,i}}\right\}_{t=0}^{T}$
	c. $error_{tr} = \left\{\frac{\hat{TR}_{t}^{\,i'} - \hat{TR}_{t}^{\,i}}{\hat{TR}_{t}^{\,i}}\right\}_{t=0}^{T}$

6. If the maximum absolute error among the three outer loop error terms is greater than some small positive tolerance $toler_{tpi,out}$,
		$$
			\max\big|\left(error_r,error_{bq},error_{tr},error_f\right)\bigr| > toler_{tpi,out}
		$$
	then update the guesses for the outer loop variables as a convex combination governed by $\xi_{tpi}\in(0,1]$ of the respective initial guesses and the new implied values and repeat steps (3) through (5).
	$$
			\left[\boldsymbol{r}^{i+1},\boldsymbol{\hat{BQ}}^{\,i+1},\boldsymbol{\hat{TR}}^{\,i+1}\right] &= \xi_{tpi}\left[\boldsymbol{r}^{i'},\boldsymbol{\hat{BQ}}^{\,i'},\boldsymbol{\hat{TR}}^{\,i'}\right] + \\
			&\qquad(1-\xi_{tpi})\left[\boldsymbol{r}^{i},\boldsymbol{\hat{BQ}}^{\,i},\boldsymbol{\hat{TR}}^{\,i}\right]
	$$
7. If the maximum absolute error among the five outer loop error terms is less-than-or-equal-to some small positive tolerance $toler_{tpi,out}$ in each period along the transition path,
		$$
			\max\big|\left(error_r,error_{bq},error_{tr},error_f\right)\bigr| \leq toler_{tpi,out}
		$$
	then the non-steady-state equilibrium has been found.
	a. Make sure that the resource constraint (goods market clearing) {eq}`EqStnrzMarkClrGoods` is satisfied in each period along the time path. It is redundant, but this is a good check as to whether everything worked correctly.
	b. Make sure that the government budget constraint {eq}`EqStnrzGovBC` binds.
	c. Make sure that all the $(T+S)\times2JS$ household Euler equations are solved to a satisfactory tolerance.

(SecNSSeqlbResults)=
### Baseline Nonsteady-state Results


[^step3_note]: The updated wage rate, $w^{i'}$, is found by using the updated interest rate, $r^{i'}$ as detailed in Step 3.

[^rootfinder_note]: In our code, there is also an option to use a Newton based root-finding algorithm to updated the outer-loop variables.  Such an algorithm is generally faster, but less robust than the functional iteration method outlined here.