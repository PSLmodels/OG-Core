---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: '1.4.1'
kernelspec:
  display_name: Python 3
  language: python
  name: ogcore-dev
---


(Chap_Demog)=
# Demographics

  We start the `OG-Core` section on modeling with a description of the household demographics of the model. {cite}`Nishiyama:2015` and {cite}`DeBackerEtAl:2019` have recently shown that demographic dynamics are likely the biggest influence on macroeconomic time series, exhibiting more influence than fiscal variables or household preference parameters.

  In this chapter, we characterize the equations and parameters that govern the transition dynamics of the population distribution by age. In `OG-Core`, we take the approach of taking mortality rates and fertility rates from outside estimates. But we estimate our immigration rates as residuals using the mortality rates, fertility rates, and at least two consecutive periods of population distribution data. This approach makes sense if one modeling a country in which in one is not confident in the immigration rate data. If the country has good immigration data, then the immigration residual approach we describe below can be skipped.

  We define $\omega_{s,t}$ as the number of households of age $s$ alive at time $t$. A measure $\omega_{1,t}$ of households is born in each period $t$ and live for up to $E+S$ periods, with $S\geq 4$.[^calibage_note] Households are termed ``youth'', and do not participate in market activity during ages $1\leq s\leq E$. The households enter the workforce and economy in period $E+1$ and remain in the workforce until they unexpectedly die or live until age $s=E+S$. We model the population with households age $s\leq E$ outside of the workforce and economy in order most closely match the empirical population dynamics.

  The population of agents of each age in each period $\omega_{s,t}$ evolves according to the following function,
  ```{math}
    :label: EqPopLawofmotion
      \omega_{1,t+1} &= (1 - \rho_0)\sum_{s=1}^{E+S} f_s\omega_{s,t} + i_1\omega_{1,t}\quad\forall t \\
      \omega_{s+1,t+1} &= (1 - \rho_s)\omega_{s,t} + i_{s+1}\omega_{s+1,t}\quad\forall t\quad\text{and}\quad 1\leq s \leq E+S-1
  ```

  where $f_s\geq 0$ is an age-specific fertility rate, $i_s$ is an age-specific net immigration rate, $\rho_s$ is an age-specific mortality hazard rate, and $\rho_0$ is an infant mortality rate.[^houseprob_note] The total population in the economy $N_t$ at any period is simply the sum of households in the economy, the population growth rate in any period $t$ from the previous period $t-1$ is $g_{n,t}$, $\tilde{N}_t$ is the working age population, and $\tilde{g}_{n,t}$ is the working age population growth rate in any period $t$ from the previous period $t-1$.

  ```{math}
    :label: EqPopN
    N_t\equiv\sum_{s=1}^{E+S} \omega_{s,t} \quad\forall t
  ```

  ```{math}
    :label: EqPopGrowth
    g_{n,t+1} \equiv \frac{N_{t+1}}{N_t} - 1 \quad\forall t
  ```

  ```{math}
    :label: EqPopNtil
    \tilde{N}_t\equiv\sum_{s=E+1}^{E+S} \omega_{s,t} \quad\forall t
  ```

  ```{math}
    :label: EqPopGrowthTil
    \tilde{g}_{n,t+1} \equiv \frac{\tilde{N}_{t+1}}{\tilde{N}_t} - 1 \quad\forall t
  ```

  The approach to estimating fertility rates $f_s$, mortality rates $\rho_s$, and immigration rates $i_s$ for a particular calibration of the model is described in the documentation for that model.


(SecDemogPopSSTP)=
## Population steady-state and transition path

  This model requires information about mortality rates $\rho_s$ in order to solve for the household's problem each period. It also requires the steady-state stationary population distribution $\bar{\omega}_{s}$ and population growth rate $\bar{g}_n$ as well as the full transition path of the stationary population distribution $\hat{\omega}_{s,t}$ and population grow rate $\tilde{g}_{n,t}$ from the current state to the steady-state. To solve for the steady-state and the transition path of the stationary population distribution, we write the stationary population dynamic equations {eq}`EqPopLawofmotionStat` and their matrix representation {eq}`EqPopLOMstatmat`.

  ```{math}
  :label: EqPopLawofmotionStat
      \hat{\omega}_{1,t+1} &= \frac{(1-\rho_0)\sum_{s=1}^{E+S} f_s\hat{\omega}_{s,t} + i_1\hat{\omega}_{1,t}}{1+\tilde{g}_{n,t+1}}\quad\forall t \\
      \hat{\omega}_{s+1,t+1} &= \frac{(1 - \rho_s)\hat{\omega}_{s,t} + i_{s+1}\hat{\omega}_{s+1,t}}{1+\tilde{g}_{n,t+1}}\qquad\quad\:\forall t\quad\text{and}\quad 1\leq s \leq E+S-1
  ```

  ```{math}
  :label: EqPopLOMstatmat
      & \begin{bmatrix}
        \hat{\omega}_{1,t+1} \\ \hat{\omega}_{2,t+1} \\ \hat{\omega}_{2,t+1} \\ \vdots \\ \hat{\omega}_{E+S-1,t+1} \\ \hat{\omega}_{E+S,t+1}
      \end{bmatrix}= \frac{1}{1 + g_{n,t+1}} \times ... \\
      & \begin{bmatrix}
        (1-\rho_0)f_1+i_1 & (1-\rho_0)f_2 & (1-\rho_0)f_3 & \cdots & (1-\rho_0)f_{E+S-1} & (1-\rho_0)f_{E+S} \\
        1-\rho_1 & i_2 & 0 & \cdots & 0 & 0 \\
        0 & 1-\rho_2 & i_3 & \cdots & 0 & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
        0 & 0 & 0 & \cdots & i_{E+S-1} & 0 \\
        0 & 0 & 0 & \cdots & 1-\rho_{E+S-1} & i_{E+S}
      \end{bmatrix}
      \begin{bmatrix}
        \hat{\omega}_{1,t} \\ \hat{\omega}_{2,t} \\ \hat{\omega}_{2,t} \\ \vdots \\ \hat{\omega}_{E+S-1,t} \\ \hat{\omega}_{E+S,t}
      \end{bmatrix}
  ```

  We can write system {eq}`EqPopLOMstatmat` more simply in the following way.

  ```{math}
  :label: EqPopLOMstatmat2
    \boldsymbol{\hat{\omega}}_{t+1} = \frac{1}{1+g_{n,t+1}}\boldsymbol{\Omega}\boldsymbol{\hat{\omega}}_t \quad\forall t
 ```

  The stationary steady-state population distribution $\boldsymbol{\bar{\omega}}$ is the eigenvector $\boldsymbol{\omega}$ with eigenvalue $(1+\bar{g}_n)$ of the matrix $\boldsymbol{\Omega}$ that satisfies the following version of {eq}`EqPopLOMstatmat2`.

  ```{math}
  :label: EqPopLOMss
    (1+\bar{g}_n)\boldsymbol{\bar{\omega}} = \boldsymbol{\Omega}\boldsymbol{\bar{\omega}}
  ```

  ```{admonition} Proposition
  :class: tip
  If the age $s=1$ immigration rate is $i_1>-(1-\rho_0)f_1$ and the other immigration rates are strictly positive $i_s>0$ for all $s\geq 2$ such that all elements of $\boldsymbol{\Omega}$ are nonnegative, then there exists a unique positive real eigenvector $\boldsymbol{\bar{\omega}}$ of the matrix $\boldsymbol{\Omega}$, and it is a stable equilibrium.

  **Proof:**
  First, note that the matrix $\boldsymbol{\Omega}$ is square and non-negative.  This is enough for a general version of the Perron-Frobenius Theorem to state that a positive real eigenvector exists with a positive real eigenvalue. This is not yet enough for uniqueness. For it to be unique by a version of the Perron-Fobenius Theorem, we need to know that the matrix is irreducible. This can be easily shown. The matrix is of the form

  $$
  \boldsymbol{\Omega} =
    \begin{bmatrix}
      * & * & * & * & \cdots & * & * & * & * \\
      * & * & 0 & 0 & \cdots & 0 & 0 & 0 & 0 \\
      0 & * & * & 0 & \cdots & 0 & 0 & 0 & 0 \\
      0 & 0 & * & * & \cdots & 0 & 0 & 0 & 0 \\
      \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots \\
      0 & 0 & 0 & 0 & \cdots & *  & 0 & 0 & 0 \\
      0 & 0 & 0 & 0 & \cdots & *  & * & 0 & 0 \\
      0 & 0 & 0 & 0 & \cdots & 0  & * & * & 0 \\
      0 & 0 & 0 & 0 & \cdots & 0  & 0 & * & *
    \end{bmatrix}
  $$

  Where each * is strictly positive. It is clear to see that taking powers of the matrix causes the sub-diagonal positive elements to be moved down a row and another row of positive entries is added at the top. None of these go to zero since the elements were all non-negative to begin with.

  $$
  \boldsymbol{\Omega}^2 =
    \begin{bmatrix}
      * & * & * & * & \cdots & * & * & * & * \\
      * & * & * & * & \cdots & * & * & * & * \\
      * & * & * & 0 & \cdots & 0 & 0 & 0 & 0 \\
      0 & * & * & * & \cdots & 0 & 0 & 0 & 0 \\
      \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots \\
      0 & 0 & 0 & 0 & \cdots & *  & 0 & 0 & 0 \\
      0 & 0 & 0 & 0 & \cdots & *  & * & 0 & 0 \\
      0 & 0 & 0 & 0 & \cdots & *  & * & * & 0 \\
      0 & 0 & 0 & 0 & \cdots & 0  & * & * & *
    \end{bmatrix}
  $$

  $$
  \boldsymbol{\Omega}^{S+E-2} =
    \begin{bmatrix}
      * & * & * & * & \cdots & * & * & * & * \\
      * & * & * & * & \cdots & * & * & * & * \\
      * & * & * & * & \cdots & * & * & * & * \\
      * & * & * & * & \cdots & * & * & * & * \\
      \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots \\
      * & * & * & * & \cdots & *  & * & * & * \\
      * & * & * & * & \cdots & *  & * & * & * \\
      * & * & * & * & \cdots & *  & * & * & 0 \\
      0 & * & * & * & \cdots & *  & * & * & *
    \end{bmatrix}
  $$

  $$
  \boldsymbol{\Omega}^{S+E-1} =
    \begin{bmatrix}
      * & * & * & * & \cdots & * & * & * & * \\
      * & * & * & * & \cdots & * & * & * & * \\
      * & * & * & * & \cdots & * & * & * & * \\
      * & * & * & * & \cdots & * & * & * & * \\
      \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \vdots \\
      * & * & * & * & \cdots & *  & * & * & * \\
      * & * & * & * & \cdots & *  & * & * & * \\
      * & * & * & * & \cdots & *  & * & * & * \\
      * & * & * & * & \cdots & *  & * & * & *
    \end{bmatrix}
  $$

  Existence of an $m \in \mathbb{N}$ such that $\left(\bf\Omega^m\right)_{ij} \neq 0 ~~ ( > 0)$ is one of the definitions of an irreducible (primitive) matrix. It is equivalent to saying that the directed graph associated with the matrix is strongly connected. Now the Perron-Frobenius Theorem for irreducible matrices gives us that the equilibrium vector is unique.

  We also know from that theorem that the eigenvalue associated with the positive real eigenvector will be real and positive. This eigenvalue, $p$, is the Perron eigenvalue and it is the steady state population growth rate of the model. By the PF Theorem for irreducible matrices, $| \lambda_i | \leq p$ for all eigenvalues $\lambda_i$ and there will be exactly $h$ eigenvalues that are equal, where $h$ is the period of the matrix. Since our matrix $\bf\Omega$ is aperiodic, the steady state growth rate is the unique largest eigenvalue in magnitude. This implies that almost all initial vectors will converge to this eigenvector under iteration.
  ```

  For a full treatment and proof of the Perron-Frobenius Theorem, see {cite}`Suzumura:1983`. Because the population growth process is exogenous to the full overlapping generations model, we calibrate it based on the annual fertility rate, mortality rate, and population rate data for age years $s=1$ to $s=100$.


  [^calibage_note]: Theoretically, the model works without loss of generality for $S\geq 3$. However, because we are calibrating the ages outside of the economy to be one-fourth of $S$ (e.g., ages 21 to 100 in the economy, and ages 1 to 20 outside of the economy), it is convenient for $S$ to be at least 4.
  [^houseprob_note]: The parameter $\rho_s$ is the probability that a household of age $s$ dies before age $s+1$.
