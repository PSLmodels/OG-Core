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
  name: ogusa-dev
---

(glue)=
(Chap_Exog)=
# Exogenous Parameters


In this chapter, list the exogenous inputs to the model, options, and where the values come from (weak calibration vs. strong calibration). Point to the respective chapters for some of the inputs. Mention the code \texttt{parameters.py}.

  List all the exogenous parameters that are outputs of the model here.

+++
```{code-cell} ogusa-dev
:tags: [hide-cell]
from myst_nb import glue
import ogusa.parameter_tables as pt
from ogusa import Specifications
p = Specifications()
table = pt.param_table(p, table_format=None, path=None)
glue("param_table", table, display=False)
```

```{glue:figure} param_table
:figwidth: 600px
:name: "TabExogVars"

List of exogenous parameters and baseline calibration values.
```

  <!-- \begin{table}[htbp] \centering \captionsetup{width=4.7in}
    \caption{\label{TabExogVars}\textbf{List of exogenous parameters and baseline calibration values}}
      \begin{threeparttable}
      \begin{tabular}{>{\footnotesize}c |>{\footnotesize}l |>{\footnotesize}c}
        \hline\hline
        Symbol & \multicolumn{1}{c}{\footnotesize{Description}} & Value \\
        \hline
        $S$ & Maximum periods in economically active & 80 \\[-1.5mm]
        & \quad household life & \\
        $E$ & Number of periods of youth economically & $\text{round}\left(\frac{S}{4}\right)=20$ \\[-1.5mm]
        & \quad outside the model & \\
        $R$ & Retirement age (period) & $E+\text{round}\left(\frac{9}{16}S\right)=65$ \\
        $T_1$ & Number of periods to steady state for initial & 160 \\[-1.5mm]
        & \quad time path guesses & \\
        $T_2$ & Maximum number of periods to steady state & 160 \\[-1.5mm]
        & \quad for nonsteady-state equilibrium & \\
        $\nu$ & Dampening parameter for TPI & 0.4 \\
        \hline
        $\{\{\omega_{s,0}\}_{s=1}^{E+S}\}_{t=0}^{T_2+S-1}$ & Initial population distribution by age & (see Ch. {ref}`Chap_Demog`) \\
        $\{f_s\}_{s=1}^{E+S}$ & Fertility rates by age & (see Sec. {ref}`SecDemogFert`) \\
        $\{i_s\}_{s=1}^{E+S}$ & Immigration rates by age & (see Sec. {ref}`SecDemogMort`) \\
        $\{\rho_s\}_{s=0}^{E+S}$ & Mortality rates by age & (see Sec. {ref}`SecDemogImm`) \\

        % $\boldsymbol{\hat{\Gamma}}_1$ & Initial distribution of savings & $\boldsymbol{\bar{\Gamma}}$ \\

        % $\{e_{j,s}\}_{j,s=1}^{J,S}$ & Deterministic ability process & (see \citealp{DEMPRW2015}) \\
        % $\{\lambda_j\}_{j=1}^J$ & Lifetime income group percentages & $[0.25,0.25,0.20,0.10,0.10,0.09,0.01]$ \\
        % $J$ & Number of lifetime income groups & 7 \\

        % \hline
        % $\tilde{l}$ & Maximum hours of labor supply & 1 \\
        % $\beta$ & Discount factor & $(0.96)^\frac{80}{S}$ \\
        % $\sigma$ & Coefficient of constant relative risk aversion & 1.5 \\
        % $b$ & Scale parameter in utility of leisure & 0.573 \\
        % $\upsilon$ & Shape parameter in utility of leisure & 2.856 \\
        % $\chi^n_s$ & Disutility of labor level parameters & [19.041, 76.623] \\
        % $\chi^b_j$ & Utility of bequests level parameters &  $[9.264 \times 10^{-5}, 118,648]$ \\ %$1.0 \ \forall j$ \\
        % \hline
        % $Z$ & Level parameter in production function & 1.0 \\
        % $\alpha$ & Capital share of income & 0.35 \\
        % $\delta$ & Capital depreciation rate & $1-(1-0.05)^\frac{80}{S}=0.05$ \\
        % $g_y$ & Growth rate of labor augmenting & $(1+0.03)^\frac{80}{S}-1 = 0.03$ \\[-2mm]
        % & \quad technological progress & \\
        % \hline

        \hline\hline
      \end{tabular}
      \end{threeparttable}
    \end{table} -->

<div id="TabExogVars">

|                                                        Symbol                                                        |                                               |                       Value                       |
|:--------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------|:-------------------------------------------------:|
|                                                         *S*                                                          | Maximum periods in economically active        |                        80                         |
|                                                                                                                      | household life                                |                                                   |
|                                                         *E*                                                          | Number of periods of youth economically       |   $\\text{round}\\left(\\frac{S}{4}\\right)=20$   |
|                                                                                                                      | outside the model                             |                                                   |
|                                                         *R*                                                          | Retirement age (period)                       | $E+\\text{round}\\left(\\frac{9}{16}S\\right)=65$ |
|                                                   *T*<sub>1</sub>                                                    | Number of periods to steady state for initial |                        160                        |
|                                                                                                                      | time path guesses                             |                                                   |
|                                                   *T*<sub>2</sub>                                                    | Maximum number of periods to steady state     |                        160                        |
|                                                                                                                      | for nonsteady-state equilibrium               |                                                   |
|                                                         *ν*                                                          | Dampening parameter for TPI                   |                        0.4                        |
| {{*ω*<sub>*s*, 0</sub>}<sub>*s* = 1</sub><sup>*E* + *S*</sup>}<sub>*t* = 0</sub><sup>*T*<sub>2</sub> + *S* − 1</sup> | Initial population distribution by age        |                 (see Ch. ref‘Chap                 |
|                              {*f*<sub>*s*</sub>}<sub>*s* = 1</sub><sup>*E* + *S*</sup>                               | Fertility rates by age                        |           (see Sec. ref‘SecDemogFert‘)            |
|                              {*i*<sub>*s*</sub>}<sub>*s* = 1</sub><sup>*E* + *S*</sup>                               | Immigration rates by age                      |           (see Sec. ref‘SecDemogMort‘)            |
|                              {*ρ*<sub>*s*</sub>}<sub>*s* = 0</sub><sup>*E* + *S*</sup>                               | Mortality rates by age                        |            (see Sec. ref‘SecDemogImm‘)            |

<span id="TabExogVars" label="TabExogVars"></span>**List
of exogenous parameters and baseline calibration values**

</div>