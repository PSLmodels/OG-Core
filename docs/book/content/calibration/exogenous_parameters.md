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

  [TODO: This chapter needs heavy updating. Would be nice to do something similar to API chapter. But it is also nice to have references and descriptions as in the table below.]

  In this chapter, list the exogenous inputs to the model, options, and where the values come from (weak calibration vs. strong calibration). Point to the respective chapters for some of the inputs. Mention the code in [`default_parameters.json`](https://github.com/PSLmodels/OG-USA/blob/master/ogusa/default_parameters.json) and [`parameters.py`](https://github.com/PSLmodels/OG-USA/blob/master/ogusa/parameters.py).

  <!-- +++
  ```{code-cell} ogusa-dev
  :tags: [hide-cell]
  from myst_nb import glue
  import ogusa.parameter_tables as pt
  from ogusa import Specifications
  p = Specifications()
  table = pt.param_table(p, table_format=None, path=None)
  glue("param_table", table, display=False)
  ```
  -->

  ```{list-table} **List of exogenous parameters and baseline calibration values.**
  :header-rows: 1
  :name: TabExogVars
  * - **Symbol**
    - **Description**
    - **Value**
  * - $S$
    - Maximum periods in economically active household life
    - 80
  * - $E$
    - Number of periods of youth economically outside the model
    - $\text{round} \frac{S}{4}$=20
  * - $R$
    - Retirement age (period)
    - $E + \text{round} (\frac{9}{16} S) = 65$
  * - $T_1$
    - Number of periods to steady state for initial time path guesses
    - 160
  * - $T_2$
    - Maximum number of periods to steady state for nonsteady-state equilibrium
    - 160
  * - $\nu$
    - Dampening parameter for TPI
    - 0.4
  * - ${ \{ { \{ \omega_{s,0} \} }_{s=1}^{E+S}  \}}_{t=0}^{T_2 + S - 1}$
    - Initial population distribution by age
    - (see Chap. {ref}`Chap_Demog`)
  * - ${ \{ f_s \}}_{s=1}^{E+S}$
    - Fertility rates by age
    - (see Sec. {ref}`SecDemogFert`)
  * - ${ \{ i_s \}}_{s=1}^{E+S}$
    - Immigration rates by age
    - (see Sec. {ref}`SecDemogMort`)
  * - ${ \{ \rho_s \}}_{s=0}^{E+S}$
    - Mortality rates by age
    - (see Sec. {ref}`SecDemogImm`)
  ```
