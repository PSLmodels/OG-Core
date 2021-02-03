(Chap_UBI)=
# Universal Basic Income (UBI)

[TODO: This section is far along but needs to be updated.]

We have included the modeling of a universal basic income (UBI) policy directly in the theory and code for `OG-USA`. We calculate the time series of a UBI matrix $ubi_{j,s,t}$ representing the UBI transfer to every household with head of household age $s$, lifetime income group $j$, in period $t$. We calculate the time series of this matrix from five parameters and some household composition data that we impose upon the existing demographics of `OG-USA`.


(SecUBIcalc)=
## Calculating UBI

  We calculate the time series of UBI household transfer in model units $ubi_{j,s,t)}$ and the time series of total UBI expenditures in model units $UBI_t$ from five parameters described in the `OG-USA` API (`ubi_growthadj`, `ubi_child`, `ubi_adult`, `ubi_senior`, and `ubi_max`) interfaced with the `OG-USA` demographic dynamics over lifetime income groups $j$ and ages $s$, and multiplied by household composition matrices from the [OG-USA-calibration](https://github.com/PSLmodels/OG-USA-Calibration) repository.

  From the [OG-USA-calibration](https://github.com/PSLmodels/OG-USA-Calibration) repository, we have three $S\times J$ matrices `ubi_num_child_mat`$_{j,s}$, `ubi_num_adult_mat`$_{j,s}$, and `ubi_num_senior_mat`$_{j,s}$ representing the number of children under age 18, the number of adults between ages 18 and 65, and the number of seniors over 65, respectively, by lifetime ability group $j$ and age $s$ of head of household. Because our demographic age data match up well with head-of-household data from other datasets, we do not have to adjust the values in these matrices.[^HOH_age_dist_note]

  Now we can solve for the dollar-valued (as opposed to model-unit-valued) UBI transfer to each household in the first period $ubi^{\$}_{j,s,t=0}$ in the following way. Let the parameter `ubi_child` be the dollar value of the UBI transfer to each household per dependent child under 18. Let `ubi_adult` and `ubi_senior` be the dollar value of UBI transfer to each household per adult between ages 18 and 65 and per senior over 65, respectively. And let `ubi_max` be the maximum UBI benefit per household.

  ```{math}
  :label: EqUBIubi_dol_jst0
    \begin{split}
      ubi^{\$}_{j,s,t=0} = \min\Bigl(&\texttt{ubi_max}, \\
      &\texttt{ubi_child} * \texttt{ubi_num_child_mat}_{j,s} + \\
      &\texttt{ubi_adult} * \texttt{ubi_num_adult_mat}_{j,s} + \\
      &\texttt{ubi_senior} * \texttt{ubi_num_senior_mat}_{j,s}\Bigr) \quad\forall j,s
    \end{split}
  ```

  The rest of the time periods of the household UBI transfer and the respective steady-states are determined by whether the UBI is growth adjusted or not as given in the `ubi_growthadjust` Boolean parameter. The following two sections cover these two cases.


(SecUBI_NonGrowthAdj)=
## UBI specification not adjusted for economic growth

  A non-growth adjusted UBI (`ubi_growthadjust=False`) is one in which the initial nonstationary dollar-valued $t=0$ UBI matrix $ubi^{\$}_{j,s,t=0}$ does not grow, while the economy's long-run growth rate is $g_y$ for the most common parameterization where the long-run growth rate is positive $g_y>0$.

  ```{math}
  :label: EqUBIubi_dol_NonGrwAdj_jst
    ubi^{\$}_{j,s,t} = ubi^{\$}_{j,s,t=0} \quad\forall j,s,t
  ```

  As described in Chapter {ref}`Chap_Stnrz`, the stationarized UBI transfer to each household $\hat{ubi}_{j,s,t}$ is the nonstationary transfer divided by the growth rate since the initial period.



  When the long-run economic growth rate is positive $g_y>0$ the steady-state stationary UBI household transfer is zero $\overline{ubi}_{j,s}=0$ for all lifetime income groups $j$ and ages $s$.

  In this case, the nonstationaryvalues in the household UBI time series matrix $ubi_{j,s,t}$ arePut description of non-growth-adjusted specification here. Note that in this case, if $g_y<0$, then the $ubi_{j,s,t}$ must be stationarized at some point before the steady-state $T$. This does not need to happen in the more common cases where $g_y\geq 0$ because the steady-state value of the UBI matrix in those cases is $\hat{ubi}_{j,s,t}=0$ for all $j$, $s$, and $t\geq T$.


(SecUBI_GrowthAdj)=
## UBI specification adjusted for economic growth

  Put description of growth-adjusted specification here.


(SecUBIfootnotes)=
## Footnotes

[^HOH_age_dist_note]: DeBacker and Evans compared the `OG-USA` age demographics $\hat{\omega}_{s,t}$ with the respective age demographics in Tax Policy Center's microsimulation model and in [Tax-Calculator](https://github.com/PSLmodels/Tax-Calculator)'s microsimulation model. The latter two microsimulation models' age demographics are based on head of household tax filer age distributions, whereas `OG-USA`'s demographics are based on the population age distribution.
