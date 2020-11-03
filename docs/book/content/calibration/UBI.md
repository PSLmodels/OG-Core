(Chap_UBI)=
# Universal Basic Income (UBI)

We have included the modeling of a universal basic income (UBI) policy directly in the theory and code for OG-USA. We calculate the time series of a UBI matrix $ubi_{j,s,t}$ representing the UBI transfer to every household with head of household age $s$, lifetime income group $j$, in period $t$. We calculate the time series of this matrix from five parameters and some household composition data that we impose upon the existing demographics of OG-USA.


(SecUBIcalc)=
## Calculating UBI

  We calculate the time series of UBI household transfer in model units $ubi_{j,s,t)}$ and the time series of total UBI expenditures in model units $UBI_t$ from five parameters described in the OG-USA API (`ubi_growthadj`, `ubi_child`, `ubi_adult`, `ubi_senior`, and `ubi_max`) interfaced with the OG-USA demographic dynamics over lifetime income groups $j$ and ages $s$, and multiplied by household composition matrices from the [OG-USA-calibration](https://github.com/PSLmodels/OG-USA-Calibration) repository.

  From the [OG-USA-calibration](https://github.com/PSLmodels/OG-USA-Calibration) repository, we have an $S\times J$ matrix `ubi_num_child_mat`$_{j,s}$ for the number of children under age 18 by lifetime ability group $j$ and age $s$ of head of household. Because the children under 18 are not economically active in the model, we do not have to adjust the size of this matrix.

  We also get from the [OG-USA-calibration](https://github.com/PSLmodels/OG-USA-Calibration) repository $S \times J$ matrices `ubi_num_adult_mat`$_{j,s}$ and `ubi_num_senior_mat`$_{j,s}$ that represent the number of adults between the ages of 18 and 65 and the number of seniors over 65, respectively, in each household with head of household in lifetime income group $j$ and age $s$. Because the number of adults in each household average more than one, simply multiplying theses by the population demographics in the OG-USA model $\lambda_j \hat{\omega}_{s,t}$ will result in too many adults. So we adjust the numbers in these `ubi_num_adult_mat`$_{j,s}$ and `ubi_num_senior_mat`$_{j,s}$ matrices so that their implied average adults in the the first period of the model (multiplied by the head of household OG-USA demographics $\lambda_j \hat{\omega}_{s,t=0}$) average to one. We do this by multiplying both matrices by the $factor_{ubi}$ that sets the average to one.

  ```{math}
  :label: EqUBInumadultsenior_hat
    \begin{split}
      &\hat{\texttt{ubi_num_adult_mat}}_{j,s} = factor_{ubi}\bigl(\texttt{ubi_num_adult_mat}_{j,s}\bigr) \\
      \text{and}\quad &\hat{\texttt{ubi_num_senior_mat}}_{j,s} = factor_{ubi}\bigl(\texttt{ubi_num_senior_mat}_{j,s}\bigr) \quad\forall j,s \\
      \text{s.t.}\quad &\sum_{s=E+1}^{E+S}\sum_{j=1}^J \lambda_j\hat{\omega}_{s,t=0}factor_{ubi}\bigl(\texttt{ubi_num_adult_mat}_{j,s} + \texttt{ubi_num_senior_mat}_{j,s}\bigr) = 1 \\
      \text{or}\quad &factor_{ubi} = \frac{1}{\sum_{s=E+1}^{E+S}\sum_{j=1}^J \lambda_j\hat{\omega}_{s,t=0}\bigl(\texttt{ubi_num_adult_mat}_{j,s} + \texttt{ubi_num_senior_mat}_{j,s}\bigr)}
    \end{split}
  ```

  It is important to note that we need the OG-USA head of household demographics $\lambda_j\hat{\omega}_{s,t=0}$ to solve for the $\hat{\texttt{ubi_num_adult_mat}}_{j,s}$ and $\hat{\texttt{ubi_num_senior_mat}}_{j,s}$ matrices in {eq}`EqUBInumadultsenior_hat`. Now we can solve for the dollar-valued (as opposed to model-unit-valued) UBI transfer to each household in the first period $ubi^{\$}_{j,s,t=0}$ in the following way. Let the parameter `ubi_child` be the dollar value of the UBI transfer to each household per dependent child under 18. Let `ubi_adult` and `ubi_senior` be the dollar value of UBI transfer to each household per adult between ages 18 and 65 and per senior over 65, respectively. And let `ubi_max` be the maximum UBI benefit per household.

  ```{math}
  :label: EqUBIubi_dol_jst0
    \begin{split}
      ubi^{\$}_{j,s,t=0} = \min\Bigl(&\texttt{ubi_max}, \\
      &\texttt{ubi_child} * \texttt{ubi_num_child_mat}_{j,s} + \\
      &\texttt{ubi_adult} * \hat{\texttt{ubi_num_adult_mat}}_{j,s} + \\
      &\texttt{ubi_senior} * \hat{\texttt{ubi_num_senior_mat}}_{j,s}\Bigr) \quad\forall j,s
    \end{split}
  ```


(SecUBI_NonGrowthAdj)=
## UBI specification not adjusted for economic growth

  A non-growth adjusted UBI is one in which the initial $t=0$ UBI matrix $ubi_{j,s,t=0}$ does not grow, while the economy's long-run growth rate is $g_y$. When the long-run economic growth rate is positive $g_y>0$ the steady-state

  Put description of non-growth-adjusted specification here. Note that in this case, if $g_y<0$, then the $ubi_{j,s,t}$ must be stationarized at some point before the steady-state $T$. This does not need to happen in the more common cases where $g_y\geq 0$ because the steady-state value of the UBI matrix in those cases is $\hat{ubi}_{j,s,t}=0$ for all $j$, $s$, and $t\geq T$.


(SecUBI_GrowthAdj)=
## UBI specification adjusted for economic growth

  Put description of growth-adjusted specification here.
