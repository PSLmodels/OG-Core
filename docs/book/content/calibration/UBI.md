(Chap_UBI)=
# Universal Basic Income (UBI)

We have included the modeling of a universal basic income (UBI) policy directly in the theory and code for OG-USA. We calculate the time series of a UBI matrix $ubi_{j,s,t}$ representing the UBI transfer to every household with head of household age $s$, lifetime income group $j$, in period $t$. We calculate the time series of this matrix from five parameters and some household composition data that we impose upon the existing demographics of OG-USA.


(SecUBI_NonGrowthAdj)=
## UBI specification not adjusted for economic growth

  Put description of non-growth-adjusted specification here. Note that in this case, if $g_y<0$, then the $ubi_{j,s,t}$ must be stationarized at some point before the steady-state $T$. This does not need to happen in the more common cases where $g_y\geq 0$ because the steady-state value of the UBI matrix in those cases is $\hat{ubi}_{j,s,t}=0$ for all $j$, $s$, and $t\geq T$.


(SecUBI_GrowthAdj)=
## UBI specification adjusted for economic growth

  Put description of growth-adjusted specification here.


(SecUBIcalc)=
## Calculating UBI

  Put description of how to calculate time series of UBI matrix $ubi_{j,s,t}$ using five parameters, household composition data, and existing OG-USA demographics.
