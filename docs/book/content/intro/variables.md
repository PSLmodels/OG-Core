Model variables
=================

This section summarizes variables output from the model in a format
that is easy to search and print.
The model variables are grouped together in common categories and are
available in both the dictionaries of steady-state and transition-path
output, unless noted otherwise in these docs.



## Economic Aggregates

####  `Y`  
_Description:_ Aggregate output, GDP  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `B`  
_Description:_ Aggregate household savings  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `K`  
_Description:_ Aggregate capital stock  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `K_f`  
_Description:_ Aggregate capital stock owned by foreign investors  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `K_d`  
_Description:_ Aggregate capital stock owned by domestic investors  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `L`  
_Description:_ Aggregate labor  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `C`  
_Description:_ Aggregate consumption  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `I`  
_Description:_ Aggregate investment  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `I_total`  
_Description:_ Aggregate investment, not account for immigrant investment  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `I_d`  
_Description:_ Aggregate investment for domestic sources  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `K_g`  
_Description:_ Aggregate stock of infrastructure  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `I_g`  
_Description:_ Aggregate investment in infrastructure  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `BQ`  
_Description:_ Aggregate bequests  
_TPI dimensions:_ TxJ or T (if use_zeta=True)  
_SS dimensions:_ J or scalar if use_zeta=True)  


####  `RM`  
_Description:_ Aggregate remittances  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


## Production Sector/Consumption Good Aggregates

####  `Y_m`  
_Description:_ Total output by industry  
_TPI dimensions:_ TxM  
_SS dimensions:_ M  


####  `K_m`  
_Description:_ Total capital stock by industry  
_TPI dimensions:_ TxM  
_SS dimensions:_ M  


####  `L_m`  
_Description:_ Total labor by industry  
_TPI dimensions:_ TxM  
_SS dimensions:_ M  


####  `C_i`  
_Description:_ Total consumption by consumption good  
_TPI dimensions:_ TxI  
_SS dimensions:_ I  


## Government Revenues, Outlays, and Debt

####  `TR`  
_Description:_ Aggregate government, non-pension transfer spending  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `agg_pension_outlays`  
_Description:_ Aggregate government pension outlays  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `G`  
_Description:_ Aggregate government consumption expenditures  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `UBI`  
_Description:_ Aggregate universal basic income outlays  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `total_tax_revenue`  
_Description:_ Total tax revenue  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `business_tax_revenue`  
_Description:_ Total corporate income tax revenue  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `iit_payroll_tax_revenue`  
_Description:_ Total tax revenue from personal income and payroll taxes  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `iit_tax_revenue`  
_Description:_ Total tax revenue from personal income taxes  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `payroll_tax_revenue`  
_Description:_ Total tax revenue from payroll taxes (i.e., social security contributions)  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `bequest_tax_revenue`  
_Description:_ Total tax revenue from taxes on bequests  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `wealth_tax_revenue`  
_Description:_ Total tax revenue from wealth taxes  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `cons_tax_revenue`  
_Description:_ Total tax revenue from consumption taxes  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `D`  
_Description:_ Total government debt  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `D_f`  
_Description:_ Total government debt held by foreign investors  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `D_d`  
_Description:_ Total government debt held by domestic investors  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `new_borrowing`  
_Description:_ New government borrowing, equal to budget deficit  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `debt_service`  
_Description:_ Total interest payments on government debt  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `new_borrowing_f`  
_Description:_ New government borrowing from foreign investors  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `debt_service_f`  
_Description:_ Government debt interest payments made to foreign investors  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


## Prices

####  `r`  
_Description:_ Real rate of return on private capital per model period.  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `r_gov`  
_Description:_ Real rate of return on government debt per model period.  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `r_p`  
_Description:_ Real rate of return on household investment portfolio per model period.  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `w`  
_Description:_ Wage rate per effective labor hour  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


####  `p_m`  
_Description:_ Price of output in each industry. Normalized so that price of industry M output is 1 ($p_{m,t}=1$)  
_TPI dimensions:_ TxM  
_SS dimensions:_ M  


####  `p_i`  
_Description:_ Price of output in each industry  
_TPI dimensions:_ TxI  
_SS dimensions:_ I  


####  `p_tilde`  
_Description:_ Price of composite output good  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  


## Household Variables

####  `b_sp1`  
_Description:_ Household savings  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `b_s`  
_Description:_ Household wealth (equal to savings from last period)  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `n`  
_Description:_ Household labor supply  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `c`  
_Description:_ Household consumption of composite good  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `c_i`  
_Description:_ Household consumption of differentiated consumption goods  
_TPI dimensions:_ TxSxJxI  
_SS dimensions:_ SxJxI  


####  `bq`  
_Description:_ Household bequest received  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `rm`  
_Description:_ Household remittances received  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `tr`  
_Description:_ Government transfers received by households  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `ubi`  
_Description:_ Universal basic income received by households  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `before_tax_income`  
_Description:_ Household before tax income  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `hh_taxes`  
_Description:_ Net taxes paid by households (taxes - transfers)  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `etr`  
_Description:_ Effective tax rate, income and payroll taxes  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `mtrx`  
_Description:_ Marginal tax rate on labor income  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `mtry`  
_Description:_ Marginal tax rate on capital income  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `theta`  
_Description:_ Replacement rate for US-style social security pension system.  Steady-state only.  
_TPI dimensions:_   
_SS dimensions:_ J  


## Model Scaling and Fit

####  `factor`  
_Description:_ Factor adjustment to convert from model units to local currency units to match mean income in local currency units. Steady-state only.  
_TPI dimensions:_   
_SS dimensions:_ scalar  


####  `euler_savings`  
_Description:_ Errors in household FOC for choice of savings  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `euler_laborleisure`  
_Description:_ Errors in household FOC for choice of labor  
_TPI dimensions:_ TxSxJ  
_SS dimensions:_ SxJ  


####  `resource_constraint_error`  
_Description:_ Errors in resource constraint  
_TPI dimensions:_ T  
_SS dimensions:_ scalar  
