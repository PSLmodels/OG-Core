Model parameters
=================

This section contains documentation of the models parameters in a format
that is easy to search and print.
The model parameters are grouped here roughly in line with the
organization of the theory in the `OG-Core` documentation.



## Household Parameters

### Behavioral Assumptions

####  `frisch`  
_Description:_ Frisch elasticity of labor supply.  
_Notes:_ See Altonji (JPE, 1986) and Peterman (Econ Inquiry, 2016) for estimates of the Frisch elasticity.  
_Value Type:_ float  
_Valid Range:_ min = 0.2 and max = 0.62  
_Out-of-Range Action:_ error  


####  `beta_annual`  
_Description:_ Annual rate of time preference for households.  
_Notes:_ Default value from Carroll (JME, 2009).  Allows for each type to have a different value.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 0.9999  
_Out-of-Range Action:_ error  


####  `sigma`  
_Description:_ Coefficient of relative risk aversion in household utility function.  
_Notes:_ Default value from Attanasio, Banks, Meghir and Weber (JEBS, 1999).  
_Value Type:_ float  
_Valid Range:_ min = 1.0 and max = 2.0  
_Out-of-Range Action:_ error  


####  `alpha_c`  
_Description:_ Share parameters for each good in the composite consumption good.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `chi_b`  
_Description:_ Household utility weight on bequests.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 10000.0  
_Out-of-Range Action:_ error  


####  `chi_n`  
_Description:_ Household utility weight on on disutility of labor supply.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 10000.0  
_Out-of-Range Action:_ error  


### Endowments

####  `e`  
_Description:_ Effective labor hours matrix.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 9e+99  
_Out-of-Range Action:_ error  


####  `ltilde`  
_Description:_ Total time endowment of household.  
_Notes:_ Can be normalized to 1.0 without loss of generality.  
_Value Type:_ float  
_Valid Range:_ min = 0.01 and max = 5.0  
_Out-of-Range Action:_ error  


####  `alpha_RM_1`  
_Description:_ Exogenous ratio of aggregate remittances to GDP in current period (t=1).  
_Value Type:_ float  
_Valid Range:_ min = -0.05 and max = 0.4  
_Out-of-Range Action:_ error  


####  `alpha_RM_T`  
_Description:_ Exogenous ratio of aggregate remittances to GDP in the long run/steady state (t>T).  
_Value Type:_ float  
_Valid Range:_ min = -0.05 and max = 0.4  
_Out-of-Range Action:_ error  


####  `g_RM`  
_Description:_ Growth rate of aggregate remittances between periods tG1 and tG2  
_Value Type:_ float  
_Valid Range:_ min = -0.02 and max = 0.15  
_Out-of-Range Action:_ error  


####  `eta_RM`  
_Description:_ Matrix that allocates aggregate remittances across households.  
_Notes:_ This SxJ matrix gives the fraction of total aggregate remittances going to each age s and ability type j.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `zeta`  
_Description:_ Matrix that describes the bequest distribution process.  
_Notes:_ This SxJ matrix gives the fraction of total aggregate bequests going to each age s and ability type j.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `use_zeta`  
_Description:_ Indicator variable for whether or not to use the zeta matrix to distribute bequests.  
_Value Type:_ bool  


### Model Dimensions

####  `S`  
_Description:_ Maximum number of periods of economic life of model agent.  
_Value Type:_ int  
_Valid Range:_ min = 3 and max = 80  
_Out-of-Range Action:_ error  


####  `J`  
_Description:_ Number of different labor productivity types of agents.  
_Value Type:_ int  
_Valid Range:_ min = 1 and max = 100  
_Out-of-Range Action:_ error  


####  `T`  
_Description:_ Number of periods until it is assumed that the model reaches its steady state.  
_Value Type:_ int  
_Valid Range:_ min = 3 and max = 1000  
_Out-of-Range Action:_ error  


####  `I`  
_Description:_ Number of different consumption goods.  
_Value Type:_ int  
_Valid Range:_ min = 1 and max = 50  
_Out-of-Range Action:_ error  


####  `lambdas`  
_Description:_ Fraction of population of each labor productivity type.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `starting_age`  
_Description:_ Age at which households become economically active.  
_Value Type:_ int  
_Valid Range:_ min = 0 and max = ending_age  
_Out-of-Range Action:_ error  


####  `ending_age`  
_Description:_ Maximum age a household can live until.  
_Value Type:_ int  
_Valid Range:_ min = starting_age and max = 120  
_Out-of-Range Action:_ error  


## Firm Parameters

### Capital Accumulation

####  `delta_annual`  
_Description:_ Annual rate of economic depreciation of capital.  
_Notes:_ Approximately the value from Kehoe calibration exercise: http://www.econ.umn.edu/~tkehoe/classes/calibration-04.pdf.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


### Model Dimensions

####  `M`  
_Description:_ Number of different production industries.  
_Value Type:_ int  
_Valid Range:_ min = 1 and max = 50  
_Out-of-Range Action:_ error  


### Production Function

####  `gamma`  
_Description:_ Capital's share of output in firm production function.  
_Notes:_ Historical value in U.S. is about 0.33, but Elsby, Hobijn, and Sahin (BPEA, 2013) find capital's share is increasing, so default value is above this.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `gamma_g`  
_Description:_ Public capital's share of output in firm production function.  
_Value Type:_ float  


####  `epsilon`  
_Description:_ Elasticity of substitution among private capital, public capital, and labor in firm production function.  
_Notes:_ If epsilon=1, then production function is Cobb-Douglas.  If epsilon=0, then production function is perfect substitutes.  
_Value Type:_ float  
_Valid Range:_ min = 0.2 and max = 10.0  
_Out-of-Range Action:_ error  


####  `io_matrix`  
_Description:_ Input-output matrix used to map production outputs into consumption goods using a fixed coefficient model.  This matrix has dimensions I x M, where I is the number of distinct consumption goods and M is the number of distinct production goods.  The sum each row of this matrix must be 1.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `Z`  
_Description:_ Total factor productivity in firm production function.  Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 5.0  
_Out-of-Range Action:_ error  


## Government Parameters

####  `delta_g_annual`  
_Description:_ Annual rate of economic depreciation of public capital.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


## Fiscal Policy Parameters

### Fiscal Closure Rules

####  `tG1`  
_Description:_ Number of model periods until budget closure rule kicks in.  
_Notes:_ It may make sense for this tG1 to exceed the length of the budget window one is looking at.  But it must also not be too late into the time path that a steady-state won't be reached by time T.  
_Value Type:_ int  
_Valid Range:_ min = 0 and max = tG2  
_Out-of-Range Action:_ error  


####  `tG2`  
_Description:_ Number of model periods until budget closure ends and budgetary variables jump to steady-state values.  
_Notes:_ The number of periods until the closure rule ends must not exceed T, the time at which the model is assumed to reach the steady-state.  
_Value Type:_ int  
_Valid Range:_ min = tG1 and max = T  
_Out-of-Range Action:_ error  


####  `rho_G`  
_Description:_ Speed of convergence to a stable government budget deficit for periods [tG1, tG2-1].  
_Notes:_ Lower rho_G => slower convergence.  
_Value Type:_ float  
_Valid Range:_ min = 0.01 and max = 0.9  
_Out-of-Range Action:_ error  


####  `debt_ratio_ss`  
_Description:_ Debt to GDP ratio to be achieved in the steady state.  
_Notes:_ Note that depending upon the tax policy parameters, some debt to GDP ratios may not be achievable with positive government spending.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 5.0  
_Out-of-Range Action:_ error  


### Government Pension Parameters

####  `retirement_age`  
_Description:_ Age at which agents can collect Social Security benefits. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Value Type:_ int  
_Valid Range:_ min = 0 and max = 100  
_Out-of-Range Action:_ error  


####  `pension_system`  
_Description:_ Pension system.  
_Value Type:_ str  
_Valid Choices:_['US-Style Social Security', 'Defined Benefits', 'Notional Defined Contribution', 'Points System']  


####  `tau_p`  
_Description:_ Pension system contribution tax rate under a notional defined contribution system.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `indR`  
_Description:_ Adjustment for survivor benefits under a notional defined contribution system.  
_Notes:_ indR = 0.0 for no survivor benefits, indR = 0.5 for survivor benefits  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `k_ret`  
_Description:_ Adjustment for frequency of pension payments under a notional defined contribution system.  
_Notes:_ k = 0.5 - (6/13n), where n is the number of payments per year  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `alpha_db`  
_Description:_ Replacement rate under a defined contribution system.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `vpoint`  
_Description:_ The value of a point under a points system pension.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `yr_contrib`  
_Description:_ Number of years of contributions made to defined benefits pension system.  
_Notes:_ Since there is not exit from the labor force in the model, the number of years of contributions is set exogenously.  
_Value Type:_ int  
_Valid Range:_ min = 0 and max = retirement_age  
_Out-of-Range Action:_ error  


####  `avg_earn_num_years`  
_Description:_ Number of years used to compute average earnings for pension benefits.  
_Notes:_ US-styel Social Security AIME is computed using average earnings from the highest earnings years for the number of years specified here.  
_Value Type:_ int  
_Valid Range:_ min = 1 and max = retirement_age  
_Out-of-Range Action:_ error  


####  `AIME_bkt_1`  
_Description:_ Upper bound to the first average index monthly earnings (AIME) bracket.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 9999999999.0  
_Out-of-Range Action:_ error  


####  `AIME_bkt_2`  
_Description:_ Upper bound to the second average index monthly earnings (AIME) bracket.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 9999999999999.9  
_Out-of-Range Action:_ error  


####  `PIA_rate_bkt_1`  
_Description:_ The rate used to determine the primary insurance amount (PIA) in the first bracket for average index monthly earnings (AIME).  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `PIA_rate_bkt_2`  
_Description:_ The rate used to determine the primary insurance amount (PIA) in the second bracket for average index monthly earnings (AIME).  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `PIA_rate_bkt_3`  
_Description:_ The rate used to determine the primary insurance amount (PIA) in the third bracket for average index monthly earnings (AIME).  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `PIA_maxpayment`  
_Description:_ The maximum primary insurance amount (PIA) payment.  
_Value Type:_ float  
_Valid Range:_ min = PIA_minpayment and max = 9999999999999.9  
_Out-of-Range Action:_ error  


####  `PIA_minpayment`  
_Description:_ The minimum primary insurance amount (PIA) payment.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = PIA_maxpayment  
_Out-of-Range Action:_ error  


####  `replacement_rate_adjust`  
_Description:_ Adjustment to the Social Security retirement benefits replacement rate for year t and household type j. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Notes:_ This parameter willy only vary along the time path.  It is assumed to be one in the steady state.  The steady state retirement rate can be adjusted by changing the parameters of the retirement benefits function.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 10.0  
_Out-of-Range Action:_ error  


### Spending

####  `alpha_T`  
_Description:_ Exogenous ratio of government transfers to GDP when budget balance is false. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 0.3  
_Out-of-Range Action:_ error  


####  `alpha_G`  
_Description:_ Exogenous ratio of government spending to GDP when budget balance is false. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 0.3  
_Out-of-Range Action:_ error  


####  `alpha_I`  
_Description:_ Exogenous fraction of GDP that goes towards government investment in infrastructure (public capital) when balanced budget is false.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 0.3  
_Out-of-Range Action:_ error  


####  `alpha_bs_T`  
_Description:_ Proportional adjustment to government transfers relative to baseline amount when budget balance is true. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 9.9999e+103  
_Out-of-Range Action:_ error  


####  `alpha_bs_G`  
_Description:_ Proportional adjustment to government consumption expenditures relative to baseline amount when budget balance is true. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 9.9999e+103  
_Out-of-Range Action:_ error  


####  `alpha_bs_I`  
_Description:_ Proportional adjustment to  infrastructure (public capital) spending relative to baseline amount when balanced budget is true.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 9.9999e+103  
_Out-of-Range Action:_ error  


####  `ubi_growthadj`  
_Description:_ Boolean indicator variable for whether or not to growth adjust universal basic income transfer.  
_Notes:_ When this Boolean = True, the ubi_{j,s,t} matrix is stationary because it is multiplied by e^{g_y t} in the nonstationary household budget constraint. When this Boolean = False, the ubi_{j,s,t} is not multiplied by e^{g_y t} and the steady-state value is a matrix of zeros.  
_Value Type:_ bool  


####  `ubi_nom_017`  
_Description:_ Dollar (nominal) amount of universal basic income (UBI) per child age 0 through 17.  
_Notes:_ The universal basic income (UBI) per child ages 0 through 17 can take positive or negative values (e.g., a negative value may be used to implement a head tax).  
_Value Type:_ float  
_Valid Range:_ min = -99000000000.0 and max = 99000000000.0  
_Out-of-Range Action:_ error  


####  `ubi_nom_1864`  
_Description:_ Dollar (nominal) amount of universal basic income (UBI) per adult age 18 to 64.  
_Notes:_ The universal basic income (UBI) per adult age 18 through 64 can take positive or negative values (e.g., a negative value may be used to implement a head tax).  
_Value Type:_ float  
_Valid Range:_ min = -99000000000.0 and max = 99000000000.0  
_Out-of-Range Action:_ error  


####  `ubi_nom_65p`  
_Description:_ Dollar (nominal) amount of universal basic income (UBI) per adult age 65 and over.  
_Notes:_ The universal basic income (UBI) per adult age 65 can take positive or negative values (e.g., a negative value may be used to implement a head tax).  
_Value Type:_ float  
_Valid Range:_ min = -99000000000.0 and max = 99000000000.0  
_Out-of-Range Action:_ error  


####  `ubi_nom_max`  
_Description:_ Maximum dollar (nominal) amount of universal basic income (UBI) per household when all UBI allowances are added up.  
_Notes:_ The maximum dollar (nominal) amount of universal basic income (UBI) per household can range from $0 per year to $40,000 per year ($3,333 per month).  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 40000.0  
_Out-of-Range Action:_ error  


####  `eta`  
_Description:_ Matrix that allocated government transfers across households.  
_Notes:_ This SxJ matrix gives the fraction of total aggregate transfers going to each age s and ability type j.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `budget_balance`  
_Description:_ Flag to balance government budget in each period.  
_Value Type:_ bool  


####  `baseline_spending`  
_Description:_ Flag for use in reform simulations to keep level of government spending and transfers constant between baseline and reform runs.  
_Value Type:_ bool  


### Taxes

####  `cit_rate`  
_Description:_ Corporate income tax rate.  Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Notes:_ This is the top marginal corporate income tax rate.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 0.99  
_Out-of-Range Action:_ error  


####  `c_corp_share_of_assets`  
_Description:_ Share of total business assets held in C corporations  
_Notes:_ This affects the effective corporate income tax rate on the representative firm in the model  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `adjustment_factor_for_cit_receipts`  
_Description:_ Adjustment to the statutory CIT rate to match effective rate on corporations  
_Notes:_ This adjustment is necessary to match corporate income tax receipts found in the data.  It is computed as the ratio of the CIT/GDP from US data to CIT/GDP in the model (with this parameter =1)  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 2.0  
_Out-of-Range Action:_ error  


####  `inv_tax_credit`  
_Description:_ Investment tax credit rate.  This credit reduced the cost of new investment by the specified rate.  
_Notes:_ This credit reduced the cost of new investment by the specified rate  
_Value Type:_ float  
_Valid Range:_ min = -1.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `tau_c`  
_Description:_ Consumption tax rate.  Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Notes:_ This policy parameter represents the effective consumption tax rate from sales taxes, VATs, and excise taxes by consumption good. Tax rates cab vary over time. It is thus a TxI array.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 5.0  
_Out-of-Range Action:_ error  


####  `delta_tau_annual`  
_Description:_ Annual rate of depreciation of capital for tax purposes.  Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Notes:_ Cost-of-Capital-Calculator can help to calibrate this parameter.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `h_wealth`  
_Description:_ Numerator in wealth tax function. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Value Type:_ float  
_Valid Range:_ min = 1e-08 and max = 10.0  
_Out-of-Range Action:_ error  


####  `m_wealth`  
_Description:_ Term in denominator in wealth tax function. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Notes:_ Must be positive to avoid division by zero when computing ETR and MTR for wealth tax  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 10.0  
_Out-of-Range Action:_ error  


####  `p_wealth`  
_Description:_ Scale parameter in wealth tax function. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Notes:_ Set to 0 to have no wealth tax.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 10.0  
_Out-of-Range Action:_ error  


####  `tau_bq`  
_Description:_ Linear tax rate on bequests. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 0.99  
_Out-of-Range Action:_ error  


####  `tau_payroll`  
_Description:_ Linear payroll tax rate. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Notes:_ Set to zero as default since tax functions include income and payroll taxes.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 0.99  
_Out-of-Range Action:_ error  


####  `constant_rates`  
_Description:_ Flag to use linear tax functions.  
_Value Type:_ bool  


####  `zero_taxes`  
_Description:_ Flag to run model without any individual income taxes.  
_Value Type:_ bool  


####  `analytical_mtrs`  
_Description:_ Flag to use analytically derived marginal tax rates in tax functions.  
_Value Type:_ bool  


####  `age_specific`  
_Description:_ Flag to use analytically derived marginal tax rates in tax functions.  
_Value Type:_ bool  


####  `tax_func_type`  
_Description:_ Functional form for individual income tax functions.  
_Value Type:_ str  
_Valid Choices:_['DEP', 'DEP_totalinc', 'GS', 'HSV', 'linear', 'mono', 'mono2D']  


####  `labor_income_tax_noncompliance_rate`  
_Description:_ Labor income tax noncompliance rate, the ratio of taxes paid to taxes owed.  
_Notes:_ Can be specified to vary by type J and over year T  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `capital_income_tax_noncompliance_rate`  
_Description:_ Capital income tax noncompliance rate, the ratio of taxes paid to taxes owed.  
_Notes:_ Can be specified to vary by type J and over year T  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `etr_params`  
_Description:_ Effective tax rate function parameters.  
_Value Type:_ float  
_Valid Range:_ min = -100.0 and max = 10000.0  
_Out-of-Range Action:_ error  


####  `mtrx_params`  
_Description:_ Marginal tax rate on labor income function parameters.  
_Value Type:_ float  
_Valid Range:_ min = -100.0 and max = 10000.0  
_Out-of-Range Action:_ error  


####  `mtry_params`  
_Description:_ Marginal tax rate on capital income function parameters.  
_Value Type:_ float  
_Valid Range:_ min = -100.0 and max = 20000.0  
_Out-of-Range Action:_ error  


####  `frac_tax_payroll`  
_Description:_ Fraction of IIT+Payroll tax revenue that is attributable to payroll taxes.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `initial_debt_ratio`  
_Description:_ Debt to GDP ratio in the initial period.  
_Notes:_ Should be calibrated based on government debt held by the public to GDP in the model start year.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 2.0  
_Out-of-Range Action:_ error  


####  `initial_Kg_ratio`  
_Description:_ Government capital (aka infrastructure) to GDP ratio in the initial period.  
_Notes:_ Should be calibrated based on the value of public infrastructure to GDP in the model start year.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 50.0  
_Out-of-Range Action:_ error  


####  `r_gov_scale`  
_Description:_ Parameter to scale the market interest rate to find interest rate on government debt.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 4.0  
_Out-of-Range Action:_ error  


####  `r_gov_shift`  
_Description:_ Parameter to shift the market interest rate to find interest rate on government debt.  
_Value Type:_ float  
_Valid Range:_ min = -0.3 and max = 0.3  
_Out-of-Range Action:_ error  


## Open Economy Parameters

####  `world_int_rate_annual`  
_Description:_ Exogenous annual world interest rate to be used when modeling a small open economy. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Value Type:_ float  
_Valid Range:_ min = 0.01 and max = 0.08  
_Out-of-Range Action:_ error  


####  `initial_foreign_debt_ratio`  
_Description:_ Share of government debt held by foreigners in model start year.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `zeta_D`  
_Description:_ Share of new debt issues from government that are purchased by foreigners. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Notes:_ This value is about 0.4 historically for the U.S., set to zero to have closed econ as default.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `zeta_K`  
_Description:_ Share of excess demand for capital that is supplied by foreigners. Set value for base year, click '+' to add value for next year.  All future years not specified are set to last value entered.  
_Notes:_ Set to 0.1 when running large-open verion of US economy, but hard to find data to pin this down.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


## Economic Assumptions

####  `g_y_annual`  
_Description:_ Growth rate of labor augmenting technological change.  
_Value Type:_ float  
_Valid Range:_ min = -0.01 and max = 0.08  
_Out-of-Range Action:_ error  


## Demographic Parameters

####  `constant_demographics`  
_Description:_ Use constant demographics.  
_Notes:_ This boolean allows one to use empirical mortality rates, but keep the population distribution constant across periods.  Note that immigration is shut off when this is true.  
_Value Type:_ bool  


####  `omega`  
_Description:_ Population distribution (fraction at each age) over the time path.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `omega_SS`  
_Description:_ Population distribution (fraction at each age) in the steady-state.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `omega_S_preTP`  
_Description:_ Population distribution (fraction at each age) in the period before the transition path begins.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `g_n`  
_Description:_ Population growth rate over the time path.  
_Value Type:_ float  
_Valid Range:_ min = -1.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `g_n_ss`  
_Description:_ Population growth rate in the steady-state.  
_Value Type:_ float  
_Valid Range:_ min = -1.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `imm_rates`  
_Description:_ Immigration rates over the time path.  
_Value Type:_ float  
_Valid Range:_ min = -1.0 and max = 1.0  
_Out-of-Range Action:_ error  


####  `rho`  
_Description:_ Age-specific mortality rates.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 1.0  
_Out-of-Range Action:_ error  


## Model Solution Parameters

####  `nu`  
_Description:_ Parameter for convergence rate of functional iteration.  
_Value Type:_ float  
_Valid Range:_ min = 0.01 and max = 0.5  
_Out-of-Range Action:_ error  


####  `SS_root_method`  
_Description:_ Root finding algorithm for outer loop of the SS solution.  
_Notes:_ Uses scipy.optimize.root, please see scipy documentation for description of methods. Note that some methods may require more arguments than are in the function calls in SS.py and TPI.py and will therefore break without modifications of the source code.  
_Value Type:_ str  
_Valid Choices:_['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']  


####  `FOC_root_method`  
_Description:_ Root finding algorithm for solving household first order conditions.   
_Notes:_ Uses scipy.optimize.root, please see scipy documentation for description of methods. Note that some methods may require more arguments than are in the function calls in SS.py and TPI.py and will therefore break without modifications of the source code.  
_Value Type:_ str  
_Valid Choices:_['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']  


####  `maxiter`  
_Description:_ Max iterations for time path iteration.  
_Value Type:_ int  
_Valid Range:_ min = 2 and max = 500  
_Out-of-Range Action:_ error  


####  `mindist_SS`  
_Description:_ Tolerance for convergence of steady state solution.  
_Value Type:_ float  
_Valid Range:_ min = 1e-13 and max = 0.001  
_Out-of-Range Action:_ error  


####  `mindist_TPI`  
_Description:_ Tolerance for convergence of time path solution.  
_Value Type:_ float  
_Valid Range:_ min = 1e-13 and max = 0.001  
_Out-of-Range Action:_ error  


####  `RC_SS`  
_Description:_ Tolerance for the resource constraint in the steady state solution.  
_Value Type:_ float  
_Valid Range:_ min = 1e-13 and max = 0.001  
_Out-of-Range Action:_ error  


####  `RC_TPI`  
_Description:_ Tolerance for the resource constraint in the time path solution.  
_Value Type:_ float  
_Valid Range:_ min = 1e-13 and max = 0.01  
_Out-of-Range Action:_ error  


####  `reform_use_baseline_solution`  
_Description:_ Whether or not the baseline SS solution is used for starting values when solving the reform.  
_Value Type:_ bool  


####  `initial_guess_r_SS`  
_Description:_ Initial guess of r for the SS solution.  
_Value Type:_ float  
_Valid Range:_ min = 0.01 and max = 0.25  
_Out-of-Range Action:_ error  


####  `initial_guess_TR_SS`  
_Description:_ Initial guess of TR for the SS solution. This value is in model units and can therefore be any large positive number. We may have to adjust the maximum for this parameter from time to time.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 2.5  
_Out-of-Range Action:_ error  


####  `initial_guess_factor_SS`  
_Description:_ Initial guess of factor for the SS solution.  
_Value Type:_ float  
_Valid Range:_ min = 1.0 and max = 500000  
_Out-of-Range Action:_ error  


## Other Parameters

####  `start_year`  
_Description:_ Calendar year in which to start model analysis.  
_Notes:_ Calendar year for initial model period  
_Value Type:_ int  
_Valid Range:_ min = 2013 and max = 2100  
_Out-of-Range Action:_ error  


####  `mean_income_data`  
_Description:_ Mean income (in current dollars) from the microdata used for tax function estimation.  
_Value Type:_ float  
_Valid Range:_ min = 0.0 and max = 9.9e+100  
_Out-of-Range Action:_ error  
