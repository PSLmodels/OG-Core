(Chap_MacroCalib)=
# Calibration of Macroeconomic Parameters

## Behavioral Assumptions

### Elasticity of labor supply

As we discuss in Chapter {ref}`Chap_House`, we use the elliptical disutility of labor function developed by {cite}`EvansPhillips:2017`.  We then fit the parameters of the elliptical utility function to match the marginal disutility from a constant Frisch elasticity function.  `OG-USA` users enter the constant Frisch elasticity as a parameter.  {cite}`Peterman:2016` finds a range of Frisch elasticities estimated from microeconomic and macroeconomic data.  These range from 0 to 4.  Peterman makes the case that in lifecycle models with out an extensive margin for employment, such as `OG-USA`, the  Frisch elasticity should be higher.  We take a default value of 0.4 from {cite}`Altonji:1986`.

### Intertemporal elasticity of substitution

The default value for the intertemporal elasticity of substitution, $\sigma$, is taken from {cite}`ABMW:1999`.  We set $\sigma=1.5$.

### Rate of time preference

We take our default value for the rate of time preference parameter, $\beta$ from {cite}`Carroll:2009`.  We set the value of to $\beta=0.96$ (on an annual basis).


## Economic Assumptions

As the default rate of labor augmenting technological change, $g_y$, we use a value of 3%.  The average annual growth rate in GDP per capita in the United States since 1948 is 2% per year.

## Aggregate Production Function and Capital Accumulation

Chapter {ref}`Chap_Firm` outlines the constant returns to scale, constant elasticity of substitution production function of the representative firm.  This function has two parameters; the elasticity of substitution and capital's share of output.  

### Elasticity of substitution

`OG-USA`'s default parameterization has an elasticity of substitution of one, which implies a Cobb-Douglas production function.  

### Capital's share of output

The historical value in U.S. is about 0.33, but {cite}`EHS:2013` find capital's share is increasing.  We therefore use the slightly higher value of 0.35.  Note that the mean of capital's share of income from 1950 onwards is 0.38 from the Penn World Tables.


## Open Economy Parameters

### Foreign holding of government debt in the initial period

The path of foreign holding of domestic debt is endogenous, but the initial period stock of debt held by foreign investors is exogenous.  We set this parameter, `initial_foreign_debt_ratio` to 0.4, consistent with data from the Financial Accounts of the United States for 2019.


### Foreign purchases of newly issued debt

We set $zeta_D = 0.4$.  This is the average share of foreign purchases of newly issued government debt found from the Financial Accounts of the United States.

### Foreign holdings of excess capital

We set $zeta_K = 0.1$.  


## Government Debt, Spending and Transfers

### Government Debt

The path of government debt is endogenous.  But the initial value is exogenous.  To avoid converting between model units and dollars, we calibrate the initial debt to GDP ratio, rather than the dollar value of the debt.  This is the model parameter $alpha_D$.  We compute this from the ratio of publicly held debt outstanding to GDP.  Based on 2019 values, this gives us a ratio of 0.78.

### Aggregate transfers

Aggregate (non-Social Security) transfers to households are set as a share of GDP with the parameter $\alpha_T$.  Using the [OMB Fiscal Year 2017 Historical Tables](https://www.whitehouse.gov/sites/default/files/omb/budget/fy2017/assets/hist.pdf), we define transfers as:
    $$Transfers = VA Benefits and Services (700) + Income Security (600) + Medicare (570) + Healthcare Services (551)$$

We exclude Social Security from transfers since it is modeled specifically. With this definition, the share of transfers to GDP in 2015 is 0.09.

### Government expenditures

Government spending on goods and services are also set as a share of GDP with the parameter $\alpha_G$.  Using the [OMB Fiscal Year 2017 Historical Tables](https://www.whitehouse.gov/sites/default/files/omb/budget/fy2017/assets/hist.pdf), we define government spending as:
    $$Government Spending = Total Outlays - Transfers - Net Interest on Debt - Social Security$$


