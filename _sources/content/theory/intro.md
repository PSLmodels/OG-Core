(Chap_Intro)=
# Model Overview

The overlapping generations model is a workhorse of dynamic fiscal analysis. `OG-USA` is dynamic in that households in the model make consumption, savings, and labor supply decisions based on their expectations over their entire lifetime, not just the current period. Because `OG-USA` is a general equilibrium model, behavioral changes by households and firms can cause macroeconomic variables and prices to adjust. This characteristic has recently become a required component of fiscal policy analysis in the United States.[^dynscore_note]

But the main characteristic that differentiates the overlapping generations model from other dynamic general equilibrium models is its realistic modeling of the finite lifetimes of individuals and the cross-sectional age heterogeneity that exists in the economy. One can make a strong case that age heterogeneity and income heterogeneity are two of the main sources of diversity that explain much of the behavior in which we are interested for policy analysis.

`OG-USA` can be summarized as having the following characteristics.

* Households
    * overlapping generations of finitely lived households
    * households are forward looking and see to maximize their expected lifetime utility, which is a function of consumption, labor supply, and bequests
    * households choose consumption, savings, and labor supply every period.
    * The only uncertainty households face is with respect to their mortality risk
    * realistic demographics: mortality rates, fertility rates, immigration rates, population growth, and population distribution dynamics
    * heterogeneous lifetime income groups within each age cohort, calibrated from U.S. tax data
    * incorporation of detailed household tax data from `Tax-Calculator` microsimulation model
    * calibrated intentional and unintentional bequests by households to surviving generations
* Firms
    * representative perfectly competitive firm maximizes static profits with general CES production function by choosing capital and labor demand
    * exogenous productivity growth is labor augmenting technological change
    * firms face a corporate income tax as well as various depreciation deductions and tax treatments
* Government
    * government collects tax revenue from households and firms
    * government distributes transfers to households
    * government spends resources on public goods
    * government can run deficits and surpluses
    * a stabilization rule (budget closure rule) must be implemented at some point in the time path if government debt is growing at a rate permanently different from GDP.
* Aggregate, market clearing, and international
    * Aggregate model is deterministic (no aggregate shocks)
    * Three markets must clear: capital, labor, and goods markets


<!-- Put summary of the general incentives in the model, overall implications of the assumptions, and particularly how these interact with tax policy -->

We will update this document as more detail is added to the model. We are currently working on adding stochastic income, aggregate shocks, multiple industries, and a large open economy multi-country version of the model. There is much to do and, as any self-respecting open source project should, we welcome outside contributions.

[^dynscore_note]: For a summary of the House rule adopted in 2015 that requires dynamic scoring of significant tax legislation see [this Politico article](http://thehill.com/blogs/floor-action/house/228684-house-adopts-dynamic-scoring-rule).