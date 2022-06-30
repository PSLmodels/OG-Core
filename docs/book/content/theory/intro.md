(Chap_Intro)=
# Model Overview

The overlapping generations model is a workhorse of dynamic fiscal analysis. The `OG-Core` model is dynamic in that households in the model make consumption, savings, and labor supply decisions based on their expectations over their entire lifetime, not just the current period. Because `OG-Core` is a general equilibrium model, behavioral changes by households and firms can cause macroeconomic variables and prices to adjust. This characteristic has recently become a required component of fiscal policy analysis in the United States.[^dynscore_note]

The main characteristic that differentiates the overlapping generations model from other dynamic general equilibrium models is its realistic modeling of the finite lifetimes of individuals and the cross-sectional age heterogeneity that exists in the economy. One can make a strong case that age heterogeneity and income heterogeneity are two of the main sources of diversity that explain much of the behavior in which we are interested for policy analysis.

`OG-Core` can be summarized as having the following characteristics.

* Households
    * overlapping generations of finitely lived households
    * households are forward looking and see to maximize their expected lifetime utility, which is a function of consumption, labor supply, and bequests
    * households choose consumption of $M$ different consumption goods, composite consumption, savings, and labor supply every period.
    * the only uncertainty households face is with respect to their mortality risk
    * realistic demographics: mortality rates, fertility rates, immigration rates, population growth, and population distribution dynamics
    * heterogeneous lifetime income groups within each age cohort, calibrated from U.S. tax data
        * each lifetime income group has its own discount factor $\beta_j$ following {cite}`CarrollEtAl:2017`
    * incorporation of detailed household tax data from specified microsimulation model
    * calibrated intentional and unintentional bequests by households to surviving generations
* Firms
    * the production side of the economy consists of $M$ different industries $m\in\{1,2,...M\}$
    * representative perfectly competitive firm in each industry maximizes static profits with general CES production function by choosing private capital and labor demand, taking public capital as given
    * exogenous productivity growth is labor augmenting technological change
    * firms face a corporate income tax as well as various depreciation deductions and tax treatments
    * only output from the $M$th industry can be used as investment
* Government
    * government collects tax revenue from households and firms
    * government distributes transfers to households
    * government supplies capital to the private firms' production process
    * government spends resources on public goods
    * government can run deficits and surpluses
    * a stabilization rule (budget closure rule) must be implemented at some point in the time path if government debt is growing at a rate permanently different from GDP.
* Aggregate, market clearing, and international
    * Aggregate model is deterministic (no aggregate shocks)
    * $M+2$ markets must clear: capital market, labor market, and $M$ goods markets


<!-- Put summary of the general incentives in the model, overall implications of the assumptions, and particularly how these interact with tax policy -->

We will update this document as more detail is added to the model. We are currently working on adding stochastic income, aggregate shocks, enhanced demographic transitions, more robust tax function estimation, and a large open economy multi-country version of the model. There is much to do and, as any self-respecting open source project should, we welcome outside contributions.

[^dynscore_note]: For a summary of the House rule adopted in 2015 that requires dynamic scoring of significant tax legislation in the United States, see [this Politico article](http://thehill.com/blogs/floor-action/house/228684-house-adopts-dynamic-scoring-rule).
