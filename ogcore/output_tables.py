import numpy as np
import pandas as pd
import os
from ogcore.constants import VAR_LABELS, DEFAULT_START_YEAR
from ogcore import tax
from ogcore.utils import save_return_table, Inequality
from ogcore.utils import pct_change_unstationarized

cur_path = os.path.split(os.path.abspath(__file__))[0]


def macro_table(
    base_tpi,
    base_params,
    reform_tpi=None,
    reform_params=None,
    var_list=["Y", "C", "K", "L", "r", "w"],
    output_type="pct_diff",
    stationarized=True,
    num_years=10,
    include_SS=True,
    include_overall=True,
    start_year=DEFAULT_START_YEAR,
    table_format=None,
    path=None,
):
    """
    Create a table of macro aggregates.

    Args:
        base_tpi (dictionary): TPI output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        reform_tpi (dictionary): TPI output from reform run
        reform_params (OG-Core Specifications class): reform parameters
            object
        var_list (list): names of variable to use in table
        output_type (string): type of plot, can be:
            'pct_diff': plots percentage difference between baselien
                and reform ((reform-base)/base)
            'diff': plots difference between baseline and reform (reform-base)
            'levels': variables in model units
        stationarized (bool): whether used stationarized variables (False
            only affects pct_diff right now)
        num_years (integer): number of years to include in table
        include_SS (bool): whether to include the steady-state results
            in the table
        include_overall (bool): whether to include results over the
            entire budget window as a column in the table
        start_year (integer): year to start table
        table_format (string): format to return table in: 'csv', 'tex',
            'excel', 'json', if None, a DataFrame is returned
        path (string): path to save table to

    Returns:
        table (various): table in DataFrame or string format or `None`
            if saved to disk

    """
    assert isinstance(start_year, (int, np.integer))
    assert isinstance(num_years, (int, np.integer))
    # Make sure both runs cover same time period
    if reform_tpi is not None:
        assert base_params.start_year == reform_params.start_year
    year_vec = np.arange(start_year, start_year + num_years)
    start_index = start_year - base_params.start_year
    # Check that reform included if doing pct_diff or diff plot
    if output_type == "pct_diff" or output_type == "diff":
        assert reform_tpi is not None
    year_list = year_vec.tolist()
    if include_overall:
        year_list.append(str(year_vec[0]) + "-" + str(year_vec[-1]))
    if include_SS:
        year_list.append("SS")
    table_dict = {"Year": year_list}
    for i, v in enumerate(var_list):
        if output_type == "pct_diff":
            # multiple by 100 so in percentage points
            if stationarized:
                results = ((reform_tpi[v] - base_tpi[v]) / base_tpi[v]) * 100
            else:
                pct_changes = pct_change_unstationarized(
                    base_tpi,
                    base_params,
                    reform_tpi,
                    reform_params,
                    output_vars=[v],
                )
                results = pct_changes[v] * 100
            results_years = results[start_index : start_index + num_years]
            results_overall = (
                (
                    reform_tpi[v][start_index : start_index + num_years].sum()
                    - base_tpi[v][start_index : start_index + num_years].sum()
                )
                / base_tpi[v][start_index : start_index + num_years].sum()
            ) * 100
            results_SS = results[-1]
            results_for_table = results_years
            if include_overall:
                results_for_table = np.append(
                    results_for_table, results_overall
                )
            if include_SS:
                results_for_table = np.append(results_for_table, results_SS)
            table_dict[VAR_LABELS[v]] = results_for_table
        elif output_type == "diff":
            results = reform_tpi[v] - base_tpi[v]
            results_years = results[start_index : start_index + num_years]
            results_overall = (
                reform_tpi[v][start_index : start_index + num_years].sum()
                - base_tpi[v][start_index : start_index + num_years].sum()
            )
            results_SS = results[-1]
            results_for_table = results_years
            if include_overall:
                results_for_table = np.append(
                    results_for_table, results_overall
                )
            if include_SS:
                results_for_table = np.append(results_for_table, results_SS)
            table_dict[VAR_LABELS[v]] = results_for_table
        else:
            results_years = base_tpi[v][start_index : start_index + num_years]
            results_overall = results_years.sum()
            results_SS = base_tpi[v][-1]
            results_for_table = results_years
            if include_overall:
                results_for_table = np.append(
                    results_for_table, results_overall
                )
            if include_SS:
                results_for_table = np.append(results_for_table, results_SS)
            table_dict[VAR_LABELS[v] + " Baseline"] = results_for_table
            if reform_tpi is not None:
                results_years = reform_tpi[v][
                    start_index : start_index + num_years
                ]
                results_overall = results_years.sum()
                results_SS = reform_tpi[v][-1]
                results_for_table = results_years
                if include_overall:
                    results_for_table = np.append(
                        results_for_table, results_overall
                    )
                if include_SS:
                    results_for_table = np.append(
                        results_for_table, results_SS
                    )
                table_dict[VAR_LABELS[v] + " Reform"] = results_for_table
        # Make df with dict so can use pandas functions
        table_df = (
            pd.DataFrame.from_dict(table_dict, orient="columns")
            .set_index("Year")
            .transpose()
        )
        table_df.reset_index(inplace=True)
        table_df.rename(columns={"index": "Variable"}, inplace=True)
        table = save_return_table(table_df, table_format, path)

    return table


def macro_table_SS(
    base_ss,
    reform_ss,
    var_list=["Yss", "Css", "Kss", "Lss", "rss", "wss"],
    table_format=None,
    path=None,
):
    """
    Create a table of macro aggregates from the steady-state solutions.

    Args:
        base_ss (dictionary): SS output from baseline run
        reform_ss (dictionary): SS output from reform run
        var_list (list): names of variable to use in table
        table_format (string): format to return table in: 'csv', 'tex',
            'excel', 'json', if None, a DataFrame is returned
        path (string): path to save table to

    Returns:
        table (various): table in DataFrame or string format or `None`
            if saved to disk

    """
    table_dict = {
        "Variable": [],
        "Baseline": [],
        "Reform": [],
        "% Change (or pp diff)": [],
    }
    for i, v in enumerate(var_list):
        table_dict["Variable"].append(VAR_LABELS[v])
        table_dict["Baseline"].append(base_ss[v])
        table_dict["Reform"].append(reform_ss[v])
        if v != "D/Y":
            diff = ((reform_ss[v] - base_ss[v]) / base_ss[v]) * 100
        else:
            diff = (
                reform_ss["Dss"] / reform_ss["Yss"]
                - base_ss["Dss"] / base_ss["Yss"]
            )
        table_dict["% Change (or pp diff)"].append(diff)
        # Make df with dict so can use pandas functions
        table_df = pd.DataFrame.from_dict(
            table_dict, orient="columns"
        ).set_index("Variable")
        table = save_return_table(table_df, table_format, path, precision=3)

    return table


def ineq_table(
    base_ss,
    base_params,
    reform_ss=None,
    reform_params=None,
    var_list=["cssmat"],
    table_format=None,
    path=None,
):
    """
    Creates table with various inequality measures in the model
    steady-state.

    Args:
        base_ss (dictionary): SS output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        reform_ss (dictionary): SS output from reform run
        reform_params (OG-Core Specifications class): reform parameters
            object
        var_list (list): names of variable to use in table
        table_format (string): format to return table in: 'csv', 'tex',
            'excel', 'json', if None, a DataFrame is returned
        path (string): path to save table to

    Returns:
        table (various): table in DataFrame or string format or `None`
            if saved to disk

    """
    table_dict = {
        "Steady-State Variable": [],
        "Inequality Measure": [],
        "Baseline": [],
    }
    if reform_ss:
        table_dict["Reform"] = []
        table_dict["% Change"] = []
    for i, v in enumerate(var_list):
        base_ineq = Inequality(
            base_ss[v],
            base_params.omega_SS,
            base_params.lambdas,
            base_params.S,
            base_params.J,
        )
        if reform_ss:
            reform_ineq = Inequality(
                reform_ss[v],
                reform_params.omega_SS,
                reform_params.lambdas,
                reform_params.S,
                reform_params.J,
            )
        table_dict["Steady-State Variable"].extend(
            [VAR_LABELS[v], "", "", "", ""]
        )
        table_dict["Inequality Measure"].extend(
            [
                "Gini Coefficient",
                "Var of Logs",
                "90/10 Ratio",
                "Top 10% Share",
                "Top 1% Share",
            ]
        )
        base_values = np.array(
            [
                base_ineq.gini(),
                base_ineq.var_of_logs(),
                base_ineq.ratio_pct1_pct2(0.90, 0.10),
                base_ineq.top_share(0.1),
                base_ineq.top_share(0.01),
            ]
        )
        table_dict["Baseline"].extend(list(base_values))
        if reform_ss:
            reform_values = np.array(
                [
                    reform_ineq.gini(),
                    reform_ineq.var_of_logs(),
                    reform_ineq.ratio_pct1_pct2(0.90, 0.10),
                    reform_ineq.top_share(0.1),
                    reform_ineq.top_share(0.01),
                ]
            )
            table_dict["Reform"].extend(list(reform_values))
            table_dict["% Change"].extend(
                list(((reform_values - base_values) / base_values) * 100)
            )
    # Make df with dict so can use pandas functions
    table_df = pd.DataFrame.from_dict(table_dict)
    table = save_return_table(table_df, table_format, path, precision=3)

    return table


def gini_table(
    base_ss,
    base_params,
    reform_ss=None,
    reform_params=None,
    var_list=["cssmat"],
    table_format=None,
    path=None,
):
    """
    Creates table with measures of the Gini coefficient: overall,
    across lifetime earnings group, and across age.

    Args:
        base_ss (dictionary): SS output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        reform_ss (dictionary): SS output from reform run
        reform_params (OG-Core Specifications class): reform parameters
            object
        var_list (list): names of variable to use in table
        table_format (string): format to return table in: 'csv', 'tex',
            'excel', 'json', if None, a DataFrame is returned
        path (string): path to save table to

    Returns:
        table (various): table in DataFrame or string format or `None`
            if saved to disk

    """
    table_dict = {"Steady-State Variable": [], "Gini Type": [], "Baseline": []}
    if reform_ss:
        table_dict["Reform"] = []
        table_dict["% Change"] = []
    for i, v in enumerate(var_list):
        base_ineq = Inequality(
            base_ss[v],
            base_params.omega_SS,
            base_params.lambdas,
            base_params.S,
            base_params.J,
        )
        if reform_ss:
            reform_ineq = Inequality(
                reform_ss[v],
                reform_params.omega_SS,
                reform_params.lambdas,
                reform_params.S,
                reform_params.J,
            )
        table_dict["Steady-State Variable"].extend([VAR_LABELS[v], "", ""])
        table_dict["Gini Type"].extend(
            ["Overall", "Lifetime Income Group, $j$", "Age , $s$"]
        )
        base_values = np.array(
            [
                base_ineq.gini(),
                base_ineq.gini(type="ability"),
                base_ineq.gini(type="age"),
            ]
        )
        table_dict["Baseline"].extend(list(base_values))
        if reform_ss:
            reform_values = np.array(
                [
                    reform_ineq.gini(),
                    reform_ineq.gini(type="ability"),
                    reform_ineq.gini(type="age"),
                ]
            )
            table_dict["Reform"].extend(list(reform_values))
            table_dict["% Change"].extend(
                list(((reform_values - base_values) / base_values) * 100)
            )
    # Make df with dict so can use pandas functions
    table_df = pd.DataFrame.from_dict(table_dict)
    table = save_return_table(table_df, table_format, path, precision=3)

    return table


def wealth_moments_table(
    base_ss, base_params, data_moments=None, table_format=None, path=None
):
    """
    Creates table with moments of the wealth distribution from the model
    and SCF data.

    Args:
        base_ss (dictionary): SS output from baseline run
        base_params (OG-Core Specifications class): baseline parameters
            object
        table_format (string): format to return table in: 'csv', 'tex',
            'excel', 'json', if None, a DataFrame is returned
        path (string): path to save table to

    Returns:
        table (various): table in DataFrame or string format or `None`
            if saved to disk

    """
    table_dict = {
        "Moment": [
            "Share 0-25%",
            "Share 25-50%",
            "Share 50-70%",
            "Share 70-80%",
            "Share 80-90%",
            "Share 90-99%",
            "Share 99-100%",
            "Gini Coefficient",
            "var(ln(Wealth))",
        ],
        "Data": [],
        "Model": [],
    }
    base_ineq = Inequality(
        base_ss["bssmat_splus1"],
        base_params.omega_SS,
        base_params.lambdas,
        base_params.S,
        base_params.J,
    )
    base_values = [
        1 - base_ineq.top_share(0.75),
        base_ineq.top_share(0.75) - base_ineq.top_share(0.5),
        base_ineq.top_share(0.5) - base_ineq.top_share(0.3),
        base_ineq.top_share(0.3) - base_ineq.top_share(0.2),
        base_ineq.top_share(0.2) - base_ineq.top_share(0.1),
        base_ineq.top_share(0.1) - base_ineq.top_share(0.01),
        base_ineq.top_share(0.01),
        base_ineq.gini(),
        base_ineq.var_of_logs(),
    ]
    table_dict["Model"].extend(base_values)
    # Add moments from the data
    if data_moments is not None:
        table_dict["Data"] = data_moments
    # Make df with dict so can use pandas functions
    table_df = pd.DataFrame.from_dict(table_dict)
    table = save_return_table(table_df, table_format, path, precision=3)

    return table


def tp_output_dump_table(
    base_params,
    base_tpi,
    reform_params=None,
    reform_tpi=None,
    table_format=None,
    path=None,
):
    """
    This function dumps many of the macro time series from the
    transition path into an output table.

    Args:
        base_params (OG-Core Specifications class): baseline parameters
            object
        base_tpi (dictionary): TP output from baseline run
        reform_params (OG-Core Specifications class): reform parameters
            object
        reform_tpi (dictionary): TP output from reform run
        table_format (string): format to return table in: 'csv', 'tex',
            'excel', 'json', if None, a DataFrame is returned
        path (string): path to save table to

    Returns:
        table (various): table in DataFrame or string format or `None`
            if saved to disk

    """
    T = base_params.T
    # keep just items of interest for final table
    vars_to_keep = [
        "Y",
        "L",
        "G",
        "TR",
        "B",
        "K",
        "K_d",
        "K_f",
        "D",
        "D_d",
        "D_f",
        "r",
        "r_gov",
        "r_p",
        "w",
        "total_tax_revenue",
        "business_tax_revenue",
    ]
    base_dict = {k: base_tpi[k] for k in vars_to_keep}
    # update key names
    base_dict_final = dict(
        (VAR_LABELS[k] + ": Baseline", v[:T]) for (k, v) in base_dict.items()
    )
    # create df
    table_df = pd.DataFrame.from_dict(base_dict_final)
    if reform_tpi is not None:
        assert base_params.start_year == reform_params.start_year
        assert base_params.T == reform_params.T
        reform_dict = {k: reform_tpi[k] for k in vars_to_keep}
        # update key names
        reform_dict_final = dict(
            (VAR_LABELS[k] + ": Reform", v[:T])
            for (k, v) in reform_dict.items()
        )
        df_reform = pd.DataFrame.from_dict(reform_dict_final)
        # merge dfs
        table_df = table_df.merge(df_reform, left_index=True, right_index=True)
    # rename index to year
    table_df.reset_index(inplace=True)
    table_df.rename(columns={"index": "Year"}, inplace=True)
    # update index to reflect years
    table_df["Year"] = table_df["Year"] + base_params.start_year

    table = save_return_table(table_df, table_format, path)

    return table


def dynamic_revenue_decomposition(
    base_params,
    base_tpi,
    base_ss,
    reform_params,
    reform_tpi,
    reform_ss,
    num_years=10,
    include_SS=True,
    include_overall=True,
    include_business_tax=True,
    full_break_out=False,
    start_year=DEFAULT_START_YEAR,
    table_format=None,
    path=None,
):
    """
    This function decomposes the source of changes in tax revenues to
    determine the percentage change in tax revenues that can be
    attributed to macroeconomic feedback effects.

    Args:
        base_params (OG-Core Specifications class): baseline parameters
            object
        base_tpi (dictionary): TP output from baseline run
        base_ss (dictionary): SS output from baseline run
        reform_params (OG-Core Specifications class): reform parameters
            object
        reform_tpi (dictionary): TP output from reform run
        reform_ss (dictionary): SS output from reform run
        num_years (integer): number of years to include in table
        include_SS (bool): whether to include the steady-state results
            in the table
        include_overall (bool): whether to include results over the
            entire budget window as a column in the table
        include_business_tax (bool): whether to include business tax
            revenue changes in result
        full_break_out (bool): whether to break out behavioral and macro
            effects
        start_year (integer): year to start table
        table_format (string): format to return table in: 'csv', 'tex',
            'excel', 'json', if None, a DataFrame is returned
        path (string): path to save table to

    Returns:
        table (various): table in DataFrame or string format or `None`
            if saved to disk

    .. note:: The decomposition is the following:
        1. Simulate the baseline and reform in OG-Core. Save the
           resulting series of tax revenues. Call these series for the
           baseline and reform A and D, respectively.
        2. Create a third revenue series that is computed using the
           baseline behavior (i.e., `bmat_s` and `n_mat`) and macro
           variables (`tr`, `bq`, `r`, `w`), but with the tax function
           parameter estimates from the reform policy.  Call this
           series B.
        3. Create a fourth revenue series that is computed using the
           reform behavior (i.e., `bmat_s` and `n_mat`) and tax
           functions estimated on the reform tax policy, but
           the macro variables (`tr`, `bq`, `r`, `w`) from the baseline.
           Call this series C.
        3. Calculate the percentage difference between B and A -- call
           this the "static" change from the macro model.  Calculate the
           percentage difference between C and B -- call this the
           behavioral effects.  Calculate the percentage difference
           between D and C -- call this the macroeconomic effect.  The
           full dynamic effect is difference between C and A.

        One can apply the percentage difference from the macro feedback
        effect to ("static") revenue estimates from the policy change
        to produce an estimate of the revenue including macro feedback.

    """
    assert isinstance(start_year, (int, np.integer))
    assert isinstance(num_years, (int, np.integer))
    # Make sure both runs cover same time period
    assert base_params.start_year == reform_params.start_year
    year_vec = np.arange(start_year, start_year + num_years)
    start_index = start_year - base_params.start_year
    year_list = year_vec.tolist()
    if include_overall:
        year_list.append(str(year_vec[0]) + "-" + str(year_vec[-1]))
    if include_SS:
        year_list.append("SS")
    table_dict = {"Year": year_list}
    T, S, J = base_params.T, base_params.S, base_params.J
    num_params = len(base_params.etr_params[0][0])
    base_etr_params_4D = [
        [
            [
                [base_params.etr_params[t][s][i] for i in range(num_params)]
                for j in range(J)
            ]
            for s in range(S)
        ]
        for t in range(T)
    ]
    reform_etr_params_4D = [
        [
            [
                [reform_params.etr_params[t][s][i] for i in range(num_params)]
                for j in range(J)
            ]
            for s in range(S)
        ]
        for t in range(T)
    ]
    tax_rev_dict = {"indiv": {}, "biz": {}, "total": {}}
    indiv_liab = {}
    # Baseline IIT + payroll tax liability
    indiv_liab["A"] = tax.income_tax_liab(
        base_tpi["r_p"][:T],
        base_tpi["w"][:T],
        base_tpi["bmat_s"],
        base_tpi["n_mat"][:T, :, :],
        base_ss["factor_ss"],
        0,
        None,
        "TPI",
        base_params.e,
        base_etr_params_4D,
        base_params,
    )
    # IIT + payroll tax liability using baseline behavior and macros
    # with the reform tax functions (this is the OG-Core static estimate)
    indiv_liab["B"] = tax.income_tax_liab(
        base_tpi["r_p"][:T],
        base_tpi["w"][:T],
        base_tpi["bmat_s"],
        base_tpi["n_mat"][:T, :, :],
        base_ss["factor_ss"],
        0,
        None,
        "TPI",
        base_params.e,
        reform_etr_params_4D,
        base_params,
    )
    # IIT + payroll tax liability using reform behavior and baseline
    # macros
    indiv_liab["C"] = tax.income_tax_liab(
        base_tpi["r_p"][:T],
        base_tpi["w"][:T],
        reform_tpi["bmat_s"],
        reform_tpi["n_mat"][:T, :, :],
        base_ss["factor_ss"],
        0,
        None,
        "TPI",
        reform_params.e,
        reform_etr_params_4D,
        reform_params,
    )
    # IIT + payroll tax liability from the reform simulation
    indiv_liab["D"] = tax.income_tax_liab(
        reform_tpi["r_p"][:T],
        reform_tpi["w"][:T],
        reform_tpi["bmat_s"],
        reform_tpi["n_mat"][:T, :, :],
        base_ss["factor_ss"],
        0,
        None,
        "TPI",
        reform_params.e,
        reform_etr_params_4D,
        reform_params,
    )
    # Business tax revenue from the baseline simulation
    tax_rev_dict["biz"]["A"] = tax.get_biz_tax(
        base_tpi["w"][:T],
        base_tpi["Y_vec"][:T, :],
        base_tpi["L_vec"][:T, :],
        base_tpi["K_vec"][:T, :],
        base_tpi["p_m"][:T],
        base_params,
        None,
        "TPI",
    ).sum(axis=-1)
    # Business tax revenue found using baseline behavior and macros with
    # the reform tax rates
    tax_rev_dict["biz"]["B"] = tax.get_biz_tax(
        base_tpi["w"][:T],
        base_tpi["Y_vec"][:T, :],
        base_tpi["L_vec"][:T, :],
        base_tpi["K_vec"][:T, :],
        base_tpi["p_m"][:T],
        reform_params,
        None,
        "TPI",
    ).sum(axis=-1)
    # Business tax revenue found using the reform behavior and baseline
    # macros with the reform tax rates
    tax_rev_dict["biz"]["C"] = tax.get_biz_tax(
        base_tpi["w"][:T],
        reform_tpi["Y_vec"][:T, :],
        reform_tpi["L_vec"][:T, :],
        reform_tpi["K_vec"][:T, :],
        reform_tpi["p_m"][:T],
        reform_params,
        None,
        "TPI",
    ).sum(axis=-1)
    # Business tax revenue from the reform
    tax_rev_dict["biz"]["D"] = tax.get_biz_tax(
        reform_tpi["w"][:T],
        reform_tpi["Y_vec"][:T, :],
        reform_tpi["L_vec"][:T, :],
        reform_tpi["K_vec"][:T, :],
        reform_tpi["p_m"][:T],
        reform_params,
        None,
        "TPI",
    ).sum(axis=-1)
    pop_weights = np.squeeze(base_params.lambdas) * np.tile(
        np.reshape(base_params.omega[:T, :], (T, S, 1)), (1, 1, J)
    )
    for k in indiv_liab.keys():
        tax_rev_dict["indiv"][k] = (indiv_liab[k] * pop_weights).sum(1).sum(1)
        tax_rev_dict["total"][k] = (
            tax_rev_dict["indiv"][k] + tax_rev_dict["biz"][k]
        )
    results_for_table = {"indiv": {}, "biz": {}, "total": {}}
    for type in ["indiv", "biz", "total"]:
        # Rate change effect
        pct_change1 = (
            (tax_rev_dict[type]["B"] - tax_rev_dict[type]["A"])
            / tax_rev_dict[type]["A"]
        ) * 100
        # Behavior effect
        pct_change2 = (
            (tax_rev_dict[type]["C"] - tax_rev_dict[type]["B"])
            / tax_rev_dict[type]["B"]
        ) * 100
        # Macro effect
        pct_change3 = (
            (tax_rev_dict[type]["D"] - tax_rev_dict[type]["C"])
            / tax_rev_dict[type]["C"]
        ) * 100
        # Dynamic effect (behavior + macro)
        pct_change4 = (
            (tax_rev_dict[type]["D"] - tax_rev_dict[type]["B"])
            / tax_rev_dict[type]["B"]
        ) * 100
        # Total change in tax revenue (rates + behavior + macro)
        pct_change5 = (
            (tax_rev_dict[type]["D"] - tax_rev_dict[type]["A"])
            / tax_rev_dict[type]["A"]
        ) * 100
        pct_change_overall1 = (
            (
                tax_rev_dict[type]["B"][
                    start_index : start_index + num_years
                ].sum()
                - tax_rev_dict[type]["A"][
                    start_index : start_index + num_years
                ].sum()
            )
            / tax_rev_dict[type]["A"][
                start_index : start_index + num_years
            ].sum()
        ) * 100
        pct_change_overall2 = (
            (
                tax_rev_dict[type]["C"][
                    start_index : start_index + num_years
                ].sum()
                - tax_rev_dict[type]["B"][
                    start_index : start_index + num_years
                ].sum()
            )
            / tax_rev_dict[type]["B"][
                start_index : start_index + num_years
            ].sum()
        ) * 100
        pct_change_overall3 = (
            (
                tax_rev_dict[type]["D"][
                    start_index : start_index + num_years
                ].sum()
                - tax_rev_dict[type]["C"][
                    start_index : start_index + num_years
                ].sum()
            )
            / tax_rev_dict[type]["C"][
                start_index : start_index + num_years
            ].sum()
        ) * 100
        pct_change_overall4 = (
            (
                tax_rev_dict[type]["D"][
                    start_index : start_index + num_years
                ].sum()
                - tax_rev_dict[type]["B"][
                    start_index : start_index + num_years
                ].sum()
            )
            / tax_rev_dict[type]["B"][
                start_index : start_index + num_years
            ].sum()
        ) * 100
        pct_change_overall5 = (
            (
                tax_rev_dict[type]["D"][
                    start_index : start_index + num_years
                ].sum()
                - tax_rev_dict[type]["A"][
                    start_index : start_index + num_years
                ].sum()
            )
            / tax_rev_dict[type]["A"][
                start_index : start_index + num_years
            ].sum()
        ) * 100
        if include_overall:
            results_for_table[type][1] = np.append(
                pct_change1[start_index : start_index + num_years],
                pct_change_overall1,
            )
            results_for_table[type][2] = np.append(
                pct_change2[start_index : start_index + num_years],
                pct_change_overall2,
            )
            results_for_table[type][3] = np.append(
                pct_change3[start_index : start_index + num_years],
                pct_change_overall3,
            )
            results_for_table[type][4] = np.append(
                pct_change4[start_index : start_index + num_years],
                pct_change_overall4,
            )
            results_for_table[type][5] = np.append(
                pct_change5[start_index : start_index + num_years],
                pct_change_overall5,
            )
        if include_SS:
            results_for_table[type][1] = np.append(
                results_for_table[type][1], pct_change1[-1]
            )
            results_for_table[type][2] = np.append(
                results_for_table[type][2], pct_change2[-1]
            )
            results_for_table[type][3] = np.append(
                results_for_table[type][3], pct_change3[-1]
            )
            results_for_table[type][4] = np.append(
                results_for_table[type][4], pct_change4[-1]
            )
            results_for_table[type][5] = np.append(
                results_for_table[type][5], pct_change5[-1]
            )
    if full_break_out:
        if include_business_tax:
            table_dict = {
                "Year": year_list,
                # IIT and Payroll Taxes
                "IIT: Pct Change due to tax rates": results_for_table["indiv"][
                    1
                ],
                "IIT: Pct Change due to behavior": results_for_table["indiv"][
                    2
                ],
                "IIT: Pct Change due to macro": results_for_table["indiv"][3],
                "IIT: Overall Pct Change in taxes": results_for_table["indiv"][
                    5
                ],
                # Business Taxes
                "CIT: Pct Change due to tax rates": results_for_table["biz"][
                    1
                ],
                "CIT: Pct Change due to behavior": results_for_table["biz"][2],
                "CIT: Pct Change due to macro": results_for_table["biz"][3],
                "CIT: Overall Pct Change in taxes": results_for_table["biz"][
                    5
                ],
                # All Taxes
                "All: Pct Change due to tax rates": results_for_table["total"][
                    1
                ],
                "All: Pct Change due to behavior": results_for_table["total"][
                    2
                ],
                "All: Pct Change due to macro": results_for_table["total"][3],
                "All: Overall Pct Change in taxes": results_for_table["total"][
                    5
                ],
            }
        else:
            table_dict = {
                "Year": year_list,
                "Pct Change due to tax rates": results_for_table["indiv"][1],
                "Pct Change due to behavior": results_for_table["indiv"][2],
                "Pct Change due to macro": results_for_table["indiv"][3],
                "Overall Pct Change in taxes": results_for_table["indiv"][5],
            }
    else:
        if include_business_tax:
            table_dict = {
                "Year": year_list,
                # 'IIT and Payroll Taxes:':
                # np.ones(results_for_table['indiv'][1].shape[0]) * np.nan,
                "IIT: Pct Change due to tax rates": results_for_table["indiv"][
                    1
                ],
                "IIT: Pct Change due to dynamics": results_for_table["indiv"][
                    4
                ],
                "IIT: Overall Pct Change in taxes": results_for_table["indiv"][
                    5
                ],
                # 'Business Taxes:':
                # np.ones(results_for_table['biz'][1].shape[0]) * np.nan,
                "CIT: Pct Change due to tax rates": results_for_table["biz"][
                    1
                ],
                "CIT: Pct Change due to dynamics": results_for_table["biz"][4],
                "CIT: Overall Pct Change in taxes": results_for_table["biz"][
                    5
                ],
                # 'All Taxes:':
                # np.ones(results_for_table['total'][1].shape[0]) * np.nan,
                "All: Pct Change due to tax rates": results_for_table["total"][
                    1
                ],
                "All: Pct Change due to dynamics": results_for_table["total"][
                    4
                ],
                "All: Overall Pct Change in taxes": results_for_table["total"][
                    5
                ],
            }
        else:
            table_dict = {
                "Year": year_list,
                "Pct Change due to tax rates": results_for_table["indiv"][1],
                "Pct Change due to dynamics": results_for_table["indiv"][4],
                "Overall Pct Change in taxes": results_for_table["indiv"][5],
            }
    # Make df with dict so can use pandas functions
    table_df = (
        pd.DataFrame.from_dict(table_dict, orient="columns")
        .set_index("Year")
        .transpose()
    )
    table_df.reset_index(inplace=True)
    table_df.rename(columns={"index": "Variable"}, inplace=True)
    table = save_return_table(table_df, table_format, path)

    return table
