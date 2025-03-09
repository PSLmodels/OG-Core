import os
import ogcore.utils as utils

# Read in json file
cur_dir = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(cur_dir, "model_variables.json")
with open(file) as f:
    json_text = f.read()
var_metadata = utils.json_to_dict(json_text)

SHOW_RUNTIME = False  # Flag to display RuntimeWarnings when run model

REFORM_DIR = "OUTPUT_REFORM"
BASELINE_DIR = "OUTPUT_BASELINE"

# Default year for model runs
DEFAULT_START_YEAR = 2025

VAR_LABELS = dict([(k, v["label"]) for k, v in var_metadata.items()])

ToGDP_LABELS = dict([(k, v["toGDP_label"]) for k, v in var_metadata.items()])

GROUP_LABELS = {
    7: {
        0: "0-25%",
        1: "25-50%",
        2: "50-70%",
        3: "70-80%",
        4: "80-90%",
        5: "90-99%",
        6: "Top 1%",
    },
    9: {
        0: "0-25%",
        1: "25-50%",
        2: "50-70%",
        3: "70-80%",
        4: "80-90%",
        5: "90-99%",
        6: "99-99.5%",
        7: "99.5-99.9%",
        8: "Top 0.1%",
    },
    10: {
        0: "0-25%",
        1: "25-50%",
        2: "50-70%",
        3: "70-80%",
        4: "80-90%",
        5: "90-99%",
        6: "99-99.5%",
        7: "99.5-99.9%",
        8: "99.9-99.99%",
        9: "Top 0.01%",
    },
}


PARAM_LABELS = {
    "start_year": ["Initial year", r"$\texttt{start\_year}$"],
    # 'Gamma': ['Initial distribution of savings', r'\hat{\Gamma}_{0}'],
    # 'N': ['Initial population', 'N_{0}'],
    "omega": ["Population by age over time", r"$\omega_{s,t}$"],
    # 'fert_rates': ['Fertility rates by age',
    #                r'$f_{s,t}$'],
    "imm_rates": ["Immigration rates by age", r"$i_{s,t}$"],
    "rho": ["Mortality rates by age", r"$\rho_{s,t}$"],
    "e": ["Deterministic ability process", r"$e_{j,s,t}$"],
    "lambdas": [
        "Lifetime income group percentages",
        r"$\lambda_{j}$",
    ],
    "J": ["Number of lifetime income groups", "$J$"],
    "S": ["Maximum periods in economically active individual life", "$S$"],
    "E": ["Number of periods of youth economically outside the model", "$E$"],
    "T": ["Number of periods to steady-state", "$T$"],
    "retirement_age": ["Retirement age", "$R$"],
    "ltilde": ["Maximum hours of labor supply", r"$\tilde{l}$"],
    "beta": ["Discount factor", r"$\beta$"],
    "sigma": ["Coefficient of constant relative risk aversion", r"$\sigma$"],
    "frisch": ["Frisch elasticity of labor supply", r"$\nu$"],
    "b_ellipse": ["Scale parameter in utility of leisure", "$b$"],
    "upsilon": ["Shape parameter in utility of leisure", r"$\upsilon$"],
    # 'k': ['Constant parameter in utility of leisure', 'k'],
    "chi_n": [
        "Disutility of labor level parameters",
        r"$\chi^{n}_{s}$",
    ],
    "chi_b": [
        "Utility of bequests level parameters",
        r"$\chi^{b}_{j}$",
    ],
    "use_zeta": [
        "Whether to distribute bequests between lifetime income groups",
        r"$\texttt{use\_zeta}$",
    ],
    "zeta": ["Distribution of bequests", r"$\zeta$"],
    "Z": ["Total factor productivity", "$Z_{t}$"],
    "gamma": ["Capital share of income", r"$\gamma$"],
    "epsilon": [
        "Elasticity of substitution between capital and labor",
        r"$\varepsilon$",
    ],
    "delta": ["Capital depreciation rate", r"$\delta$"],
    "g_y": [
        "Growth rate of labor augmenting technological progress",
        r"$g_{y}$",
    ],
    "tax_func_type": [
        "Functional form used for income tax functions",
        r"$\texttt{tax\_func\_type}$",
    ],
    "analytical_mtrs": [
        "Whether use analytical MTRs or estimate MTRs",
        r"$\texttt{analytical\_mtrs}$",
    ],
    "age_specific": [
        "Whether use age-specific tax functions",
        r"$\texttt{age\_specific}$",
    ],
    "tau_payroll": ["Payroll tax rate", r"$\tau^{p}_{t}$"],
    # 'theta': ['Replacement rate by average income',
    #           r'\left{\theta_{j}\right}_{j=1}^{J}'],
    "tau_bq": ["Bequest (estate) tax rate", r"$\tau^{BQ}_{t}$"],
    "tau_b": ["Entity-level business income tax rate", r"$\tau^{b}_{t}$"],
    "delta_tau": [
        "Rate of depreciation for tax purposes",
        r"$\delta^{\tau}_{t}$",
    ],
    "tau_c": ["Consumption tax rates", r"$\tau^{c}_{t,s,j}$"],
    "h_wealth": ["Coefficient on linear term in wealth tax function", "$H$"],
    "m_wealth": ["Constant in wealth tax function", "$M$"],
    "p_wealth": ["Coefficient on level term in wealth tax function", "$P$"],
    "budget_balance": [
        "Whether have a balanced budget in each period",
        r"$\texttt{budget\_balance}$",
    ],
    "baseline_spending": [
        "Whether level of spending constant between "
        + "the baseline and reform runs",
        r"$\texttt{baseline\_spending}$",
    ],
    "alpha_T": ["Transfers as a share of GDP", r"$\alpha^{T}_{t}$"],
    "eta": ["Distribution of transfers", r"$\eta_{j,s,t}$"],
    "eta_RM": ["Distribution of remittances", r"$\eta_{RM,j,s,t}$"],
    "alpha_G": ["Government spending as a share of GDP", r"$\alpha^{G}_{t}$"],
    "alpha_RM_1": [
        "Remittances as a share of GDP in initial period",
        r"$\alpha_{RM,1}$",
    ],
    "alpha_RM_T": [
        "Remittances as a share of GDP in long run",
        r"$\alpha_{RM,T}$",
    ],
    "g_RM": ["Growth rate of remittances in initial periods", r"$g_{RM,t}$"],
    "tG1": ["Model period in which budget closure rule starts", r"$t_{G1}$"],
    "tG2": ["Model period in which budget closure rule ends", r"$t_{G2}$"],
    "rho_G": ["Budget closure rule smoothing parameter", r"$\rho_{G}$"],
    "debt_ratio_ss": ["Steady-state Debt-to-GDP ratio", r"$\bar{\alpha}_{D}$"],
    "initial_debt_ratio": [
        "Initial period Debt-to-GDP ratio",
        r"$\alpha_{D,0}$",
    ],
    "r_gov_scale": [
        "Scale parameter in government interest rate wedge",
        r"$\tau_{d,t}$",
    ],
    "r_gov_shift": [
        "Shift parameter in government interest rate wedge",
        r"$\mu_{d,t}$",
    ],
    "avg_earn_num_years": [
        "Number of years over which compute average earnings for pension benefit",
        r"$\texttt{avg\_earn\_num\_years}$",
    ],
    "AIME_bkt_1": ["First AIME bracket threshold", r"$\texttt{AIME\_bkt\_1}$"],
    "AIME_bkt_2": [
        "Second AIME bracket threshold",
        r"$\texttt{AIME\_bkt\_2}$",
    ],
    "PIA_rate_bkt_1": [
        "First AIME bracket PIA rate",
        r"$\texttt{PIA\_rate\_bkt\_1}$",
    ],
    "PIA_rate_bkt_2": [
        "Second AIME bracket PIA rate",
        r"$\texttt{PIA\_rate\_bkt\_2}$",
    ],
    "PIA_rate_bkt_3": [
        "Third AIME bracket PIA rate",
        r"$\texttt{PIA\_rate\_bkt\_3}$",
    ],
    "PIA_maxpayment": ["Maximum PIA payment", r"$\texttt{PIA\_maxpayment}$"],
    "PIA_minpayment": ["Minimum PIA payment", r"$\texttt{PIA\_minpayment}$"],
    "replacement_rate_adjust": [
        "Adjustment to replacement rate",
        r"$\theta_{adj,t}$",
    ],
    "world_int_rate": ["World interest rate", r"$r^{*}_{t}$"],
    "initial_foreign_debt_ratio": [
        "Share of government debt held by foreigners in initial period",
        r"$D_{f,0}$",
    ],
    "zeta_D": [
        "Share of new debt issues purchased by foreigners",
        r"$\zeta_{D, t}$",
    ],
    "zeta_K": [
        "Share of excess capital demand satisfied by foreigners",
        r"$\zeta_{K, t}$",
    ],
    "nu": ["Dampening parameter for TPI", r"$\xi$"],
    "maxiter": ["Maximum number of iterations for TPI", r"$\texttt{maxiter}$"],
    "mindist_SS": ["SS solution tolerance", r"$\texttt{mindist\_SS}$"],
    "mindist_TPI": ["TPI solution tolerance", r"$\texttt{mindist\_TPI}$"],
}

# Ignoring the following:
# 'starting_age', 'ending_age', 'constant_demographics',
# 'constant_rates', 'zero_taxes'
