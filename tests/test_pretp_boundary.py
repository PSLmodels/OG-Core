import numpy as np
from types import SimpleNamespace

from ogcore import aggregates as aggr
from ogcore import demographics


def make_mock_params():
    """
    Create a small parameter object that isolates the pre-time-path
    boundary from the later TPI path.
    """
    p = SimpleNamespace()
    p.S = 3
    p.J = 1
    p.T = 4
    p.use_zeta = False
    p.lambdas = np.array([1.0]).reshape(1, 1)
    p.omega_S_preTP = np.array([0.2, 0.3, 0.5])
    p.omega_SS = np.array([0.25, 0.35, 0.4])
    p.omega = np.array(
        [
            [0.3, 0.3, 0.4],
            [0.25, 0.35, 0.4],
            [0.2, 0.4, 0.4],
            [0.2, 0.4, 0.4],
        ]
    )
    # Explicit pre-period objects intentionally differ from the path
    # objects so the boundary bug is visible in a small unit test.
    p.imm_rates_preTP = np.array([0.11, 0.12, 0.13])
    p.rho_preTP = np.array([0.41, 0.42, 0.43])
    p.g_n_preTP = 0.21
    p.imm_rates = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06],
            [0.07, 0.08, 0.09],
            [0.1, 0.11, 0.12],
        ]
    )
    p.rho = np.array(
        [
            [0.11, 0.12, 0.13],
            [0.21, 0.22, 0.23],
            [0.31, 0.32, 0.33],
            [0.41, 0.42, 0.43],
        ]
    )
    p.g_n = np.array([0.01, 0.06, 0.07, 0.08])
    p.g_n_ss = 0.09
    p.g_y = 0.0
    p.delta = 0.1
    p.delta_g = 0.05
    p.infra_investment_leakage_rate = 0.0
    p.alpha_RM_1 = 0.05
    p.alpha_RM_T = 0.05
    p.g_RM = np.array([0.02, 0.02, 0.02, 0.02])
    p.tG1 = 2
    p.tG2 = 4
    return p


def step_population(pop_t, fert_t, mort_t, infmort_t, imm_t):
    """
    One-period demographic transition used to build a consistent toy path.
    """
    pop_tp1 = np.zeros_like(pop_t)
    newborns = np.dot(fert_t, pop_t)
    pop_tp1[0] = (1.0 - infmort_t) * newborns + imm_t[0] * pop_t[0]
    pop_tp1[1:] = pop_t[:-1] * (1.0 - mort_t[:-1]) + pop_t[1:] * imm_t[1:]
    return pop_tp1


def test_pretp_ss_aggregates_use_explicit_boundary_objects():
    """
    The preTP SS accounting objects should use the explicit pre-period
    demographic rates, not the first row of the TPI path.
    """
    p = make_mock_params()
    b = np.array([[1.0], [2.0], [3.0]])
    r = 0.04

    omega_shift = np.append(p.omega_S_preTP[1:], [0.0])
    imm_shift = np.append(p.imm_rates_preTP[1:], [0.0])
    expected_B = (
        b[:, 0] * (p.omega_S_preTP + (omega_shift * imm_shift))
    ).sum() / (1.0 + p.g_n_preTP)
    expected_BQ = (
        (p.omega_S_preTP * p.rho_preTP * b[:, 0]).sum()
        * (1.0 + r)
        / (1.0 + p.g_n_preTP)
    )

    assert np.allclose(aggr.get_B(b, p, "SS", True), expected_B)
    assert np.allclose(aggr.get_BQ(r, b, None, p, "SS", True), expected_BQ)


def test_pretp_tpi_bequests_change_only_boundary_period():
    """
    A boundary-only fix should change only the prepended preTP element of
    the TPI bequest path. Later periods should keep the existing path
    indexing contract.
    """
    p = make_mock_params()
    r = np.array([0.01, 0.02, 0.03, 0.04])
    b_splus1 = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[1.5], [2.5], [3.5]],
            [[2.0], [3.0], [4.0]],
            [[2.5], [3.5], [4.5]],
        ]
    )

    expected = np.zeros(p.T)
    expected[0] = (
        (p.omega_S_preTP * p.rho_preTP * b_splus1[0, :, 0]).sum()
        * (1.0 + r[0])
        / (1.0 + p.g_n_preTP)
    )
    for t in range(1, p.T):
        expected[t] = (
            (p.omega[t - 1, :] * p.rho[t - 1, :] * b_splus1[t, :, 0]).sum()
            * (1.0 + r[t])
            / (1.0 + p.g_n[t])
        )

    assert np.allclose(
        np.asarray(aggr.get_BQ(r, b_splus1, None, p, "TPI", False)).reshape(-1),
        expected,
    )


def test_pretp_fields_are_optional_for_legacy_callers():
    """
    Existing callers that do not yet provide explicit preTP demographic
    objects should keep the legacy behavior.
    """
    p = make_mock_params()
    del p.imm_rates_preTP
    del p.rho_preTP
    del p.g_n_preTP

    b = np.array([[1.0], [2.0], [3.0]])
    r = 0.04

    omega_shift = np.append(p.omega_S_preTP[1:], [0.0])
    imm_shift = np.append(p.imm_rates[0, 1:], [0.0])
    expected_B = (b[:, 0] * (p.omega_S_preTP + (omega_shift * imm_shift))).sum()
    expected_B /= 1.0 + p.g_n[0]
    expected_BQ = (
        (p.omega_S_preTP * p.rho[0, :] * b[:, 0]).sum()
        * (1.0 + r)
        / (1.0 + p.g_n[0])
    )

    assert np.allclose(aggr.get_B(b, p, "SS", True), expected_B)
    assert np.allclose(aggr.get_BQ(r, b, None, p, "SS", True), expected_BQ)


def test_get_pop_objs_pins_growth_timing_contract():
    """
    The existing downstream code expects the master storage convention:
    `g_n[0]` is preTP->0 growth and `g_n[1]` is 0->1 growth.
    """
    E = 1
    S = 4
    T = 8
    fert_rates = np.array(
        [
            [0.20, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    mort_rates = np.array(
        [
            [0.05, 0.10, 0.15, 0.20, 1.0],
            [0.06, 0.11, 0.16, 0.21, 1.0],
            [0.06, 0.11, 0.16, 0.21, 1.0],
        ]
    )
    infmort_rates = np.array([0.02, 0.03, 0.03])
    imm_rates = np.array(
        [
            [0.30, 0.31, 0.32, 0.33, 0.34],
            [0.01, 0.02, 0.03, 0.04, 0.05],
            [0.01, 0.02, 0.03, 0.04, 0.05],
        ]
    )
    pre_pop = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    pop_dist = np.zeros((4, E + S))
    pop_dist[0] = np.array([24.8, 59.9, 45.6, 32.1, 19.4])
    pop_dist[1] = step_population(
        pop_dist[0],
        fert_rates[0],
        mort_rates[0],
        infmort_rates[0],
        imm_rates[0],
    )
    pop_dist[2] = step_population(
        pop_dist[1],
        fert_rates[1],
        mort_rates[1],
        infmort_rates[1],
        imm_rates[1],
    )
    pop_dist[3] = step_population(
        pop_dist[2],
        fert_rates[2],
        mort_rates[2],
        infmort_rates[2],
        imm_rates[2],
    )

    pop_dict = demographics.get_pop_objs(
        E=E,
        S=S,
        T=T,
        min_age=0,
        max_age=4,
        fert_rates=fert_rates,
        mort_rates=mort_rates,
        infmort_rates=infmort_rates,
        imm_rates=imm_rates,
        infer_pop=False,
        pop_dist=pop_dist,
        pre_pop_dist=pre_pop,
        initial_data_year=2020,
        final_data_year=2022,
        GraphDiag=False,
    )

    pre_growth = (pop_dist[0, -S:].sum() - pre_pop[-S:].sum()) / pre_pop[-S:].sum()
    zero_to_one_growth = (
        pop_dist[1, -S:].sum() - pop_dist[0, -S:].sum()
    ) / pop_dist[0, -S:].sum()

    assert np.allclose(pop_dict["omega_S_preTP"], pre_pop[-S:] / pre_pop[-S:].sum())
    assert np.allclose(pop_dict["g_n_preTP"], pre_growth)
    assert np.allclose(pop_dict["g_n"][0], pre_growth)
    assert np.allclose(pop_dict["g_n"][1], zero_to_one_growth)
    # Legacy: with user-supplied rates and no pre_mort_rates kwarg, the
    # boundary objects alias the first in-window row.
    assert np.allclose(pop_dict["rho_preTP"], mort_rates[0, E:])
    assert np.allclose(pop_dict["imm_rates_preTP"], imm_rates[0, E:])


def test_get_pop_objs_fetches_distinct_prior_year_boundary_rates(monkeypatch):
    """
    If OG-Core is sourcing demographic rates itself, it should populate
    the preTP boundary objects from a genuine prior-year row rather than
    aliasing the first in-window path row.
    """
    E = 1
    S = 4
    T = 8
    initial_data_year = 2020
    final_data_year = 2022
    pop_dist = np.array(
        [
            [24.8, 59.9, 45.6, 32.1, 19.4],
            [12.3008, 42.129, 68.502, 49.353, 32.276],
            [3.105952, 12.405332, 39.54987, 59.5158, 40.60267],
            [0.78425288, 3.16770152, 12.22724158, 35.6025228, 49.0476155],
        ]
    )
    pre_pop = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    main_fert = np.array(
        [
            [0.20, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    prior_fert = np.array([[0.15, 0.0, 0.0, 0.0, 0.0]])
    main_mort = np.array(
        [
            [0.05, 0.10, 0.15, 0.20, 1.0],
            [0.06, 0.11, 0.16, 0.21, 1.0],
            [0.06, 0.11, 0.16, 0.21, 1.0],
        ]
    )
    prior_mort = np.array([[0.01, 0.02, 0.03, 0.04, 1.0]])
    main_infmort = np.array([0.02, 0.03, 0.03])
    prior_infmort = np.array([0.005])
    main_imm = np.array(
        [
            [0.30, 0.31, 0.32, 0.33, 0.34],
            [0.01, 0.02, 0.03, 0.04, 0.05],
            [0.01, 0.02, 0.03, 0.04, 0.05],
        ]
    )
    prior_imm = np.array([[0.20, 0.21, 0.22, 0.23, 0.24]])

    def _resolve_start_year(args, kwargs, positional_index):
        if "start_year" in kwargs:
            return kwargs["start_year"]
        if len(args) > positional_index:
            return args[positional_index]
        raise TypeError("start_year not supplied")

    def fake_get_fert(*args, **kwargs):
        # get_fert(totpers, min_age, max_age, country_id, start_year, ...)
        start_year = _resolve_start_year(args, kwargs, 4)
        if start_year == initial_data_year - 1:
            return prior_fert
        return main_fert

    def fake_get_mort(*args, **kwargs):
        # get_mort(totpers, min_age, max_age, country_id, start_year, ...)
        start_year = _resolve_start_year(args, kwargs, 4)
        if start_year == initial_data_year - 1:
            return prior_mort, prior_infmort
        return main_mort, main_infmort

    def fake_get_imm_rates(*args, **kwargs):
        # get_imm_rates(totpers, min_age, max_age, fert_rates, mort_rates,
        #               infmort_rates, pop_dist, country_id, start_year, ...)
        start_year = _resolve_start_year(args, kwargs, 8)
        if start_year == initial_data_year - 1:
            return prior_imm
        return main_imm

    monkeypatch.setattr(demographics, "get_fert", fake_get_fert)
    monkeypatch.setattr(demographics, "get_mort", fake_get_mort)
    monkeypatch.setattr(demographics, "get_imm_rates", fake_get_imm_rates)

    pop_dict = demographics.get_pop_objs(
        E=E,
        S=S,
        T=T,
        min_age=0,
        max_age=4,
        fert_rates=None,
        mort_rates=None,
        infmort_rates=None,
        imm_rates=None,
        infer_pop=False,
        pop_dist=pop_dist,
        pre_pop_dist=pre_pop,
        initial_data_year=initial_data_year,
        final_data_year=final_data_year,
        GraphDiag=False,
    )

    assert np.allclose(pop_dict["rho_preTP"], prior_mort[0, E:])
    assert np.allclose(pop_dict["imm_rates_preTP"], prior_imm[0, E:])
    assert np.allclose(pop_dict["rho"][0], main_mort[0, E:])
    assert np.allclose(pop_dict["imm_rates"][0], main_imm[0, E:])


def test_pre_mort_rates_kwarg_collapses_boundary_identity():
    """
    Supplying ``pre_mort_rates`` to ``get_pop_objs`` should populate the
    boundary objects from genuine prior-year data so that the preTP -> 0
    demographic identity holds at machine precision (s in [1, S-1]).

    The synthetic universe has rates that vary year over year, so the
    legacy fallback (preTP rates = period-0 rates) leaves a residual
    proportional to the year-over-year rate change.
    """
    E = 0
    S = 5
    T = 8
    initial_data_year = 2020
    final_data_year = 2022

    mort_yrm1 = np.array([0.05, 0.10, 0.15, 0.30, 1.00])
    imm_yrm1 = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
    fert_yrm1 = np.array([0.20, 0.30, 0.10, 0.00, 0.00])
    infmort_yrm1 = 0.04

    main_mort = np.array(
        [
            [0.07, 0.13, 0.18, 0.34, 1.00],   # year 2020
            [0.08, 0.14, 0.19, 0.35, 1.00],   # year 2021
            [0.08, 0.14, 0.19, 0.35, 1.00],   # year 2022
        ]
    )
    main_fert = np.array(
        [
            [0.18, 0.28, 0.09, 0.00, 0.00],
            [0.17, 0.27, 0.08, 0.00, 0.00],
            [0.17, 0.27, 0.08, 0.00, 0.00],
        ]
    )
    main_infmort = np.array([0.03, 0.03, 0.03])
    main_imm = np.array(
        [
            [0.05, 0.06, 0.07, 0.08, 0.09],
            [0.05, 0.06, 0.07, 0.08, 0.09],
            [0.05, 0.06, 0.07, 0.08, 0.09],
        ]
    )

    pre_pop = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
    pop0 = step_population(pre_pop, fert_yrm1, mort_yrm1, infmort_yrm1, imm_yrm1)
    pop1 = step_population(
        pop0, main_fert[0], main_mort[0], main_infmort[0], main_imm[0]
    )
    pop2 = step_population(
        pop1, main_fert[1], main_mort[1], main_infmort[1], main_imm[1]
    )
    pop3 = step_population(
        pop2, main_fert[2], main_mort[2], main_infmort[2], main_imm[2]
    )
    pop_dist = np.vstack((pop0, pop1, pop2, pop3))

    pop_dict = demographics.get_pop_objs(
        E=E,
        S=S,
        T=T,
        min_age=0,
        max_age=4,
        fert_rates=main_fert,
        mort_rates=main_mort,
        infmort_rates=main_infmort,
        imm_rates=main_imm,
        infer_pop=False,
        pop_dist=pop_dist,
        pre_pop_dist=pre_pop,
        pre_mort_rates=mort_yrm1,
        initial_data_year=initial_data_year,
        final_data_year=final_data_year,
        GraphDiag=False,
    )

    # Boundary identity, S-space, s in [1, S-1]:
    #   omega[0, s] * (1 + g_n_preTP)
    #     == (1 - rho_preTP[s-1]) * omega_S_preTP[s-1]
    #        + imm_rates_preTP[s] * omega_S_preTP[s]
    omega_0 = pop_dict["omega"][0, :]
    omega_S_preTP = pop_dict["omega_S_preTP"]
    g_n_preTP = pop_dict["g_n_preTP"]
    rho_preTP = pop_dict["rho_preTP"]
    imm_preTP = pop_dict["imm_rates_preTP"]

    lhs = omega_0[1:] * (1.0 + g_n_preTP)
    rhs = (1.0 - rho_preTP[:-1]) * omega_S_preTP[:-1] + imm_preTP[1:] * omega_S_preTP[1:]
    assert np.allclose(lhs, rhs, atol=1e-13)
    assert np.allclose(rho_preTP, mort_yrm1[E:])


