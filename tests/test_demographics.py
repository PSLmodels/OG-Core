import numpy as np
from ogcore import demographics


def test_get_pop_objs():
    """
    Test of the that omega_SS and the last period of omega_path_S are
    close to each other.
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2019

    pop_dict = demographics.get_pop_objs(
        E, S, T, 0, 99, start_year - 1, start_year, GraphDiag=False
    )

    assert np.allclose(pop_dict["omega_SS"], pop_dict["omega"][-1, :])


def test_pop_smooth():
    """
    Test that population distribution and pop growth rates evolve smoothly.
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2019
    fixper = int(1.5 * S)
    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        start_year - 1,
        start_year,
        country_id="840",
        GraphDiag=False,
    )

    # assert diffs are small
    # note that in the "fixper" we impost a jump in immigration rates
    # to achieve the SS more quickly so the min dist is not super small
    # in that period
    assert np.all(
        np.abs(
            pop_dict["omega"][: fixper - 2, :]
            - pop_dict["omega"][1 : fixper - 1, :]
        )
        < 0.003
    )
    assert np.all(
        np.abs(
            pop_dict["omega"][fixper:-1, :]
            - pop_dict["omega"][fixper + 1 :, :]
        )
        < 0.0001
    )


def test_pop_growth_smooth():
    """
    Test that population distribution and pop growth rates evolve smoothly.
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2019
    fixper = int(1.5 * S)
    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        start_year - 1,
        start_year,
        country_id="840",
        GraphDiag=False,
    )

    # assert diffs are small
    # note that in the "fixper" we impost a jump in immigration rates
    # to achieve the SS more quickly so the min dist is not super small
    # in that period
    assert np.all(
        np.abs(pop_dict["g_n"][: fixper - 2] - pop_dict["g_n"][1 : fixper - 1])
        < 0.003
    )
    assert np.all(
        np.abs(pop_dict["g_n"][fixper:-1] - pop_dict["g_n"][fixper + 1 :])
        < 0.003
    )


def test_imm_smooth():
    """
    Test that immigration rates evolve smoothly.
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2019
    fixper = int(1.5 * S)
    pop_dict = demographics.get_pop_objs(
        E, S, T, 0, 99, start_year - 1, start_year, GraphDiag=False
    )
    # assert diffs are small
    # note that in the "fixper" we impost a jump in immigration rates
    # to achieve the SS more quickly so the min dist is not super small
    # in that period
    assert np.all(
        np.abs(
            pop_dict["imm_rates"][: fixper - 2, :]
            - pop_dict["imm_rates"][1 : fixper - 1, :]
        )
        < 0.03
    )
    assert np.all(
        np.abs(
            pop_dict["imm_rates"][fixper:-1, :]
            - pop_dict["imm_rates"][fixper + 1 :, :]
        )
        < 0.0001
    )


def test_get_fert():
    """
    Test of function to get fertility rates from data
    """
    S = 100
    fert_rates = demographics.get_fert(S, 0, 99, graph=False)
    assert fert_rates.shape[1] == S


def test_get_mort():
    """
    Test of function to get mortality rates from data
    """
    S = 100
    mort_rates, infmort_rate = demographics.get_mort(S, 0, 99, graph=False)
    assert mort_rates.shape[1] == S


def test_infant_mort():
    """
    Test of function to get mortality rates from data
    """
    mort_rates, infmort_rate = demographics.get_mort(100, 0, 99, graph=False)
    # check that infant mortality equals rate hardcoded into
    # demographics.py
    assert infmort_rate == 0.00491958


def test_pop_rebin():
    """
    Test of population rebin function
    """
    curr_pop_dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    totpers_new = 5
    rebinned_data = demographics.pop_rebin(curr_pop_dist, totpers_new)
    assert rebinned_data.shape[0] == totpers_new


def test_get_imm_rates():
    """
    Test of function to solve for immigration rates from population data
    """
    S = 100
    imm_rates = demographics.get_imm_rates(S, 0, 99)
    assert imm_rates.shape[1] == S


# Test functionality when passing in a custom series of fertility rates
def test_custom_series():
    """
    Test of the get pop objects function when passing in a custom series
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2019
    imm_rates = np.ones((1, E + S)) * 0.01
    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        start_year - 1,
        start_year,
        GraphDiag=False,
        imm_rates=imm_rates,
    )
    assert np.allclose(pop_dict["imm_rates"][0, :], imm_rates[0, E:])


# Test that SS solved for
def test_SS_dist():
    """
    Test of the that omega_SS is found by period T (so in SS for last
    S periods of the T+S transition path)
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2019

    pop_dict = demographics.get_pop_objs(
        E, S, T, 0, 99, start_year - 1, start_year, GraphDiag=False
    )
    # Assert that S reached by period T
    assert np.allclose(pop_dict["omega_SS"], pop_dict["omega"][-S, :])
    assert np.allclose(pop_dict["omega_SS"], pop_dict["omega"][-1, :])
