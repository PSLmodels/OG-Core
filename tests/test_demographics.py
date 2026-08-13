import numpy as np
import pytest
import os
import base64
import datetime
import json
from ogcore import demographics

# Read in some test population data to use in select tests below
# Tests that ping the UN data portal are marked with the "local" mark
data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_io_data"
)
fert_rates = np.loadtxt(
    os.path.join(data_dir, "fert_rates.csv"), delimiter=","
)
mort_rates = np.loadtxt(
    os.path.join(data_dir, "mort_rates.csv"), delimiter=","
)
infmort_rates = np.loadtxt(
    os.path.join(data_dir, "infmort_rates.csv"), delimiter=","
)
imm_rates = np.loadtxt(
    os.path.join(data_dir, "immigration_rates.csv"), delimiter=","
)
pop_dist = np.loadtxt(
    os.path.join(data_dir, "population_distribution.csv"), delimiter=","
)


@pytest.mark.local
def test_get_pop_objs_read_UN_data():
    """
    Test of the that omega_SS and the last period of omega_path_S are
    close to each other.
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2024

    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        initial_data_year=start_year - 1,
        final_data_year=start_year,
        GraphDiag=False,
    )

    assert isinstance(pop_dict, dict)


def test_get_pop_objs():
    """
    Test of the that omega_SS and the last period of omega_path_S are
    close to each other.
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2024

    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        fert_rates=fert_rates,
        mort_rates=mort_rates,
        infmort_rates=infmort_rates,
        imm_rates=imm_rates,
        infer_pop=True,
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        pop_dist=pop_dist[0, :].reshape(1, E + S),
        initial_data_year=start_year - 1,
        final_data_year=start_year,
        GraphDiag=False,
    )
    assert np.allclose(pop_dict["omega_SS"], pop_dict["omega"][-1, :, :])


def test_pop_smooth():
    """
    Test that population distribution and pop growth rates evolve smoothly.
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2024
    fixper = int(1.5 * S)
    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        fert_rates=fert_rates,
        mort_rates=mort_rates,
        infmort_rates=infmort_rates,
        imm_rates=imm_rates,
        infer_pop=True,
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        pop_dist=pop_dist[0, :].reshape(1, E + S),
        initial_data_year=start_year - 1,
        final_data_year=start_year,
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
    start_year = 2024
    fixper = int(1.5 * S)
    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        fert_rates=fert_rates,
        mort_rates=mort_rates,
        infmort_rates=infmort_rates,
        imm_rates=imm_rates,
        infer_pop=True,
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        pop_dist=pop_dist[0, :].reshape(1, E + S),
        initial_data_year=start_year - 1,
        final_data_year=start_year,
        country_id="840",
        GraphDiag=False,
    )

    # assert diffs are small
    # note that in the "fixper" we impost a jump in immigration rates
    # to achieve the SS more quickly so the min dist is not super small
    # in that period
    print("first few of g_n = ", pop_dict["g_n"][:5])
    assert np.all(
        np.abs(pop_dict["g_n"][: fixper - 2] - pop_dict["g_n"][1 : fixper - 1])
        < 0.003
    )
    assert np.all(
        np.abs(pop_dict["g_n"][fixper:-1] - pop_dict["g_n"][fixper + 1 :])
        < 0.003
    )


@pytest.mark.local
def test_imm_smooth():
    """
    Test that immigration rates evolve smoothly.
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2024
    fixper = int(1.5 * S + 2)
    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        initial_data_year=start_year - 1,
        final_data_year=start_year,
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        GraphDiag=False,
    )
    # assert diffs are small
    # note that in the "fixper" we impost a jump in immigration rates
    # to achieve the SS more quickly so the min dist is not super small
    # in that period
    print(
        "Max diff before = ",
        np.abs(
            pop_dict["imm_rates"][: fixper - 2, :]
            - pop_dict["imm_rates"][1 : fixper - 1, :]
        ).max(),
    )

    print(
        "Max diff after = ",
        np.abs(
            pop_dict["imm_rates"][fixper:-1, :]
            - pop_dict["imm_rates"][fixper + 1 :, :]
        ).max(),
    )
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
        < 0.00001
    )


@pytest.mark.local
def test_get_fert():
    """
    Test of function to get fertility rates from data
    """
    S = 100
    fert_rates, fig = demographics.get_fert(S, 0, 99, graph=True)
    assert fert_rates.shape[1] == S
    assert fig


@pytest.mark.local
def test_get_fert_multi_year():
    """
    Test get_fert with multiple years (start_year != end_year) and graph=True.
    Covers the else branch that builds years_list and fert_rates_list for
    multiple years.
    """
    S = 100
    fert_rates, fig = demographics.get_fert(
        S, 0, 99, start_year=2023, end_year=2024, graph=True
    )
    assert fert_rates.shape == (2, S)
    assert fig


@pytest.mark.local
def test_get_fert_graph_with_plot_path(tmpdir):
    """
    Test get_fert with graph=True and a plot_path.
    Covers the plot_path is not None branch (saves figure, returns only
    fert_rates_2D without a fig object).
    """
    import os
    import matplotlib.image as mpimg

    S = 100
    result = demographics.get_fert(S, 0, 99, graph=True, plot_path=str(tmpdir))
    # When plot_path is given, only fert_rates_2D is returned (not a tuple)
    assert result.shape[1] == S
    img = mpimg.imread(os.path.join(str(tmpdir), "fert_rates.png"))
    assert isinstance(img, np.ndarray)


@pytest.mark.local
def test_get_mort():
    """
    Test of function to get mortality rates from data
    """
    S = 100
    mort_rates, infmort_rate, fig = demographics.get_mort(S, 0, 99, graph=True)
    assert mort_rates.shape[1] == S
    assert fig


@pytest.mark.local
def test_infant_mort():
    """
    Test of function to get mortality rates from data
    """
    mort_rates, infmort_rate = demographics.get_mort(100, 0, 99, graph=False)
    # check that infant mortality equals rate hardcoded into
    # demographics.py
    assert infmort_rate == 0.00496921


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
    imm_rates, fig = demographics.get_imm_rates(
        S,
        0,
        99,
        fert_rates=fert_rates,
        mort_rates=mort_rates,
        infmort_rates=infmort_rates,
        pop_dist=pop_dist,
        start_year=2024,
        end_year=2025,
        graph=True,
    )
    assert imm_rates.shape[1] == S
    assert fig


# Test functionality when passing in a custom series of immigration rates
@pytest.mark.local
def test_custom_series():
    """
    Test of the get pop objects function when passing in a custom series
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2024
    imm_rates = np.ones((2, E + S)) * 0.01
    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        initial_data_year=start_year,
        final_data_year=start_year + 1,
        GraphDiag=False,
        imm_rates=imm_rates,
        infer_pop=True,
    )
    assert np.allclose(pop_dict["imm_rates"][0, :, 0], imm_rates[0, E:])


def test_custom_series_fail():
    """
    Test of the get pop objects function when passing in custom series
    for fertility, mortality, immigration, and population

    This test gives a pop dist that doesn't result from the fert, mort,
    and immigration rates.  This should raise an error.
    """
    with pytest.raises(Exception) as e_info:
        E = 20
        S = 80
        T = int(round(4.0 * S))
        start_year = 2024
        fert_rates = demographics.get_fert(
            E + S,
            0,
            99,
            start_year=start_year,
            end_year=start_year + 1,
            graph=False,
        )
        mort_rates, infmort_rates = demographics.get_mort(
            E + S,
            0,
            99,
            start_year=start_year,
            end_year=start_year + 1,
            graph=False,
        )
        imm_rates = np.ones((2, E + S)) * 0.01
        pop_dist = np.zeros((2, E + S))
        for t in range(pop_dist.shape[0]):
            df = demographics.get_un_data(
                "47", start_year=start_year + t, end_year=start_year + t
            )
            pop = df[(df.age < 100) & (df.age >= 0)].value.values
            pop_dist[t, :] = demographics.pop_rebin(pop, E + S)
        df = demographics.get_un_data(
            "47", start_year=start_year - 1, end_year=start_year - 1
        )
        pop = df[(df.age < 100) & (df.age >= 0)].value.values
        pop_dict = demographics.get_pop_objs(
            E,
            S,
            T,
            0,
            99,
            fert_rates=fert_rates,
            mort_rates=mort_rates,
            infmort_rates=infmort_rates,
            imm_rates=imm_rates,
            income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
            pop_dist=pop_dist,
            initial_data_year=start_year,
            final_data_year=start_year + 1,
            GraphDiag=False,
        )


# Test that SS solved for
def test_SS_dist():
    """
    Test of the that omega_SS is found by period T (so in SS for last
    S periods of the T+S transition path)
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2024

    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        fert_rates=fert_rates,
        mort_rates=mort_rates,
        infmort_rates=infmort_rates,
        imm_rates=imm_rates,
        infer_pop=True,
        pop_dist=pop_dist[0, :].reshape(1, E + S),
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        initial_data_year=start_year - 1,
        final_data_year=start_year,
        GraphDiag=False,
    )
    # Assert that S reached by period T
    assert np.allclose(pop_dict["omega_SS"], pop_dict["omega"][-S, :])
    assert np.allclose(pop_dict["omega_SS"], pop_dict["omega"][-1, :])


# Test all time path variables returned are of T+S length in the time dimension
def test_time_path_length():
    """
    Test of the that omega_SS is found by period T (so in SS for last
    S periods of the T+S transition path)
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2024

    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        fert_rates=fert_rates,
        mort_rates=mort_rates,
        infmort_rates=infmort_rates,
        imm_rates=imm_rates,
        infer_pop=True,
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        pop_dist=pop_dist[0, :].reshape(1, E + S),
        initial_data_year=start_year - 1,
        final_data_year=start_year,
        GraphDiag=False,
    )
    # Assert that S reached by period T
    assert pop_dict["omega"].shape[0] == T + S
    assert pop_dict["g_n"].shape[0] == T + S
    assert pop_dict["imm_rates"].shape[0] == T + S
    assert pop_dict["rho"].shape[0] == T + S
    assert np.isscalar(pop_dict["g_n_preTP"])


# test of get pop when infer population, but don't pass initial pop
@pytest.mark.local
def test_infer_pop_nones():
    """
    Test of the get pop objects function when passing in custom series
    for fertility, mortality, immigration, and population
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2024
    fert_rates = demographics.get_fert(
        E + S,
        0,
        99,
        start_year=start_year,
        end_year=start_year + 1,
        graph=False,
    )
    mort_rates, infmort_rates = demographics.get_mort(
        E + S,
        0,
        99,
        start_year=start_year,
        end_year=start_year + 1,
        graph=False,
    )
    imm_rates = demographics.get_imm_rates(
        E + S,
        0,
        99,
        fert_rates=fert_rates,
        mort_rates=mort_rates,
        infmort_rates=infmort_rates,
        start_year=start_year,
        end_year=start_year + 1,
        graph=False,
    )

    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        fert_rates=fert_rates,
        mort_rates=mort_rates,
        infmort_rates=infmort_rates,
        imm_rates=imm_rates,
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        infer_pop=True,
        pop_dist=None,
        initial_data_year=start_year,
        final_data_year=start_year + 1,
        GraphDiag=False,
    )
    assert pop_dict is not None


# Test data download option
@pytest.mark.local
def test_data_download(tmpdir):
    """
    Test of the data download function passing through get_pop_objs
    to all other functions that use it
    """
    E = 20
    S = 80
    T = int(round(4.0 * S))
    start_year = 2024

    pop_dict = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        initial_data_year=start_year,
        final_data_year=start_year + 1,
        download_path=tmpdir,
    )

    # Now read in each file and call get_pop_objs again with the data
    fert_rates = np.loadtxt(
        os.path.join(tmpdir, "fert_rates.csv"), delimiter=","
    )
    mort_rates = np.loadtxt(
        os.path.join(tmpdir, "mort_rates.csv"), delimiter=","
    )
    infmort_rates = np.loadtxt(
        os.path.join(tmpdir, "infmort_rates.csv"), delimiter=","
    )
    imm_rates = np.loadtxt(
        os.path.join(tmpdir, "immigration_rates.csv"), delimiter=","
    )
    pop_dist = np.loadtxt(
        os.path.join(tmpdir, "population_distribution.csv"), delimiter=","
    )
    pop_dict2 = demographics.get_pop_objs(
        E,
        S,
        T,
        0,
        99,
        fert_rates=fert_rates[:, :],
        mort_rates=mort_rates[:, :],
        infmort_rates=infmort_rates[:],
        imm_rates=imm_rates[:, :],
        infer_pop=True,
        income_percentiles=[0.25, 0.25, 0.2, 0.2, 0.05, 0.04, 0.01],
        pop_dist=pop_dist[0, :].reshape(1, E + S),
        initial_data_year=start_year,
        final_data_year=start_year + 2,
        GraphDiag=False,
    )

    # Assert that the two pop_dicts are the same
    for key in pop_dict:
        print(key)
        print("Diff =", np.abs(pop_dict[key] - pop_dict2[key]).max())
        if key == "imm_rates":
            # print the max diff for each T
            for t in range(pop_dict[key].shape[0]):
                print(
                    "Max diff for imm_rates at T = ",
                    t,
                    " is ",
                    np.abs(pop_dict[key][t, :] - pop_dict2[key][t, :]).max(),
                )
    for key in pop_dict:
        print(key)
        assert np.allclose(pop_dict[key], pop_dict2[key], atol=7e-5)


def test_expand_pop_obj_J_tiles_when_no_income_inputs():
    """
    Test that income_percentiles expands aggregate objects by J when no
    income-specific gradients or immigrant shares are supplied.
    """
    E = 1
    S = 3
    J = 3
    num_periods = 5
    fixper = 2
    income_percentiles = np.array([0.5, 0.3, 0.2])
    omega_SSfx = np.array([0.2, 0.3, 0.3, 0.2])
    omega_path_lev = np.tile(omega_SSfx.reshape(1, E + S), (num_periods, 1))
    omega_path_S = omega_path_lev[:, E:] / omega_path_lev[:, E:].sum(
        axis=1
    ).reshape(num_periods, 1)
    fert_rates = np.zeros((num_periods, E + S))
    mort_rates = np.tile(
        np.array([0.01, 0.02, 0.03, 1.0]).reshape(1, E + S),
        (num_periods, 1),
    )
    infmort_rates = np.ones(num_periods) * 0.005
    imm_rates = np.zeros((num_periods, E + S))
    mort_rates_S = mort_rates[:, E:]
    imm_rates_mat = imm_rates[:, E:]

    pop_objs = demographics.expand_pop_obj_J(
        omega_path_lev,
        omega_path_S,
        omega_SSfx,
        fert_rates,
        mort_rates,
        infmort_rates,
        imm_rates,
        mort_rates_S,
        imm_rates_mat,
        E,
        S,
        0.0,
        fixper,
        income_percentiles=income_percentiles,
    )

    assert pop_objs["omega_path_S"].shape == (num_periods, S, J)
    assert np.allclose(pop_objs["omega_path_S"].sum(axis=2), omega_path_S)
    assert np.allclose(pop_objs["mort_rates_S"][:, :, 0], mort_rates_S)
    assert np.allclose(pop_objs["imm_rates_mat"][:, :, 0], imm_rates_mat)
    assert np.allclose(
        pop_objs["omega_SS"],
        omega_path_S[fixper, :, None] * income_percentiles,
    )


def test_expand_pop_obj_J_defaults_to_single_group():
    """
    Regression test for Issue #1180: with income_percentiles=None and no
    income-specific inputs (the pre-0.18 call signature), the aggregate
    objects must be broadcast across a single income group rather than
    raising an AssertionError.
    """
    E = 1
    S = 3
    num_periods = 5
    fixper = 2
    omega_SSfx = np.array([0.2, 0.3, 0.3, 0.2])
    omega_path_lev = np.tile(omega_SSfx.reshape(1, E + S), (num_periods, 1))
    omega_path_S = omega_path_lev[:, E:] / omega_path_lev[:, E:].sum(
        axis=1
    ).reshape(num_periods, 1)
    fert_rates = np.zeros((num_periods, E + S))
    mort_rates = np.tile(
        np.array([0.01, 0.02, 0.03, 1.0]).reshape(1, E + S),
        (num_periods, 1),
    )
    infmort_rates = np.ones(num_periods) * 0.005
    imm_rates = np.zeros((num_periods, E + S))
    mort_rates_S = mort_rates[:, E:]
    imm_rates_mat = imm_rates[:, E:]

    pop_objs = demographics.expand_pop_obj_J(
        omega_path_lev,
        omega_path_S,
        omega_SSfx,
        fert_rates,
        mort_rates,
        infmort_rates,
        imm_rates,
        mort_rates_S,
        imm_rates_mat,
        E,
        S,
        0.0,
        fixper,
    )

    assert pop_objs["omega_path_S"].shape == (num_periods, S, 1)
    assert np.allclose(pop_objs["omega_path_S"][:, :, 0], omega_path_S)
    assert pop_objs["omega_SS"].shape == (S, 1)
    assert np.allclose(pop_objs["mort_rates_S"][:, :, 0], mort_rates_S)
    assert np.allclose(pop_objs["imm_rates_mat"][:, :, 0], imm_rates_mat)


def test_expand_pop_obj_J_income_inputs_require_percentiles():
    """
    income_percentiles stays required when income-specific inputs are
    supplied: providing a gradient without percentiles must raise.
    """
    E = 1
    S = 3
    num_periods = 5
    fixper = 2
    omega_SSfx = np.array([0.2, 0.3, 0.3, 0.2])
    omega_path_lev = np.tile(omega_SSfx.reshape(1, E + S), (num_periods, 1))
    omega_path_S = omega_path_lev[:, E:] / omega_path_lev[:, E:].sum(
        axis=1
    ).reshape(num_periods, 1)
    fert_rates = np.zeros((num_periods, E + S))
    mort_rates = np.tile(
        np.array([0.01, 0.02, 0.03, 1.0]).reshape(1, E + S),
        (num_periods, 1),
    )
    infmort_rates = np.ones(num_periods) * 0.005
    imm_rates = np.zeros((num_periods, E + S))

    with pytest.raises(AssertionError, match="income_percentiles"):
        demographics.expand_pop_obj_J(
            omega_path_lev,
            omega_path_S,
            omega_SSfx,
            fert_rates,
            mort_rates,
            infmort_rates,
            imm_rates,
            mort_rates[:, E:],
            imm_rates[:, E:],
            E,
            S,
            0.0,
            fixper,
            mort_gradient=np.zeros(E + S),
        )


def test_expand_pop_obj_J_preserves_aggregate_mortality_with_gradients():
    """
    Test that log-odds gradients generate bounded J-specific rates whose
    within-age weighted means recover aggregate mortality rates.
    """
    E = 1
    S = 3
    J = 3
    num_periods = 6
    fixper = 3
    g_n_ss = 0.01
    income_percentiles = np.array([0.5, 0.3, 0.2])
    omega_SSfx = np.array([0.2, 0.3, 0.3, 0.2])
    omega_path_lev = np.zeros((num_periods, E + S))
    for t in range(num_periods):
        omega_path_lev[t] = 100 * ((1 + g_n_ss) ** t) * omega_SSfx
    omega_path_S = omega_path_lev[:, E:] / omega_path_lev[:, E:].sum(
        axis=1
    ).reshape(num_periods, 1)
    omega_path_S[fixper:] = omega_path_S[fixper]
    fert_rates = np.tile(
        np.array([0.0, 0.02, 0.01, 0.0]).reshape(1, E + S),
        (num_periods, 1),
    )
    mort_rates = np.tile(
        np.array([0.01, 0.02, 0.03, 1.0]).reshape(1, E + S),
        (num_periods, 1),
    )
    infmort_rates = np.ones(num_periods) * 0.005
    imm_rates = np.zeros((num_periods, E + S))
    mort_rates_S = mort_rates[:, E:]
    imm_rates_mat = imm_rates[:, E:]
    mort_gradient = np.array([-0.01, -0.005, 0.0])
    fert_gradient = np.array([0.002, 0.001, 0.0])

    pop_objs = demographics.expand_pop_obj_J(
        omega_path_lev,
        omega_path_S,
        omega_SSfx,
        fert_rates,
        mort_rates,
        infmort_rates,
        imm_rates,
        mort_rates_S,
        imm_rates_mat,
        E,
        S,
        g_n_ss,
        fixper,
        income_percentiles=income_percentiles,
        fert_gradient=fert_gradient,
        mort_gradient=mort_gradient,
    )

    omega = pop_objs["omega_path_S"]
    rho = pop_objs["mort_rates_S"]
    within_age_weights = omega / omega.sum(axis=2, keepdims=True)
    assert omega.shape == (num_periods, S, J)
    assert rho.shape == (num_periods, S, J)
    assert np.allclose(omega.sum(axis=2), omega_path_S)
    assert np.allclose((within_age_weights * rho).sum(axis=2), mort_rates_S)
    assert np.all((rho >= 0) & (rho <= 1))


"""
------------------------------------------------------------------------
Tests of the UN Data Portal API token resolution
------------------------------------------------------------------------
"""


@pytest.fixture
def isolated_token_env(monkeypatch, tmp_path):
    """
    Point token resolution at a temporary home and working directory so
    the tests never read or write the developer's real token.
    """
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.delenv("UN_API_TOKEN", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("APPDATA", str(home / "AppData"))
    monkeypatch.setattr(demographics, "_WARNED_LEGACY_UN_TOKEN", False)
    monkeypatch.setattr(demographics, "_HINTED_NO_UN_TOKEN", False)
    monkeypatch.chdir(cwd)
    return cwd


class _Stdin:
    """Stand-in for sys.stdin with a controllable isatty()."""

    def __init__(self, interactive):
        self.interactive = interactive

    def isatty(self):
        return self.interactive


def _set_tty(monkeypatch, interactive):
    """
    Control whether resolution believes the session is interactive.
    pytest captures stdin, so isatty() is False by default and the prompt
    path has to be switched on explicitly.
    """
    monkeypatch.setattr("sys.stdin", _Stdin(interactive))


def test_un_token_path_follows_platform_convention(monkeypatch, tmp_path):
    """The token file sits under the user's config directory."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "cfg"))
    path = demographics.un_token_path()
    assert path.startswith(str(tmp_path / "cfg"))
    assert path.endswith(demographics.UN_TOKEN_FILENAME)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("abc", "abc"),
        ("  abc  ", "abc"),
        ("Bearer abc", "abc"),
        ("bearer  abc ", "abc"),
        (None, ""),
        ("", ""),
    ],
    ids=["plain", "padded", "bearer", "lower_bearer", "none", "empty"],
)
def test_clean_un_token(raw, expected):
    """Whitespace and any 'Bearer ' prefix are removed."""
    assert demographics._clean_un_token(raw) == expected


def test_argument_wins_over_every_other_source(
    isolated_token_env, monkeypatch
):
    """An explicit token beats the environment and both files."""
    monkeypatch.setenv("UN_API_TOKEN", "from_env")
    (isolated_token_env / demographics.UN_TOKEN_FILENAME).write_text(
        "from_cwd"
    )
    assert demographics.resolve_un_token("from_arg") == "from_arg"


def test_env_var_wins_over_files(isolated_token_env, monkeypatch):
    """The environment variable beats both the user file and the cwd file."""
    monkeypatch.setenv("UN_API_TOKEN", "Bearer from_env")
    user_path = demographics.un_token_path()
    os.makedirs(os.path.dirname(user_path), exist_ok=True)
    with open(user_path, "w") as f:
        f.write("from_user_file")
    (isolated_token_env / demographics.UN_TOKEN_FILENAME).write_text(
        "from_cwd"
    )
    assert demographics.resolve_un_token() == "from_env"


def test_user_file_wins_over_cwd_file(isolated_token_env):
    """The per-user file beats the deprecated working-directory file."""
    user_path = demographics.un_token_path()
    os.makedirs(os.path.dirname(user_path), exist_ok=True)
    with open(user_path, "w") as f:
        f.write("from_user_file\n")
    (isolated_token_env / demographics.UN_TOKEN_FILENAME).write_text(
        "from_cwd"
    )
    assert demographics.resolve_un_token() == "from_user_file"


def test_empty_user_file_is_authoritative(isolated_token_env, monkeypatch):
    """
    A user who declined the prompt is not asked again, and the cwd file is
    not consulted behind their back.
    """
    user_path = demographics.un_token_path()
    os.makedirs(os.path.dirname(user_path), exist_ok=True)
    with open(user_path, "w") as f:
        f.write("")
    (isolated_token_env / demographics.UN_TOKEN_FILENAME).write_text(
        "from_cwd"
    )

    def _fail(*args, **kwargs):
        raise AssertionError("the user should not be prompted again")

    monkeypatch.setattr(demographics.getpass, "getpass", _fail)
    assert demographics.resolve_un_token() == ""


def test_cwd_file_still_works_and_warns(isolated_token_env, capsys):
    """The old location keeps working, with a one-time deprecation notice."""
    (isolated_token_env / demographics.UN_TOKEN_FILENAME).write_text(
        "from_cwd"
    )
    assert demographics.resolve_un_token() == "from_cwd"
    assert "deprecated" in capsys.readouterr().out
    # the notice is not repeated on later calls in the same session
    assert demographics.resolve_un_token() == "from_cwd"
    assert "deprecated" not in capsys.readouterr().out


def test_prompt_saves_to_the_user_file_not_the_working_directory(
    isolated_token_env, monkeypatch
):
    """This is the behavior change: answering the prompt no longer leaves a
    copy of the token in whatever directory the run started from."""
    _set_tty(monkeypatch, True)
    monkeypatch.setattr(
        demographics.getpass, "getpass", lambda *a, **k: "typed_token"
    )

    assert demographics.resolve_un_token() == "typed_token"

    user_path = demographics.un_token_path()
    assert os.path.exists(user_path)
    with open(user_path) as f:
        assert f.read() == "typed_token"
    assert not os.path.exists(
        isolated_token_env / demographics.UN_TOKEN_FILENAME
    )


def test_no_prompt_when_not_interactive(isolated_token_env, monkeypatch):
    """A scheduled run gets an empty token instead of hanging, and writes
    nothing to disk."""
    _set_tty(monkeypatch, False)

    def _fail(*args, **kwargs):
        raise AssertionError("a non-interactive session must not prompt")

    monkeypatch.setattr(demographics.getpass, "getpass", _fail)
    assert demographics.resolve_un_token() == ""
    assert not os.path.exists(demographics.un_token_path())


def test_get_un_data_sends_the_resolved_token(isolated_token_env, monkeypatch):
    """The token reaches the request header, so the wiring from argument to
    Authorization is covered and not just the resolver in isolation."""
    sent = {}

    class _Response:
        status_code = 401  # short-circuits before any parsing

    class _Session:
        def get(self, target, headers=None, data=None):
            sent["headers"] = headers
            return _Response()

    monkeypatch.setattr(demographics, "get_legacy_session", lambda: _Session())
    demographics.get_un_data("47", un_token="Bearer explicit_token")
    assert sent["headers"]["Authorization"] == "Bearer explicit_token"

    monkeypatch.setenv("UN_API_TOKEN", "env_token")
    demographics.get_un_data("47")
    assert sent["headers"]["Authorization"] == "Bearer env_token"


def test_un_token_cli_set_show_rm(isolated_token_env, monkeypatch, capsys):
    """The og-token command round-trips: save, report, delete."""
    monkeypatch.setattr(
        demographics.getpass, "getpass", lambda *a, **k: "Bearer cli_tok1234"
    )

    assert demographics.un_token_cli(["set"]) == 0
    path = demographics.un_token_path()
    with open(path) as f:
        assert f.read() == "cli_tok1234"  # the Bearer prefix is stripped
    assert demographics.resolve_un_token() == "cli_tok1234"

    assert demographics.un_token_cli(["show"]) == 0
    out = capsys.readouterr().out
    assert path in out
    assert "a token is stored" in out
    assert "cli_tok1234" not in out  # no part of the token is printed
    assert "1234" not in out

    assert demographics.un_token_cli(["rm"]) == 0
    assert not os.path.exists(path)
    assert demographics.un_token_cli(["rm"]) == 1  # nothing left to remove


def test_un_token_cli_set_rejects_an_empty_answer(
    isolated_token_env, monkeypatch, capsys
):
    """Pressing return at the prompt writes nothing and reports failure."""
    monkeypatch.setattr(demographics.getpass, "getpass", lambda *a, **k: "  ")

    assert demographics.un_token_cli(["set"]) == 1
    assert not os.path.exists(demographics.un_token_path())
    assert "Nothing was saved" in capsys.readouterr().out


def test_un_token_cli_show_flags_the_environment_override(
    isolated_token_env, monkeypatch, capsys
):
    """show warns when the environment variable will win over the file."""
    monkeypatch.setenv("UN_API_TOKEN", "env_token")
    assert demographics.un_token_cli(["show"]) == 0
    assert "takes precedence" in capsys.readouterr().out


@pytest.mark.parametrize(
    "interrupt", [EOFError, KeyboardInterrupt], ids=["eof", "ctrl_c"]
)
def test_un_token_cli_set_handles_an_interrupted_prompt(
    isolated_token_env, monkeypatch, capsys, interrupt
):
    """`og-token set < /dev/null` or Ctrl-C reports cleanly instead of
    printing a traceback."""

    def _raise(*args, **kwargs):
        raise interrupt()

    monkeypatch.setattr(demographics.getpass, "getpass", _raise)
    assert demographics.un_token_cli(["set"]) == 1
    assert "Cancelled" in capsys.readouterr().out
    assert not os.path.exists(demographics.un_token_path())


def test_prompt_offers_both_ways_out(isolated_token_env, monkeypatch, capsys):
    """The prompt tells the user where to get a token and what happens if
    they decline, rather than leaving them to guess."""
    _set_tty(monkeypatch, True)
    monkeypatch.setattr(demographics.getpass, "getpass", lambda *a, **k: "")

    assert demographics.resolve_un_token() == ""
    out = capsys.readouterr().out
    assert demographics.UN_TOKEN_URL in out
    assert demographics.UN_DATA_ARCHIVE_URL in out


def test_hint_names_a_runnable_command_and_the_file(
    isolated_token_env, monkeypatch, capsys
):
    """Falling back to the archive tells the user how to register a token.
    `og-token` is installed beside the interpreter and is normally not on
    the shell's PATH, so the hint has to give the full path, plus the file
    as a way that needs no environment at all."""
    _set_tty(monkeypatch, False)  # no prompt, straight to the fallback

    assert demographics.resolve_un_token() == ""
    out = capsys.readouterr().out
    assert demographics.UN_TOKEN_URL in out
    assert demographics.un_token_path() in out  # always actionable

    command = demographics.og_token_command()
    if command is not None:
        assert command in out
        assert os.path.isabs(command)  # copy-pasteable from any shell


def test_hint_is_printed_once_per_session(
    isolated_token_env, monkeypatch, capsys
):
    """get_pop_objs resolves a token once per series, so the hint must not
    repeat three times in a single run."""
    _set_tty(monkeypatch, False)

    demographics.resolve_un_token()
    first = capsys.readouterr().out
    assert "No UN API token registered" in first

    demographics.resolve_un_token()
    demographics.resolve_un_token()
    assert "No UN API token registered" not in capsys.readouterr().out


def test_no_hint_when_a_token_is_present(
    isolated_token_env, monkeypatch, capsys
):
    """Users who have a token are not nagged."""
    monkeypatch.setenv("UN_API_TOKEN", "a_real_token")
    assert demographics.resolve_un_token() == "a_real_token"
    assert "No UN API token registered" not in capsys.readouterr().out


def test_token_is_never_echoed_at_the_prompt(
    isolated_token_env, monkeypatch, capsys
):
    """The token is a secret, so it is read with getpass and never reaches
    the terminal or its scrollback. Reverting to input() would echo it."""
    _set_tty(monkeypatch, True)

    def _must_not_be_used(*args, **kwargs):
        raise AssertionError("input() echoes; the token must use getpass")

    monkeypatch.setattr("builtins.input", _must_not_be_used)
    monkeypatch.setattr(
        demographics.getpass, "getpass", lambda *a, **k: "s3cret_token"
    )

    assert demographics.resolve_un_token() == "s3cret_token"
    out = capsys.readouterr().out
    assert "s3cret_token" not in out  # not printed back either


def _make_jwt(days_from_today):
    """Build a token shaped like the portal's: three base64url segments
    with an `exp` claim. Only the payload matters here; the signature is
    never checked."""
    exp = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        days=days_from_today
    )

    def seg(obj):
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return ".".join(
        [
            seg({"alg": "HS256", "typ": "JWT"}),
            seg({"exp": int(exp.timestamp()), "unique_name": "someone"}),
            "not_a_real_signature",
        ]
    )


def test_un_token_expiry_reads_the_claim():
    """The portal issues JWTs, so the expiry is readable without a call."""
    expiry = demographics.un_token_expiry(_make_jwt(30))
    assert (
        expiry
        == (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(30)
        ).date()
    )


@pytest.mark.parametrize(
    "value",
    ["opaque-token-with-no-dots", "a.b.c", "", None, 12345, "a..c"],
    ids=["opaque", "not_base64", "empty", "none", "not_a_string", "empty_seg"],
)
def test_un_token_expiry_degrades_quietly(value):
    """Anything not a readable JWT returns None rather than raising, so a
    model run never dies because the portal changed its token format."""
    assert demographics.un_token_expiry(value) is None


def test_expired_token_is_reported_once(
    isolated_token_env, monkeypatch, capsys
):
    """An expired token otherwise fails the same silent-looking way as a
    missing one."""
    monkeypatch.setattr(demographics, "_WARNED_EXPIRED_UN_TOKEN", False)
    expired = _make_jwt(-5)
    monkeypatch.setenv("UN_API_TOKEN", expired)

    assert demographics.resolve_un_token() == expired  # still returned
    out = capsys.readouterr().out
    assert "expired on" in out
    assert demographics.UN_TOKEN_URL in out
    assert expired not in out  # the token itself is never printed

    demographics.resolve_un_token()
    assert "expired on" not in capsys.readouterr().out  # once per session


def test_valid_token_is_not_flagged(isolated_token_env, monkeypatch, capsys):
    """A token with time left produces no noise."""
    monkeypatch.setattr(demographics, "_WARNED_EXPIRED_UN_TOKEN", False)
    monkeypatch.setenv("UN_API_TOKEN", _make_jwt(60))

    demographics.resolve_un_token()
    assert "expired" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "days,expected", [(-5, "expired on"), (60, "valid until")]
)
def test_cli_show_reports_expiry(
    isolated_token_env, monkeypatch, capsys, days, expected
):
    """og-token show says whether the stored token is still usable."""
    token = _make_jwt(days)
    monkeypatch.setattr(demographics.getpass, "getpass", lambda *a, **k: token)
    assert demographics.un_token_cli(["set"]) == 0
    capsys.readouterr()

    assert demographics.un_token_cli(["show"]) == 0
    out = capsys.readouterr().out
    assert expected in out
    assert token not in out
