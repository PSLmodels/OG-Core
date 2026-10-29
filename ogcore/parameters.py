import os
import numpy as np
import scipy.interpolate as si
import paramtools
import ogcore
from ogcore import elliptical_u_est
from ogcore.utils import (
    rate_conversion,
    extrapolate_array,
    extrapolate_nested_list,
)
from ogcore.constants import BASELINE_DIR

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

# Pension parameters that a pension regime may set. A regime assigns its
# values by birth cohort (see Specifications._build_cohort_pension_arrays),
# so each parameter listed here is expanded to a (T + S) x S array indexed
# by period t and age s.
COHORT_PENSION_PARAMS = ("retirement_age",)
# The keys of a pension regime that describe when and to whom it applies
PENSION_REGIME_KEYS = ("year", "max_age_at_start")


class Specifications(paramtools.Parameters):
    """
    Inherits ParamTools Parameters abstract base class.
    """

    defaults = os.path.join(CURRENT_PATH, "default_parameters.json")
    array_first = True

    def __init__(
        self,
        output_base=BASELINE_DIR,
        baseline_dir=BASELINE_DIR,
        baseline=False,
        num_workers=1,
    ):
        super().__init__()

        self.output_base = output_base
        self.baseline_dir = baseline_dir
        self.baseline = baseline
        self.num_workers = num_workers

        # put OG-Core version in parameters to save for reference
        self.ogcore_version = ogcore.__version__

        # Pension regimes are entered through update_specifications and
        # kept outside of paramtools, which cannot validate a list of
        # dicts. See _build_cohort_pension_arrays for their meaning.
        self.pension_regimes = []

        # does cheap calculations to find parameter values
        self.initialize()

        self.parameter_warnings = ""
        self.parameter_errors = ""
        self._ignore_errors = False

    def initialize(self):
        """
        ParametersBase reads JSON file and sets attributes to self
        Next call self.compute_default_params for further initialization

        Args:
            None

        Returns:
            None

        """

        self.compute_default_params()

    def compute_default_params(self):
        """
        Does cheap calculations to return parameter values

        Args:
            None

        Returns:
            None

        """
        # Catch error if baseline_spending=True and baseline=True
        if self.baseline_spending and self.baseline:
            err_msg = (
                "Parameter baseline_spending=True cannot coincide with "
                + "baseline=True."
            )
            raise ValueError(err_msg)

        # reshape lambdas
        self.lambdas = self.lambdas.reshape(self.lambdas.shape[0], 1)
        # cast integers as integers
        self.S = int(self.S)
        self.T = int(self.T)
        self.J = len(self.lambdas)

        # The initial household wealth parameters are optional: a null
        # value in the JSON (or None in a revision) means the time path
        # starts from the steady-state distribution of savings, which is
        # what TPI does when the attribute is None. paramtools stores a
        # null float as NaN and a cleared value as an empty array, so
        # normalize both to None here, and check a supplied distribution.
        for name in ("initial_wealth_ratio", "initial_hhwealth_dist"):
            value = getattr(self, name, None)
            if (
                value is None
                or np.size(value) == 0
                or np.all(np.isnan(np.asarray(value, dtype=float)))
            ):
                setattr(self, name, None)
            elif name == "initial_hhwealth_dist":
                dist = np.asarray(value, dtype=float)
                if dist.shape != (self.S, self.J):
                    raise ValueError(
                        "initial_hhwealth_dist must have shape (S, J) = "
                        f"({self.S}, {self.J}); got {dist.shape}"
                    )
                if not np.isclose(dist.sum(), 1.0, atol=1e-6):
                    raise ValueError(
                        "initial_hhwealth_dist must sum to one; got "
                        f"{dist.sum()}"
                    )
                self.initial_hhwealth_dist = dist
            else:
                self.initial_wealth_ratio = float(np.squeeze(value))

        # beta_annual and chi_b carry one value per lifetime-income group, but
        # paramtools validates values and nesting depth, not list length, so a
        # wrong-length vector used to flow through silently: a short one
        # crashed much later (IndexError deep in the SS solve, far from the
        # cause) and a long one was silently truncated to the first J entries
        # -- wrong economics with no symptom. Guard here, where J is first
        # known: one value broadcasts to all J groups, J values pass through,
        # a uniform vector of any other length is reshaped losslessly
        # (smaller-J runs that keep the uniform defaults rely on this), and a
        # non-uniform mismatch raises immediately.
        for param in ("beta_annual", "chi_b"):
            val = np.asarray(getattr(self, param)).ravel()
            if val.shape[0] != self.J:
                if val.shape[0] == 1 or np.all(val == val[0]):
                    setattr(self, param, np.full(self.J, val[0]))
                else:
                    raise ValueError(
                        f"{param} must have either 1 value (applied to all "
                        f"J groups) or J={self.J} values; got "
                        f"{val.shape[0]} non-uniform values."
                    )

        # get parameters of elliptical utility function
        self.b_ellipse, self.upsilon = elliptical_u_est.estimation(
            self.frisch, self.ltilde
        )
        # determine length of budget window from individual income tax
        # parameters passed in
        self.BW = len(self.etr_params)
        # Find number of economically active periods of life
        self.E = int(
            self.starting_age
            * (self.S / (self.ending_age - self.starting_age))
        )
        # Find rates in model periods from annualized rates
        self.beta = 1 / (
            rate_conversion(
                1 / self.beta_annual - 1,
                self.starting_age,
                self.ending_age,
                self.S,
            )
            + 1
        )
        self.delta = -1 * rate_conversion(
            -1 * self.delta_annual, self.starting_age, self.ending_age, self.S
        )
        self.delta_g = -1 * rate_conversion(
            -1 * self.delta_g_annual,
            self.starting_age,
            self.ending_age,
            self.S,
        )
        self.g_y = rate_conversion(
            self.g_y_annual, self.starting_age, self.ending_age, self.S
        )

        # set fraction of income taxes from payroll to zero initially
        # will be updated when read in tax function parameters
        if self.T + self.S > self.BW:
            self.frac_tax_payroll = np.append(
                self.frac_tax_payroll,
                np.ones(self.T + self.S - self.BW) * self.frac_tax_payroll[-1],
            )

        # Interpolate chi_n and create omega_SS_80 if necessary
        if self.S < 80 and self.chi_n.shape[-1] == 80:
            self.age_midp_80 = np.linspace(20.5, 99.5, 80)
            reshape_chi_n = np.zeros((self.T + self.S, self.S))
            for t in range(self.T + self.S):
                if self.chi_n.ndim == 1:
                    self.chi_n_interp = si.interp1d(
                        self.age_midp_80,
                        np.squeeze(self.chi_n),
                        kind="cubic",
                    )
                else:
                    self.chi_n_interp = si.interp1d(
                        self.age_midp_80,
                        np.squeeze(self.chi_n[t, :]),
                        kind="cubic",
                    )
                self.newstep = 80.0 / self.S
                self.age_midp_S = np.linspace(
                    20 + 0.5 * self.newstep, 100 - 0.5 * self.newstep, self.S
                )
                reshape_chi_n[t, :] = self.chi_n_interp(self.age_midp_S)
            self.chi_n = reshape_chi_n

        # Extend parameters that may vary over the time path
        tp_param_list = [
            "alpha_G",
            "alpha_T",
            "alpha_I",
            "alpha_FA",
            "alpha_bs_G",
            "alpha_bs_T",
            "alpha_bs_I",
            "world_int_rate_annual",
            "adjustment_factor_for_cit_receipts",
            "tau_bq",
            "h_wealth",
            "m_wealth",
            "p_wealth",
            "zeta_D",
            "zeta_K",
            "r_gov_scale",
            "r_gov_shift",
            "g_RM",
        ]
        for item in tp_param_list:
            param_in = getattr(self, item)
            param_out = extrapolate_array(
                param_in, dims=(self.T + self.S,), item=item
            )
            setattr(self, item, param_out)
        # Deal with parameters that vary across industry and over time
        tp_param_list2 = [
            "Z",
            "delta_tau_annual",
            "cit_rate",
            "inv_tax_credit",
        ]
        for item in tp_param_list2:
            param_in = getattr(self, item)
            param_out = extrapolate_array(
                param_in, dims=(self.T + self.S, self.M), item=item
            )
            setattr(self, item, param_out)
        # Deal with parameters that vary across consumption good and over time
        tp_param_list3 = ["tau_c"]
        for item in tp_param_list3:
            param_in = getattr(self, item)
            param_out = extrapolate_array(
                param_in, dims=(self.T + self.S, self.I), item=item
            )
            setattr(self, item, param_out)
        # Deal with parameters that vary across J and over time
        tp_param_list3 = [
            "labor_income_tax_noncompliance_rate",
            "capital_income_tax_noncompliance_rate",
            "replacement_rate_adjust",
            "income_tax_filer",
            "wealth_tax_filer",
            "tau_payroll",
            "alpha_db",
        ]
        for item in tp_param_list3:
            param_in = np.asarray(getattr(self, item))
            # If J has changed since this parameter was last extrapolated
            # (e.g., via update_specifications), a previously extrapolated
            # T+S x J_old array will not match the new J.  When the values
            # are the same across the old J groups, collapse to one column
            # so that it can be re-extrapolated to the new J.  Otherwise,
            # leave it to extrapolate_array to raise the error.
            if (
                param_in.ndim == 2
                and param_in.shape[1] != self.J
                and np.all(param_in == param_in[:, :1])
            ):
                param_in = param_in[:, :1]
            param_out = extrapolate_array(
                param_in, dims=(self.T + self.S, self.J), item=item
            )
            setattr(self, item, param_out)
        # Deal with parameters that vary across age and over time
        # None at the moment - moved to vary by J also
        # tp_param_list4 = [
        #     "rho",
        # ]
        # for item in tp_param_list4:
        #     param_in = getattr(self, item)
        #     param_out = extrapolate_array(
        #         param_in, dims=(self.T + self.S, self.S), item=item
        #     )
        #     setattr(self, item, param_out)
        # Deal with tax parameters that maybe age and time specific
        tax_params_to_TP = [
            "etr_params",
            "mtrx_params",
            "mtry_params",
        ]
        for item in tax_params_to_TP:
            tax_to_set_in = getattr(self, item)
            try:
                len(tax_to_set_in[0][0])
            except TypeError:
                print(
                    "please give a "
                    + item
                    + " that is a nested lists of"
                    + " lists that is three lists deep"
                )
                assert False
            tax_to_set_out = extrapolate_nested_list(
                tax_to_set_in, dims=(self.T, self.S, len(tax_to_set_in[0][0]))
            )
            setattr(self, item, tax_to_set_out)

        # Try to deal with size of eta and eta_RM. They may vary by S, J, T,
        # but want to allow user to enter one that varies by only S, S and J,
        # S and T, or T and S and J.
        eta_params_to_TP = ["eta", "eta_RM", "omega", "rho", "imm_rates"]
        for item in eta_params_to_TP:
            param_in = getattr(self, item)
            param_out = extrapolate_array(
                param_in, dims=(self.T + self.S, self.S, self.J), item=item
            )
            setattr(self, item, param_out)
        # extrapolate SS demog objects
        demog_obj_to_SS = [
            "omega_SS",
            "omega_S_preTP",
            "imm_rates_preTP",
            "rho_preTP",
        ]
        for item in demog_obj_to_SS:
            param_in = getattr(self, item)
            param_out = extrapolate_array(
                param_in, dims=(self.S, self.J), item=item
            )
            setattr(self, item, param_out)
        # extrapolate lifetime ability e matrix over time dimension
        param_in = getattr(self, "e")
        param_out = extrapolate_array(
            param_in, dims=(self.T, self.S, self.J), item="e"
        )
        setattr(self, "e", param_out)
        # Extrapolate chi_n over T + S
        param_in = getattr(self, "chi_n")
        param_out = extrapolate_array(
            param_in, dims=(self.T + self.S, self.S), item="chi_n"
        )
        setattr(self, "chi_n", param_out)

        # make sure zeta matrix sums to one (e.g., default off due to rounding)
        self.zeta = self.zeta / self.zeta.sum()

        # open economy parameters
        self.world_int_rate = rate_conversion(
            self.world_int_rate_annual,
            self.starting_age,
            self.ending_age,
            self.S,
        )

        # set period of retirement for each period t and age s
        self._build_cohort_pension_arrays()

        # Calculations for business income taxes
        # at some point, we will want to make Cost of Capital Calculator
        # a dependency to compute tau_b
        # this adjustment factor has as the numerator CIT receipts/GDP
        # and as the denominator CIT receipts/GDP from the
        # model with baseline parameterization and no adjustment to the
        # CIT_rate
        self.tau_b = (
            self.cit_rate
            * self.c_corp_share_of_assets
            * self.adjustment_factor_for_cit_receipts.reshape(
                self.adjustment_factor_for_cit_receipts.shape[0], 1
            )
        )
        self.delta_tau = -1 * rate_conversion(
            -1 * self.delta_tau_annual,
            self.starting_age,
            self.ending_age,
            self.S,
        )

        # for constant demographics
        if self.constant_demographics:
            self.g_n_ss = 0.0
            self.g_n_preTP = 0.0
            self.g_n = np.zeros(self.T + self.S)
            # Average rho over J for scalar survival computation
            rho_ss = self.rho[-1, :, :].mean(axis=-1)  # (S,)
            surv_rate1 = np.ones((self.S,))  # prob start at age S
            surv_rate1[1:] = np.cumprod(1 - rho_ss[:-1], dtype=float)
            # number of each age alive at any time
            omega_SS_1d = surv_rate1 / surv_rate1.sum()  # (S,)
            # Expand to (S, J) joint distribution
            lambdas_arr = np.array(self.lambdas).reshape(1, self.J)
            self.omega_SS = omega_SS_1d.reshape(self.S, 1) * lambdas_arr
            self.imm_rates = np.zeros((self.T + self.S, self.S, self.J))
            self.omega = np.tile(
                self.omega_SS.reshape(1, self.S, self.J),
                (self.T + self.S, 1, 1),
            )
            self.omega_S_preTP = self.omega_SS.copy()
            self.imm_rates_preTP = np.zeros((self.S, self.J))
            self.rho_preTP = self.rho[0, :, :]

        # Create time series of stationarized UBI transfers
        self.ubi_nom_array = self.get_ubi_nom_objs()

    def get_ubi_nom_objs(self):
        """
        Generate time series of nominal SxJ UBI household matrix and aggregate
        UBI expenditure over necessary time periods. Also generate steady-state
        versions

        Args:
            self: OG-Core Specifications class object

        Returns:
            ubi_nom_array (array): T+S x S x J array time series of UBI
                transfers in dollars for each type-j age-s household in every
                period t
        """
        # Get matrices of number of children 0-17, number of dependents 18-20,
        # number of adults 21-64, and number of seniors >= 65 from
        # OG-Core-Calibration package
        ubi_num_017_mat = 1.1 * np.ones((self.S, self.J))
        ubi_num_1864_mat = 0.85 * np.ones((self.S, self.J))
        ubi_num_65p_mat = 0.15 * np.ones((self.S, self.J))

        # Calculate the UBI transfers to each household type in the first
        # period t=0
        ubi_nom_init = np.tile(
            np.reshape(
                np.minimum(
                    (
                        self.ubi_nom_017 * ubi_num_017_mat
                        + self.ubi_nom_1864 * ubi_num_1864_mat
                        + self.ubi_nom_65p * ubi_num_65p_mat
                    ),
                    self.ubi_nom_max,
                ),
                (1, self.S, self.J),
            ),
            (self.T + self.S, 1, 1),
        )

        # Calculate steady-state and transition path of stationary individual
        # household UBI payments and stationary aggregate UBI outlays
        if self.ubi_growthadj or self.g_y_annual == 0:
            # If ubi_growthadj=True or if g_y_annual<0, then ubi_arr is
            # just a copy of the initial UBI matrix for T periods.
            ubi_nom_array = ubi_nom_init
        else:
            # If ubi_growthadj=False, and g_y_annual>=0, then must divide
            # by e^{g_y t} every period, then set the steady-state matrix
            # to its value close to zero at t=T
            ubi_nom_array = np.zeros_like(ubi_nom_init)
            discount_factor = np.exp(
                self.g_y * np.linspace(0, self.T, self.T + 1)
            )
            ubi_nom_array[: self.T + 1, :, :] = ubi_nom_init[
                : self.T + 1, :, :
            ] / discount_factor.reshape(discount_factor.shape[0], 1, 1)
            ubi_nom_array[self.T + 1 :, :, :] = ubi_nom_array[self.T, :, :]

        return ubi_nom_array

    def _validate_pension_regimes(self, regimes):
        """
        Check the structure and values of the pension regimes list.

        Args:
            regimes (list): list of dicts, each with a ``year``, an
                optional ``max_age_at_start``, and the pension parameters
                the regime sets (see COHORT_PENSION_PARAMS)

        Returns:
            None

        Raises:
            ValueError: naming the offending regime and the problem

        """
        allowed = set(PENSION_REGIME_KEYS) | set(COHORT_PENSION_PARAMS)
        if not isinstance(regimes, (list, tuple)):
            raise ValueError(
                "pension_regimes must be a list of dicts, each with a "
                "'year', an optional 'max_age_at_start', and the pension "
                "parameters the regime sets: "
                + ", ".join(COHORT_PENSION_PARAMS)
            )
        ypp = 80.0 / self.S
        last_year = self.start_year + (self.T + self.S - 1) * ypp
        for k, regime in enumerate(regimes):
            label = f"pension_regimes[{k}]"
            if not isinstance(regime, dict):
                raise ValueError(f"{label} must be a dict, got {regime!r}")
            unknown = set(regime) - allowed
            if unknown:
                raise ValueError(
                    f"{label} has unknown keys {sorted(unknown)}; allowed "
                    f"keys are {sorted(allowed)}"
                )
            if "year" not in regime:
                raise ValueError(f"{label} is missing the required 'year'")
            year = regime["year"]
            if not (
                isinstance(year, (int, np.integer))
                and not isinstance(year, bool)
            ):
                raise ValueError(
                    f"{label}['year'] must be an integer calendar year, "
                    f"got {year!r}"
                )
            if year > last_year:
                raise ValueError(
                    f"{label}['year']={year} is after the end of the time "
                    f"path ({int(last_year)})"
                )
            max_age = regime.get("max_age_at_start")
            if max_age is not None and not (
                isinstance(max_age, (int, float, np.integer, np.floating))
                and not isinstance(max_age, bool)
                and max_age >= 0
            ):
                raise ValueError(
                    f"{label}['max_age_at_start'] must be a non-negative "
                    f"age in years or None, got {max_age!r}"
                )
            set_params = [q for q in COHORT_PENSION_PARAMS if q in regime]
            if not set_params:
                raise ValueError(
                    f"{label} sets none of the pension parameters "
                    f"{list(COHORT_PENSION_PARAMS)}"
                )
            for q in set_params:
                value = regime[q]
                if not (
                    isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, bool)
                ):
                    raise ValueError(
                        f"{label}['{q}'] must be a single number, got "
                        f"{value!r}"
                    )
                rng = self._data[q]["validators"].get("range", {})
                lo, hi = rng.get("min"), rng.get("max")
                if isinstance(lo, (int, float)) and value < lo:
                    raise ValueError(
                        f"{label}['{q}']={value} is below the minimum {lo}"
                    )
                if isinstance(hi, (int, float)) and value > hi:
                    raise ValueError(
                        f"{label}['{q}']={value} is above the maximum {hi}"
                    )

    def _build_cohort_pension_arrays(self):
        """
        Expand the cohort-specific pension parameters to (T + S) x S
        arrays indexed by model period t and age s, and set ``retire``.

        Each pension parameter in COHORT_PENSION_PARAMS has a single base
        value (``retirement_age``). Changes to it are entered through
        ``pension_regimes``, a list of dicts applied in order. A regime
        has a calendar ``year`` from which it applies, an optional
        ``max_age_at_start`` (the oldest age, in years, at that calendar
        year to which it applies; None means everyone, including those
        already retired), and the new values of the parameters it sets.
        A regime that leaves a parameter out leaves it unchanged, so a
        later regime can change one parameter for a narrow set of
        cohorts while an earlier one still governs the rest. The regime
        applies to the household of age s in period t when the calendar
        year of t is at or after ``year`` and the household's age in
        that calendar year is at or below ``max_age_at_start``. Cohorts
        not yet born at ``year`` always qualify.

        With S model periods over 80 years of life, the calendar year of
        period t is start_year + t * 80 / S, the age of the household at
        (t, s) is starting_age + s * 80 / S, and a cutoff by birth year
        is the same as a cutoff by age at start: born in or after
        ``year - max_age_at_start``.

        ``retire`` is the (T + S) x S integer array of the model period
        at which each household becomes eligible for the public
        pension. The steady state uses the final row.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: for a year-varying ``retirement_age`` (changes
                go through ``pension_regimes``), for a malformed regime,
                for a regime that raises a cohort's retirement age after
                it has retired, and for a final period that spans more
                than one regime

        """
        regimes = getattr(self, "pension_regimes", None) or []
        self._validate_pension_regimes(regimes)
        base = np.asarray(self.retirement_age, dtype=float).reshape(-1)
        if base.size != 1:
            if np.all(base == base[0]):
                base = base[:1]
            else:
                raise ValueError(
                    "retirement_age takes a single value. A retirement age "
                    "that changes along the time path is a property of "
                    "birth cohorts, so enter the change as a pension "
                    "regime: pension_regimes=[{'year': Y, "
                    "'max_age_at_start': A, 'retirement_age': R}] applies "
                    "R to households at or below age A in calendar year "
                    "Y, from that year on."
                )
        self.retirement_age = base
        n_rows = self.T + self.S
        ypp = 80.0 / self.S
        t_idx, s_idx = np.indices((n_rows, self.S))
        year_t = self.start_year + t_idx * ypp
        ret_age = np.full((n_rows, self.S), base[0])
        for regime in regimes:
            if "retirement_age" not in regime:
                continue
            mask = year_t >= regime["year"]
            max_age = regime.get("max_age_at_start")
            if max_age is not None:
                age_at_start = (
                    self.starting_age
                    + (s_idx - t_idx) * ypp
                    + (regime["year"] - self.start_year)
                )
                mask &= age_at_start <= max_age
            ret_age[mask] = regime["retirement_age"]
        self.retire = (
            np.round(((ret_age - self.starting_age) * self.S) / 80.0) - 1
        ).astype(int)
        self._check_retire_schedule()

    def _check_retire_schedule(self):
        """
        Check that the (T + S) x S ``retire`` array describes a
        schedule the model can solve.

        Args:
            None

        Returns:
            None

        Raises:
            ValueError: if any retirement period is outside the working
                life, if a cohort's retirement period rises after the
                cohort has already retired (its benefits would be
                withdrawn), or if the final period spans more than one
                regime (the steady state needs a single retirement age)

        """
        retire = self.retire
        n_rows, S = retire.shape
        if np.any(retire < 1) or np.any(retire >= S):
            raise ValueError(
                "retirement must fall inside the working life: the "
                "retirement period must be between 1 and S - 1 = "
                f"{S - 1}, but retire ranges from {retire.min()} to "
                f"{retire.max()} (retirement_age {self.retirement_age[0]} "
                f"and pension_regimes {self.pension_regimes})"
            )
        if np.any(retire[-1, :] != retire[-1, 0]):
            raise ValueError(
                "the final period of the time path spans more than one "
                "pension regime, so the steady state has no single "
                "retirement age. Increase T or lower max_age_at_start in "
                "pension_regimes."
            )
        ypp = 80.0 / S
        # Each diagonal of retire is one birth cohort: offset k holds
        # retire[i, i + k], the cohort of age k in period 0 (k >= 0) or
        # born in period -k (k < 0).
        for k in range(-(n_rows - 1), S):
            diag = np.diagonal(retire, offset=k)
            if diag.size < 2:
                continue
            ages = np.arange(diag.size) + max(k, 0)
            for i in np.nonzero(np.diff(diag))[0]:
                age, prev, new = ages[i + 1], diag[i], diag[i + 1]
                if age >= prev and new > age:
                    period = i + 1 + max(-k, 0)
                    raise ValueError(
                        "pension_regimes raises the retirement age of a "
                        "cohort after it has retired: in model period "
                        f"{period} (calendar year "
                        f"{int(self.start_year + period * ypp)}) the "
                        f"cohort aged {int(self.starting_age + age * ypp)} "
                        f"years, retired since period {prev}, moves to "
                        f"retirement period {new}. Lower max_age_at_start "
                        "(the oldest age the regime covers) so the regime "
                        "excludes cohorts already retired."
                    )

    def update_specifications(self, revision, raise_errors=True):
        """
        Updates parameter specification with values in revision
        dictionary.

        Args:
            revision (dict): dictionary or JSON string with one or more
                `PARAM: VALUE` pairs
            raise_errors (boolean):
                    if True (the default), raises ValueError when
                       `parameter_errors` exists;
                    if False, does not raise ValueError when
                       `parameter_errors` exists and leaves error
                       handling to caller of the update_specifications
                       method.

        Returns:
            None

        Raises:
            ValueError: if raise_errors is True AND
              `_validate_parameter_names_types` generates errors OR
              `_validate_parameter_values` generates errors.

        Notes:
            Given a reform dictionary, typical usage of the
            Specifications class is as follows::

                >>> specs = Specifications()
                >>> specs.update_specifications(revision)

            An example of a multi-parameter specification is as follows::

                >>> revision = {
                    frisch: [0.03]
                }

        """
        if not (isinstance(revision, dict) or isinstance(revision, str)):
            raise ValueError("ERROR: revision is not a dictionary or string")
        if isinstance(revision, str):
            # a JSON string or a path to a JSON file
            revision = self.read_params(revision)
        # Pension regimes are a list of dicts, which paramtools cannot
        # validate, so set them aside and check them in
        # _build_cohort_pension_arrays (called by compute_default_params)
        if "pension_regimes" in revision:
            revision = dict(revision)  # leave the caller's dict alone
            self.pension_regimes = revision.pop("pension_regimes")
        # Skip over the adjust method if the tax parameters passed in
        # are functions (e.g., in the case of tax_func_type = mono)
        tax_update_dict = {}
        tax_func_params_functions = False
        try:
            if revision["tax_func_type"] in ["mono", "mono2D"]:
                tax_func_params_functions = True
        except (KeyError, TypeError):
            pass
        if self.tax_func_type in ["mono", "mono2D"]:
            tax_func_params_functions = True
        if tax_func_params_functions:
            try:
                for item in ["etr_params", "mtrx_params", "mtry_params"]:
                    if item in revision.keys():
                        tax_update_dict[item] = revision[item]
                        del revision[item]
            except (KeyError, TypeError):
                pass
        self.adjust(revision, raise_errors=raise_errors)
        # put tax values skipped over in the adjust method back in so
        # they are in the parameters class.
        if tax_update_dict != {}:
            for key, value in tax_update_dict.items():
                setattr(self, key, value)
        self.compute_default_params()


def revision_warnings_errors(spec_revision):
    """
    Generate warnings and errors for OG-Core parameter specifications

    Args:
        spec_revision (dict): dictionary suitable for use with the
            `Specifications.update_specifications method`.

    Returns:
        rtn_dict (dict): with endpoint specific warning and error messages

    """
    rtn_dict = {"warnings": "", "errors": ""}
    spec = Specifications()
    spec.update_specifications(spec_revision, raise_errors=False)
    if spec._errors:
        rtn_dict["errors"] = spec._errors
    return rtn_dict
