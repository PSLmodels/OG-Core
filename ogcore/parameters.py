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
            "world_int_rate_annual",
            "adjustment_factor_for_cit_receipts",
            "tau_bq",
            "tau_payroll",
            "h_wealth",
            "m_wealth",
            "p_wealth",
            "retirement_age",
            "replacement_rate_adjust",
            "zeta_D",
            "zeta_K",
            "r_gov_scale",
            "r_gov_shift",
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
        ]
        for item in tp_param_list3:
            param_in = getattr(self, item)
            param_out = extrapolate_array(
                param_in, dims=(self.T + self.S, self.J), item=item
            )
            setattr(self, item, param_out)
        # Deal with parameters that vary across age and over time
        tp_param_list4 = [
            "rho",
        ]
        for item in tp_param_list4:
            param_in = getattr(self, item)
            param_out = extrapolate_array(
                param_in, dims=(self.T + self.S, self.S), item=item
            )
            setattr(self, item, param_out)
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

        # Try to deal with size of eta.  It may vary by S, J, T, but
        # want to allow user to enter one that varies by only S, S and J,
        # S and T, or T and S and J.
        param_in = getattr(self, "eta")
        param_out = extrapolate_array(
            param_in, dims=(self.T + self.S, self.S, self.J), item="eta"
        )
        setattr(self, "eta", param_out)
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

        # set period of retirement
        self.retire = (
            np.round(
                ((self.retirement_age - self.starting_age) * self.S) / 80.0
            )
            - 1
        ).astype(int)

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
            self.g_n = np.zeros(self.T + self.S)
            surv_rate = np.ones_like(self.rho) - self.rho
            surv_rate1 = np.ones((self.S,))  # prob start at age S
            surv_rate1[1:] = np.cumprod(surv_rate[-1, :-1], dtype=float)
            # number of each age alive at any time
            omega_SS = np.ones(self.S) * surv_rate1
            self.omega_SS = omega_SS / omega_SS.sum()
            self.imm_rates = np.zeros((self.T + self.S, self.S))
            self.omega = np.tile(
                np.reshape(self.omega_SS, (1, self.S)), (self.T + self.S, 1)
            )
            self.omega_S_preTP = self.omega_SS

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
