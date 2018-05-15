import json
import os
import collections as collect
import six
import numpy as np

from taxcalc.growfactors import Growfactors
from taxcalc.policy import Policy

import ogusa
from ogusa.parametersbase import ParametersBase

class Specifications(ParametersBase):
    DEFAULTS_FILENAME = 'default_parameters.json'
    LAST_BUDGET_YEAR = 2027  # increases by one every calendar year
    JSON_START_YEAR = 2013

    def __init__(self,
                 start_year=JSON_START_YEAR,
                 num_years=None,
                 initial_estimates=False):
        super(Specifications, self).__init__()

        if num_years is None:
            num_years = Specifications.LAST_BUDGET_YEAR - start_year
        # reads in default data
        self._vals = self._params_dict_from_json_file()

        if num_years < 1:
            raise ValueError('num_years cannot be less than one')

        # does cheap calculations such as growth
        self.initialize(start_year, num_years, initial_estimates=False)

        self.parameter_warnings = ''
        self.parameter_errors = ''
        self._ignore_errors = False

    def initialize(self, start_year, num_years, initial_estimates=False):
        """
        ParametersBase reads JSON file and sets attributes to self
        Next call self.ogusa_set_default_vals for further initialization
        If estimate_params is true, then run long running estimation routines
        """
        super(Specifications, self).initialize(start_year, num_years)
        self.ogusa_set_default_vals()
        if initial_estimates:
            self.estimate_parameters()

    def ogusa_set_default_vals(self):
        """
        Does cheap calculations such as calculating/applying growth rates
        """
        # self.b_ellipse, self.upsilon = ogusa.elliptical_u_est.estimation(
        #     self.frisch[0],
        #     self.ltilde[0]
        # )
        # call some more functions
        pass

    def estimate_parameters(self, data=None, reform={}):
        """
        Runs long running parameter estimatation routines such as estimating
        tax function parameters
        """
        # self.tax_func_estimate = tax_func_estimate(self.BW, self.S, self.starting_age, self.ending_age,
        #                                 self.start_year, self.baseline,
        #                                 self.analytical_mtrs, self.age_specific,
        #                                 reform=None, data=data)
        pass

    def default_parameters(self):
        """
        Return Policy object same as self except with current-law policy.
        """
        dp = Specifications(start_year=self.start_year,
                            num_years=self.num_years)
        dp.set_year(self.current_year)
        return dp

    def update_specifications(self, revision, raise_errors=True):
        """
        copied from TC behavior.py-update_behavior
        """
        # check that all revisions dictionary keys are integers
        if not isinstance(revision, dict):
            raise ValueError('ERROR: YYYY PARAM revision is not a dictionary')
        if not revision:
            return  # no revision to implement
        revision_years = sorted(list(revision.keys()))
        for year in revision_years:
            if not isinstance(year, int):
                msg = 'ERROR: {} KEY {}'
                details = 'KEY in revision is not an integer calendar year'
                raise ValueError(msg.format(year, details))
        # check range of remaining revision_years
        first_revision_year = min(revision_years)
        if first_revision_year < self.start_year:
            msg = 'ERROR: {} YEAR revision provision in YEAR < start_year={}'
            raise ValueError(msg.format(first_revision_year, self.start_year))
        if first_revision_year < self.current_year:
            msg = 'ERROR: {} YEAR revision provision in YEAR < current_year={}'
            raise ValueError(
                msg.format(first_revision_year, self.current_year)
            )
        last_revision_year = max(revision_years)
        if last_revision_year > self.end_year:
            msg = 'ERROR: {} YEAR revision provision in YEAR > end_year={}'
            raise ValueError(msg.format(last_revision_year, self.end_year))
        # validate revision parameter names and types
        self.parameter_errors = ''
        self.parameter_warnings = ''
        self._validate_parameter_names_types(revision)
        if not self._ignore_errors and self.parameter_errors:
            raise ValueError(self.parameter_errors)
        # implement the revision year by year
        precall_current_year = self.current_year
        revision_parameters = set()
        for year in revision_years:
            self.set_year(year)
            revision_parameters.update(revision[year].keys())
            self._update({year: revision[year]})
        self.set_year(precall_current_year)
        # validate revision parameter values
        self._validate_parameter_values(revision_parameters)
        if self.parameter_errors and raise_errors:
            raise ValueError('\n' + self.parameter_errors)

    def read_json_parameters_object(self, parameters):
        raise NotImplementedError()

    def _validate_parameter_names_types(self, revision):
        """
        Check validity of parameter names and parameter types used
        in the specified revision dictionary.

        copied from taxcalc.Behavior._validate_parameter_names_types
        """
        # pylint: disable=too-many-branches,too-many-nested-blocks
        # pylint: disable=too-many-locals
        param_names = set(self._vals.keys())
        for year in sorted(revision.keys()):
            for name in revision[year]:
                if name not in param_names:
                    msg = '{} {} unknown parameter name'
                    self.parameter_errors += (
                        'ERROR: ' + msg.format(year, name) + '\n'
                    )
                else:
                    # check parameter value type avoiding use of isinstance
                    # because isinstance(True, (int,float)) is True, which
                    # makes it impossible to check float parameters
                    bool_param_type = self._vals[name]['boolean_value']
                    int_param_type = self._vals[name]['integer_value']
                    assert isinstance(revision[year][name], list)
                    pvalue = revision[year][name][0]
                    if isinstance(pvalue, list):
                        scalar = False  # parameter value is a list
                    else:
                        scalar = True  # parameter value is a scalar
                        pvalue = [pvalue]  # make scalar a single-item list
                    # pylint: disable=consider-using-enumerate
                    for idx in range(0, len(pvalue)):
                        if scalar:
                            pname = name
                        else:
                            pname = '{}_{}'.format(name, idx)
                        pval = pvalue[idx]
                        # pylint: disable=unidiomatic-typecheck
                        pval_is_bool = type(pval) == bool
                        pval_is_int = type(pval) == int
                        pval_is_float = type(pval) == float
                        if bool_param_type:
                            if not pval_is_bool:
                                msg = '{} {} value {} is not boolean'
                                self.parameter_errors += (
                                    'ERROR: ' +
                                    msg.format(year, pname, pval) +
                                    '\n'
                                )
                        elif int_param_type:
                            if not pval_is_int:  # pragma: no cover
                                msg = '{} {} value {} is not integer'
                                self.parameter_errors += (
                                    'ERROR: ' +
                                    msg.format(year, pname, pval) +
                                    '\n'
                                )
                        else:  # param is float type
                            if not (pval_is_int or pval_is_float):
                                msg = '{} {} value {} is not a number'
                                self.parameter_errors += (
                                    'ERROR: ' +
                                    msg.format(year, pname, pval) +
                                    '\n'
                                )
        del param_names


    def _validate_parameter_values(self, parameters_set):
        """
        Check values of parameters in specified parameter_set using
        range information from the current_law_policy.json file.

        copied from taxcalc.Policy._validate_parameter_values
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-nested-blocks
        rounding_error = 100.0
        # above handles non-rounding of inflation-indexed parameter values
        dp = self.default_parameters()
        parameters = sorted(parameters_set)
        syr = Policy.JSON_START_YEAR
        for pname in parameters:
            ###################################
            # don't need this part
            # if pname.endswith('_cpi'):
            #     continue  # *_cpi parameter values validated elsewhere
            ###################################
            pvalue = getattr(self, pname)
            for vop, vval in self._vals[pname]['range'].items():
                if isinstance(vval, six.string_types):
                    if vval == 'default':
                        vvalue = getattr(dp, pname)
                        if vop == 'min':
                            vvalue -= rounding_error
                        # the follow branch can never be reached, so it
                        # is commented out because it can never be tested
                        # (see test_range_infomation in test_policy.py)
                        # --> elif vop == 'max':
                        # -->    vvalue += rounding_error
                    else:
                        vvalue = self.simple_eval(vval)
                else:
                    vvalue = np.full(pvalue.shape, vval)
                assert pvalue.shape == vvalue.shape
                assert len(pvalue.shape) <= 2
                if len(pvalue.shape) == 2:
                    scalar = False  # parameter value is a list
                else:
                    scalar = True  # parameter value is a scalar
                for idx in np.ndindex(pvalue.shape):
                    out_of_range = False
                    if vop == 'min' and pvalue[idx] < vvalue[idx]:
                        out_of_range = True
                        msg = '{} {} value {} < min value {}'
                        extra = self._vals[pname]['out_of_range_minmsg']
                        if extra:
                            msg += ' {}'.format(extra)
                    if vop == 'max' and pvalue[idx] > vvalue[idx]:
                        out_of_range = True
                        msg = '{} {} value {} > max value {}'
                        extra = self._vals[pname]['out_of_range_maxmsg']
                        if extra:
                            msg += ' {}'.format(extra)
                    if out_of_range:
                        action = self._vals[pname]['out_of_range_action']
                        if scalar:
                            name = pname
                        else:
                            name = '{}_{}'.format(pname, idx[1])
                            if extra:
                                msg += '_{}'.format(idx[1])
                        if action == 'warn':
                            self.parameter_warnings += (
                                'WARNING: ' + msg.format(idx[0] + syr, name,
                                                         pvalue[idx],
                                                         vvalue[idx]) + '\n'
                            )
                        if action == 'stop':
                            self.parameter_errors += (
                                'ERROR: ' + msg.format(idx[0] + syr, name,
                                                       pvalue[idx],
                                                       vvalue[idx]) + '\n'
                            )
        del dp
        del parameters

# copied from taxcalc.tbi.tbi.reform_errors_warnings--probably needs further
# changes
def reform_warnings_errors(user_mods):
    """
    The reform_warnings_errors function assumes user_mods is a dictionary
    returned by the Calculator.read_json_param_objects() function.
    This function returns a dictionary containing two STR:STR pairs:
    {'warnings': '<empty-or-message(s)>', 'errors': '<empty-or-message(s)>'}
    In each pair the second string is empty if there are no messages.
    Any returned messages are generated using current_law_policy.json
    information on known policy parameter names and parameter value ranges.
    Note that this function will return one or more error messages if
    the user_mods['policy'] dictionary contains any unknown policy
    parameter names or if any *_cpi parameters have values other than
    True or False.  These situations prevent implementing the policy
    reform specified in user_mods, and therefore, no range-related
    warnings or errors will be returned in this case.
    """
    rtn_dict = {'ogusa': {'warnings': '', 'errors': ''}}

    # create Specifications object and implement reform
    specs = Specifications(2017)
    specs._ignore_errors = True
    try:
        specs.update_specifications(user_mods['ogusa'], raise_errors=False)
        rtn_dict['ogusa']['warnings'] = specs.parameter_warnings
        rtn_dict['ogusa']['errors'] = specs.parameter_errors
    except ValueError as valerr_msg:
        rtn_dict['ogusa']['errors'] = valerr_msg.__str__()
    return rtn_dict

if __name__ == '__main__':
    specs = Specifications(2017)
    specs._ignore_errors = True
    reform = {
        2017: {
            "tG1": [50],
            "T": [80]
        }
    }
    specs.update_specifications(reform, raise_errors=False)
    print('errors', specs.parameter_errors)
    print('warnings', specs.parameter_warnings)

    for name in specs._vals:
        item = getattr(specs, name[1:], None)
        if item is not None:
            print(name, item)
