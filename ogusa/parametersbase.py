"""
Tax-Calculator abstract base parameters class.
"""
# CODING-STYLE CHECKS:
# pep8 parameters.py

import os
import json
import six
import abc
import ast
import collections as collect
import numpy as np
from taxcalc.utils import read_egg_json


class ParametersBase(object):
    """
    Inherit from this class for Policy, Behavior, Consumption, Growdiff, and
    other groups of parameters that need to have a set_year method.
    Override this __init__ method and DEFAULTS_FILENAME.
    """
    __metaclass__ = abc.ABCMeta

    DEFAULTS_FILENAME = None

    @classmethod
    def default_data(cls, metadata=False):
        """
        Return parameter data read from the subclass's json file.
        Parameters
        ----------
        metadata: boolean
        start_year: int or None
        Returns
        -------
        params: dictionary of data
        """
        params = cls._params_dict_from_json_file()
        # return different data from params dict depending on metadata value
        if metadata:
            return params
        else:
            return {name: data['value'] for name, data in params.items()}

    def __init__(self):
        pass

    def initialize(self, ):
        """
        Called from subclass __init__ function.

        Added expand=True option to turn off expansion of arrays
        """
        self.set_default_vals()

    def set_default_vals(self, known_years=999999):
        """
        Called by initialize method and from some subclass methods.
        """
        if hasattr(self, '_vals'):
            for name, data in self._vals.items():
                intg_val = data.get('integer_value', None)
                bool_val = data.get('boolean_value', None)
                values = data.get('value', None)
                if values:
                    setattr(self, name,
                            self._expand_array(values, intg_val, bool_val))

    # ----- begin private methods of ParametersBase class -----

    @classmethod
    def _params_dict_from_json_file(cls):
        """
        Read DEFAULTS_FILENAME file and return complete dictionary.
        Parameters
        ----------
        nothing: void
        Returns
        -------
        params: dictionary
            containing complete contents of DEFAULTS_FILENAME file.
        """
        if cls.DEFAULTS_FILENAME is None:
            msg = 'DEFAULTS_FILENAME must be overridden by inheriting class'
            raise NotImplementedError(msg)
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            cls.DEFAULTS_FILENAME)
        if os.path.exists(path):
            with open(path) as pfile:
                params_dict = json.load(pfile,
                                        object_pairs_hook=collect.OrderedDict)
        else:
            # cannot call read_egg_ function in unit tests
            params_dict = read_egg_json(
                cls.DEFAULTS_FILENAME)  # pragma: no cover
        return params_dict

    def _update(self, mods):
        """
        Private method used by public implement_reform and update_* methods
        in inheriting classes.
        Parameters
        ----------
        mods: dictionary containing a parameter:value pairs
        Raises
        ------
        ValueError:
            if mods is not a dict.
        Returns
        -------
        nothing: void
        Notes
        -----
        """
        # check YEAR value in the single YEAR:MODS dictionary parameter
        if not isinstance(mods, dict):
            msg = 'mods is not a dictionary'
            raise ValueError(msg)
        all_names = set(mods.keys())  # no duplicate keys in a dict
        used_names = set()  # set of used parameter names in MODS dict
        for name, values in mods.items():
            intg_val = self._vals[name].get('integer_value', None)
            bool_val = self._vals[name].get('boolean_value', None)
            # set post-reform values of parameter with name
            used_names.add(name)
            cval = getattr(self, name, None)
            nval = self._expand_array(values, intg_val, bool_val)
            cval[0] = nval
        # confirm that all names have been used
        assert len(used_names) == len(all_names)

    @staticmethod
    def _expand_array(x, x_int, x_bool):
        """
        Private method called only within this abstract base class.
        Dispatch to either _expand_1D or _expand_2D given dimension of x.
        Parameters
        ----------
        x : value to expand
            x must be either a scalar list or a 1D numpy array, or
            x must be either a list of scalar lists or a 2D numpy array
        x_int : boolean
            True implies x has dtype=np.int8;
            False implies x has dtype=np.float64 or dtype=np.bool_
        x_bool : boolean
            True implies x has dtype=np.bool_;
            False implies x has dtype=np.float64 or dtype=np.int8
        Returns
        -------
        expanded numpy array with specified dtype
        """
        assert not (x_int and x_bool)
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            msg = '_expand_array expects x to be a list or numpy array'
            raise ValueError(msg)
        if isinstance(x, list):
            if x_int:
                x = np.array(x, np.int8)
            elif x_bool:
                x = np.array(x, np.bool_)
            else:
                x = np.array(x, np.float64)
        return x

    OP_DICT = {
        '+': lambda pvalue, val: pvalue + val,
        '-': lambda pvalue, val: pvalue - val,
        '*': lambda pvalue, val: pvalue * val,
        '/': lambda pvalue, val: pvalue / val if val > 0 else 'ERROR: Cannot divide by zero',
    }

    def simple_eval(self, param_string):
        """
        Parses `param_string` and returns result. `param_string can be either:
            1. `param_name op scalar` -- this will be parsed into param, op, and scalar
                    where `op` is a key in `OP_DICT`. The corresponding function is
                    applied to the parameter value and the scalar value.
            2. `param_name` -- simply return the parameter value that is retrieved
                    from the object
        returns: float used for validation
        """
        pieces = param_string.split(' ')
        validate_against = pieces[0]
        # param_string is of the form 'param_name op scalar'
        if len(pieces) > 1:
            op = pieces[1]
            # parse string to python type (i.e. str --> int, float, bool)
            scalar = ast.literal_eval(pieces[2])
            value_against = getattr(self, validate_against)
            assert value_against is not None and isinstance(value_against, (int, float, np.ndarray))
            assert op in ParametersBase.OP_DICT
            return ParametersBase.OP_DICT[op](value_against, scalar)
        else:
            # vval is just the parameter name
            return getattr(self, param_string)
