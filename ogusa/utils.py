'''
------------------------------------------------------------------------
Miscellaneous functions used in the OG-USA model.
------------------------------------------------------------------------
'''
# Packages
import os
from io import StringIO
import numpy as np
import taxcalc
import pickle
from pkg_resources import resource_stream, Requirement

EPSILON = 1e-10
PATH_EXISTS_ERRNO = 17

REFORM_DIR = "./OUTPUT_REFORM"
BASELINE_DIR = "./OUTPUT_BASELINE"

# Default year for model runs
DEFAULT_START_YEAR = 2018

# Latest year TaxData extrapolates to
TC_LAST_YEAR = 2028

# Year of data used (e.g. PUF or CPS year)
CPS_START_YEAR = taxcalc.Records.CPSCSV_YEAR
PUF_START_YEAR = taxcalc.Records.PUFCSV_YEAR


def mkdirs(path):
    '''
    Makes directories to save output.

    Args:
        path (str): path name for new directory

    Returns:
        None

    '''

    try:
        os.makedirs(path)
    except OSError as oe:
        if oe.errno == PATH_EXISTS_ERRNO:
            pass


def pct_diff_func(simul, data):
    '''
    Used to calculate the absolute percent difference between data
    moments and model moments.

    Args:
        simul (array_like): any shape, model moments
        data (array_like): same shape as simul, data moments

    Functions called: None

    Returns:
        output (array_like): percentage differences between model and
            data moments
    '''
    frac = (simul - data) / data
    output = np.abs(frac)
    return output


def convex_combo(var1, var2, nu):
    '''
    Takes the convex combination of two variables, where nu is in [0,1].

    Args:
        var1 (array_like): any shape, variable 1
        var2 (array_like): same shape as var1, variable 2
        nu (scalar): weight on var1 in convex combination, in [0, 1]

    Returns:
        combo (array_like): same shape as var1, convex combination of
        var1 and var2

    '''
    combo = nu * var1 + (1 - nu) * var2
    return combo


def read_file(path, fname):
    '''
    Read the contents of 'path'. If it does not exist, assume the file
    is installed in a .egg file, and adjust accordingly.

    Args:
        path (str): path name for new directory
        fname (str): filename

    Returns:
        file contents (str)

    '''

    if not os.path.exists(os.path.join(path, fname)):
        path_in_egg = os.path.join("ogusa", fname)
        buf = resource_stream(Requirement.parse("ogusa"), path_in_egg)
        _bytes = buf.read()
        return StringIO(_bytes.decode("utf-8"))
    else:
        return open(os.path.join(path, fname))


def pickle_file_compare(fname1, fname2, tol=1e-3, exceptions={},
                        relative=False):
    '''
    Read two pickle files and unpickle each. We assume that each
    resulting object is a dictionary. The values of each dict are either
    numpy arrays or else types that are comparable with the == operator.

    Args:
        fname1 (str): file name of file 1
        fname2 (str): file name of file 2
        tol (scalar): tolerance
        exceptions (dict): exceptions
        relative (bool): whether comparison compares relative values

    Returns:
        comparison (bool): whether therea two dictionaries are the same
    '''
    try:
        pkl1 = pickle.load(open(fname1, 'rb'), encoding='latin1')
    except TypeError:
        pkl1 = pickle.load(open(fname1, 'rb'))
    try:
        pkl2 = pickle.load(open(fname2, 'rb'), encoding='latin1')
    except TypeError:
        pkl2 = pickle.load(open(fname2, 'rb'))
    comparison = dict_compare(fname1, pkl1, fname2, pkl2, tol=tol,
                              exceptions=exceptions, relative=relative)

    return comparison


def comp_array(name, a, b, tol, unequal, exceptions={}, relative=False):
    '''
    Compare two arrays in the L inifinity norm
    Return True if | a - b | < tol, False otherwise
    If not equal, add items to the unequal list
    name: the name of the value being compared

    Args:
        name (str): name of variable being compared
        a (array_like): first array to compare
        b (array_like): second array to compare
        tol (scalar): tolerance used for comparison
        unequal (dict): dict of variables that are not equal
        exceptions (dict): exceptions
        relative (bool): whether comparison compares relative values

    Returns:
        (bool): whether two arrays are the same or not

    '''

    if name in exceptions:
        tol = exceptions[name]

    if not a.shape == b.shape:
        print("unequal shapes for {0} comparison ".format(str(name)))
        unequal.append((str(name), a, b))
        return False

    else:

        if np.all(a < EPSILON) and np.all(b < EPSILON):
            return True

        if relative:
            err = abs(a - b)
            mn = np.mean(b)
            err = np.max(err / mn)
        else:
            err = np.max(abs(a - b))

        if not err < tol:
            print("diff for {0} is {1} which is NOT OK".format(str(name), err))
            unequal.append((str(name), a, b))
            return False
        else:
            print("err is {0} which is OK".format(err))
            return True


def comp_scalar(name, a, b, tol, unequal, exceptions={}, relative=False):
    '''
    Compare two scalars in the L inifinity norm
    Return True if abs(a - b) < tol, False otherwise
    If not equal, add items to the unequal list

    Args:
        name (str): name of variable being compared
        a (scalar): first scalar to compare
        b (scalra): second scalar to compare
        tol (scalar): tolerance used for comparison
        unequal (dict): dict of variables that are not equal
        exceptions (dict): exceptions
        relative (bool): whether comparison compares relative values

    Returns:
        (bool): whether two arrays are the same or not

    '''

    if name in exceptions:
        tol = exceptions[name]

    if (a < EPSILON) and (b < EPSILON):
        return True

    if relative:
        err = float(abs(a - b)) / float(b)
    else:
        err = abs(a - b)

    if not err < tol:
        print("err for {0} is {1} which is NOT OK".format(str(name), err))
        unequal.append((str(name), str(a), str(b)))
        return False
    else:
        print("err is {0} which is OK".format(err))
        return True


def dict_compare(fname1, pkl1, fname2, pkl2, tol, verbose=False,
                 exceptions={}, relative=False):
    '''
    Compare two dictionaries. The values of each dict are either
    numpy arrays
    or else types that are comparable with the == operator.
    For arrays, they are considered the same if |x - y| < tol in
    the L_inf norm.
    For scalars, they are considered the same if x - y < tol

    Args:
        fname1 (str): files name for pickle file
        pk1 (dict): first dictionary to compare
        fname2 (str): files name for pickle file
        pk2 (dict): second dictionary to compare
        tol (scalar): tolerance used for comparison
        verbose (bool): whether print messages
        exceptions (dict): exceptions
        relative (bool): whether comparison compares relative values

    Returns:
        (bool): whether two dictionaries are the same or not

    '''

    keys1 = set(pkl1.keys())
    keys2 = set(pkl2.keys())
    check = True
    if keys1 != keys2:
        if len(keys1) == len(keys2):
            extra1 = keys1 - keys2
            extra2 = keys2 - keys1
            msg1 = "extra items in {0}: {1}"
            print(msg1.format(fname1, extra1))
            print(msg1.format(fname2, extra2))
            return False
        elif len(keys1) > len(keys2):
            bigger = keys1
            bigger_file = fname1
            smaller = keys2
        else:
            bigger = keys2
            bigger_file = fname2
            smaller = keys1
        res = bigger - smaller
        msg = "more items in {0}: {1}"
        print(msg.format(bigger_file, res))
        return False
    else:
        unequal_items = []
        for k, v in pkl1.items():
            if type(v) == np.ndarray:
                check &= comp_array(k, v, pkl2[k], tol, unequal_items,
                                    exceptions=exceptions,
                                    relative=relative)
            else:
                try:
                    check &= comp_scalar(k, v, pkl2[k], tol, unequal_items,
                                         exceptions=exceptions,
                                         relative=relative)
                except TypeError:
                    check &= comp_array(k, np.array(v), np.array(pkl2[k]), tol,
                                        unequal_items,
                                        exceptions=exceptions,
                                        relative=relative)

        if verbose and unequal_items:
            frmt = "Name {0}"
            res = [frmt.format(x[0]) for x in unequal_items]
            print("Different arrays: ", res)
            return False

    return check


def to_timepath_shape(some_array, p):
    '''
    This function takes an vector of length T and tiles it to fill a
    Tx1x1 array for time path computations.

    Args:
        some_array (Numpy array): array to reshape
        p (OG-USA Specifcations object): model parameters

    Returns:
        tp_array (Numpy  array): reshaped array

    '''
    tp_array = some_array.reshape(some_array.shape[0], 1, 1)
    return tp_array


def get_initial_path(x1, xT, T, spec):
    '''
    This function generates a path from point x1 to point xT such that
    that the path x is a linear or quadratic function of time t.
        linear:    x = d*t + e
        quadratic: x = a*t^2 + b*t + c
    The identifying assumptions for quadratic are the following:
        (1) x1 is the value at time t=0: x1 = c
        (2) xT is the value at time t=T-1: xT = a*(T-1)^2 + b*(T-1) + c
        (3) the slope of the path at t=T-1 is 0: 0 = 2*a*(T-1) + b

    Args:
        x1 (scalar): initial value of the function x(t) at t=0
        xT (scalar): value of the function x(t) at t=T-1
        T (int): number of periods of the path, must be >= 3
        spec (str): shape of guess for time path, "linear" or "quadratic"

    Returns:
        xpath (Numpy array): guess of variable over the time path

    '''
    if spec == "linear":
        xpath = np.linspace(x1, xT, T)
    elif spec == "quadratic":
        cc = x1
        bb = 2 * (xT - x1) / (T - 1)
        aa = (x1 - xT) / ((T - 1) ** 2)
        xpath = (aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) +
                 cc)

    return xpath


def safe_read_pickle(file_path):
    '''
    This function reads a pickle from Python 2 into Python 2 or Python 3

    Args:
        file_path (str): path to pickle file

    Returns:
        obj (object): object saved in pickle file

    '''
    with open(file_path, 'rb') as f:
        try:
            obj = pickle.load(f, encoding='latin1')
        except TypeError:
            obj = pickle.load(f)
    return obj


def rate_conversion(annual_rate, start_age, end_age, S):
    '''
    This function converts annual rates to model period ratesself.

    Args:
        annual_rate (array_like): annualized rates
        start_age (int): age at which agents become economically active
        end_age (int): maximum age of agents
        S (int): number of model periods in agents life

    Returns:
        rate (array_like): model period rates

    '''
    rate = (1 + annual_rate) ** ((end_age - start_age) / S) - 1
    return rate


def save_return_table(table_df, output_type, path, precision=0):
    '''
    Function to save or return a table of data.

    Args:
        table_df (Pandas DataFrame): table
        output_type (string): specifies the type of file to save
            table to: 'csv', 'tex', 'excel', 'json'
        path (string): specifies path to save file with table to
        precision (integer): number of significant digits to print.
            Defaults to 0.

    Returns:
        table_df (Pandas DataFrame): table

    '''
    if path is None:
        if output_type == 'tex':
            tab_str = table_df.to_latex(
                buf=path, index=False, na_rep='',
                float_format=lambda x: '%.' + str(precision) + '0f' % x)
            return tab_str
        elif output_type == 'json':
            tab_str = table_df.to_json(
                path_or_buf=path, double_precision=0)
            return tab_str
        elif output_type == 'html':
            print('Output html...')
            # with pd.option_context('display.precision', precision):
            tab_html = (
                table_df.round(2).style
                # .format({'': '', '%.' + str(precision) + '0f')
                .set_properties(**{'font-size': '9pt',
                                   'font-family': 'Calibri',
                                   'text-align': 'left'})
                .hide_index()
                .render()
            )
            return tab_html
        else:
            return table_df
    else:
        if output_type == 'tex':
            table_df.to_latex(buf=path, index=False, na_rep='',
                              float_format=lambda x: '%.' +
                              str(precision) + '0f' % x)
        elif output_type == 'csv':
            table_df.to_csv(path_or_buf=path, index=False, na_rep='',
                            float_format='%.' + str(precision) + '0f')
        elif output_type == 'json':
            table_df.to_json(path_or_buf=path,
                             double_precision=precision)
        elif output_type == 'excel':
            table_df.to_excel(excel_writer=path, index=False, na_rep='',
                              float_format='%.' + str(precision) + '0f')
        else:
            print('Please enter a valid output format')
            assert(False)
>>>>>>> upstream/master
