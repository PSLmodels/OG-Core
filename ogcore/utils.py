# Packages
import os
import sys
import requests
from zipfile import ZipFile
import urllib.request
from tempfile import NamedTemporaryFile
from io import BytesIO, StringIO
import numpy as np
import pandas as pd
import pickle
from pkg_resources import resource_stream, Requirement

EPSILON = 1e-10  # tolerance or comparison functions


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
        if oe.errno == 17:  # 17 is an error code if can't make path
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
    if np.asarray(data).all() != 0:
        frac = (simul - data) / data
        output = np.abs(frac)
    else:
        output = np.abs(simul - data)
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
        path_in_egg = os.path.join("ogcore", fname)
        buf = resource_stream(Requirement.parse("ogcore"), path_in_egg)
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
    pkl1 = safe_read_pickle(fname1)
    pkl2 = safe_read_pickle(fname2)
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
        unequal (list):  list of variables that are not equal
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


def dict_compare(name1, dict1, name2, dict2, tol, verbose=False,
                 exceptions={}, relative=False):
    r'''
    Compare two dictionaries. The values of each dict are either
    numpy arrays or else types that are comparable with the `==` operator.
    For arrays, they are considered the same if `|x - y| < tol` in
    the L_inf norm. For scalars, they are considered the same if
    `x - y < tol`.

    Args:
        name1 (str): name of dictionary 1
        dict1 (dict): first dictionary to compare
        name2 (str): name of dictionary 2
        dict2 (dict): second dictionary to compare
        tol (scalar): tolerance used for comparison
        verbose (bool): whether print messages
        exceptions (dict): exceptions
        relative (bool): whether comparison compares relative values

    Returns:
        (bool): whether two dictionaries are the same or not

    '''

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    check = True
    if keys1 != keys2:
        if len(keys1) == len(keys2):
            extra1 = keys1 - keys2
            extra2 = keys2 - keys1
            msg1 = "extra items in {0}: {1}"
            print(msg1.format(name1, extra1))
            print(msg1.format(name2, extra2))
            return False
        elif len(keys1) > len(keys2):
            bigger = keys1
            bigger_file = name1
            smaller = keys2
        else:
            bigger = keys2
            bigger_file = name2
            smaller = keys1
        res = bigger - smaller
        msg = "more items in {0}: {1}"
        print(msg.format(bigger_file, res))
        return False
    else:
        unequal_items = []
        for k, v in dict1.items():
            if type(v) == np.ndarray:
                check &= comp_array(k, v, dict2[k], tol, unequal_items,
                                    exceptions=exceptions,
                                    relative=relative)
            else:
                try:
                    check &= comp_scalar(
                        k, v, dict2[k], tol, unequal_items,
                        exceptions=exceptions, relative=relative)
                except TypeError:
                    check &= comp_array(
                        k, np.array(v), np.array(dict2[k]), tol,
                        unequal_items, exceptions=exceptions,
                        relative=relative)

        if verbose and unequal_items:
            frmt = "Name {0}"
            res = [frmt.format(x[0]) for x in unequal_items]
            print("Different arrays: ", res)
            return False

    return check


def to_timepath_shape(some_array):
    '''
    This function takes an vector of length T and tiles it to fill a
    Tx1x1 array for time path computations.

    Args:
        some_array (Numpy array): array to reshape

    Returns:
        tp_array (Numpy  array): reshaped array

    '''
    tp_array = some_array.reshape(some_array.shape[0], 1, 1)
    return tp_array


def get_initial_path(x1, xT, p, shape):
    r'''
    This function generates a path from point x1 to point xT such that
    that the path x is a linear or quadratic function of time t.

        * linear:    `x = d*t + e`
        * quadratic: `x = a*t^2 + b*t + c`

    Args:
        x1 (scalar): initial value of the function x(t) at t=0
        xT (scalar): value of the function x(t) at t=T-1
        T (int): number of periods of the path, must be >= 3
        shape (str): shape of guess for time path, "linear", "ratio",
            or "quadratic"

    Returns:
        xpath (Numpy array): guess of variable over the time path

    Notes:
        The identifying assumptions for quadratic are the following:
            1. `x1` is the value at time `t=0: x1 = c
            2. `xT` is the value at time `t=T-1: xT = a*(T-1)^2 + b*(T-1) + c`
            3. the slope of the path at `t=T-1` is 0: 0 = 2*a*(T-1) + b`

    '''
    if shape == "linear":
        xpath = np.linspace(x1, xT, p.T)
    elif shape == "ratio":
        domain = np.linspace(0, p.T, p.T)
        domain2 = np.tile(domain.reshape(p.T, 1, 1), (1, p.S, p.J))
        xpath = (-1 / (domain2 + 1)) * (xT - x1) + xT
    elif shape == "quadratic":
        cc = x1
        bb = 2 * (xT - x1) / (p.T - 1)
        aa = (x1 - xT) / ((p.T - 1) ** 2)
        xpath = (aa * (np.arange(0, p.T).reshape(p.T, 1, 1) ** 2) +
                 (bb * np.arange(0, p.T).reshape(p.T, 1, 1)) + cc)
    ending_x_tail = np.tile(xT.reshape(1, p.S, p.J),
                            (p.S, 1, 1))
    xpath_full = np.append(xpath, ending_x_tail, axis=0)

    return xpath_full


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
        except TypeError:  # pragma no cover
            obj = pickle.load(f)  # pragma no cover
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


def save_return_table(table_df, output_type, path, precision=2):
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
    pd.options.display.float_format = (
        '{:,.' + str(precision) + 'f}').format
    if path is None:
        if output_type == 'tex':
            tab_str = table_df.to_latex(index=False, na_rep='')
            return tab_str
        elif output_type == 'json':
            tab_str = table_df.to_json(double_precision=precision)
            return tab_str
        elif output_type == 'html':
            tab_html = table_df.to_html(
                classes="table table-striped table-hover"
                ).replace('\n', '')
            tab_html.replace('\n', '')
            return tab_html
        else:
            return table_df
    else:
        if output_type == 'tex':
            table_df.to_latex(buf=path, index=False, na_rep='')
        elif output_type == 'csv':
            table_df.to_csv(path_or_buf=path, index=False, na_rep='')
        elif output_type == 'json':
            table_df.to_json(path_or_buf=path,
                             double_precision=precision)
        elif output_type == 'excel':
            table_df.to_excel(excel_writer=path, index=False, na_rep='')
        else:
            print('Please enter a valid output format')  # pragma no cover
            assert(False)  # pragma no cover


class Inequality():
    '''
    A class with methods to compute different measures of inequality.
    '''

    def __init__(self, dist, pop_weights, ability_weights, S, J):
        '''
        Args:
            dist (Numpy array): distribution of endogenous variables
                over age and lifetime income group, size, SxJ
            pop_weights (Numpy array): fraction of population by each
                age, length S
            ability_weights (Numpy array): fraction of population for
                each lifetime income group, length J
            S (int): number of economically active periods in lifetime
            J (int): number of ability types

        Returns:
            None

        '''
        self.dist = dist
        self.pop_weights = pop_weights
        self.ability_weights = ability_weights
        weights = (np.tile(pop_weights.reshape(S, 1), (1, J)) *
                   ability_weights.reshape(1, J))
        flattened_dist = dist.flatten()
        flattened_weights = weights.flatten()
        idx = np.argsort(flattened_dist)
        self.sort_dist = flattened_dist[idx]
        self.sort_weights = flattened_weights[idx]
        self.cum_weights = np.cumsum(self.sort_weights)

    def gini(self, type='overall'):
        '''
        Compute the Gini coefficient

        Args:
            None

        Returns:
            gini_coeff (scalar): Gini coefficient
        '''
        if type == 'overall':
            p = np.cumsum(self.sort_weights)
            nu = np.cumsum(self.sort_dist * self.sort_weights)
        elif type == 'age':
            flattened_dist = self.dist.sum(axis=1).flatten()
            flattened_weights = self.pop_weights.flatten()
            idx = np.argsort(flattened_dist)
            sort_dist = flattened_dist[idx]
            sort_weights = flattened_weights[idx]/flattened_weights.sum()
            p = np.cumsum(sort_weights)
            nu = np.cumsum(sort_dist*sort_weights)
        elif type == 'ability':
            flattened_dist = self.dist.sum(axis=0).flatten()
            flattened_weights = self.ability_weights.flatten()
            idx = np.argsort(flattened_dist)
            sort_dist = flattened_dist[idx]
            sort_weights = flattened_weights[idx]/flattened_weights.sum()
            p = np.cumsum(sort_weights)
            nu = np.cumsum(sort_dist*sort_weights)
        nu = nu / nu[-1]
        gini_coeff = (nu[1:] * p[:-1]).sum() - (nu[:-1] * p[1:]).sum()

        return gini_coeff

    def var_of_logs(self):
        '''
        Compute the variance of logs

        Args:
            None

        Returns:
            var_ln_dist (scalar): variance of logs

        '''
        ln_dist = np.log(self.sort_dist)
        weight_mean = ((
            ln_dist * self.sort_weights).sum() / self.sort_weights.sum())
        var_ln_dist = ((
            (self.sort_weights * ((ln_dist - weight_mean) ** 2)).sum())
                       * (1. / (self.sort_weights.sum())))

        return var_ln_dist

    def ratio_pct1_pct2(self, pct1, pct2):
        '''
        Compute the pct1/pct2 percentile ratio

        Args:
            pct1 (scalar): percentile to compute the top pctile% for,
                in (0, 1).
            pct2 (scalar): percentile to compute the top pctile% for,
                in (0, 1)

        Returns:
            pct_ratio (scalar): ratio of pct1 to pct2

        Notes:
            usually pct1 > pct 2
        '''
        assert pct1 > 0
        assert pct1 < 1
        assert pct2 > 0
        assert pct2 < 1
        loc_pct1 = np.argmin(np.abs(self.cum_weights - pct1))
        loc_pct2 = np.argmin(np.abs(self.cum_weights - pct2))
        pct_ratio = self.sort_dist[loc_pct1] / self.sort_dist[loc_pct2]

        return pct_ratio

    def pct(self, pct):
        '''
        Returns value at given percentile

        Args:
            pct1 (scalar): percentile to compute the value at,
                in (0, 1).

        Returns:
            value (scalar): value of variable at pct

        '''
        assert pct > 0
        assert pct < 1
        loc_pct = np.argmin(np.abs(self.cum_weights - pct))
        value = self.sort_dist[loc_pct]

        return value

    def top_share(self, pctile):
        '''
        Compute the top X% share

        Args:
            pctile (scalar): percentile to compute the top pctile% for,
                in (0, 1).

        Returns:
            pctile_share (scalar): share of variable attributed to the
                top pctile group
        '''
        assert pctile > 0
        assert pctile < 1
        loc_pctile = np.argmin(np.abs(self.cum_weights - (1 - pctile)))
        pctile_share = ((
            self.sort_dist[loc_pctile:] *
            self.sort_weights[loc_pctile:]).sum() /
                        (self.sort_dist * self.sort_weights).sum())

        return pctile_share


def print_progress(iteration, total, source_name='', prefix='Progress:',
                   suffix='Complete', decimals=1, bar_length=50):
    '''
    Prints a progress bar to the terminal when completing small tasks
    of a larger job.

    Args:
        iteration (int>=1): which task the job is currently doing
        total (int>=1): how many tasks are in the job
        source_name (string): name of source data
        prefix (string): what to print before the progress bar
        suffix (string): what to print after the progress bar
        decimals (int>=0): how many decimals in the percentage
        bar_length (int>=3): how many boxes in the progress bar

    Functions called: None

    Objects created within function:
        status (string): status of download
        str_format (string): string containing percentage completed
        percents (string): percentage completed
        filled_length (int): number of boxes in the progress bar to fill
        bar (string): progress bar

    Returns: status
    '''
    status = 'Incomplete'
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    if iteration == 0:
        if source_name == '':
            sys.stdout.write('Accessing data files...\n')
        else:
            sys.stdout.write('Accessing ' + source_name +
                             ' data files...\n')

    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.write('Computing...\n')
        status = 'Complete'
    sys.stdout.flush()

    return status


def fetch_files_from_web(file_urls):
    '''
    Fetches zip files from respective web addresses and saves them as
    temporary files. Prints progress bar as it downloads the files.

    Args:
        file_urls (list of strings): list of URLs of respective data zip
            files

    Functions called:
        print_progress()

    Objects created within function:
        local_paths = list, local paths for teporary files
        iteration   = int, the number of files that have been downloaded
        total       = total, the total number of files to download
        f           = temporary file of monthly CPS survey
        path        = string, local path for temporary file
        zipped_file = ZipFile object, opened zipfile

    Files created by this function:
        .dta file for each year of SCF data

    Returns:
        local_paths (list of strings): local paths of temporary data
            files
    '''
    local_paths = []

    iteration = 0
    total = len(file_urls)
    _ = print_progress(iteration, total, source_name='SCF')

    for file_url in file_urls:
        # url = requests.get(file_url) (if using reuests package)
        url = urllib.request.urlopen(file_url)

        f = NamedTemporaryFile(delete=False)
        path = f.name

        # url.content (if using requests package)
        with ZipFile(BytesIO(url.read())) as zipped_file:
            for contained_file in zipped_file.namelist():
                f.write(zipped_file.open(contained_file).read())
                # for line in zipped_file.open(contained_file).readlines():
                #     f.write(line)

        local_paths.append(path)

        f.close()

        iteration += 1
        _ = print_progress(iteration, total, source_name='SCF')

    return local_paths


def not_connected(url='http://www.google.com/', timeout=5):
    '''
    Checks for internet connection status of machine.

    Args:
        url (string): url used to check connectivity
        timeout (float>0): time to wait for timeout

    Functions called: None

    Returns:
        Boolean singleton: =True if connection was made within timeout

    Raises:
        ConnectionError: If no response from url withing timeout
    '''
    try:
        _ = requests.get(url, timeout=timeout)
        return False
    except requests.ConnectionError:
        return True
