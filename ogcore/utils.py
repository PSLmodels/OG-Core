# Packages
import os
import sys
import requests
from zipfile import ZipFile
import urllib
from tempfile import NamedTemporaryFile
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import pickle
import urllib3
import ssl
import json

EPSILON = 1e-10  # tolerance or comparison functions


def mkdirs(path):
    """
    Makes directories to save output.

    Args:
        path (str): path name for new directory

    Returns:
        None

    """

    try:
        os.makedirs(path)
    except OSError as oe:
        if oe.errno == 17:  # 17 is an error code if can't make path
            pass


def pct_diff_func(simul, data):
    """
    Used to calculate the absolute percent difference between data
    moments and model moments.

    Args:
        simul (array_like): any shape, model moments
        data (array_like): same shape as simul, data moments

    Functions called: None

    Returns:
        output (array_like): percentage differences between model and
            data moments
    """
    if np.asarray(data).all() != 0:
        frac = (simul - data) / data
        output = np.abs(frac)
    else:
        output = np.abs(simul - data)
    return output


def convex_combo(var1, var2, nu):
    """
    Takes the convex combination of two variables, where nu is in [0,1].

    Args:
        var1 (array_like): any shape, variable 1
        var2 (array_like): same shape as var1, variable 2
        nu (scalar): weight on var1 in convex combination, in [0, 1]

    Returns:
        combo (array_like): same shape as var1, convex combination of
        var1 and var2

    """
    combo = nu * var1 + (1 - nu) * var2
    return combo


def pickle_file_compare(
    fname1, fname2, tol=1e-3, exceptions={}, relative=False
):
    """
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
    """
    pkl1 = safe_read_pickle(fname1)
    pkl2 = safe_read_pickle(fname2)
    comparison = dict_compare(
        fname1,
        pkl1,
        fname2,
        pkl2,
        tol=tol,
        exceptions=exceptions,
        relative=relative,
    )

    return comparison


def comp_array(name, a, b, tol, unequal, exceptions={}, relative=False):
    """
    Compare two arrays in the L inifinity norm. Return True if | a - b | < tol,
    False otherwise. If not equal, add items to the unequal list name: the name
    of the value being compared

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
    """

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
    """
    Compare two scalars in the L inifinity norm. Return True if
    abs(a - b) < tol, False otherwise. If not equal, add items to the unequal
    list.

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
    """

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


def dict_compare(
    name1,
    dict1,
    name2,
    dict2,
    tol,
    verbose=False,
    exceptions={},
    relative=False,
):
    r"""
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

    """

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
                check &= comp_array(
                    k,
                    v,
                    dict2[k],
                    tol,
                    unequal_items,
                    exceptions=exceptions,
                    relative=relative,
                )
            else:
                try:
                    check &= comp_scalar(
                        k,
                        v,
                        dict2[k],
                        tol,
                        unequal_items,
                        exceptions=exceptions,
                        relative=relative,
                    )
                except TypeError:
                    check &= comp_array(
                        k,
                        np.array(v),
                        np.array(dict2[k]),
                        tol,
                        unequal_items,
                        exceptions=exceptions,
                        relative=relative,
                    )

        if verbose and unequal_items:
            frmt = "Name {0}"
            res = [frmt.format(x[0]) for x in unequal_items]
            print("Different arrays: ", res)
            return False

    return check


def to_timepath_shape(some_array):
    """
    This function takes an vector of length T and tiles it to fill a
    Tx1x1 array for time path computations.

    Args:
        some_array (Numpy array): array to reshape

    Returns:
        tp_array (Numpy  array): reshaped array

    """
    tp_array = some_array.reshape(some_array.shape[0], 1, 1)
    return tp_array


def get_initial_path(x1, xT, p, shape):
    r"""
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
            1. `x1` is the value at time `t=0: x1 = c`
            2. `xT` is the value at time `t=T-1: xT = a*(T-1)^2 + b*(T-1) + c`
            3. the slope of the path at `t=T-1` is 0: `0 = 2*a*(T-1) + b`

    """
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
        xpath = (
            aa * (np.arange(0, p.T).reshape(p.T, 1, 1) ** 2)
            + (bb * np.arange(0, p.T).reshape(p.T, 1, 1))
            + cc
        )
    ending_x_tail = np.tile(xT.reshape(1, p.S, p.J), (p.S, 1, 1))
    xpath_full = np.append(xpath, ending_x_tail, axis=0)

    return xpath_full


def safe_read_pickle(file_path):
    """
    This function reads a pickle from Python 2 into Python 2 or Python 3

    Args:
        file_path (str): path to pickle file

    Returns:
        obj (object): object saved in pickle file

    """
    with open(file_path, "rb") as f:
        try:
            obj = pickle.load(f, encoding="latin1")
        except TypeError:  # pragma no cover
            obj = pickle.load(f)  # pragma no cover
    return obj


def rate_conversion(annual_rate, start_age, end_age, S):
    """
    This function converts annual rates to model period ratesself.

    Args:
        annual_rate (array_like): annualized rates
        start_age (int): age at which agents become economically active
        end_age (int): maximum age of agents
        S (int): number of model periods in agents life

    Returns:
        rate (array_like): model period rates

    """
    rate = (1 + annual_rate) ** ((end_age - start_age) / S) - 1
    return rate


def save_return_table(table_df, output_type, path, precision=2):
    """
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

    """
    pd.options.display.float_format = ("{:,." + str(precision) + "f}").format
    if path is None:
        if output_type == "tex":
            tab_str = table_df.to_latex(index=False, na_rep="")
            return tab_str
        elif output_type == "json":
            tab_str = table_df.to_json(double_precision=precision)
            return tab_str
        elif output_type == "html":
            tab_html = table_df.to_html(
                classes="table table-striped table-hover"
            ).replace("\n", "")
            tab_html.replace("\n", "")
            return tab_html
        else:
            return table_df
    else:
        if output_type == "tex":
            table_df.to_latex(buf=path, index=False, na_rep="")
        elif output_type == "csv":
            table_df.to_csv(path_or_buf=path, index=False, na_rep="")
        elif output_type == "json":
            table_df.to_json(path_or_buf=path, double_precision=precision)
        elif output_type == "excel":
            table_df.to_excel(excel_writer=path, index=False, na_rep="")
        else:
            print("Please enter a valid output format")  # pragma no cover
            assert False  # pragma no cover


class Inequality:
    """
    A class with methods to compute different measures of inequality.
    """

    def __init__(self, dist, pop_weights, ability_weights, S, J):
        """
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

        """
        self.dist = dist
        self.pop_weights = pop_weights
        self.ability_weights = ability_weights
        weights = np.tile(
            pop_weights.reshape(S, 1), (1, J)
        ) * ability_weights.reshape(1, J)
        flattened_dist = dist.flatten()
        flattened_weights = weights.flatten()
        idx = np.argsort(flattened_dist)
        self.sort_dist = flattened_dist[idx]
        self.sort_weights = flattened_weights[idx]
        self.cum_weights = np.cumsum(self.sort_weights)

    def gini(self, type="overall"):
        """
        Compute the Gini coefficient

        Args:
            None

        Returns:
            gini_coeff (scalar): Gini coefficient
        """
        if type == "overall":
            p = np.cumsum(self.sort_weights)
            nu = np.cumsum(self.sort_dist * self.sort_weights)
        elif type == "age":
            flattened_dist = self.dist.sum(axis=1).flatten()
            flattened_weights = self.pop_weights.flatten()
            idx = np.argsort(flattened_dist)
            sort_dist = flattened_dist[idx]
            sort_weights = flattened_weights[idx] / flattened_weights.sum()
            p = np.cumsum(sort_weights)
            nu = np.cumsum(sort_dist * sort_weights)
        elif type == "ability":
            flattened_dist = self.dist.sum(axis=0).flatten()
            flattened_weights = self.ability_weights.flatten()
            idx = np.argsort(flattened_dist)
            sort_dist = flattened_dist[idx]
            sort_weights = flattened_weights[idx] / flattened_weights.sum()
            p = np.cumsum(sort_weights)
            nu = np.cumsum(sort_dist * sort_weights)
        nu = nu / nu[-1]
        gini_coeff = (nu[1:] * p[:-1]).sum() - (nu[:-1] * p[1:]).sum()

        return gini_coeff

    def var_of_logs(self):
        """
        Compute the variance of logs

        Args:
            None

        Returns:
            var_ln_dist (scalar): variance of logs

        """
        ln_dist = np.log(self.sort_dist)
        weight_mean = (
            ln_dist * self.sort_weights
        ).sum() / self.sort_weights.sum()
        var_ln_dist = (
            (self.sort_weights * ((ln_dist - weight_mean) ** 2)).sum()
        ) * (1.0 / (self.sort_weights.sum()))

        return var_ln_dist

    def ratio_pct1_pct2(self, pct1, pct2):
        """
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
        """
        assert pct1 > 0
        assert pct1 < 1
        assert pct2 > 0
        assert pct2 < 1
        loc_pct1 = np.argmin(np.abs(self.cum_weights - pct1))
        loc_pct2 = np.argmin(np.abs(self.cum_weights - pct2))
        pct_ratio = self.sort_dist[loc_pct1] / self.sort_dist[loc_pct2]

        return pct_ratio

    def pct(self, pct):
        """
        Returns value at given percentile

        Args:
            pct1 (scalar): percentile to compute the value at,
                in (0, 1).

        Returns:
            value (scalar): value of variable at pct

        """
        assert pct > 0
        assert pct < 1
        loc_pct = np.argmin(np.abs(self.cum_weights - pct))
        value = self.sort_dist[loc_pct]

        return value

    def top_share(self, pctile):
        """
        Compute the top X% share

        Args:
            pctile (scalar): percentile to compute the top pctile% for,
                in (0, 1).

        Returns:
            pctile_share (scalar): share of variable attributed to the
                top pctile group
        """
        assert pctile > 0
        assert pctile < 1
        loc_pctile = np.argmin(np.abs(self.cum_weights - (1 - pctile)))
        pctile_share = (
            self.sort_dist[loc_pctile:] * self.sort_weights[loc_pctile:]
        ).sum() / (self.sort_dist * self.sort_weights).sum()

        return pctile_share


def print_progress(
    iteration,
    total,
    source_name="",
    prefix="Progress:",
    suffix="Complete",
    decimals=1,
    bar_length=50,
):
    """
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
    """
    status = "Incomplete"
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    if iteration == 0:
        if source_name == "":
            sys.stdout.write("Accessing data files...\n")
        else:
            sys.stdout.write("Accessing " + source_name + " data files...\n")

    sys.stdout.write(
        "\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)
    ),

    if iteration == total:
        sys.stdout.write("\n")
        sys.stdout.write("Computing...\n")
        status = "Complete"
    sys.stdout.flush()

    return status


def fetch_files_from_web(file_urls):
    """
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
    """
    local_paths = []

    iteration = 0
    total = len(file_urls)
    _ = print_progress(iteration, total, source_name="SCF")

    for file_url in file_urls:
        request = urllib.request.Request(file_url)
        request.add_header("User-Agent", "Mozilla/5.0")
        url = urllib.request.urlopen(request)

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
        _ = print_progress(iteration, total, source_name="SCF")

    return local_paths


def not_connected(url="http://www.google.com/", timeout=5):
    """
    Checks for internet connection status of machine.

    Args:
        url (string): url used to check connectivity
        timeout (float>0): time to wait for timeout

    Functions called: None

    Returns:
        Boolean singleton: =True if connection was made within timeout

    Raises:
        ConnectionError: If no response from url withing timeout
    """
    try:
        _ = requests.get(url, timeout=timeout)
        return False
    except requests.ConnectionError:
        return True


def avg_by_bin(x, y, weights=None, bins=10, eql_pctl=True):
    """
    For an x series and y series (vectors), this function computes vectors of
    weighted average values in specified bins.

    Args:
        x (array_like, either pd.Series or np.array): x values
        y (array_like, either pd.Series or np.array): y values
        weights (array_like, either pd.Series or np.array): weights
        bins (scalar, list, or np.array): number of bins or specific bin edges
        eql_pctl (boolean): if True, bins are equal percentiles of x

    Returns:
        x_binned(array_like, np.array): weighted average x values in bins
        y_binned (array_like, np.array): weighted average y values in bins
        weights_binned (array_like, np.array): total weights in bins
    """
    # If vectors are pandas Series objects, convert to numpy arrays
    if isinstance(x, pd.Series):
        x = x.to_numpy()
        y = y.to_numpy()
        if isinstance(weights, pd.Series):
            weights = weights.to_numpy()

    # Set original observations number and weights
    obs = len(x)
    if weights is None:
        weights = np.ones_like(x)

    # Case of bins=K bins that are either equal percentiles or equal x-width
    if np.isscalar(bins):
        x_binned = np.zeros(bins)
        y_binned = np.zeros(bins)
        weights_binned = np.zeros(bins)

        pctl_cuts = np.append(0, np.linspace(1 / bins, 1, bins))

        # Case of bins=K bins that are equal percentiles
        if eql_pctl:
            # Sort x and weights by x in ascending order
            df = pd.DataFrame(
                data=np.hstack(
                    (
                        x.reshape((-1, 1)),
                        y.reshape((-1, 1)),
                        weights.reshape((-1, 1)),
                    )
                ),
                columns=["x", "y", "weights"],
            ).sort_values(by=["x"])
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()
            weights = df["weights"].to_numpy()

            weights_norm_cum = weights.cumsum() / weights.sum()
            weights_norm_cum[-1] = 1.0

            bin = 1
            i_part_last = int(0)
            pct_ind_end = 0.0
            for i in range(obs):
                if weights_norm_cum[i] >= pctl_cuts[bin]:
                    pct_ind_beg = 1 - pct_ind_end
                    pct_ind_end = 1 - (
                        (weights_norm_cum[i] - pctl_cuts[bin])
                        / (weights_norm_cum[i] - weights_norm_cum[i - 1])
                    )
                    weights_vec_bin = np.concatenate(
                        [
                            [pct_ind_beg * weights[i_part_last]],
                            weights[i_part_last + 1 : i],
                            [pct_ind_end * weights[i]],
                        ]
                    )
                    weights_binned[bin - 1] = weights_vec_bin.sum()
                    x_binned[bin - 1] = np.average(
                        x[i_part_last : i + 1], weights=weights_vec_bin
                    )
                    y_binned[bin - 1] = np.average(
                        y[i_part_last : i + 1], weights=weights_vec_bin
                    )

                    i_part_last = i
                    bin += 1

        # Case of bins=K bins that are equal x-width
        else:
            bin_edges = np.linspace(x.min(), x.max(), bins + 1)
            for bin in range(bins):
                if bin == 0:
                    bin_ind = x >= bin_edges[bin] & x <= bin_edges[bin + 1]
                else:
                    bin_ind = x > bin_edges[bin] & x <= bin_edges[bin + 1]
                x_binned[bin] = np.average(
                    x[bin_ind], weights=weights[bin_ind]
                )
                y_binned[bin] = np.average(
                    y[bin_ind], weights=weights[bin_ind]
                )
                weights_binned[bin] = weights[bin_ind].sum()

    # Case of bin edges specified, eql_pctl must be False
    elif isinstance(bins, list) or isinstance(bins, np.ndarray):
        if eql_pctl:
            err_msg = (
                "avg_by_bin ERROR: eql_pctl=True with bins given as "
                + "list or ndarray. In this case, eql_pctl must be set "
                + "to False"
            )
            raise ValueError(err_msg)
        bin_num = len(bins) - 1
        x_binned = np.zeros(bin_num)
        y_binned = np.zeros(bin_num)
        weights_binned = np.zeros(bin_num)
        for bin in range(bin_num):
            if bin == 0:
                bin_ind = x >= bins[bin] & x <= bins[bin + 1]
            else:
                bin_ind = x > bins[bin] & x <= bins[bin + 1]
            x_binned[bin] = np.average(x[bin_ind], weights=weights[bin_ind])
            y_binned[bin] = np.average(y[bin_ind], weights=weights[bin_ind])
            weights_binned[bin] = weights[bin_ind].sum()

    else:
        err_msg = (
            "avg_by_bin ERROR: bins value is type "
            + str(type(bins))
            + ", but needs to be either scalar, list, or ndarray."
        )
        raise ValueError(err_msg)

    return x_binned, y_binned, weights_binned


def extrapolate_array(param_in, dims=None, item="Parameter Name"):
    """
    Extrapolates input values to fit model dimensions. Using this allows
    users to input smaller dimensional arrays and have the model infer
    that they want these extrapolated.

    Example: User enters a constant for total factor productivity, Z_mt.
    This function will create an array of size TxM where TFP is the same
    for all industries in all years.

    Args:
        param_in (array_like): input parameter value
        dims (tuple): size of each dimension param_out should take, note
            that the first dimension is always T+S
        item (str): parameter name, used to inform user if exception

    Returns:
        param_out (array_like): reshape parameter for use in OG-Core
    """
    if len(dims) == 1:
        # this is the case for 1D arrays, they vary over the time path
        if param_in.ndim > 1:
            param_in = np.squeeze(param_in, axis=1)
        if param_in.size > dims[0]:
            param_in = param_in[: dims[0]]
        param_out = np.concatenate(
            (
                param_in,
                np.ones((dims[0] - param_in.size)) * param_in[-1],
            )
        )
    elif len(dims) == 2:
        # this is the case for 2D arrays, they vary over the time path
        # and some other dimension (e.g., by industry)
        if param_in.ndim == 1:
            # case where enter single number, so assume constant
            # across all dimensions
            if param_in.shape[0] == 1:
                param_out = np.ones((dims)) * param_in[0]
            # case where user enters just one year for all types in 2nd dim
            if param_in.shape[0] == dims[1]:
                param_out = np.tile(param_in.reshape(1, dims[1]), (dims[0], 1))
            else:
                # case where user enters multiple years, but not full time
                # path
                # will assume they implied values the same across 2nd dim
                # and will fill in all periods
                param_out = np.concatenate(
                    (
                        param_in,
                        np.ones((dims[0] - param_in.size)) * param_in[-1],
                    )
                )
                param_out = np.tile(
                    param_out.reshape(dims[0], 1), (1, dims[1])
                )
        elif param_in.ndim == 2:
            # case where enter values along 2 dimensions, but those aren't
            # complete
            if param_in.shape[1] > 1 and param_in.shape[1] != dims[1]:
                print(
                    "please provide values of "
                    + item
                    + " for element in the 2nd dimension (or enter a "
                    + "constant if the value is common across elements in "
                    + "the second dimension}"
                )
                assert False
            if param_in.shape[1] == 1:
                # Case where enter just one value along the 2nd dim, will
                # assume constant across it
                param_in = np.tile(
                    param_in.reshape(param_in.shape[0], 1), (1, dims[1])
                )
            if param_in.shape[0] > dims[0]:
                # Case where enter a number of values along the time
                # dimension that exceeds the length of the time path
                param_in = param_in[: dims[0], :]
            param_out = np.concatenate(
                (
                    param_in,
                    np.ones(
                        (
                            dims[0] - param_in.shape[0],
                            param_in.shape[1],
                        )
                    )
                    * param_in[-1, :],
                )
            )
    elif len(dims) == 3:
        # this is the case for 3D arrays, they vary over the time path
        # and two other dimensions (e.g., by age and ability type)
        if param_in.ndim == 1:
            # case if S x 1 input
            assert param_in.shape[0] == dims[1]
            param_out = np.tile(
                (
                    np.tile(param_in.reshape(dims[1], 1), (1, dims[2]))
                    / dims[2]
                ).reshape(1, dims[1], dims[2]),
                (dims[0], 1, 1),
            )
        # this could be where vary by S and J or T and S
        elif param_in.ndim == 2:
            # case if S by J input
            if param_in.shape[0] == dims[1]:
                param_out = np.tile(
                    param_in.reshape(1, dims[1], dims[2]),
                    (dims[0], 1, 1),
                )
            # case if T by S input
            elif param_in.shape[0] == dims[0] - dims[1]:
                param_in = (
                    np.tile(
                        param_in.reshape(dims[0] - dims[1], dims[1], 1),
                        (1, 1, dims[2]),
                    )
                    / dims[2]
                )
                param_out = np.concatenate(
                    (
                        param_in,
                        np.tile(
                            param_in[-1, :, :].reshape(1, dims[1], dims[2]),
                            (dims[1], 1, 1),
                        ),
                    ),
                    axis=0,
                )
            else:
                print(item + " dimensions are: ", param_in.shape)
                print("please give an " + item + " that is either SxJ or TxS")
                assert False
        elif param_in.ndim == 3:
            # this is the case where input varies by T, S, J
            if param_in.shape[0] > dims[0]:
                param_out = param_in[: dims[0], :, :]
            else:
                param_out = np.concatenate(
                    (
                        param_in,
                        np.tile(
                            param_in[-1, :, :].reshape(1, dims[1], dims[2]),
                            (dims[0] - param_in.shape[0], 1, 1),
                        ),
                    ),
                    axis=0,
                )

    return param_out


def extrapolate_nested_list(list_in, dims=(400, 80, 1)):
    """
    Function to extrapolate a nested list to a specified size.

    Currently only set up for 3 deep nested lists, but could be
    generalized to deeper or shallower lists.

    Args:
        list_in (list): list to extrapolate
        dims (tuple): dimensions of the output list

    Returns:
        list_out (list): extrapolated list
    """
    T, S, num_params = dims
    try:
        list_in = list_in.tolist()  # in case parameters are numpy arrays
    except AttributeError:  # catches if they are lists already
        pass
    assert isinstance(list_in, list), "please give a list"

    def depth(L):
        return isinstance(L, list) and max(map(depth, L)) + 1

    # for now, just have this work for 3 deep lists since
    # the only OG-Core use case is for tax function parameters
    assert depth(list_in) == 3, "please give a list that is three lists deep"
    assert depth(list_in) == len(
        dims
    ), "please make sure the depth of nested list is equal to the length of dims to extrapolate"
    # Extrapolate along the first dimension
    if len(list_in) > T + S:
        list_in = list_in[: T + S]
    if len(list_in) < T + S:
        params_to_add = [list_in[-1]] * (T + S - len(list_in))
        list_in.extend(params_to_add)
    # Extrapolate along the second dimension
    for t in range(len(list_in)):
        if len(list_in[t]) > S:
            list_in[t] = list_in[t][:S]
        if len(list_in[t]) < S:
            params_to_add = [list_in[t][-1]] * (S - len(list_in[t]))
            list_in[t].extend(params_to_add)

    return list_in


class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    """
    The UN Data Portal server doesn't support "RFC 5746 secure renegotiation". This causes and error when the client is using OpenSSL 3, which enforces that standard by default.
    The fix is to create a custom SSL context that allows for legacy connections. This defines a function get_legacy_session() that should be used instead of requests().
    """

    # "Transport adapter" that allows us to use custom ssl_context.
    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=self.ssl_context,
        )


def get_legacy_session():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT  #in Python 3.12 you will be able to switch from 0x4 to ssl.OP_LEGACY_SERVER_CONNECT.
    session = requests.session()
    session.mount("https://", CustomHttpAdapter(ctx))
    return session


def shift_bio_clock(
    param_in,
    initial_effect_period=1,
    final_effect_period=1,
    total_effect=1,
    min_age_effect_felt=20,
    bound_below=False,
    bound_above=False,
    use_spline=False,
):
    """
    This function takes an initial array of parameters that has a time
    dimension and an age dimension. It then applies a shift along the
    age dimension (i.e., a change in the "biological clock") and phases
    this shift in over some period of time.

    Args:
        param_in (Numpy array): initial parameter values, first two
            dimensions must be time then age, respectively
        initial_effect_period (int): first model period when transition
            to new parameter values occurs
        final_effect_period (int): model period when effect is fully
            phased in (so transition from param_in to param_out happens
            between model periods `initial_effect_period` and
            `final_effect_period`)
        total_effect (float): total number of model periods to shift
            the biological clock (allows for fractions of periods)
        min_age_effect_felt (int): minimum age at which the effect of
            the transition is felt in model periods
        bound_below (bool): whether param_out bounded below by param_in
        bound_above (bool): whether param_out bounded above by param_in
        use_spline (bool): whether to use a cubic spline to interpolate
            tail of param_out

    Returns:
        param_out (Numpy array): updated parameter values
    """
    assert (
        total_effect >= 0
    )  # this code would need to change to accommodate effects < 0
    n_dims = param_in.ndim
    assert n_dims >= 2
    T = param_in.shape[1]
    S = param_in.shape[1]

    # create a linear transition path
    t = (
        final_effect_period - initial_effect_period + 1
    )  # number of periods transition over
    transition_path = np.linspace(0, 1.0, t)
    transition_arr = np.zeros_like(param_in, dtype=float)
    for i in range(t):
        transition_arr[initial_effect_period + i, ...] = transition_path[
            i
        ] * np.ones_like(param_in[0, ...])
    transition_arr[final_effect_period:, ...] = np.ones_like(
        param_in[final_effect_period:, ...]
    )

    param_shift = param_in.copy()
    # Accounting for shifts that are fractions of a model period
    total_effect_ru = int(np.ceil(total_effect))
    if total_effect > 0:
        pct_effect = total_effect / total_effect_ru
    else:
        pct_effect = 0
    # apply the transition path to the initial parameters
    # find diff from shifting bio clock back total_effect years
    param_shift[:, min_age_effect_felt:, ...] = param_in[
        :, (min_age_effect_felt - total_effect_ru) : S - total_effect_ru, ...
    ]
    if use_spline:
        # use cubic spline to avoid plateau at end of lifecycle profile
        T = param_in.shape[0]
        if n_dims == 3:
            J = param_in.shape[-1]
            for k in range(initial_effect_period, T):
                for j in range(J):
                    spline = CubicSpline(
                        np.arange(S - total_effect),
                        param_shift[k, :-total_effect, j],
                    )
                    param_shift[k, -total_effect:, j] = spline(
                        np.arange(S - total_effect, S)
                    )
        else:
            for k in range(initial_effect_period, T):
                spline = CubicSpline(
                    np.arange(S - total_effect), param_shift[k, :-total_effect]
                )
                param_shift[k, -total_effect:] = spline(
                    np.arange(S - total_effect, S)
                )
    # make sure values are not lower after shift
    if bound_below:
        param_shift = np.maximum(param_shift, param_in)
    # make sure values are not higher after shift
    if bound_above:
        param_shift = np.minimum(param_shift, param_in)
    # Now transition the shift over time using the transition path
    param_out = param_in + (
        transition_arr * pct_effect * (param_shift - param_in)
    )

    return param_out


def pct_change_unstationarized(
    tpi_base,
    param_base,
    tpi_reform,
    param_reform,
    output_vars=["K", "Y", "C", "L", "r", "w"],
):
    """
    This function takes the time paths of variables from the baseline
    and reform and parameters from the baseline and reform runs and
    computes percent changes for each variable in the output_vars list.
    The function first unstationarizes the time paths of the variables
    and then computes the percent changes.

    Args:
        tpi_base (Numpy array): time path of the output variables from
            the baseline run
        param_base (Specifications object): dictionary of parameters
            from the baseline run
        tpi_reform (Numpy array): time path of the output variables from
            the reform run
        param_reform (Specifications object): dictionary of parameters
            from the reform run
        output_vars (list): list of variables for which to compute
            percent changes

    Returns:
        pct_changes (dict): dictionary of percent changes for each
            variable in output_vars list
    """
    # compute non-stationary variables
    non_stationary_output = {"base": {}, "reform": {}}
    pct_changes = {}
    T = param_base.T
    for var in output_vars:
        if var in [
            "Y",
            "B",
            "K",
            "K_f",
            "K_d",
            "C",
            "I",
            "K_g",
            "I_g",
            "Y_vec",
            "K_vec",
            "C_vec",
            "I_total",
            "I_d",
            "BQ",
            "TR",
            "total_tax_revenue",
            "business_tax_revenue",
            "iit_payroll_tax_revenue",
            "iit_revenue",
            "payroll_tax_revenue",
            "agg_pension_outlays",
            "bequest_tax_revenue",
            "wealth_tax_revenue",
            "cons_tax_revenue",
            "G",
            "D",
            "D_f",
            "D_d",
            "UBI_path",
            "new_borrowing_f",
            "debt_service_f",
        ]:
            non_stationary_output["base"][var] = (
                tpi_base[var][:T]
                * np.cumprod(1 + param_base.g_n[:T])
                * np.exp(param_base.g_y * np.arange(param_base.T))
            )
            non_stationary_output["reform"][var] = (
                tpi_reform[var][:T]
                * np.cumprod(1 + param_reform.g_n[:T])
                * np.exp(param_reform.g_y * np.arange(param_reform.T))
            )
        elif var in [
            "L",
            "L_vec",
        ]:
            non_stationary_output["base"][var] = tpi_base[var][
                :T
            ] * np.cumprod(1 + param_base.g_n[:T])
            non_stationary_output["reform"][var] = tpi_reform[var][
                :T
            ] * np.cumprod(1 + param_reform.g_n[:T])
        elif var in [
            "w",
            "ubi_path",
            "tr_path",
            "bq_path",
            "bmat_splus1",
            "bmat_s",
            "c_path",
            "y_before_tax_path",
            "tax_path",
        ]:
            non_stationary_output["base"][var] = tpi_base[var][:T] * np.exp(
                param_base.g_y * np.arange(param_base.T)
            )
            non_stationary_output["reform"][var] = tpi_reform[var][
                :T
            ] * np.exp(param_reform.g_y * np.arange(param_reform.T))
        else:
            non_stationary_output["base"][var] = tpi_base[var][:T]
            non_stationary_output["reform"][var] = tpi_reform[var][:T]

        # calculate percent change
        pct_changes[var] = (
            non_stationary_output["reform"][var]
            / non_stationary_output["base"][var]
            - 1
        )

    return pct_changes


def param_dump_json(p, path=None):
    """
    This function creates a JSON file with the model parameters of the
    format used for the default_parameters.json file.

    Args:
        p (OG-Core Specifications class): model parameters object
        path (string): path to save JSON file to

    Returns:
        JSON (string): JSON on model parameters
    """
    converted_data = {}
    spec = p.specification(
        meta_data=False, include_empty=True, serializable=True, use_state=True
    )
    for key in p.keys():
        val = dict(spec[key][0])["value"]
        if isinstance(val, np.ndarray):
            converted_data[key] = val.tolist()
        else:
            converted_data[key] = val

    # Parameters that need to be turned into annual rates for default_parameters.json
    # g_y_annual
    # beta_annual
    # delta_annual
    # delta_tau_annual
    # delta_g_annual
    # world_int_rate_annual

    # Convert to JSON string
    json_str = json.dumps(converted_data, indent=4)

    if path is not None:
        with open(path, "w") as f:
            f.write(json_str)
    else:
        return json_str
