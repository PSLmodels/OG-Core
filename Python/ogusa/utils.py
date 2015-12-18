'''
------------------------------------------------------------------------
Last updated 7/16/2015

Miscellaneous functions for SS and TPI.

This python files calls:
    OUTPUT/Saved_moments/wealth_data_moments.pkl

------------------------------------------------------------------------
'''

# Packages
import os
from io import StringIO
import numpy as np
import cPickle as pickle
from pkg_resources import resource_stream, Requirement

EPSILON = 1e-10
PATH_EXISTS_ERRNO = 17


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as oe:
        if oe.errno == PATH_EXISTS_ERRNO:
            pass


def perc_dif_func(simul, data):
    '''
    Used to calculate the absolute percent difference between the data
    moments and model moments
    Inputs:
        simul = model moments (any shape)
        data  = data moments (same shape as simul)
    Output:
        output = absolute percent difference between data and model moments (same shape as simul)
    '''
    frac = (simul - data) / data
    output = np.abs(frac)
    return output


def convex_combo(var1, var2, nu):
    '''
    Takes the convex combination of two variables, where nu is the value
    between 0 and 1 in params.
    Inputs:
        var1 = (any shape)
        var2 = (same shape as var1)
        params = parameters list from model (list) (only nu is needed...perhaps it should just take that as an input)
    Outputs:
        combo = convex combination of var1 and var2 (same shape as var1)
    '''
    combo = nu * var1 + (1 - nu) * var2
    return combo


def check_wealth_calibration(wealth_model, factor_model, params, wealth_dir):
    '''
    Creates a vector of the percent differences between the
    model and data wealth moments for the two age groups for
    each J group.
    Inputs:
        wealth_model = model wealth levels (SxJ array)
        factor_model = factor to convert wealth levels to dollars (scalar)
        params = parameters list from model (list)
        wealth_dir = path to the wealth data momenets
    Outputs:
        wealth_fits = Fits for how well the model wealth levels match the data wealth levels ((2*J)x1 array)
    '''
    # Import the wealth data moments
    wealth_path = os.path.join(
        wealth_dir, "Saved_moments/wealth_data_moments.pkl")
    wealth_dict = pickle.load(open(wealth_path, "rb"))
    # Set lowest ability group's wealth to be a positive, not negative, number
    # for the calibration
    wealth_dict['wealth_data_array'][2:26, 0] = 500.0

    J, S, T, BW, beta, sigma, alpha, Z, delta, ltilde, nu, g_y,\
                  g_n_ss, tau_payroll, retire, mean_income_data,\
                  h_wealth, p_wealth, m_wealth, b_ellipse, upsilon = params

    # Convert wealth levels from model to dollar terms
    wealth_model_dollars = wealth_model * factor_model
    wealth_fits = np.zeros(2 * J)
    # Look at the percent difference between the fits for the first age group (20-44) and second age group (45-65)
    #   The wealth_data_array moment indices are shifted because they start at age 18
    # The :: indices is so that we look at the 2 moments for the lowest group,
    # the 2 moments for the second lowest group, etc in order
    wealth_fits[0::2] = perc_dif_func(np.mean(wealth_model_dollars[
                                      :24], axis=0), np.mean(wealth_dict['wealth_data_array'][2:26], axis=0))
    wealth_fits[1::2] = perc_dif_func(np.mean(wealth_model_dollars[
                                      24:45], axis=0), np.mean(wealth_dict['wealth_data_array'][26:47], axis=0))
    return wealth_fits


def read_file(path, fname):
    '''
    Read the contents of 'path'. If it does not exist, assume the file
    is installed in a .egg file, and adjust accordingly
    '''
    if not os.path.exists(os.path.join(path, fname)):
        path_in_egg = os.path.join("ogusa", fname)
        buf = resource_stream(Requirement.parse("ogusa"), path_in_egg)
        _bytes = buf.read()
        return StringIO(_bytes.decode("utf-8"))
    else:
        return open(os.path.join(path, fname))


def pickle_file_compare(fname1, fname2, tol=1e-3, exceptions={}, relative=False):
    '''
    Read two pickle files and unpickle each. We assume that each resulting
    object is a dictionary. The values of each dict are either numpy arrays
    or else types that are comparable with the == operator.
    '''

    pkl1 = pickle.load(open(fname1, 'rb'))
    pkl2 = pickle.load(open(fname2, 'rb'))

    return dict_compare(fname1, pkl1, fname2, pkl2, tol=tol,
                        exceptions=exceptions, relative=relative)


def comp_array(name, a, b, tol, unequal, exceptions={}, relative=False):
    '''
        Compare two arrays in the L inifinity norm
        Return True if | a - b | < tol, False otherwise
        If not equal, add items to the unequal list
        name: the name of the value being compared
    '''

    if name in exceptions:
        tol = exceptions[name]

    if not a.shape == b.shape:
        print "unequal shpaes for {0} comparison ".format(str(name))
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
            print "diff for {0} is {1} which is NOT OK".format(str(name), err)
            unequal.append((str(name), a, b))
            return False
        else:
            print "err is {0} which is OK".format(err)
            return True


def comp_scalar(name, a, b, tol, unequal, exceptions={}, relative=False):
    '''
        Compare two scalars in the L inifinity norm
        Return True if abs(a - b) < tol, False otherwise
        If not equal, add items to the unequal list
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
        print "err for {0} is {1} which is NOT OK".format(str(name), err)
        unequal.append((str(name), str(a), str(b)))
        return False
    else:
        print "err is {0} which is OK".format(err)
        return True


def dict_compare(fname1, pkl1, fname2, pkl2, tol, verbose=False, exceptions={}, relative=False):
    '''
    Compare two dictionaries. The values of each dict are either
    numpy arrays
    or else types that are comparable with the == operator.
    For arrays, they are considered the same if |x - y| < tol in
    the L_inf norm.
    For scalars, they are considered the same if x - y < tol
    '''

    keys1 = set(pkl1.keys())
    keys2 = set(pkl2.keys())
    check = True
    if keys1 != keys2:
        if len(keys1) == len(keys2):
            extra1 = keys1 - keys2
            extra2 = keys2 - keys1
            msg1 = "extra items in {0}: {1}"
            print msg1.format(fname1, extra1)
            print msg1.format(fname2, extra2)
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
        print msg.format(bigger_file, res)
        return False
    else:
        unequal_items = []
        for k, v in pkl1.items():
            if type(v) == np.ndarray:
                check &= comp_array(k, v, pkl2[k], tol, unequal_items,
                                    exceptions=exceptions, relative=relative)
            else:
                try:
                    check &= comp_scalar(k, v, pkl2[k], tol, unequal_items,
                                         exceptions=exceptions, relative=relative)
                except TypeError:
                    check &= comp_array(k, np.array(v), np.array(pkl2[k]), tol,
                                        unequal_items, exceptions=exceptions,
                                        relative=relative)

        if verbose == True and unequal_items:
            frmt = "Name {0}"
            res = [frmt.format(x[0]) for x in unequal_items]
            print "Different arrays: ", res
            return False

    return check
