'''
------------------------------------------------------------------------
Household functions for taxes in SS and TPI.

This file calls the following files:
    tax.py
------------------------------------------------------------------------
'''

# Import packages
import numpy as np

import tax

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def get_NetTaxLiab(tfparams, inc_lab_m, inc_cap_m, factor, T_H):
    '''
    --------------------------------------------------------------------
    This function generates net tax liability for each age (s)
    individual and ability type (j) given capital and labor income.
    --------------------------------------------------------------------
    tfparams  = (S x 10) matrix, 10 tax function parameters
                (A,B,D,C,E,F,max_x,min_x,max_y,min_y) for each age from
                final year of estimated data
    inc_lab_m = (S x J) matrix, labor income amounts from model for age
                (s) and ability (j) individuals
    inc_cap_m = (S x J) matrix, capital income amounts from model for
                age (s) and ability (j) individuals
    factor    = scalar > 0,  scaling factor to make average model income
                equal average data income in tax function
    T_H       = scalar, lump sum transfer to each household
    inc_lab   = (S x J) matrix, labor income amounts inflated by factor
                to match income amounts from the data
    inc_cap   = (S x J) matrix, capital income amounts inflated by
                factor to match income amounts from the data
    phi       = (S x J) matrix, labor income share of total income for
                each individual of age (s) and ability (j)
    A_mat     = (S x J) matrix, column vector of A tax function
                parameters by age copied across J columns
    B_mat     = (S x J) matrix, column vector of B tax function
                parameters by age copied across J columns
    C_mat     = (S x J) matrix, column vector of C tax function
                parameters by age copied across J columns
    D_mat     = (S x J) matrix, column vector of D tax function
                parameters by age copied across J columns
    E_mat     = (S x J) matrix, column vector of E tax function
                parameters by age copied across J columns
    F_mat     = (S x J) matrix, column vector of F tax function
                parameters by age copied across J columns
    max_x_mat = (S x J) matrix, column vector of max_x tax function
                parameters by age copied across J columns
    min_x_mat = (S x J) matrix, column vector of min_x tax function
                parameters by age copied across J columns
    max_y_mat = (S x J) matrix, column vector of max_y tax function
                parameters by age copied across J columns
    min_y_mat = (S x J) matrix, column vector of min_y tax function
                parameters by age copied across J columns
    Phi       = (S x J) matrix, convex combination of the labor income
                difference and the capital income difference between max
                rate and min rate
    P_num     = (S x J) matrix, numerator of ratio of polynomials
    P_den     = (S x J) matrix, denominator of ratio of polynomials
    MinConst  = (S x J) matrix, convex combination of the labor income
                and capital income minimum tax rates
    EffTxRt   = (S x J) matrix, tau(x,y) effective tax rate
    TaxLiab   = (S x J) matrix, total tax liability
    NetTxLiab = (S x J) matrix, net tax liability for each age (s)
                individual and ability type (j)

    returns: NetTxLiab
    --------------------------------------------------------------------
    '''
    J = inc_lab_m.shape[1]
    inc_lab = factor * inc_lab_m
    inc_cap = factor * inc_cap_m
    phi = inc_lab / (inc_lab + inc_cap)
    A_mat = np.tile(tfparams[:, 0], (1, J))
    B_mat = np.tile(tfparams[:, 1], (1, J))
    C_mat = np.tile(tfparams[:, 2], (1, J))
    D_mat = np.tile(tfparams[:, 3], (1, J))
    E_mat = np.tile(tfparams[:, 4], (1, J))
    F_mat = np.tile(tfparams[:, 5], (1, J))
    max_x_mat = np.tile(tfparams[:, 6], (1, J))
    min_x_mat = np.tile(tfparams[:, 7], (1, J))
    max_y_mat = np.tile(tfparams[:, 8], (1, J))
    min_y_mat = np.tile(tfparams[:, 9], (1, J))
    Phi = (phi * (max_x_mat - min_x_mat) +
          (1 - phi) * (max_y_mat - min_y_mat))
    P_num = (A_mat * inc_lab **2 + B_mat * inc_cap ** 2 +
        C_mat * inc_lab * inc_cap + D_mat * inc_lab + E_mat * inc_cap)
    P_den = (A_mat * inc_lab **2 + B_mat * inc_cap ** 2 +
        C_mat * inc_lab * inc_cap + D_mat * inc_lab + E_mat * inc_cap +
        F_mat)
    MinConst = phi * min_x_mat + (1 - phi) * min_y_mat
    EffTxRt = Phi * (P_num / P_den) + MinConst
    TaxLiab = EffTxRt * (inc_lab_m + inc_cap_m)
    NetTxLiab = TaxLiab - T_H

    return NetTxLiab


def euler_labor(params, r, w, T_H, factor, n_js, b_js, meth):
    '''
    --------------------------------------------------------------------
    This function generates Euler errors (either percent deviation or
    difference) for labor_leisure Euler equation.
    --------------------------------------------------------------------
    params      = length ? tuple, parameters and objects
    ???
    r           = scalar > 0, real return on capital
    w           = scalar > 0, real wage
    T_H         = scalar, lump sum transfer to each household
    factor      = scalar > 0, scaling factor to make average model
                  income equal average data income in tax function
    n_js        = (N x S) matrix, labor supply decisions over an
                  individual's lifetime (s) for each ability type (j)
    b_js        = (N x S) matrix, savings decisions over an individual's
                  lifetime (s) for each ability type (j)
    meth        = string, "pct" = percent deviation, "dif" = difference
    Tmat        = (S x J) matrix, net tax liability for each age (s)
                  individual and ability type (j)
    ???
    BQ          =
    c_js        = (S x J) matrix, consumption over individual's lifetime
    MU_c_n      = (S x J) matrix, derivative of utility of consumption
                  with respect to labor
    MU_dis_n    = (S x J) matrix, derivative of disutility of labor with
                  respect to labor
    eul_err_mat = (S x J) matrix, labor supply euler errors for each age
                  (s) individual of ability type (j)
    eul_err_lab = (S*J,) vector, euler errors from labor equations

    This function calls the following functions:
        get_tax
        get_dtax_n
        get_BQ
        get_cons
        get_MU_c_n
        get_MU_dis_n

    returns: eul_err_lab
    --------------------------------------------------------------------
    '''
    S, J, emat, tfparams = params
    inc_lab = w * emat * n_js
    inc_cap = r * b_js
    Tmat = get_NetTaxLiab(tfparams, inc_lab, inc_cap, factor, T_H)
    DtaxDn = get_dtax_n()
    BQ = get_BQ()
    c_js = get_cons()
    MU_c_n = get_MU_c_n()
    MU_dis_n = get_MU_dis_n()
    if meth == "pct":
        eul_err_lab = (MU_n / MU_c_n) - 1
    elif meth == "dif":
        eul_err_lab = MU_n - MU_c_n
    eul_err_lab = eul_err_mat.reshape(S*J)

    return eul_err_lab
