'''
------------------------------------------------------------------------
Last updated 4/7/2016

This script takes a Frisch elasticity parameter from the OSPC's Tax Brain 
and then estimates the parameters of the elliptical utility fuction that
correspond to a constant Frisch elasticity function with a Frisch elasticity
as input into Tax Brain. 

This Python script calls the following functions:

This Python script outputs the following:

------------------------------------------------------------------------
'''
# Import packages
import time
import numpy as np
import numpy.random as rnd
import scipy.optimize as opt
try:
    import cPickle as pickle
except:
    import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



def sumsq(params, *objs):
    '''
    --------------------------------------------------------------------
    This function generates the sum of squared deviations between the 
    constant Frisch elasticity function and the elliptical utility function
    --------------------------------------------------------------------
    Inputs:
        params  = length 3 tuple, (b, k, upsilon)
        b       = scalar, scaling parameter for elliptical utility function
        k       = scalar, shift parameter for elliptical utility function
        upsilon = scalar, curvature parameter for elliptical utility function
        objs    = lenght 3 tuple, (theta, l_tilde, n_grid)
        theta   = scalar, inverse of the Frisch elasticity
        l_tilde = scalar, max labor supply
        n_grid  = [N,] vector, grid of labor supply values

    Functions called: None

    Objects in function:
        CFE      = [N,] vector, constant Frisch elasticity function evaluated at points on n_grid
        ellipse = [N,] vector, elliptical utility function evaluated at points on n_grid
        error    = [N,] vector, differences between CFE and ellipse on each point in n_grid
        ssqdev   = [N,] vector, sum of squared errors between two utility functions at each point in n_grid

    Returns: ssqdev
    --------------------------------------------------------------------
    '''
    theta, l_tilde, n_grid = objs
    b, k, upsilon = params
    CFE = ((n_grid/l_tilde)**(1+theta))/(1+theta)
    ellipse = b*((1-((n_grid/l_tilde)**upsilon))**(1/upsilon)) + k
    errors = CFE - ellipse
    ssqdev = (errors ** 2).sum()
    #ssqdev = (np.abs(errors)).sum()
    return ssqdev

def sumsq_MU(params, *objs):
    '''
    --------------------------------------------------------------------
    This function generates the sum of squared deviations between the marginals of the
    constant Frisch elasticity function and the elliptical utility function
    --------------------------------------------------------------------
    Inputs:
        params  = length 3 tuple, (b, k, upsilon)
        b       = scalar, scaling parameter for elliptical utility function
        k       = scalar, shift parameter for elliptical utility function
        upsilon = scalar, curvature parameter for elliptical utility function
        objs    = lenght 3 tuple, (theta, l_tilde, n_grid)
        theta   = scalar, inverse of the Frisch elasticity
        l_tilde = scalar, max labor supply
        n_grid  = [N,] vector, grid of labor supply values

    Functions called: None

    Objects in function:
        CFE_MU     = [N,] vector, marginal utility from the constant Frisch elasticity function evaluated at points on n_grid
        ellipse_MU = [N,] vector, marginal utility from the elliptical utility function evaluated at points on n_grid
        error      = [N,] vector, differences between the MU from the CFE and ellipse on each point in n_grid
        ssqdev     = [N,] vector, sum of squared errors between marginals of the two utility functions at each point in n_grid

    Returns: ssqdev
    --------------------------------------------------------------------
    '''
    theta, l_tilde, n_grid = objs
    b, upsilon = params
    CFE_MU = (1.0/l_tilde)*((n_grid/l_tilde)**theta)
    ellipse_MU =  b * (1.0 / l_tilde) * ((1.0 - (n_grid / l_tilde) ** upsilon) ** ((1.0 / upsilon) - 1.0)) * (n_grid / l_tilde) ** (upsilon - 1.0)
    errors = CFE_MU - ellipse_MU
    ssqdev = (errors ** 2).sum()
    #ssqdev = (np.abs(errors)).sum()
    return ssqdev



def estimation(frisch, l_tilde):
    '''
    --------------------------------------------------------------------
    This function estimates the parameters of an elliptical utility 
    funcion that fits a constant frisch elasticty function

    Inputs:
        frisch  = scalar, Frisch elasticity of labor supply
        l_tilde = scalar, maximum amount of labor supply

    Functions called: 
        sumsq_MU
        sumsq

    Objects in function:
        theta               = inverse of the Frisch elasticity of labor supply
        l_tilde             = max labor supply 
        N                   = number of grid points used in function estimation
        graph               = boolean, =True if print graph with CFE and elliptical functions
        start_time          = scalar, current processor time in seconds (float)
        elapsed_time        = scalar, processor time for estimation
        b                   = scalar >0, vertical radius of ellipitcal utility function
        k                   = scalar, centroid of elliptical utility function
        upsilon             = scalar > 0, curvature parameter in ellipitcal utility function
        b_init              = scalar > 0, initial guess at b for minimizer
        k_init              = scalar, initial guess at k for minimizer
        upsilon_init        = scalar > 0, initial guess at upsilon for minimizer
        ellipse_params_init = length 3 tuple, initial guesses for parameter values
        n_grid              = [N,] vector, grid of labor supply values over which function evaluated
        ellipse_obj         = length 3 tuple, objects passed to minimizer function
        bnds                = length 3 tuple, bounds for parameters of elliptical utility 
        ellipse_params_til  = tuple with full output of minimizer
        b_til               = scalar > 0, estimate of b
        k_til               = scalar, estimate of k
        upsilon_til         = scalar > 0, estimate of upsilon

    Returns: b_MU_til, upsilon_MU_til

    --------------------------------------------------------------------
    '''

    '''
    ------------------------------------------------------------------------
    Set parameters
    ------------------------------------------------------------------------
    '''

    theta = 1/frisch
    N = 101
    graph = False
    # start_time = time.clock()


    '''
    ------------------------------------------------------------------------
    Estimate parameters of ellipitical utility function
    ------------------------------------------------------------------------
    '''

    # Initial guesses
    b_init = .6701
    k_init = -.6548
    upsilon_init = 2.3499
    #n_grid = np.linspace(0.01, l_tilde, num=N)
    n_grid = np.linspace(0.01, 0.8, num=N) # don't estimate near edge of range of labor supply

    #Estimating using levels of utility function
    # ellipse_params_init = np.array([b_init, k_init, upsilon_init])
    # ellipse_objs = (theta, l_tilde, n_grid)
    # bnds = ((None, None), (None, None), (1e-12, None))
    # ellipse_params_til = opt.minimize(sumsq, ellipse_params_init,
    #                     args=(ellipse_objs), method="L-BFGS-B", bounds=bnds,
    #                     tol=1e-15)
    # (b_til, k_til, upsilon_til) = ellipse_params_til.x

    # elapsed_time = time.clock() - start_time

    ### Estimate params using marginal utilities
    ellipse_MU_params_init = np.array([b_init, upsilon_init])
    ellipse_MU_objs = (theta, l_tilde, n_grid)
    bnds_MU = ((None, None), (None, None))
    ellipse_MU_params_til = opt.minimize(sumsq_MU, ellipse_MU_params_init,
                        args=(ellipse_MU_objs), method="L-BFGS-B", bounds=bnds_MU,
                        tol=1e-15)
    (b_MU_til, upsilon_MU_til) = ellipse_MU_params_til.x

    # Print tax function computation time
    # if elapsed_time < 60: # seconds
    #     secs = round(elapsed_time, 3)
    #     print 'Elliptical utility estimation time: ', secs, ' sec.'
    # elif elapsed_time >= 60 and elapsed_time < 3600: # minutes
    #     mins = int(elapsed_time / 60)
    #     secs = round(((elapsed_time / 60) - mins) * 60, 1)
    #     print 'Elliptical utility estimation time: ', mins, ' min, ', secs, ' sec'

    #print 'Ellipse parameters; b, k, upsilon: ', b_til, k_til, upsilon_til, ellipse_params_til
    #print 'Ellipse MU parameters; b, upsilon: ', b_MU_til, upsilon_MU_til, ellipse_MU_params_til


    if graph == True:
        '''
        ------------------------------------------------------------------------
        Plot CFE vs Elliptical Function
        ------------------------------------------------------------------------
        '''
        CFE = ((n_grid/l_tilde)**(1+theta))/(1+theta)
        ellipse_til = b_til*((1-((n_grid/l_tilde)**upsilon_til))**(1/upsilon_til)) + k_til
        fig, ax = plt.subplots()
        plt.plot(n_grid, CFE, 'r--', label='CFE')
        plt.plot(n_grid, ellipse_til, 'b', label='Elliptical U')
        # for the minor ticks, use no labels; default NullFormatter
        # ax.xaxis.set_minor_locator(MinorLocator)
        # plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.legend(loc='center right')
        plt.title('Constant Frisch Elasticity vs. Elliptical Utility')
        plt.xlabel(r'Labor Supply')
        plt.ylabel(r'Utility')
        # plt.savefig('cm_ss_Chap11')
        plt.show()



        CFE_MU = (1.0/l_tilde)*((n_grid/l_tilde)**theta)
        ellipse_MU = (1.0*b_MU_til * (1.0 / l_tilde) * ((1.0 - (n_grid / l_tilde) ** upsilon_MU_til) 
                     ** ((1.0 / upsilon_MU_til) - 1.0)) * (n_grid / l_tilde) ** (upsilon_MU_til - 1.0))
        fig, ax = plt.subplots()
        plt.plot(n_grid, CFE_MU, 'r--', label='CFE')
        plt.plot(n_grid, ellipse_MU, 'b', label='Elliptical U')
        # for the minor ticks, use no labels; default NullFormatter
        # ax.xaxis.set_minor_locator(MinorLocator)
        # plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.legend(loc='center right')
        plt.title('Marginal Utility of CFE and Elliptical')
        plt.xlabel(r'Labor Supply')
        plt.ylabel(r'Utility')
        # plt.savefig('cm_ss_Chap11')
        plt.show()


    return b_MU_til, upsilon_MU_til
