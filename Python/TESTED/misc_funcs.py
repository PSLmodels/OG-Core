'''
------------------------------------------------------------------------
Last updated 6/3/2015

Miscellaneous functions for taxes in SS and TPI.

------------------------------------------------------------------------
'''

# Packages
import numpy as np

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''


def perc_dif_func(simul, data):
    '''
    Used to calculate the absolute percent difference between the data and
        simulated data
    '''
    frac = (simul - data)/data
    output = np.abs(frac)
    return output
