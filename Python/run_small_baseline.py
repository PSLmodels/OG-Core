'''
A 'smoke test' for the ogusa package. Uses a fake data set to run the
baseline
'''

import cPickle as pickle
import os
import numpy as np
import time

import ogusa
ogusa.parameters.DATASET = 'SMALL'

import ogusa.SS
import ogusa.TPI
from ogusa import parameters, wealth, labor, demographics, income, SS, TPI

from ogusa import txfunc

#txfunc.get_tax_func_estimate(baseline=True)

from execute import runner


if __name__ == "__main__":
    output_base = "./OUTPUT_BASELINE"
    input_dir = "./OUTPUT_BASELINE"
    runner(output_base=output_base, input_dir=input_dir, baseline=True)
