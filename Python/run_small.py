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

#txfunc.get_tax_func_estimate(baseline=False)

from execute import runner

if __name__ == "__main__":
    output_base = "./OUTPUT"
    input_dir = "./OUTPUT"

    reform = {
                2015: {
                    '_II_rt1': [.09],
                    '_II_rt2': [.135],
                    '_II_rt3': [.225],
                    '_II_rt4': [.252],
                    '_II_rt5': [.297],
                    '_II_rt6': [.315],
                    '_II_rt7': [0.3564],
                    },
              }

    runner(output_base=output_base, input_dir=input_dir, baseline=False, reform=reform, run_micro=False)
