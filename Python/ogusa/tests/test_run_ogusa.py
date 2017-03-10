import os
import sys
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_PATH, "../../"))
import pytest
import tempfile
import pickle
import numpy as np
import pandas as pd
from ogusa.utils import comp_array
from ogusa.utils import comp_scalar
from ogusa.utils import dict_compare
from ogusa.get_micro_data import get_calculator
from ogusa import SS
from ogusa import TPI
import uuid
import time

import postprocess
from execute import runner
SS.ENFORCE_SOLUTION_CHECKS = False
TPI.ENFORCE_SOLUTION_CHECKS = False

def run_micro_macro(reform, user_params, guid):

    guid = ''
    start_time = time.time()

    REFORM_DIR = "./OUTPUT_REFORM_" + guid
    BASELINE_DIR = "./OUTPUT_BASELINE_" + guid

    # Add start year from reform to user parameters
    start_year = sorted(reform.keys())[0]
    user_params['start_year'] = start_year

    with open("log_{}.log".format(guid), 'w') as f:
        f.write("guid: {}\n".format(guid))
        f.write("reform: {}\n".format(reform))
        f.write("user_params: {}\n".format(user_params))


    '''
    ------------------------------------------------------------------------
        Run baseline
    ------------------------------------------------------------------------
    '''
    output_base = BASELINE_DIR
    kwargs={'output_base':output_base, 'baseline_dir':BASELINE_DIR, 'test':True,
            'time_path':True, 'baseline':True, 'analytical_mtrs':False,
            'user_params':user_params, 'age_specific':False, 'run_micro':False,
            'guid':guid}
    runner(**kwargs)

    '''
    ------------------------------------------------------------------------
        Run reform
    ------------------------------------------------------------------------
    '''

    output_base = REFORM_DIR
    kwargs={'output_base':output_base, 'baseline_dir':BASELINE_DIR,
            'test':True, 'time_path':True, 'baseline':False,
            'analytical_mtrs':False, 'reform':reform, 'user_params':user_params,
            'age_specific':False, 'guid': guid, 'run_micro':False}
    runner(**kwargs)

    time.sleep(0.5)
    ans = postprocess.create_diff(baseline_dir=BASELINE_DIR, policy_dir=REFORM_DIR)
    print "total time was ", (time.time() - start_time)

    return ans

def test_run_micro_macro():

    reform = {
    2017: {
        '_II_rt1': [.09],
        '_II_rt2': [.135],
        '_II_rt3': [.225],
        '_II_rt4': [.252],
        '_II_rt5': [.297],
        '_II_rt6': [.315],
        '_II_rt7': [0.3564],
    }, }
    run_micro_macro(reform=reform, user_params={'frisch': 0.44, 'g_y_annual': 0.021}, guid='abc')
