import ogusa
import os
import sys
import time
import numpy as np
import pandas as pd

from ogusa.scripts import postprocess
from ogusa.scripts.execute import runner
from ogusa.utils import REFORM_DIR, BASELINE_DIR

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
PUF_PATH = os.path.join(CUR_PATH, '../ogusa/puf.csv')

def run_micro_macro(user_params, reform=None, baseline_dir=BASELINE_DIR,
                    reform_dir=REFORM_DIR, guid='', data=PUF_PATH):

    start_time = time.time()

    T_shifts = np.zeros(50)
    T_shifts[2:10] = 0.01
    T_shifts[10:40] = -0.01
    G_shifts = np.zeros(6)
    G_shifts[0:3] = -0.01
    G_shifts[3:6] = -0.005
    small_open = dict(world_int_rate=0.04)  # Alternatively
                                            # small_open can be False/None
                                            # (if False/None then 0.04 is used)
    user_params = {'frisch': 0.41, 'start_year': 2017, 'tau_b': 0.20,
                   'debt_ratio_ss': 1.0, 'T_shifts': T_shifts,
                   'G_shifts': G_shifts, 'small_open': small_open}

    '''
    ------------------------------------------------------------------------
        Run SS for Baseline first - so can run baseline and reform in parallel if want
    ------------------------------------------------------------------------
    '''
    output_base = BASELINE_DIR
    input_dir = BASELINE_DIR
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': False, 'time_path': True, 'baseline': True,
              'analytical_mtrs': False, 'age_specific': True,
              'user_params': user_params, 'guid': '', 'run_micro': False,
              'small_open': small_open, 'budget_balance': False,
              'baseline_spending': False}
    runner(**kwargs)


    '''
    ------------------------------------------------------------------------
        Run baseline
    ------------------------------------------------------------------------
    '''

    if not os.path.exists(baseline_dir):
        output_base = baseline_dir
        input_dir = baseline_dir
        kwargs = {'output_base': output_base, 'baseline_dir': baseline_dir,
                  'test': False, 'time_path': True, 'baseline': True,
                  'analytical_mtrs': False, 'age_specific': True,
                  'user_params': user_params, 'guid': 'baseline',
                  'run_micro': True, 'small_open': small_open,
                  'budget_balance': False, 'baseline_spending': False,
                  'data': data}
        runner(**kwargs)


    '''
    ------------------------------------------------------------------------
        Run reform
    ------------------------------------------------------------------------
    '''
    output_base = reform_dir
    input_dir = reform_dir
    guid_iter = 'reform_' + str(0)
    user_params = {'frisch': 0.41, 'start_year': 2017, 'tau_b': 0.20,
                   'debt_ratio_ss': 1.0, 'T_shifts': T_shifts,
                   'G_shifts': G_shifts, 'small_open': small_open}
    kwargs = {'output_base': output_base, 'baseline_dir': baseline_dir,
              'test': False, 'time_path': True, 'baseline': False,
              'analytical_mtrs': False, 'age_specific': True,
              'user_params': user_params, 'guid': guid, 'reform': reform,
              'run_micro': True, 'small_open': small_open,
              'budget_balance': False, 'baseline_spending': False,
              'data': data}
    runner(**kwargs)

    ans = postprocess.create_diff(baseline_dir=baseline_dir,
                                  policy_dir=reform_dir)

    print "total time was ", (time.time() - start_time)
    print 'Percentage changes in aggregates:', ans

    # return ans

if __name__ == "__main__":
    data = pd.read_csv(PUF_PATH)
    reform0 = {
        2016: {
            '_II_rt1': [.09],
            '_II_rt2': [.135],
            '_II_rt3': [.225],
            '_II_rt4': [.252],
            '_II_rt5': [.297],
            '_II_rt6': [.315],
            '_II_rt7': [0.3564],
        },
    }
    reform1 = {
        2016: {
            '_II_rt7': [0.35],
        },
    }
    reform2 = {
    2016: {
        '_II_rt7': [0.34],
    }, }

    reform3 = {
    2016: {
        '_CG_rt3': [0.25],
    }, }

    reform4 = {
    2016: {
        '_CG_rt3': [0.24],
    }, }

    reform5 = {
    2016: {
        '_CG_rt3': [0.16],
    }, }

    reform6 = {
    2016: {
        '_STD': [ [6100*2, 12200*2, 6100*2, 8950*2, 12200*2, 6100*2, 1000*2],
                    [6200*2, 12400*2, 6200*2, 9100*2, 12400*2, 6200*2, 1000*2],
                    [6300*2, 12600*2, 6300*2, 9250*2, 12600*2, 6300*2, 1050*2]],
    }, }

    reform7 = {
    2016: {
        '_STD': [ [6100*2.1, 12200*2.1, 6100*2.1, 8950*2.1, 12200*2.1, 6100*2.1, 1000*2.1],
                    [6200*2.1, 12400*2.1, 6200*2.1, 9100*2.1, 12400*2.1, 6200*2.1, 1000*2.1],
                    [6300*2.1, 12600*2.1, 6300*2.1, 9250*2.1, 12600*2.1, 6300*2.1, 1050*2.1]],
    }, }

    reform8 = {
    2016: {
        '_II_rt3': [.15],
        '_II_rt4': [.15],
        '_II_rt5': [.15],
        '_II_brk5':[[250000, 250000, 125000, 250000, 250000, 250000]]
    }, }


    reform9 = {
    2016: {
            '_STD': [[12600, 25200, 12600, 18600, 25300, 12600, 2100]],
            '_II_brk1': [[27825, 55650, 27825, 39750, 55650, 27825]],
            '_II_brk2': [[65005, 130010, 65005, 88180, 130010, 65005]],
            '_AMT_trt1': [.0],
           '_AMT_trt2': [.0]
    },}

    reforms = [reform0, reform1, reform2, reform3, reform4, reform5, reform6,
               reform7, reform8, reform9]

    for i in range(len(reforms)):
        run_micro_macro({},
                        reforms[i],
                        "./OUTPUT_BASELINE",
                        "./OUTPUT_REFORM_" + str(i),
                        str(i),
                        data)
