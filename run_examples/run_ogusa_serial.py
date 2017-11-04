import ogusa
import os
import sys
from multiprocessing import Process
import time
import numpy as np

#OGUSA_PATH = os.environ.get("OGUSA_PATH", "../../ospc-dynamic/dynamic")

#sys.path.append(OGUSA_PATH)

from ogusa.scripts import postprocess
from ogusa.scripts.execute import runner
from ogusa.utils import REFORM_DIR, BASELINE_DIR

def run_micro_macro(user_params):

    # reform = {
    # 2015: {
    #     '_II_rt1': [.09],
    #     '_II_rt2': [.135],
    #     '_II_rt3': [.225],
    #     '_II_rt4': [.252],
    #     '_II_rt5': [.297],
    #     '_II_rt6': [.315],
    #     '_II_rt7': [0.3564],
    # }, }

    # reform = {
    # 2015: {
    #     '_II_rt1': [0.045]
    # }, }

    reform = {
    2017: {
       '_II_rt5': [.3],
       '_II_rt6': [.3],
       '_II_rt7': [0.3]
    } }


    start_time = time.time()

    T_shifts = np.zeros(50)
    T_shifts[2:10] = 0.01
    T_shifts[10:40]= -0.01
    G_shifts = np.zeros(6)
    G_shifts[0:3]  = -0.01
    G_shifts[3:6]  = -0.005
    user_params = {'frisch':0.41, 'start_year':2017, 'tau_b':0.20,
                   'debt_ratio_ss':1.0, 'T_shifts':T_shifts,
                   'G_shifts':G_shifts}

    '''
    ------------------------------------------------------------------------
        Run SS for Baseline first - so can run baseline and reform in parallel if want
    ------------------------------------------------------------------------
    '''
    output_base = BASELINE_DIR
    input_dir = BASELINE_DIR
    kwargs={'output_base':output_base, 'baseline_dir':BASELINE_DIR,
           'test':False, 'time_path':False, 'baseline':True, 'analytical_mtrs':False, 'age_specific':True,
           'user_params':user_params,'guid':'',
           'run_micro':False, 'small_open':False, 'budget_balance':False, 'baseline_spending':False}
    #p1 = Process(target=runner, kwargs=kwargs)
    #p1.start()
    runner(**kwargs)
    quit()


    '''
    ------------------------------------------------------------------------
        Run baseline
    ------------------------------------------------------------------------
    '''


    output_base = BASELINE_DIR
    input_dir = BASELINE_DIR
    kwargs={'output_base':output_base, 'baseline_dir':BASELINE_DIR,
            'test':False, 'time_path':True, 'baseline':True,
            'analytical_mtrs':False, 'age_specific':True,
            'user_params':user_params,'guid':'',
            'run_micro':False, 'small_open': False, 'budget_balance':False, 'baseline_spending':False}
    #p1 = Process(target=runner, kwargs=kwargs)
    #p1.start()
    runner(**kwargs)


    '''
    ------------------------------------------------------------------------
        Run reform
    ------------------------------------------------------------------------
    '''
    output_base = REFORM_DIR
    input_dir = REFORM_DIR
    guid_iter = 'reform_' + str(0)
    kwargs={'output_base':output_base, 'baseline_dir':BASELINE_DIR,
            'test':True, 'time_path':True, 'baseline':False,
            'analytical_mtrs':False, 'age_specific':True,
            'user_params':user_params,'guid':'', 'reform':reform ,
            'run_micro':False, 'small_open': False, 'budget_balance':False,
            'baseline_spending':False}
    #p2 = Process(target=runner, kwargs=kwargs)
    #p2.start()
    runner(**kwargs)







    #p1.join()
    # print "just joined"
    #p2.join()

    # time.sleep(0.5)

    ans = postprocess.create_diff(baseline_dir=BASELINE_DIR, policy_dir=REFORM_DIR)

    print "total time was ", (time.time() - start_time)
    print 'Percentage changes in aggregates:', ans

    # return ans

if __name__ == "__main__":
    run_micro_macro(user_params={})
