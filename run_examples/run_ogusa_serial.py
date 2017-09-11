import ogusa
import os
import sys
from multiprocessing import Process, Pool
import time
import numpy as np

#OGUSA_PATH = os.environ.get("OGUSA_PATH", "../../ospc-dynamic/dynamic")

#sys.path.append(OGUSA_PATH)

from ogusa.scripts import postprocess
from ogusa.scripts.execute import runner
from ogusa.utils import REFORM_DIR, BASELINE_DIR

CPU_COUNT = 4

def run_micro_macro(user_params, reform=None, baseline_dir=BASELINE_DIR, reform_dir=REFORM_DIR):

    # reform = {
    # 2015: {
    #     '_II_rt1': [0.045]
    # }, }

    # reform = {
    # 2017: {
    #    '_II_rt5': [.3],
    #    '_II_rt6': [.3],
    #    '_II_rt7': [0.3]
    # } }


    start_time = time.time()

    T_shifts = np.zeros(50)
    T_shifts[2:10] = 0.01
    T_shifts[10:40]= -0.01
    G_shifts = np.zeros(6)
    G_shifts[0:3]  = -0.01
    G_shifts[3:6]  = -0.005
    user_params = {'frisch':0.41, 'start_year':2017, 'debt_ratio_ss':1.0, 'T_shifts':T_shifts, 'G_shifts':G_shifts}

    '''
    ------------------------------------------------------------------------
        Run SS for Baseline first - so can run baseline and reform in parallel if want
    ------------------------------------------------------------------------
    '''
    # output_base = BASELINE_DIR
    # input_dir = BASELINE_DIR
    # kwargs={'output_base':output_base, 'baseline_dir':BASELINE_DIR,
    #        'test':False, 'time_path':False, 'baseline':True, 'analytical_mtrs':False, 'age_specific':True,
    #        'user_params':user_params,'guid':'',
    #        'run_micro':False, 'small_open':False, 'budget_balance':False, 'baseline_spending':False}
    # #p1 = Process(target=runner, kwargs=kwargs)
    # #p1.start()
    # runner(**kwargs)
    # # quit()


    '''
    ------------------------------------------------------------------------
        Run baseline
    ------------------------------------------------------------------------
    '''


    output_base = baseline_dir
    input_dir = baseline_dir
    kwargs={'output_base':output_base, 'baseline_dir':baseline_dir,
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
    output_base = reform_dir
    input_dir = reform_dir
    guid_iter = 'reform_' + str(0)
    kwargs={'output_base':output_base, 'baseline_dir':baseline_dir,
            'test':False, 'time_path':True, 'baseline':False,
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

    ans = postprocess.create_diff(baseline_dir=baseline_dir, policy_dir=reform_dir)

    print "total time was ", (time.time() - start_time)
    print 'Percentage changes in aggregates:', ans

    # return ans

if __name__ == "__main__":
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

    pool = Pool(processes=CPU_COUNT)
    results = []
    for i in range(len(reforms)):
        args = ({},
                reforms[i],
                "./OUTPUT_BASELINE_" + str(i),
                "./OUTPUT_REFORM_" + str(i), )

        async_result = pool.apply_async(run_micro_macro, args)
        results.append(async_result)

    for result in results:
        result.get()

    pool.close()
    pool.join()


    # run_micro_macro(user_params={},
    #                 reform=reform0,
    #                 baseline_dir="./OUTPUT_BASELINE_0",
    #                 reform_dir="./OUTPUT_REFORM_0")
    # run_micro_macro(user_params={},
    #                 reform=reform1,
    #                 baseline_dir="./OUTPUT_BASELINE_1",
    #                 reform_dir="./OUTPUT_REFORM_1")
    # run_micro_macro(user_params={},
    #                 reform=reform2,
    #                 baseline_dir="./OUTPUT_BASELINE_2",
    #                 reform_dir="./OUTPUT_REFORM_2")
    # run_micro_macro(user_params={},
    #                 reform=reform3,
    #                 baseline_dir="./OUTPUT_BASELINE_3",
    #                 reform_dir="./OUTPUT_REFORM_3")
    # run_micro_macro(user_params={},
    #                 reform=reform4,
    #                 baseline_dir="./OUTPUT_BASELINE_4",
    #                 reform_dir="./OUTPUT_REFORM_4")
    # run_micro_macro(user_params={},
    #                 reform=reform5,
    #                 baseline_dir="./OUTPUT_BASELINE_5",
    #                 reform_dir="./OUTPUT_REFORM_5")
    # run_micro_macro(user_params={},
    #                 reform=reform6,
    #                 baseline_dir="./OUTPUT_BASELINE_6",
    #                 reform_dir="./OUTPUT_REFORM_6")
    # run_micro_macro(user_params={},
    #                 reform=reform7,
    #                 baseline_dir="./OUTPUT_BASELINE_7",
    #                 reform_dir="./OUTPUT_REFORM_7")
    # run_micro_macro(user_params={},
    #                 reform=reform8,
    #                 baseline_dir="./OUTPUT_BASELINE_8",
    #                 reform_dir="./OUTPUT_REFORM_8")
    # run_micro_macro(user_params={},
    #                 reform=reform9,
    #                 baseline_dir="./OUTPUT_BASELINE_9",
    #                 reform_dir="./OUTPUT_REFORM_9")
