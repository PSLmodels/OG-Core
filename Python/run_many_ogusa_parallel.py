import ogusa
import os
import sys
from multiprocessing import Process
import time

#OGUSA_PATH = os.environ.get("OGUSA_PATH", "../../ospc-dynamic/dynamic/Python")

#sys.path.append(OGUSA_PATH)

import postprocess
#from execute import runner # change here for small jobs
from execute_large import runner, runner_SS


def run_micro_macro(user_params):

    reform0 = {
    2016: {
        '_II_rt1': [.09],
        '_II_rt2': [.135],
        '_II_rt3': [.225],
        '_II_rt4': [.252],
        '_II_rt5': [.297],
        '_II_rt6': [.315],
        '_II_rt7': [0.3564],
    }, }

    reform1 = {
    2016: {
        '_II_rt7': [0.35],
    }, }

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


    start_time = time.time()

    BASELINE_DIR = "./OUTPUT_BASELINE"
    output_base = BASELINE_DIR
    input_dir = BASELINE_DIR

    user_params = {'frisch':0.41, 'start_year':2016}

    '''
    ------------------------------------------------------------------------
        Run SS for Baseline first - so can run baseline and reform in parallel if want 
    ------------------------------------------------------------------------
    '''
    output_base = BASELINE_DIR
    input_dir = BASELINE_DIR
    kwargs={'output_base':output_base, 'baseline_dir':BASELINE_DIR,
            'baseline':True, 'analytical_mtrs':False, 'age_specific':False,
            'user_params':user_params,'guid':'_base_flattax',
            'run_micro':True}
    runner_SS(**kwargs)


    '''
    ------------------------------------------------------------------------
        Run baseline
    ------------------------------------------------------------------------
    '''
    output_base = BASELINE_DIR
    input_dir = BASELINE_DIR
    kwargs={'output_base':output_base, 'baseline_dir':BASELINE_DIR,
            'baseline':True, 'analytical_mtrs':False, 'age_specific':False,
            'user_params':user_params,'guid':'_base_flattax',
            'run_micro':False}
    runner(**kwargs)


    '''
    ------------------------------------------------------------------------
        Run reforms
    ------------------------------------------------------------------------
    '''
    reforms = (reform0, reform1, reform2, reform3, reform4, reform5, reform6, reform7, reform8, reform9)

    counter = 0 
    for x in reforms:
        print 'Running reform ', counter

        REFORM_DIR = './OUTPUT_REFORM/' + str(counter) + '/'

        reform = x 
        guid_iter = 'reform_' + str(counter)

        output_base = REFORM_DIR
        input_dir = REFORM_DIR
        guid_iter = 'reform_' + str(counter)
        kwargs={'output_base':output_base, 'baseline_dir':BASELINE_DIR,
            'baseline':False, 'analytical_mtrs':False, 'age_specific':False, 
            'reform':reform, 'user_params':user_params,'guid':guid_iter, 'run_micro':True}
        runner(**kwargs)

        counter = counter + 1

    print "total time was ", (time.time() - start_time)


if __name__ == "__main__":
    run_micro_macro(user_params={})
