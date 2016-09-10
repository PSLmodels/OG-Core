#!/usr/bin/env python
from __future__ import print_function
import ogusa
import os
import sys
sys.path.append("../")
from multiprocessing import Process
import time
import json
import uuid
from ogusa import SS
from ogusa import TPI

import postprocess
import pandas as pd
from execute import runner # change here for small jobs
#from execute_small import runner

VERSION = "0.5.5"
QUICK_RUN = False


def run_micro_macro(reform, user_params, guid, solution_checks, run_micro):

    # Turn off checks for now
    SS.ENFORCE_SOLUTION_CHECKS = solution_checks
    TPI.ENFORCE_SOLUTION_CHECKS = solution_checks

    start_time = time.time()

    REFORM_DIR = "./OUTPUT_REFORM" + guid
    BASELINE_DIR = "./OUTPUT_BASELINE" + guid

    # Add start year from reform to user parameters
    start_year = sorted(reform.keys())[0]
    user_params['start_year'] = start_year

    input_dir = BASELINE_DIR

    kwargs={'output_base':BASELINE_DIR, 'baseline_dir':BASELINE_DIR,
            'baseline':True, 'analytical_mtrs':False, 'age_specific':False,
            'user_params':user_params, 'guid':guid, 'run_micro':run_micro}

    #p1 = Process(target=runner, kwargs=kwargs)
    #p1.start()
    runner(**kwargs)

    kwargs={'output_base':REFORM_DIR, 'baseline_dir':BASELINE_DIR, 
             'baseline':False, 'analytical_mtrs':False, 'user_params':user_params,
             'reform':reform, 'age_specific':False, 'guid':guid,'run_micro':run_micro}

    #p2 = Process(target=runner, kwargs=kwargs)
    #p2.start()
    runner(**kwargs)

    #p1.join()
    #print("just joined")
    #p2.join()

    #time.sleep(0.5)

    ans = postprocess.create_diff(baseline_dir=BASELINE_DIR, policy_dir=REFORM_DIR)

    print("total time was ", (time.time() - start_time))
    print(ans)

    return ans

if __name__ == "__main__":
    with open("reforms.json", "r") as f:
        reforms = json.loads(f.read())

    reform_num = sys.argv[1]
    # Run the given reform
    if QUICK_RUN:
        guid = ''
        solution_checks = False
        run_micro = False
    else:
        guid = uuid.uuid1().hex
        solution_checks = True
        run_micro = True

    reform = {int(k):v for k,v in reforms[reform_num].items()}

    ans = run_micro_macro(reform=reform, user_params={}, guid=guid,
                          solution_checks=solution_checks,
                          run_micro=run_micro)
    as_percent = ans * 100

    # Dump a "pretty print" version of the answer provided to the web app
    cols = list(map(str, range(2016, 2026))) + ["2016-2025"] + ["Steady State"]
    rows = ["GDP", "Consumption", "Investment", "Hours Worked", "Wages",
            "Interest Rates", "Total Taxes"]
    df = pd.DataFrame(data=ans, columns=cols, index=rows)
    pd.options.display.float_format = '{:12,.3f}'.format
    with open("results_pprint_{}.txt".format(reform_num), 'w') as h:
        h.write(df.__repr__())

    # Dump the actual data
    df.to_csv("results_data_{}.csv".format(reform_num))

    if len(sys.argv) > 2 and sys.argv[2] == "diff":
        regression_data = "results_data_{0}_{1}.csv".format(VERSION, reform_num)
        df_released = pd.read_csv(regression_data, index_col=0)
        df_diff = df_released - df
        # Dump the diff data
        df_diff.to_csv("diff_{0}_v{1}_to_master.csv".format(reform_num, VERSION))

    print("END")


