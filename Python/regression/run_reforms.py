#!/usr/bin/env python
from __future__ import print_function
from multiprocessing import Process
import argparse
import json
import os
import sys
import time
import uuid
sys.path.append("../")

import pandas as pd

import ogusa
from ogusa import SS
from ogusa import TPI
import postprocess
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

def cli():
    parser = argparse.ArgumentParser(description='Run reforms baseline or difference result sets')
    parser.add_argument('reform', help='Reform name such as "reform0"')
    parser.add_argument('--against-taxcalc', default='0.6.6', help="Tax-Calculator version as basis for differencing")
    parser.add_argument('--against-ogusa', default='0.5.5', help='OG-USA version as basis for differencing')
    parser.add_argument('--diff', action='store_true', help='Run difference')
    args = parser.parse_args()
    args.folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'standards',
                               'tc{}_og{}'.format(args.against_taxcalc,
                                                  args.against_ogusa))
    if args.diff:
        args.standard = os.path.join(args.folder, 'results_data_{}.csv'.format(args.reform))
        if not os.path.exists(args.folder):
            raise ValueError('Cannot diff against Tax-Calculator {} '
                             'and OG-USA {} because {} does not '
                             'exist'.format(args.against_taxcalc,
                                            args.against_ogusa,
                                            args.standard))
    return args


def main():
    args = cli()
    with open("reforms.json", "r") as f:
        reforms = json.loads(f.read())

    reform_num = args.reform
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

    if args.diff:
        df_released = pd.read_csv(args.standard, index_col=0)
        df_diff = df_released - df
        # Dump the diff data
        fname = "diff_{0}_tc{1}_og{2}_to_master.csv".format(args.reform, args.against_taxcalc, args.against_ogusa)
        df_diff.to_csv(fname)

    print("END")


if __name__ == "__main__":
    main()