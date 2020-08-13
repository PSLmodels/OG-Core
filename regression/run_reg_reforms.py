import os
import multiprocessing
from dask.distributed import Client
import time
import numpy as np
import pandas as pd
import argparse
from ogusa.execute import runner
import ogusa.output_tables as ot
from ogusa.utils import REFORM_DIR, BASELINE_DIR, safe_read_pickle

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
PUF_PATH = os.path.join(CUR_PATH, '..' , 'ogusa', 'puf.csv')

client = Client(processes=False)
CPU_COUNT = 4  # multiprocessing.cpu_count()
REF_IDXS = [0, 1, 2, 3, 4, 5, 6, 7]

DATA = pd.read_csv(PUF_PATH)
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
        '_STD': [[6100*2, 12200*2, 6100*2, 8950*2, 12200*2],
                 [6200*2, 12400*2, 6200*2, 9100*2, 12400*2],
                 [6300*2, 12600*2, 6300*2, 9250*2, 12600*2]]
    }, }

reform7 = {
    2016: {
        '_STD': [[6100*2.1, 12200*2.1, 6100*2.1, 8950*2.1, 12200*2.1],
                 [6200*2.1, 12400*2.1, 6200*2.1, 9100*2.1, 12400*2.1],
                 [6300*2.1, 12600*2.1, 6300*2.1, 9250*2.1, 12600*2.1]]
    }, }

reform8 = {
    2016: {
        '_II_rt3': [.15],
        '_II_rt4': [.15],
        '_II_rt5': [.15],
        '_II_brk5': [[250000, 250000, 125000, 250000, 250000]]
    }, }


reform9 = {
    2016: {
            '_STD': [[12600, 25200, 12600, 18600, 25300]],
            '_II_brk1': [[27825, 55650, 27825, 39750, 55650]],
            '_II_brk2': [[65005, 130010, 65005, 88180, 130010]],
            '_AMT_rt1': [.0],
            '_AMT_rt2': [.0]
    }, }

reforms = [reform0, reform1, reform2, reform3, reform4, reform5, reform6,
           reform7]


def run_micro_macro(user_params, reform=None, baseline_dir=BASELINE_DIR,
                    reform_dir=REFORM_DIR, guid='', data=PUF_PATH,
                    ok_to_run_baseline=True):

    start_time = time.time()

    T_shifts = np.zeros(50)
    T_shifts[2:10] = 0.01
    T_shifts[10:40] = -0.01
    G_shifts = np.zeros(6)
    G_shifts[0:3] = -0.01
    G_shifts[3:6] = -0.005
    user_params = {'frisch': 0.41, 'start_year': 2017, 'tau_b': 0.20,
                   'debt_ratio_ss': 1.0, 'delta_tau_annual': 0.05,
                   'T_shifts': T_shifts.tolist(),
                   'G_shifts': G_shifts.tolist()}

    '''
    ------------------------------------------------------------------------
        Run baseline
    ------------------------------------------------------------------------
    '''
    print('path exists', not os.path.exists(baseline_dir), ok_to_run_baseline)
    # if not os.path.exists(baseline_dir) and ok_to_run_baseline:
    if ok_to_run_baseline:
        output_base = baseline_dir
        kwargs = {'output_base': baseline_dir,
                  'baseline_dir': baseline_dir, 'test': False,
                  'time_path': True, 'baseline': True,
                  'analytical_mtrs': False, 'age_specific': True,
                  'user_params': user_params, 'guid': '',
                  'run_micro': False, 'small_open': False,
                  'budget_balance': False, 'baseline_spending': False,
                  'data': data, 'client': client, 'num_workers': 4}
        runner(**kwargs)

    '''
    ------------------------------------------------------------------------
        Run reform
    ------------------------------------------------------------------------
    '''
    output_base = reform_dir
    kwargs = {'output_base': output_base, 'baseline_dir': baseline_dir,
              'test': False, 'time_path': True, 'baseline': False,
              'analytical_mtrs': False, 'age_specific': True,
              'user_params': user_params, 'guid': guid,
              'reform': reform, 'run_micro': False, 'small_open': False,
              'budget_balance': False, 'baseline_spending': False,
              'data': data, 'client': client, 'num_workers': 4}
    runner(**kwargs)

    base_tpi = safe_read_pickle(
        os.path.join(baseline_dir, 'TPI', 'TPI_vars.pkl'))
    base_params = safe_read_pickle(
        os.path.join(baseline_dir, 'model_params.pkl'))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, 'TPI', 'TPI_vars.pkl'))
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, 'model_params.pkl'))
    ans = ot.macro_table(
        base_tpi, base_params, reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=['Y', 'C', 'K', 'L', 'r', 'w'], output_type='pct_diff',
        num_years=10, start_year=2017)

    print("total time was ", (time.time() - start_time))
    print('Percentage changes in aggregates:', ans)


def run_reforms(ref_idxs=REF_IDXS, path_prefix="", cpu_count=CPU_COUNT,
                data=DATA):
    # make sure we have a baseline result before other reforms are run
    ok_to_run_baseline = True
    run_micro_macro({},
                    reforms[0],
                    "./{0}OUTPUT_BASELINE".format(path_prefix),
                    "./{0}OUTPUT_REFORM_{1}".format(path_prefix, 0),
                    str(0),
                    data,
                    ok_to_run_baseline,)
    # run reforms in parallel
    pool = multiprocessing.pool.ThreadPool(processes=2)
    results = []
    ok_to_run_baseline = False
    for i in range(1, len(reforms)):
        args = ({},
                reforms[i],
                "./{0}OUTPUT_BASELINE".format(path_prefix),
                "./{0}OUTPUT_REFORM_{1}".format(path_prefix, i),
                str(i),
                data,
                ok_to_run_baseline,)
        async_result = pool.apply_async(run_micro_macro, args)
        results.append(async_result)
    for result in results:
        result.get()
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_idxs", default=REF_IDXS,
                        help="List of reform indices to run")
    parser.add_argument("--set_reg_data", action="store_true",
                        help=("If used, we are creating regression data and "
                              "results are written to './REG_*'"))
    parser.add_argument("--cpu_count", default=CPU_COUNT, type=int,
                        help="Number CPUs to use")
    args = parser.parse_args()

    ref_idxs = args.ref_idxs
    if ref_idxs != REF_IDXS:
        ref_idxs = [int(idx) for idx in ref_idxs.split(',')]
    path_prefix = "regression_results/REG_" if args.set_reg_data else ""
    cpu_count = args.cpu_count

    run_reforms(ref_idxs=ref_idxs,
                path_prefix=path_prefix,
                cpu_count=cpu_count)
