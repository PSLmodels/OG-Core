import multiprocessing
import time
import os, sys
from pathlib import Path
# import modules from run_ogusa_example.py
import multiprocessing
from distributed import Client
import time
import numpy as np
import os, sys
import taxcalc
from taxcalc import Calculator
from ogusa import output_tables as ot
from ogusa import output_plots as op
from ogusa.execute import runner
from ogusa.constants import REFORM_DIR, BASELINE_DIR
from ogusa.utils import safe_read_pickle

def call_run_ogusa_example():
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    path = Path(cur_path)
    run_example_dir = os.path.join(path.parent.parent, "run_examples")
    run_example_file = os.path.join(run_example_dir, "run_ogusa_example.py")
    print('run_example_file: ', run_example_file)
    exec(open(run_example_file).read())


def test_run_ogusa_example(f = call_run_ogusa_example):
    '''
    test that run_ogusa_example runs for at least 5 minutes
    '''
    p = multiprocessing.Process(target = f, 
                                name="run_ogusa_example", args=())
    p.start()
    time.sleep(30)
    if p.is_alive():
        p.terminate() 
        p.join()
        timetest = True
    else:
        print("run_ogusa_example did not run for minimum time")
        timetest = False
    print('timetest ==', timetest)
    
    assert timetest == True

