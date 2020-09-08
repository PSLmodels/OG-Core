#### import modules (same as run_ogusa_example)
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

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from run_examples import run_ogusa_example


def call_run_ogusa_example():
    run_ogusa_example.main()


def test_run_ogusa_example(f = call_run_ogusa_example):
    '''
    test that run_ogusa_example runs for at least 5 minutes
    '''
    p = multiprocessing.Process(target = f, 
                                name="run_ogusa_example", args=())
    p.start()
    time.sleep(300)
    if p.is_alive():
        p.terminate() 
        p.join()
        timetest = True
    else:
        print("run_ogusa_example did not run for minimum time")
        timetest = False
    print('timetest ==', timetest)
    
    assert timetest == True

