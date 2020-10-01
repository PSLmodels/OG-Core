'''
This test tests whether starting a `run_ogusa_example.py` run of the model does
not break down (is still running) after 5 minutes or 300 seconds.
'''

import multiprocessing
import time
import os, sys
import importlib.util
from pathlib import Path


def call_run_ogusa_example():
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    path = Path(cur_path)
    roe_fldr = os.path.join(path.parent.parent, "run_examples")
    roe_file_path = os.path.join(roe_fldr, "run_ogusa_example.py")
    spec = importlib.util.spec_from_file_location('run_ogusa_example.py',
                                                   roe_file_path)
    roe_module = importlib.util.module_from_spec(spec)
    sys.modules['run_ogusa_example.py'] = roe_module
    spec.loader.exec_module(roe_module)
    roe_module.main()


def test_run_ogusa_example(f = call_run_ogusa_example):
    '''
    test that run_ogusa_example runs for at least 5 minutes
    '''
    p = multiprocessing.Process(target = f,
                                name="run_ogusa_example", args=())
    p.start()

    for i in range(60):
        time.sleep(5)
        print("Still here!")

    if p.is_alive():
        p.terminate()
        p.join()
        timetest = True
    else:
        print("run_ogusa_example did not run for minimum time")
        timetest = False
    print('timetest ==', timetest)

    assert timetest == True
