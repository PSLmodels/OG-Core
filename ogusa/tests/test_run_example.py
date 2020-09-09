#### import modules (same as run_ogusa_example)
import multiprocessing
import time
import os, sys

# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)
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
