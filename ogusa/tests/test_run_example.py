'''
This test checks whether starting the run_ogusa_example.py script run without
shutting down for 5 minutes (300 seconds)
'''
# Import packages
import multiprocessing
import time
import importlib
import os
from pathlib import Path

# Import run_ogusa_example.py, which is not part of the ogusa package
OG_USA_path = Path(__file__).parents[2]
run_examples_path = os.path.join(OG_USA_path, 'run_examples')
print('Test directory is :', run_examples_path)
module_path = os.path.join(run_examples_path, 'run_ogusa_example.py')
run_ogusa_example = importlib.import_module(module_path)


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
