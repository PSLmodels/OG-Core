"""
This test tests whether starting a `run_ogcore_example.py` run of the model does
not break down (is still running) after 5 minutes or 300 seconds.
"""

import multiprocessing
import time
import os
import sys
import pandas as pd
import importlib.util
import shutil
from pathlib import Path
import pytest


def call_run_ogcore_example():
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    path = Path(cur_path)
    roe_fldr = os.path.join(path.parent, "run_examples")
    roe_file_path = os.path.join(roe_fldr, "run_ogcore_example.py")
    spec = importlib.util.spec_from_file_location(
        "run_ogcore_example.py", roe_file_path
    )
    roe_module = importlib.util.module_from_spec(spec)
    sys.modules["run_ogcore_example.py"] = roe_module
    spec.loader.exec_module(roe_module)
    roe_module.main()


@pytest.mark.local
def test_run_ogcore_example(f=call_run_ogcore_example):
    p = multiprocessing.Process(target=f, name="run_ogcore_example", args=())
    p.start()
    time.sleep(300)
    if p.is_alive():
        p.terminate()
        p.join()
        timetest = True
    else:
        print("run_ogcore_example did not run for minimum time")
        timetest = False
    print("timetest ==", timetest)
    # Delete directory created by run_ogcore_example.py
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    path = Path(cur_path)
    roe_output_dir = os.path.join(
        path.parent, "run_examples", "OG-Core-Example", "OUTPUT_BASELINE"
    )
    shutil.rmtree(roe_output_dir)

    assert timetest


@pytest.mark.local
def test_run_ogcore_example_output(f=call_run_ogcore_example):
    p = multiprocessing.Process(target=f, name="run_ogcore_example", args=())
    p.start()
    p.join()  # this makes sure process finished running before going on
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    path = Path(cur_path)
    expected_df = pd.read_csv(
        os.path.join(
            path.parent, "run_examples", "expected_ogcore_example_output.csv"
        )
    )
    # read in output from this run
    test_df = pd.read_csv(
        os.path.join(
            path.parent,
            "run_examples",
            "OG-Core-Example",
            "OG-Core_example_output.csv",
        )
    )
    # Delete directory created by run_ogcore_example.py
    roe_output_dir = os.path.join(
        path.parent, "run_examples", "OG-Core-Example", "OUTPUT_BASELINE"
    )
    shutil.rmtree(roe_output_dir)

    pd.testing.assert_frame_equal(expected_df, test_df)
