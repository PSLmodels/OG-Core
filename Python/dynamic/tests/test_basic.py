import os
import sys
CUR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_PATH, "../../"))

def test_import_ok():
    import dynamic

def test_run_small():
    from run_small import runner
    runner()
