import ogusa
import os
import sys
from multiprocessing import Process
import time

#OGUSA_PATH = os.environ.get("OGUSA_PATH", "../../ospc-dynamic/dynamic")

#sys.path.append(OGUSA_PATH)

from ogusa.scripts import postprocess
#from execute import runner # change here for small jobs
from ogusa.scripts.execute_large import runner


def run_micro_macro(user_params):

    start_time = time.time()

    BASELINE_DIR = "./OUTPUT_BASELINE4"

    output_base = BASELINE_DIR
    input_dir = BASELINE_DIR

    kwargs={'output_base':output_base, 'input_dir':input_dir, 'baseline':True, 'user_params':user_params,
            'guid':'abc123', 'run_micro':False}
    runner(**kwargs)
    time.sleep(0.5)

    print "total time for baseline was ", (time.time() - start_time)


if __name__ == "__main__":
    run_micro_macro(user_params={})
