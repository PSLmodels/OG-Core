import ogusa
import os
import sys
from multiprocessing import Process
import time

#OGUSA_PATH = os.environ.get("OGUSA_PATH", "../../ospc-dynamic/dynamic/Python")

#sys.path.append(OGUSA_PATH)

import postprocess
#from execute import runner # change here for small jobs
from execute_large import runner


def run_micro_macro(user_params):

    reform = {
    2015: {
        '_II_rt1': [.09],
        '_II_rt2': [.135],
        '_II_rt3': [.225],
        '_II_rt4': [.252],
        '_II_rt5': [.297],
        '_II_rt6': [.315],
        '_II_rt7': [0.3564],
    }, }

    start_time = time.time()

    REFORM_DIR = "./OUTPUT_REFORM4"

    output_base = REFORM_DIR
    input_dir = REFORM_DIR

    kwargs={'output_base':output_base, 'input_dir':input_dir,
            'baseline':False, 'reform':reform, 'user_params':user_params,
            'guid':'abc123', 'run_micro':False}
    runner(**kwargs)

    time.sleep(0.5)

    print "total time for policy run was ", (time.time() - start_time)


if __name__ == "__main__":
    run_micro_macro(user_params={})
