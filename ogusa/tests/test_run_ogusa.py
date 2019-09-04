import pytest
from ogusa import SS, TPI, postprocess
import time
from ogusa.execute import runner
SS.ENFORCE_SOLUTION_CHECKS = False
TPI.ENFORCE_SOLUTION_CHECKS = False


def run_micro_macro(iit_reform, og_spec, guid):

    guid = ''
    start_time = time.time()

    REFORM_DIR = "./OUTPUT_REFORM_" + guid
    BASELINE_DIR = "./OUTPUT_BASELINE_" + guid

    # Add start year from reform to user parameters
    start_year = sorted(iit_reform.keys())[0]
    og_spec['start_year'] = start_year

    with open("log_{}.log".format(guid), 'w') as f:
        f.write("guid: {}\n".format(guid))
        f.write("iit_reform: {}\n".format(iit_reform))
        f.write("og_spec: {}\n".format(og_spec))

    '''
    ------------------------------------------------------------------------
        Run baseline
    ------------------------------------------------------------------------
    '''
    output_base = BASELINE_DIR
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': True, 'time_path': True, 'baseline': True,
              'og_spec': og_spec, 'run_micro': False,
              'guid': guid}
    runner(**kwargs)

    '''
    ------------------------------------------------------------------------
        Run reform
    ------------------------------------------------------------------------
    '''

    output_base = REFORM_DIR
    kwargs = {'output_base': output_base, 'baseline_dir': BASELINE_DIR,
              'test': True, 'time_path': True, 'baseline': False,
              'iit_reform': iit_reform, 'og_spec': og_spec,
              'guid': guid, 'run_micro': False}
    runner(**kwargs)
    time.sleep(0.5)
    ans = postprocess.create_diff(baseline_dir=BASELINE_DIR,
                                  policy_dir=REFORM_DIR)
    print("total time was ", (time.time() - start_time))

    return ans


@pytest.mark.full_run
def test_run_micro_macro():

    iit_reform = {
        2018: {
            '_II_rt1': [.09],
            '_II_rt2': [.135],
            '_II_rt3': [.225],
            '_II_rt4': [.252],
            '_II_rt5': [.297],
            '_II_rt6': [.315],
            '_II_rt7': [0.3564],
            }, }
    run_micro_macro(iit_reform=iit_reform, og_spec={
        'frisch': 0.44, 'g_y_annual': 0.021}, guid='abc')
