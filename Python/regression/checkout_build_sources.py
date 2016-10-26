from __future__ import print_function
import argparse
import os
import shutil
import subprocess as sp
import sys

import yaml

def run_cmd(args, cwd='.', raise_err=True):
    if isinstance(args, str):
        args = args.split()
    print("RUN CMD", args, file=sys.stderr)
    proc =  sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT, cwd=cwd)
    lines = []
    while proc.poll() is None:
        line = proc.stdout.readline()
        print(line, end='', file=sys.stderr)
        lines.append(line)
    new_lines = proc.stdout.readlines()
    print(''.join(new_lines), file=sys.stderr)
    if proc.poll() and raise_err:
        raise ValueError("Subprocess failed {}".format(proc.poll()))
    return lines

_d = os.path.dirname
REGRESSION_CONFIG = os.path.join(_d(_d(_d(os.path.abspath(__file__)))), '.regression.yml')
REGRESSION_CONFIG = yaml.load(open(REGRESSION_CONFIG))
REQUIRED = set(('compare_taxcalc_version',
                'compare_ogusa_version',
                'install_taxcalc_version',
                'diff',
                'numpy_version'))
if not set(REGRESSION_CONFIG) >= REQUIRED:
    raise ValueError('.regression.yml at top level of repo needs to define: '.format(REQUIRED - set(REGRESSION_CONFIG)))

OGUSA_ENV_PATH = os.path.join(os.environ['WORKSPACE'], 'ogusa_env')


def checkout_build_sources():
    parser = argparse.ArgumentParser(description='Get install OG-USA branch')
    parser.add_argument('ogusabranch')
    numpy_vers = REGRESSION_CONFIG['numpy_version']
    install_ogusa_version = parser.parse_args().ogusabranch
    install_taxcalc_version = REGRESSION_CONFIG['install_taxcalc_version']
    compare_ogusa_version = REGRESSION_CONFIG['compare_ogusa_version']
    compare_taxcalc_version = REGRESSION_CONFIG['compare_taxcalc_version']
    print('CHECKOUT_BUILD_SOURCES')
    run_cmd('wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh')
    miniconda_path = os.path.join(os.environ['WORKSPACE'], 'miniconda')
    run_cmd('bash miniconda.sh -b -p {}'.format(miniconda_path))
    run_cmd('conda config --set always_yes yes --set changeps1 no')
    run_cmd('conda update conda -n root')
    lines = ' '.join(run_cmd('conda env list')).lower()
    if 'ogusa_env' in lines:
        run_cmd('conda env remove --name ogusa_env')
    run_cmd('conda install nomkl')
    run_cmd('conda create --force python=2.7 --name ogusa_env')
    line = [line for line in run_cmd('conda env list')
            if 'ogusa_env' in line][0]
    conda_path = os.path.join(line.strip().split()[-1].strip(), 'bin', 'conda')
    print('Using conda {}'.format(conda_path))
    run_cmd('{} install --force -c ospc openblas pytest toolz scipy numpy={} pandas=0.18.1 matplotlib'.format(conda_path, numpy_vers))
    run_cmd('{} remove mkl mkl-service'.format(conda_path), raise_err=False)
    run_cmd('{} install -c ospc taxcalc={} --force'.format(conda_path, install_taxcalc_version))
    run_cmd('git fetch --all', cwd=cwd)
    run_cmd('git checkout regression', cwd=cwd)
    regression_tmp = os.path.join(cwd, '..', 'regression')
    if os.path.exists(regression_tmp):
        shutil.rmtree(regression_tmp)
    shutil.copytree(os.path.join(cwd, 'Python', 'regression'), regression_tmp)
    run_cmd('git checkout {}'.format(install_ogusa_version), cwd=cwd)
    run_cmd('python setup.py install', cwd=cwd)
    puf_choices = (os.path.join(cwd, '..', '..', 'puf.csv'),
                   os.path.join('Python', 'regression', 'puf.csv'),
                   os.path.join('/home', 'ubuntu', 'deploy', 'puf.csv'))

    for puf in puf_choices:
        if os.path.exists(puf):
            print('puf from', puf)
        shutil.copy(puf, os.path.join('Python', 'regression', 'puf.csv'))
    print("CHECKOUT_BUILD_SOURCES OK")
    return cwd


if __name__ == "__main__":
    checkout_build_sources()