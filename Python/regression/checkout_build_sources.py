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

def cli():
    parser = argparse.ArgumentParser(description='Get install OG-USA branch')
    parser.add_argument('action', choices=['make_ogusa_env', 'customize_ogusa_env'])
    parser.add_argument('ogusabranch')
    return parser.parse_args()


def make_ogusa_env(args):
    numpy_vers = REGRESSION_CONFIG['numpy_version']
    install_ogusa_version = args.ogusabranch
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
    run_cmd('conda create --name ogusa_env --force python=2.7 yaml')
    line = [line for line in run_cmd('conda env list')
            if 'ogusa_env' in line][0]
    conda_path = os.path.join(line.strip().split()[-1].strip(), 'bin', 'conda')
    print('Using conda {}'.format(conda_path))

def customize_ogusa_env(args):

    numpy_vers = REGRESSION_CONFIG['numpy_version']
    install_ogusa_version = args.ogusabranch
    install_taxcalc_version = REGRESSION_CONFIG['install_taxcalc_version']
    compare_ogusa_version = REGRESSION_CONFIG['compare_ogusa_version']
    compare_taxcalc_version = REGRESSION_CONFIG['compare_taxcalc_version']
    run_cmd('conda install --force -c ospc openblas pytest toolz scipy numpy={} pandas=0.18.1 matplotlib'.format(numpy_vers))
    run_cmd('conda remove mkl mkl-service', raise_err=False)
    run_cmd('conda install -c ospc taxcalc={} --force'.format(install_taxcalc_version))
    run_cmd('git fetch --all')
    run_cmd('git checkout regression')
    regression_tmp = os.path.join('..', 'regression')
    if os.path.exists(regression_tmp):
        shutil.rmtree(regression_tmp)
    src = os.path.join('Python', 'regression')
    shutil.copytree(src, regression_tmp)
    run_cmd('git checkout {}'.format(install_ogusa_version))
    if not os.path.exists(src):
        shutil.copytree(regression_tmp, src)
    run_cmd('python setup.py install')
    puf_choices = (os.path.join('..', '..', 'puf.csv'),
                   os.path.join('Python', 'regression', 'puf.csv'),
                   os.path.join('/home', 'ubuntu', 'deploy', 'puf.csv'))

    puf = [puf for puf in puf_choices if os.path.exists(puf)]
    if not puf:
        raise ValueError('Could not find puf.csv at {}'.format(puf_choices))
    puf = puf[0]
    shutil.copy(puf, os.path.join('Python', 'regression', 'puf.csv'))
    print("CHECKOUT_BUILD_SOURCES OK")
    return 0


if __name__ == "__main__":
    args = cli()
    if args.action == 'make_ogusa_env':
        make_ogusa_env(args)
    else:
        customize_ogusa_env(args)