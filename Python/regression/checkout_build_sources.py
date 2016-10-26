from __future__ import print_function
import os
import shutil
import subprocess as sp

import matplotlib
import yaml

def run_cmd(args, cwd='.', raise_err=True):
    if isinstance(args, str):
        args = args.split()
    proc =  sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT, cwd=cwd)
    lines = []
    while proc.poll() is None:
        line = proc.stdout.readline().decode()
        print(line, end='')
        lines.append(line)
    new_lines = proc.stdout.readlines()
    print(''.join(new_lines))
    if proc.poll() and raise_err:
        raise ValueError("Subprocess failed {}".format(proc.poll()))
    return lines

def get_ogusa_git_branch():
    return [line for line in run_cmd('git branch')
            if line.strip() and '*' == line.strip()[0]][0]

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
REGRESSION_CONFIG['install_ogusa_version'] = get_ogusa_git_branch()

OGUSA_ENV_PATH = os.path.join(os.environ['WORKSPACE'], 'ogusa_env')


def checkout_build_sources():
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
    conda_path = line.strip().split()[-1].strip()
    run_cmd('{} install --force -c ospc openblas pytest toolz scipy numpy={} pandas=0.18.1 matplotlib'.format(conda_path, numpy_vers))
    run_cmd('{} remove mkl mkl-service'.format(conda_path), raise_err=False)
    run_cmd('{} install -c ospc taxcalc={} --force'.format(conda_path, install_taxcalc_version))
    if ogusainstallmethod == 'conda':
        run_cmd('{} install -c ospc ogusa={}'.format(conda_path, install_ogusa_version))
    run_cmd('git clone https://github.com/open-source-economics/OG-USA OG-USA')
    cwd = os.path.join(os.path.dirname(__file__), 'OG-USA')
    run_cmd('git fetch --all', cwd=cwd)
    run_cmd('git checkout regression', cwd=cwd)
    regression_tmp = os.path.join(cwd, '..', 'regression')
    if os.path.exists(regression_tmp):
        shutil.rmtree(regression_tmp)
    shutil.copytree(os.path.join(cwd, 'Python', 'regression'), regression_tmp)
    run_cmd('git checkout {}'.format(install_ogusa_version), cwd=cwd)
    if ogusainstallmethod == 'git':
        run_cmd('python setup.py install', cwd=cwd)
    puf_choices = (os.path.join(cwd, '..', '..', 'puf.csv'),
                   os.path.join('Python', 'regression', 'puf.csv'),
                   os.path.join('/home', 'ubuntu', 'deploy', 'puf.csv'))

    for puf in puf_choices:
        if os.path.exists(puf):
            print('puf from', puf)
        shutil.copy(puf, os.path.join('Python', 'regression', 'puf.csv'))
    mfile = matplotlib.matplotlib_fname()
    with open(mfile, 'w') as f:
        f.write('backend    : Agg')
    cwd = os.path.join(cwd, 'Python', 'regression')
    print("CHECKOUT_BUILD_SOURCES OK")
    return cwd


if __name__ == "__main__":
    checkout_build_sources()