try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as f:
    longdesc = f.read()

version = '0.7.0'

config = {
    'description': 'General equilibribum, overlapping generations model for the USA',
    'long_description': longdesc,
    'url': 'https://github.com/PSLmodels/OG-Core/',
    'download_url': 'https://github.com/PLSmodels/OG-Core/',
    'version': version,
    'license': 'CC0 1.0 Universal public domain dedication',
    'packages': ['ogcore'],
    'include_package_data': True,
    'name': 'ogcore',
    'install_requires': [
        'mkl', 'psutil', 'scipy', 'pandas', 'matplotlib', 'dask',
        'dask-core', 'distributed', 'paramtools'],
    'package_data': {
                     'ogcore': [
                               'default_parameters.json',
                               'data/ability/*',
                               'data/demographic/*',
                               'data/labor/*',
                               'data/wealth/*']
                     },
    'classifiers': [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: CC0 1.0 Universal public domain dedication',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules'],
    'tests_require': ['pytest']
}

setup(**config)
