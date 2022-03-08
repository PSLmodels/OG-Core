try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as f:
    longdesc = f.read()

version = '0.8.0'

config = {
    'description': 'A general equilibribum overlapping generations model for fiscal policy analysis',
    'long_description_content_type': 'text/markdown',
    'long_description': longdesc,
    'url': 'https://github.com/PSLmodels/OG-Core/',
    'download_url': 'https://github.com/PLSmodels/OG-Core/',
    'version': version,
    'license': 'CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
    'packages': ['ogcore'],
    'include_package_data': True,
    'name': 'ogcore',
    'install_requires': [
        'mkl', 'psutil', 'scipy', 'pandas', 'matplotlib', 'dask',
        'distributed', 'paramtools'],
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
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
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
