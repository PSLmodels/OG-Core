try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as f:
        longdesc = f.read()

version = '0.0.0'


config = {
    'description': 'dynamic scoring model using Overlapping Generations model for the USA',
    'url': 'https://github.com/open-source-economics/OG-USA/',
    'download_url': 'https://github.com/open-source-economics/OG-USA/',
    'install_requires': ['numpy', 'pandas', 'taxcalc'],
    'license': 'MIT',
    'packages': ['ogusa'],
    'package_data': {
                     'ogusa': [
                               'parameters_metadata.json',
                               'data/ability/*',
                               'data/demographic/*',
                               'data/labor/*',
                               'data/wealth/*']
                     },
    'include_package_data': True,
    'name': 'ogusa',
    'version': version,
    'classifiers': [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules'],
    'tests_require': ['pytest']
}

setup(**config)
