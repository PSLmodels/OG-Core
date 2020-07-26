try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as f:
    longdesc = f.read()

version = '0.6.3'

config = {
    'description': 'General equilibribum, overlapping generations model for the USA',
    'long_description': longdesc,
    'url': 'https://github.com/PSLmodels/OG-USA/',
    'download_url': 'https://github.com/PLSmodels/OG-USA/',
    'version': version,
    'license': 'CC0 1.0 Universal public domain dedication',
    'packages': ['ogusa'],
    'include_package_data': True,
    'name': 'ogusa',
    'install_requires': [],
    'package_data': {
                     'ogusa': [
                               'parameters_metadata.json',
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
        'Topic :: Software Development :: Libraries :: Python Modules'],
    'tests_require': ['pytest']
}

setup(**config)
