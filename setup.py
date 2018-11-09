try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import versioneer


config = {
    'description': 'dynamic scoring model using Overlapping Generations model for the USA',
    'url': 'https://github.com/open-source-economics/OG-USA/',
    'download_url': 'https://github.com/open-source-economics/OG-USA/',
    'description': 'ogusa',
    'install_requires': [],
    'license': 'MIT',
    'packages': ['ogusa'],
    'package_data': {
                     'ogusa': [
                               '../TxFuncEst_baseline.pkl',
                               '../TxFuncEst_policy.pkl',
                               'parameters_metadata.json',
                               'data/ability/*',
                               'data/demographic/*',
                               'data/labor/*',
                               'data/wealth/*']
                     },
    'include_package_data': True,
    'name': 'ogusa',
    'version': versioneer.get_version(),
    'cmdclass': versioneer.get_cmdclass(),
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
