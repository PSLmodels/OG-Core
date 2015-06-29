try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


config = {
    'description': 'dynamic scoring model',
    'url': 'https://github.com/OpenSourcePolicyCenter/dynamic',
    'download_url': 'https://github.com/OpenSourcePolicyCenter/dynamic',
    'description':'dynamic',
    'install_requires': ["scipy", "numpy"],
    'version': '0.1',
    'license': 'MIT',
    'packages': ['dynamic'],
    'include_package_data': True,
    'name': 'dynamic',
    'classifiers': [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries :: Python Modules'],
    'tests_require': ['pytest']
}

setup(**config)
