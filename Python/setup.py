try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


config = {
    'description': 'dynamic scoring model using Overlapping Generations model for the USA',
    'url': 'https://github.com/open-source-economics/OG-USA/',
    'download_url': 'https://github.com/open-source-economics/OG-USA/',
    'description':'ogusa',
    'install_requires': ['scipy', 'numpy'],
    'version': '0.1',
    'license': 'MIT',
    'packages': ['ogusa'],
    'include_package_data': True,
    'name': 'ogusa',
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
