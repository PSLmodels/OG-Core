import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    longdesc = fh.read()

setuptools.setup(
    name="ogcore",
    version="0.14.2",
    author="Jason DeBacker and Richard W. Evans",
    license="CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    description="A general equilibrium overlapping generations model for fiscal policy analysis",
    long_description_content_type="text/markdown",
    long_description=longdesc,
    url="https://github.com/PSLmodels/OG-Core/",
    download_url="https://github.com/PLSmodels/OG-Core/",
    project_urls={
        "Issue Tracker": "https://github.com/PSLmodels/OG-Core/issues",
    },
    packages=["ogcore"],
    package_data={
        "ogcore": [
            "default_parameters.json",
            "model_variables.json",
            "OGcorePlots.mplstyle",
        ]
    },
    include_packages=True,
    python_requires=">=3.7.7, <3.13",
    install_requires=[
        "numpy",
        "scipy>=1.7.1",
        "pandas>=1.2.5",
        "numba",
        "matplotlib",
        "dask>=2.30.0",
        "distributed>=2.30.1",
        "paramtools>=0.15.0",
        "requests",
        "pip",
        "pygam",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    tests_require=["pytest"],
)
