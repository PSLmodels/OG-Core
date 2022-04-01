import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    longdesc = fh.read()

setuptools.setup(
    name="ogcore",
    version="0.8.1",
    author="Jason DeBacker and Richard W. Evans",
    license="CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    description="A general equilibribum overlapping generations model for fiscal policy analysis",
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
            "data/ability/*",
            "data/demographic/*",
            "data/labor/*",
            "data/wealth/*"
        ]
    },
    include_packages=True,
    python_requires=">=3.7.7",
    install_requires=[
        "mkl>=2021.4.0",
        "psutil",
        "scipy>=1.7.1",
        "pandas>=1.2.5",
        "matplotlib",
        "dask>=2.30.0",
        "distributed>=2.30.1",
        "paramtools>=0.15.0",
        "requests"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    tests_require=["pytest"]
)
