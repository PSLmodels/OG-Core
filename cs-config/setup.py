"""
setup.py is used to build a light-weight python package around your
Compute Studio code. Add a MANIFEST.in file to specify data files that should
be included with this package. Read more here:
https://docs.python.org/3.8/distutils/sourcedist.html#specifying-the-files-to-distribute
"""

import setuptools
import os

setuptools.setup(
    name="cs-config",
    description="Compute Studio configuration files.",
    url="https://github.com/compute-tooling/compute-studio-kit",
    packages=setuptools.find_packages(),
    include_package_data=True,
)
