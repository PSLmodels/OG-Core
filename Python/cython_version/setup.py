from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'SS solver',
  ext_modules = cythonize("SS_cy.pyx"),
)