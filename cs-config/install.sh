# bash commands for installing your package

git clone -b master --depth 1 https://github.com/PSLmodels/OG-USA
cd OG-USA

# Explicitly add channels for looking up dependencies outside of
# taxcalc and paramtools. If the channels are not specified like this,
# the tests fail due to not being able to converge on a solution.
conda config --add channels PSLmodels
conda config --add channels conda-forge
conda install scipy mkl dask matplotlib PSLmodels::taxcalc conda-forge::paramtools
pip install -e .
