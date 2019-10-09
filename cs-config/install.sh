# bash commands for installing your package

# point at my branch to test my changes
git clone https://github.com/hdoupe/OG-USA
cd OG-USA
git fetch origin
git checkout cs-tweaks
git fetch origin
git merge origin/cs-tweaks

# Explicitly add channels for looking up dependencies outside of
# taxcalc and paramtools. If the channels are not specified like this,
# the tests fail due to not being able to converge on a solution.
conda config --add channels PSLmodels
conda config --add channels conda-forge
conda install scipy mkl dask matplotlib PSLmodels::taxcalc conda-forge::paramtools
pip install -e .
