# bash commands for installing your package

git clone https://github.com/PSLmodels/OG-USA
cd OG-USA
conda install scipy mkl dask PSLmodels::taxcalc conda-forge::paramtools
pip install -e .
