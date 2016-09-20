taxcalc_vers=$1
ogusa_vers=$2
numpy_vers=$3
install_env(){
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $WORKSPACE/miniconda
    export PATH="$WORKSPACE/miniconda/bin:$PATH"
    conda config --set always_yes yes --set changeps1 no
    conda update conda
    conda create -n ogusa_env -c ospc taxcalc=$taxcalc_vers ogusa=$ogusa_vers numba python=2.7 pytest nomkl scipy numpy=$numpy_vers pandas=0.18.1 matplotlib
    source activate ogusa_env
    export mfile=$(python -c "import matplotlib;print matplotlib.matplotlib_fname()")
    echo "backend    : Agg" > ${mfile}
}

install_env && python ./run_reforms.py $4 $5 $6 $7 $8 $9 $10 # there may not be that many args passed (placeholders)
