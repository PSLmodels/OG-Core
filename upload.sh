#!/bin/sh
cd $OGUSA_PACKAGE_DIR
anaconda upload ~/miniconda3/conda-bld/osx-64/ogusa-$1-py27_0.tar.bz2
anaconda upload ./linux-32/ogusa-$1-py27_0.tar.bz2
anaconda upload ./linux-64/ogusa-$1-py27_0.tar.bz2
anaconda upload ./win-64/ogusa-$1-py27_0.tar.bz2
anaconda upload ./win-32/ogusa-$1-py27_0.tar.bz2
echo "FINISHED package upload"
