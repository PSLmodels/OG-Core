#!/bin/sh
# Run this at the root of the repo!
source activate ospcdyn
git tag $1
echo "finished tag, now distributing to PyPI"
python setup.py register sdist upload
echo "FINISHED UPLOAD TO PYPI"
git push; git push --tags
echo "FINISHED push "
rm ~/code/og.tar
echo "FINISHED rm tart "
rm -rf ~/code/ogusa
echo "FINISHED rm dir "
git archive --prefix=ogusa/ -o ~/code/og.tar $1
echo "FINISHED git archive"
cd ~/code/
tar xvf og.tar
echo "FINISHED tar extract"
cd ogusa/conda.recipe
sed -i '' 's/version: 0.1/version: '${1}'/g' meta.yaml
echo "FINISHED changing meta.yaml"
conda build --python 2.7 .
echo "FINISHED CONDA BUILD"
cd $OGUSA_PACKAGE_DIR
conda convert -p all ~/miniconda3/conda-bld/osx-64/ogusa-$1-py27_0.tar.bz2 -o .
echo "FINISHED CONDA CONVERT"
