#!/bin/sh
git tag $1
echo "finished tag, now distributing to PyPI"
python setup.py register sdist upload
echo "FINISHED UPLOAD TO PYPI"
git push; git push --tags
echo "FINISHED push "
rm ~/code/og.tar
rm -rf ~/code/ogusa
git archive --prefix=ogusa/ -o ~/code/og.tar $1
cd ~/code/
tar xvf og.tar
source deactivate
cd ogusa/Python/conda.recipe
#sed -i '' 's/version: 0.1/version: $1/g' meta.yaml
sed -i '' 's/version: 0.1/version: '${1}'/g' meta.yaml
echo "FINISHED changing meta.yaml"
conda build --python 2.7 .
echo "FINISHED CONDA BUILD"
cd $OGUSA_PACKAGE_DIR
conda convert -p all ~/miniconda3/conda-bld/osx-64/ogusa-$1-py27_0.tar.bz2 -o .
echo "FINISHED CONDA CONVERT"
