#! /usr/bin/env bash

git tag -a v`python setup.py --version`
git push --tags || true  # update the repository version
