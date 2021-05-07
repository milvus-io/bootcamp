#!/bin/bash

set -e

# axel or wget.
#  axel -n 10 -a [file] or wget [file]
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
wget http://www.openslr.org/resources/12/dev-other.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
wget http://www.openslr.org/resources/12/test-other.tar.gz
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
wget http://www.openslr.org/resources/12/train-clean-360.tar.gz
wget http://www.openslr.org/resources/12/train-other-500.tar.gz

echo "extracting dev-clean..."
tar xzf dev-clean.tar.gz
echo "extracting dev-other..."
tar xzf dev-other.tar.gz
echo "extracting test-clean..."
tar xzf test-clean.tar.gz
echo "extracting test-other..."
tar xzf test-other.tar.gz
echo "extracting train-clean-100..."
tar xzf train-clean-100.tar.gz
echo "extracting train-clean-360..."
tar xzf train-clean-360.tar.gz
echo "extracting train-other-500..."
tar xzf train-other-500.tar.gz
