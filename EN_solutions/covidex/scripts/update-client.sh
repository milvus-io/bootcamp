#!/bin/bash

echo "Building client for production..."

export BUILD_PATH=client/build
export STATIC_PATH=api/app/static

rm -rf $BUILD_PATH

cd client && yarn install --silent && yarn build && cd ..

rm -rf $STATIC_PATH
mv $BUILD_PATH $STATIC_PATH
mv $STATIC_PATH/static/* $STATIC_PATH

echo "Client built successfully to ${STATIC_PATH}!"
