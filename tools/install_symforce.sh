#!/bin/sh

mkdir -p ./cpp_installs

cmake --install external/symforce/build/temp*/ --prefix ./cpp_installs/ 