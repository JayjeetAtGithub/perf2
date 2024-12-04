#!/bin/bash
set -ex

g++ -std=c++11 -S -o core_avx.s core.cc
g++ -std=c++11 -o core_avx core.cc
./core_avx
