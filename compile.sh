#!/bin/bash
set -ex

g++ -S -o core_avx.s core.cc
g++ -o core_avx core.cc
./core_avx
