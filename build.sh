#!/bin/bash
set -ex

BINARY_DIR=/usr/local/bin

g++ -O3 \
    -std=c++17 \
    perf_cpu.cc \
    -fomit-frame-pointer \
    -march=native \
    -o ${BINARY_DIR}/perf_cpu

g++ -O3 \
    -std=c++17 \
    perf_amx.cc \
    -ldnnl \
    -fomit-frame-pointer \
    -fopenmp \
    -lopenblas \
    -march=native \
    -o ${BINARY_DIR}/perf_amx
