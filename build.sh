#!/bin/bash
set -ex

BINARY_DIR=bin

g++ -O3 \
    -std=c++17 \
    perf_amx.cpp \
    -ldnnl \
    -fomit-frame-pointer \
    -lopenblas \
    -march=native \
    -o ${BIN}/perf_amx
