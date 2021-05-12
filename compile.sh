#!/bin/bash
echo 'Compiling with master'
g++ -O3 -I../eigen-master -std=c++11 new_gemm_test.cpp -o gto
echo 'Compiling current'
g++ -O3 -I. -std=c++14 new_gemm_test.cpp -D__ENABLE_VECTOR_KERNELS__ -o gt