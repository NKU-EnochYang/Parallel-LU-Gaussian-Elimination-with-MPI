//此头文件的成员变量包含两个矩阵、矩阵维度、循环划分任务量、OpenMP线程数等全局变量
//成员函数包括矩阵初始化、重置函数、串行高斯消去法以及打印函数等常用函数

#ifndef MPI_LU_MATRIX_H
#define MPI_LU_MATRIX_H

#include <iostream>
#include <mpi.h>
#include <sys/time.h>
#include <stdlib.h>
#include <pmmintrin.h>
#include <omp.h>

using namespace std;
using namespace MPI;

static const int N = 1000;
static const int task = 1;
static const int thread_count = 4;

extern float test[N][N];
extern float mat[N][N];

class Matrix
{
public:
    static void init_mat(float test[][N]);

    static void reset_mat(float mat[][N], float test[][N]);

    static void naive_lu(float mat[][N]);

    static void print_mat(float mat[][N]);
};


#endif //MPI_LU_MATRIX_H
