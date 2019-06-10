//块划分的MPI_LU，包含未优化及优化（SSE及OpenMP）的程序
//未优化高斯消去法函数：eliminate()
//优化后高斯消去法函数：eliminate_opt()，逻辑与未优化版基本一致，看eliminate()的注释即可

#ifndef MPI_LU_MPI_LU_BLOCK_H
#define MPI_LU_MPI_LU_BLOCK_H

#include "Matrix.h"

class MPI_LU_Block
{
private:
    void eliminate(float mat[][N], int rank, int num_proc);

    void eliminate_opt(float mat[][N], int rank, int num_proc);

public:
    void run();

    void run_opt();
};


#endif //MPI_LU_MPI_LU_BLOCK_H
