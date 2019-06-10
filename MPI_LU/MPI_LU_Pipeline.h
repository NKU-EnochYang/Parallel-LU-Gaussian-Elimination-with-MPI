#ifndef MPI_LU_MPI_LU_PIPELINE_H
#define MPI_LU_MPI_LU_PIPELINE_H

#include "Matrix.h"

class MPI_LU_Pipeline
{
private:
    void eliminate(float mat[][N], int rank, int num_proc);

    void eliminate_opt(float mat[][N], int rank, int num_proc);

public:
    void run();

    void run_opt();
};


#endif //MPI_LU_MPI_LU_PIPELINE_H
