//测试说明：
//调用相应类的接口函数即可，其中run表示未经SSE和OpenMP优化，run_opt表示经两者优化过
//如需调整矩阵规模N、循环划分任务量task及OpenMP线程数，在Matrix.h文件中进行
//如需调整MPI进程数，在命令行参数中进行修改

#include "Matrix.h"
#include "MPI_LU_Block.h"
#include "MPI_LU_Pipeline.h"
#include "MPI_LU_Recycle.h"

int main()
{
    Matrix::init_mat(test);

    MPI_Init(NULL, NULL);

    MPI_LU_Block mpi_lu_block;
    mpi_lu_block.run();
    mpi_lu_block.run_opt();

    MPI_LU_Recycle mpi_lu_recycle;
    mpi_lu_recycle.run();
    mpi_lu_recycle.run_opt();

    MPI_LU_Pipeline mpi_lu_pipeline;
    mpi_lu_pipeline.run();
    mpi_lu_pipeline.run_opt();

    MPI_Finalize();
    return 0;
}