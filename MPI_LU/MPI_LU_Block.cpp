//块划分的MPI_LU，包含未优化及优化（SSE及OpenMP）的程序
//未优化高斯消去法函数：eliminate()
//优化后高斯消去法函数：eliminate_opt()，逻辑与未优化版基本一致，看eliminate()的注释即可

#include "MPI_LU_Block.h"

void MPI_LU_Block::eliminate(float mat[][N], int rank, int num_proc)
{
    int block = N / num_proc;
//    未能整除划分的剩余部分
    int remain = N % num_proc;

    int begin = rank * block;
//    当前进程为最后一个进程时，需处理剩余部分
    int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
    for (int k = 0; k < N; k++)
    {
//        判断当前行是否是自己的任务
        if (k >= begin && k < end)
        {
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
//            向之后的进程发送消息
            for (int p = rank + 1; p < num_proc; p++)
                MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
//            当所处行属于当前进程前一进程的任务，需接收消息
            if (cur_p < rank)
                MPI_Recv(&mat[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0.0;
            }
        }
    }
}

void MPI_LU_Block::run()
{
    timeval t_start;
    timeval t_end;

    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / num_proc;
    int remain = N % num_proc;
    if (rank == 0)
    {
        Matrix::reset_mat(mat, test);
        gettimeofday(&t_start, NULL);
//        在0号进程进行任务划分
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }
        eliminate(mat, rank, num_proc);
//        处理完0号进程自己的任务后需接收其他进程处理之后的结果
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        gettimeofday(&t_end, NULL);
        cout << "Block MPI LU time cost: "
             << 1000 * (t_end.tv_sec - t_start.tv_sec) +
                0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
        Matrix::print_mat(mat);
    }
    else
    {
//        非0号进程先接收任务
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        eliminate(mat, rank, num_proc);
//        处理完后向零号进程返回结果
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

void MPI_LU_Block::eliminate_opt(float mat[][N], int rank, int num_proc)
{
    __m128 t1, t2, t3;
    int block = N / num_proc;
    int remain = N % num_proc;
    int begin = rank * block;
    int end = rank != num_proc - 1 ? begin + block : begin + block + remain;
#pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < N; k++)
    {
        if (k >= begin && k < end)
        {
            float temp1[4] = {mat[k][k], mat[k][k], mat[k][k], mat[k][k]};
            t1 = _mm_loadu_ps(temp1);
            int j = k + 1;
#pragma omp for schedule(guided, 20)
            for (j; j < N - 3; j += 4)
            {
                t2 = _mm_loadu_ps(mat[k] + j);
                t3 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(mat[k] + j, t3);
            }
#pragma omp for schedule(guided, 20)
            for (j; j < N; j++)
            {
                mat[k][j] = mat[k][j] / mat[k][k];
            }
            mat[k][k] = 1.0;
            for (int p = rank + 1; p < num_proc; p++)
                MPI_Send(&mat[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        else
        {
            int cur_p = k / block;
            if (cur_p < rank)
                MPI_Recv(&mat[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                float temp2[4] = {mat[i][k], mat[i][k], mat[i][k], mat[i][k]};
                t1 = _mm_loadu_ps(temp2);
                int j = k + 1;
#pragma omp for schedule(guided, 20)
                for (j; j <= N - 3; j += 4)
                {
                    t2 = _mm_loadu_ps(mat[i] + j);
                    t3 = _mm_loadu_ps(mat[k] + j);
                    t3 = _mm_mul_ps(t1, t3);
                    t2 = _mm_sub_ps(t2, t3);
                    _mm_storeu_ps(mat[i] + j, t2);
                }
#pragma omp for schedule(guided, 20)
                for (j; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0;
            }
        }
    }
}

void MPI_LU_Block::run_opt()
{
    timeval t_start;
    timeval t_end;

    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / num_proc;
    int remain = N % num_proc;
    if (rank == 0)
    {
        Matrix::reset_mat(mat, test);
        gettimeofday(&t_start, NULL);
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&mat[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }
        eliminate_opt(mat, rank, num_proc);
        for (int i = 1; i < num_proc; i++)
        {
            if (i != num_proc - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&mat[i * block + j], N, MPI_FLOAT, i, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        gettimeofday(&t_end, NULL);
        cout << "Block MPI LU with SSE and OpenMP time cost: "
             << 1000 * (t_end.tv_sec - t_start.tv_sec) +
                0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
        Matrix::print_mat(mat);
    }
    else
    {
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&mat[rank * block + j], N, MPI_FLOAT, 0, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        eliminate_opt(mat, rank, num_proc);
        if (rank != num_proc - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&mat[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}