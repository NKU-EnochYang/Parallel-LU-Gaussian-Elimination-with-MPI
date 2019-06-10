//基于循环划分实现的流水线形式的MPI_LU，包含未优化及优化（SSE及OpenMP）的程序
//未优化高斯消去法函数：eliminate()
//优化后高斯消去法函数：eliminate_opt()，逻辑与未优化版基本一致，看eliminate()的注释即可

#include "MPI_LU_Pipeline.h"

void MPI_LU_Pipeline::eliminate(float mat[][N], int rank, int num_proc)
{
    int seg = task * num_proc;
//    计算当前进程的前一进程及下一进程
    int pre_proc = (rank + (num_proc - 1)) % num_proc;
    int next_proc = (rank + 1) % num_proc;
    for (int k = 0; k < N; k++)
    {
//        判断当前行是否是自己的任务
        if (int((k % seg) / task) == rank)
        {
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
//            处理完自己的任务后向下一进程发送消息
            MPI_Send(&mat[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
        }
        else
        {
//            如果当前行不是当前进程的任务，则接收前一进程的消息
            MPI_Recv(&mat[k], N, MPI_FLOAT, pre_proc, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            如果当前行不是下一进程的任务，需将消息进行传递
            if (int((k % seg) / task) != next_proc)
                MPI_Send(&mat[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
        }
        for (int i = k + 1; i < N; i++)
        {
            if (int((i % seg) / task) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0.0;
            }
        }
    }
}

void MPI_LU_Pipeline::run()
{
    timeval t_start;
    timeval t_end;

    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seg = task * num_proc;
    if (rank == 0)
    {
        Matrix::reset_mat(mat, test);
        gettimeofday(&t_start, NULL);
//        在0号进程进行任务划分
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Send(&mat[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        eliminate(mat, rank, num_proc);
//        处理完0号进程自己的任务后需接收其他进程处理之后的结果
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Recv(&mat[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        gettimeofday(&t_end, NULL);
        cout << "Pipeline MPI LU time cost: "
             << 1000 * (t_end.tv_sec - t_start.tv_sec) +
                0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
        Matrix::print_mat(mat);
    }
    else
    {
//        非0号进程先接收任务
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Recv(&mat[i + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        eliminate(mat, rank, num_proc);
//        处理完后向零号进程返回结果
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Send(&mat[i + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

void MPI_LU_Pipeline::eliminate_opt(float mat[][N], int rank, int num_proc)
{
    __m128 t1, t2, t3;
    int seg = task * num_proc;
    int pre_proc = (rank + (num_proc - 1)) % num_proc;
    int next_proc = (rank + 1) % num_proc;
#pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < N; k++)
    {
        if (int((k % seg) / task) == rank)
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
            MPI_Send(&mat[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Recv(&mat[k], N, MPI_FLOAT, pre_proc, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (int((k % seg) / task) != next_proc)
                MPI_Send(&mat[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
        }
        for (int i = k + 1; i < N; i++)
        {
            if (int((i % seg) / task) == rank)
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

void MPI_LU_Pipeline::run_opt()
{
    timeval t_start;
    timeval t_end;

    int num_proc;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seg = task * num_proc;
    if (rank == 0)
    {
        Matrix::reset_mat(mat, test);
        gettimeofday(&t_start, NULL);
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Send(&mat[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        eliminate_opt(mat, rank, num_proc);
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Recv(&mat[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        gettimeofday(&t_end, NULL);
        cout << "Pipeline MPI LU with SSE and OpenMP time cost: "
             << 1000 * (t_end.tv_sec - t_start.tv_sec) +
                0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
        Matrix::print_mat(mat);
    }
    else
    {
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Recv(&mat[i + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        eliminate_opt(mat, rank, num_proc);
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Send(&mat[i + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}