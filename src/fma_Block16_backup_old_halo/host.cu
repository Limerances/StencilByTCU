#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <vector>
#include <map>
#include "../common/common.hpp"
#include "../common/stencil_part.cpp"

using namespace nvcuda;
using namespace std;

__global__ void mma_run(\
    half * A_Padding, float *S, float *C,\
    half *matrix_row, \
    half *matrix_col,\
    int col_addr,\
    int halo_addr,\
    int S_addr,\
    int row_pos_min , int row_pos_max, int col_pos_min, int col_pos_max,\
    int stencil_part_size, int *stencil_part_type,int *stencil_part_pos, int *stencil_part_order,\
    int N, int padding);//块边长大于16是必然的

void stencil_on_cpu(float *A, float *S, float *C)
{
    for(int i = 0; i < MESH_SIZE; i++)
    {
        for(int j = 0; j < MESH_SIZE; j++)
        {
            for(int m = -stencil_core_M; m < stencil_shape_M - stencil_core_M; m++)
            {
                for(int n = -stencil_core_N; n < stencil_shape_N - stencil_core_N; n++)
                {
                    if(i + m >= 0 && i + m < MESH_SIZE && j + n >= 0 && j + n < MESH_SIZE)
                    {
                        C[i * MESH_SIZE + j] += A[(i + m) * MESH_SIZE + j + n] * S[(m + stencil_core_M) * stencil_shape_N + n + stencil_core_N];
                    }
                }
            }
        }
    }
}

void fill_padding(half *A_h_padding, float* A_h, int size, int padding)
{
    memset(A_h_padding, 0, (size + 2*padding)*(size + 2*padding)*sizeof(half));
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            A_h_padding[(i + padding)*(size + 2*padding) + j + padding] = __float2half(A_h[i*size + j]); 
        }
    }
}

void stencil2matrix_row(float *S_h, StencilPart stencil_part, half *stencil_matrix, int size)
{
    if(stencil_part.type.size() == 0)
        return;
    memset(stencil_matrix, 0, (int)stencil_part.type.size()*size*size*sizeof(half));
    for(int i = 0; i < stencil_part.type.size(); ++i)//有stencil_part.type.size()个从上到下重叠的计算矩阵
    {
        if(stencil_part.type[i] == 0)//确认一下为行
        {
            int begin = (stencil_part.pos[i] + stencil_core_M)*stencil_shape_N;//这一行填充stencil的起点
            int stride = 1;//因为是一行，所有间距为1
            for(int j = 0; j < size; ++j)//把stencil行一列一列填充
            {
                for(int k = 0; k < stencil_shape_N; ++k)//遍历一行
                {
                    int stencil_data_addr = begin + k*stride;//stencil(S_h)准备放进去的数据的地址
                    int matrix_m = k - stencil_core_N + j;//放进去位置的行号
                    int matrix_n = j;//放进去位置的列号
                    if(matrix_m >= 0 && matrix_m < size && matrix_n >= 0 && matrix_n < size)//必须填在对应单个计算矩阵内部
                    {
                        int matrix_addr = i*size*size + matrix_m*size + matrix_n;//计算矩阵的地址
                        stencil_matrix[matrix_addr] = __float2half(S_h[stencil_data_addr]); 
                    }
                }
            }
            memset(S_h + begin, 0, stencil_shape_N*sizeof(float));//把已经填充的stencil行清零，防止后面的列重复计算.
        }
    }
}

void stencil2matrix_col(float *S_h, StencilPart stencil_part, half *stencil_matrix, int size)
{
    if(stencil_part.type.size() == 0)
        return;
    memset(stencil_matrix, 0, (int)stencil_part.type.size()*size*size*sizeof(half));
    for(int i = 0; i < stencil_part.type.size(); ++i)//有stencil_part.type.size()个从上到下重叠的计算矩阵
    {
        if(stencil_part.type[i] == 1)//确认一下为列
        {
            int begin = stencil_part.pos[i] + stencil_core_N;//这一列填充stencil的起点
            int stride = stencil_shape_N;//因为是一列，所有间距为stencil_shape_N
            for(int j = 0; j < size; ++j)//把stencil列一行一行填充
            {
                for(int k = 0; k < stencil_shape_M; ++k)//遍历一列
                {
                    int stencil_data_addr = begin + k*stride;//stencil(S_h)准备放进去的数据的地址
                    int matrix_m = j;//放进去位置的行号
                    int matrix_n = k - stencil_core_M + j;//放进去位置的列号
                    if(matrix_m >= 0 && matrix_m < size && matrix_n >= 0 && matrix_n < size)//必须填在对应单个计算矩阵内部
                    {
                        int matrix_addr = i*size*size + matrix_m*size + matrix_n;//计算矩阵的地址
                        stencil_matrix[matrix_addr] = __float2half(S_h[stencil_data_addr]); 
                    }
                }
            }
        }
    }
}

void show_stencil_matrix(half *stencil_matrix, int n, int size)
{
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < size; ++j)
        {
            for(int k = 0; k < size; ++k)
            {
                printf("%6.1f", __half2float(stencil_matrix[i*size*size + j*size + k]));
            }
            printf("\n");
        }
        if(i != n - 1)
        {
            for(int k = 0; k < size; ++k)
            {
                printf("------");
            }
        }
        printf("\n");
    }
}

void show_data_matrix(half *A_h_padding, int M, int N)
{
    printf("data: \n");
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            printf("%6.0f", __half2float(A_h_padding[i*N + j]));
        }
        printf("\n");
    }
    printf("\n");
}

void stencil_on_gpu(float *A_h, float *S_h, float *C)
{
    half *A_h_padding = (half*)malloc((MESH_SIZE + 2*Padding)*(MESH_SIZE + 2*Padding)*sizeof(half));
    // half *S_h_half = (half*)malloc(stencil_shape_M*stencil_shape_N*sizeof(half));

    fill_padding(A_h_padding, A_h, MESH_SIZE, Padding);
    //展示填充后的网格
    // show_data_matrix(A_h_padding, MESH_SIZE + 2*Padding, MESH_SIZE + 2*Padding);
    // array_float2half(S_h, S_h_half, stencil_shape_M*stencil_shape_N);

    half *A_d_padding;
    float *S_d;
    float *C_d;
    CHECK(cudaMalloc((half**)&A_d_padding, (MESH_SIZE + 2*Padding)*(MESH_SIZE + 2*Padding)*sizeof(half)));
    CHECK(cudaMalloc((float**)&S_d, stencil_shape_M*stencil_shape_N*sizeof(float)));
    CHECK(cudaMalloc((float**)&C_d, MESH_SIZE*MESH_SIZE*sizeof(float)));

    CHECK(cudaMemcpy(A_d_padding, A_h_padding, (MESH_SIZE + 2*Padding)*(MESH_SIZE + 2*Padding)*sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(S_d, S_h, stencil_shape_M*stencil_shape_N*sizeof(float), cudaMemcpyHostToDevice));


    //切分数据矩阵，现在采取未处理地方硬算的策略
    dim3 block(Warp_Size);
    dim3 grid(MESH_SIZE/(Block_N), MESH_SIZE/(Block_M));

    //重组Stencil计算矩阵 根据ipad推导过程 利用多种算法计算最小复杂度，这样就不用增大搜索空间了
    StencilPart stencil_part = find_best_stencil_part(S_h,Block_M,Block_N);//拿到拆分结果

    //分开行列
    StencilPart stencil_part_row;
    StencilPart stencil_part_col;
    int row_pos_min = 0, row_pos_max = 0;
    int col_pos_min = 0, col_pos_max = 0;
    for(int i = 0; i < stencil_part.type.size(); ++i)
    {
        if(stencil_part.type[i] == 0)
        {
            stencil_part_row.type.push_back(0);
            stencil_part_row.pos.push_back(stencil_part.pos[i]);
            row_pos_min = Min(row_pos_min, stencil_part.pos[i]);
            row_pos_max = Max(row_pos_max, stencil_part.pos[i]);
        }
        else
        {
            stencil_part_col.type.push_back(1);
            stencil_part_col.pos.push_back(stencil_part.pos[i]);
            col_pos_min = Min(col_pos_min, stencil_part.pos[i]);
            col_pos_max = Max(col_pos_max, stencil_part.pos[i]);
        }
    }
    //混合起来(放在一起，准备传参)，添加order，我怎么把这么简单的东西写一大堆代码呢 :(
    int stencil_part_size = stencil_part.type.size();
    int *stencil_part_type = (int*)malloc(stencil_part_size*sizeof(int));
    int *stencil_part_pos = (int*)malloc(stencil_part_size*sizeof(int));
    int *stencil_part_order = (int*)malloc(stencil_part_size*sizeof(int));
    for(int i = 0; i < stencil_part_row.type.size(); ++i)
    {
        stencil_part_type[i] = stencil_part_row.type[i];
        stencil_part_pos[i] = stencil_part_row.pos[i];
        stencil_part_order[i] = i;
    }
    for(int i = 0; i < stencil_part_col.type.size(); ++i)
    {
        stencil_part_type[i + stencil_part_row.type.size()] = stencil_part_col.type[i];
        stencil_part_pos[i + stencil_part_row.type.size()] = stencil_part_col.pos[i];
        stencil_part_order[i + stencil_part_row.type.size()] = i;
    }
    int *stencil_part_type_d, *stencil_part_pos_d, *stencil_part_order_d;
    CHECK(cudaMalloc((int**)&stencil_part_type_d, stencil_part_size*sizeof(int)));
    CHECK(cudaMalloc((int**)&stencil_part_pos_d, stencil_part_size*sizeof(int)));
    CHECK(cudaMalloc((int**)&stencil_part_order_d, stencil_part_size*sizeof(int)));
    CHECK(cudaMemcpy(stencil_part_type_d, stencil_part_type, stencil_part_size*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(stencil_part_pos_d, stencil_part_pos, stencil_part_size*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(stencil_part_order_d, stencil_part_order, stencil_part_size*sizeof(int), cudaMemcpyHostToDevice));


    //初始化stencil计算矩阵，行列分别放在一起，之后就可以通过offset来访问了
    int stencil_row_size = Block_N;
    int stencil_col_size = Block_M;

    half *stencil_matrix_row = (half*)malloc((int)stencil_part_row.type.size()*stencil_row_size*stencil_row_size*sizeof(half));
    half *stencil_matrix_col = (half*)malloc((int)stencil_part_col.type.size()*stencil_col_size*stencil_col_size*sizeof(half));
    half *stencil_matrix_row_d, *stencil_matrix_col_d;
    CHECK(cudaMalloc((half**)&stencil_matrix_row_d, (int)stencil_part_row.type.size()*stencil_row_size*stencil_row_size*sizeof(half)));
    CHECK(cudaMalloc((half**)&stencil_matrix_col_d, (int)stencil_part_col.type.size()*stencil_col_size*stencil_col_size*sizeof(half)));
    
    float *S_h_copy = (float*)malloc(stencil_shape_M*stencil_shape_N*sizeof(float));//建立一个S_h的副本，因为在第一次提取后会被修改（置0）
    memcpy(S_h_copy, S_h, stencil_shape_M*stencil_shape_N*sizeof(float));//因为非0值总不能同时被行列取吧，这样就多算了
    stencil2matrix_row(S_h_copy, stencil_part_row, stencil_matrix_row, stencil_row_size);
    stencil2matrix_col(S_h_copy, stencil_part_col, stencil_matrix_col, stencil_col_size);
    CHECK(cudaMemcpy(stencil_matrix_row_d, stencil_matrix_row, (int)stencil_part_row.type.size()*stencil_row_size*stencil_row_size*sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(stencil_matrix_col_d, stencil_matrix_col, (int)stencil_part_col.type.size()*stencil_col_size*stencil_col_size*sizeof(half), cudaMemcpyHostToDevice));
    
    //展示重组后的stencil参数矩阵
    // printf("\nstencil按行拆分矩阵：数据块大小为:%d %d\n", Block_M, Block_N);
    // show_stencil_matrix(stencil_matrix_row, (int)stencil_part_row.type.size(), stencil_row_size);
    // printf("\nstencil按列拆分矩阵：数据块大小为:%d %d\n", Block_M, Block_N);
    // show_stencil_matrix(stencil_matrix_col, (int)stencil_part_col.type.size(), stencil_col_size);

    //设置五块共享内存的大小，因为共享内存只能一次写入，所以获取第二、三、四、五块区域（col、store，halo,S）的指针
    //halo其实包括了四块内容，上下左右，虽然与一、二块有所重复，不过我觉得问题不大，不然写得有点复杂
    int shared_mem_row_size = sizeof(half)*(Block_M*Block_N + Block_N*(row_pos_max - row_pos_min));//行拆分的stencil所计算的网格
    int shared_mem_col_size = sizeof(half)*(Block_M*Block_N + Block_M*(col_pos_max - col_pos_min));//列拆分的stencil所计算的网格
    int shared_mem_halo = sizeof(half)*((Block_M + stencil_shape_M - 1)*(Block_N + stencil_shape_N - 1) - Block_M*Block_N);//halo区域
    int shared_mem_S = sizeof(float)*(stencil_shape_M*stencil_shape_N);//stencil区域

    int shared_mem_size = shared_mem_row_size + shared_mem_col_size + shared_mem_halo + shared_mem_S;

    int shared_mem_col_addr = (Block_M*Block_N + Block_N*(row_pos_max - row_pos_min));//col指针位置
    int shared_mem_halo_addr = shared_mem_col_addr + (Block_M*Block_N + Block_M*(col_pos_max - col_pos_min));//halo指针位置
    int shared_mem_S_addr = shared_mem_halo_addr + ((Block_M + stencil_shape_M - 1)*(Block_N + stencil_shape_N - 1) - Block_M*Block_N);//stencil指针位置

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed = 0.0;
    double sum = 0.0;

    for (int test_t = 0; test_t < RUN_TIMES; test_t++)
    {
        cudaDeviceSynchronize();
        cudaEventRecord(start, 0);

        mma_run<<<grid, block, shared_mem_size>>>(\
            A_d_padding, S_d, C_d ,\
            stencil_matrix_row_d, \
            stencil_matrix_col_d, \
            shared_mem_col_addr, \
            shared_mem_halo_addr, \
            shared_mem_S_addr, \
            row_pos_min , row_pos_max, col_pos_min, col_pos_max, \
            stencil_part_size, stencil_part_type_d, stencil_part_pos_d, stencil_part_order_d, \
            MESH_SIZE, Padding);

        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);

        sum += elapsed;
    }
    printf("[Time] Time used: %lf ms\n", sum / RUN_TIMES);

    CHECK(cudaMemcpy(C, C_d, MESH_SIZE*MESH_SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    
    
    //free
    free(A_h_padding);
    free(stencil_matrix_row);
    free(stencil_matrix_col);
    free(S_h_copy);
    free(stencil_part_type);
    free(stencil_part_pos);
    free(stencil_part_order);
    CHECK(cudaFree(A_d_padding));
    CHECK(cudaFree(S_d));
    CHECK(cudaFree(C_d));
    CHECK(cudaFree(stencil_part_type_d));
    CHECK(cudaFree(stencil_part_pos_d));
    CHECK(cudaFree(stencil_part_order_d));
    CHECK(cudaFree(stencil_matrix_row_d));
    CHECK(cudaFree(stencil_matrix_col_d));

    return;
}


int main(int argc,char** argv)
{
    float *A_h, *S_h, *C_h, *C_from_gpu;
    A_h = (float*)malloc(MESH_SIZE*MESH_SIZE*sizeof(float));
    S_h = (float*)malloc(stencil_shape_M*stencil_shape_N*sizeof(float));
    C_h = (float*)malloc(MESH_SIZE*MESH_SIZE*sizeof(float));
    C_from_gpu = (float*)malloc(MESH_SIZE*MESH_SIZE*sizeof(float));

    init_data(A_h, MESH_SIZE*MESH_SIZE);
    init_stencil(S_h, stencil_shape_M*stencil_shape_N);
    memset(C_h, 0, MESH_SIZE*MESH_SIZE*sizeof(float));
    memset(C_from_gpu, 0, MESH_SIZE*MESH_SIZE*sizeof(float));

    stencil_on_cpu(A_h, S_h, C_h);
    
    stencil_on_gpu(A_h, S_h, C_from_gpu);

    checkResult_wmma(C_h, C_from_gpu, MESH_SIZE*MESH_SIZE);
    // showDifference(C_h, C_from_gpu, MESH_SIZE, MESH_SIZE ,MESH_SIZE);

    // printf2D(C_h, MESH_SIZE, MESH_SIZE);
    // printf("\n");
    // printf2D(C_from_gpu, MESH_SIZE, MESH_SIZE);

    free(A_h);
    free(S_h);
    free(C_h);
    free(C_from_gpu);
    return 0;
}