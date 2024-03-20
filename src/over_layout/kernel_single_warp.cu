#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "../common/param.h"
// #include "common.hpp"
// #include "stencil_part.cpp"

using namespace nvcuda;
using namespace std;

__global__ void mma_run(\
    half * A_Padding, float *S, float *C,\
    int offset_n, int offset_m,\
    half *matrix_row, \
    half *matrix_col,\
    int row_addr,\
    int col_addr, \
    int store_addr, \
    int row_pos_min , int row_pos_max, int col_pos_min, int col_pos_max,\
    int stencil_part_size, int *stencil_part_type,int *stencil_part_pos, int *stencil_part_order,\
    int N, int padding)//分块边长大于16是必然的
{

    extern __shared__ half data[];
    half *row_data = (half *)(data + row_addr);
    half *col_data = (half *)(data + col_addr);
    float *store_data = (float *)(data + store_addr);

    ////////////////////Block_N == stencil_row_size; Block_M == stencil_col_size
    int size = N + 2 * padding;
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int offset = (padding + blockIdx.y*Core_M)*size + padding + blockIdx.x*Core_N;//当前线程块对应的数据位置因为横向和纵向填充都是padding，但分块其实不是方形

    //一次性加载行拆分部分的全部
    int row_offset = offset - offset_m*size - offset_n;//row_pos_min和col_pos_min是负数
    int times = (Block_N*Block_M - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < Block_N*Block_M)
        {
            (row_data)[i*blockDim.x*blockDim.y + tid] = \
            (A_Padding + row_offset)[(i*blockDim.x*blockDim.y + tid)%Block_N + (i*blockDim.x*blockDim.y + tid)/Block_N*size];    
        }
    }
    __syncthreads();

    // 对列拆分参数矩阵对应的网格进行加载，直接从刚才读取的行拆分部分进行转置，采取行读列存的形式，而且用不了伪向量化
    // 一次性加载列拆分部分的全部
    times = (Block_M*Block_N - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < Block_M*Block_N)
        {
            col_data[(i*blockDim.x*blockDim.y + tid)%Block_N*Block_M + (i*blockDim.x*blockDim.y + tid)/Block_N] = \
            row_data[i*blockDim.x*blockDim.y + tid];
        }
    }
    __syncthreads();

    // 调用tcu进行计算(注意本实现只有一个warp)
    #pragma unroll
    for(int m = 0; m < Block_M; m += WMMA_M)//总所周知，stencil被拆分成了多个部分，每个部分都是一个参数矩阵和对应的网格一一对应，这一共有stencil_part_size个对应
    {
        #pragma unroll
        for(int n = 0; n < Block_N; n += WMMA_N)//这里采取对分块中一个一个WMMA_M*WMMA_N的小块依次计算，每个小块需要将所有对应全部遍历一遍
        {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            #pragma unroll
            for(int i = 0; i < stencil_part_size; ++i)
            {
                if(stencil_part_type[i] == 0)//横向
                {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                    int data_offset = (stencil_part_pos[i])*Block_N;//row_data
                    int stencil_offset = stencil_part_order[i]*Block_N*Block_N;//matrix_row
                    #pragma unroll
                    for(int k = 0; k < Block_N; k += WMMA_K)
                    {
                        wmma::load_matrix_sync(a_frag, row_data + data_offset + m*Block_N + k, Block_N);//网格
                        wmma::load_matrix_sync(b_frag, matrix_row + stencil_offset + k*Block_N + n, Block_N);//参数矩阵
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
                }
                else
                {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;//参数矩阵
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;//网格 注意这里应该是col_major 我们构造的
                    int data_offset = (stencil_part_pos[i])*Block_M;//col_data
                    int stencil_offset = stencil_part_order[i]*Block_M*Block_M;//matrix_col
                    #pragma unroll
                    for(int k = 0; k < Block_M; k += WMMA_K)
                    {
                        wmma::load_matrix_sync(a_frag,matrix_col + stencil_offset + m*Block_M + k,Block_M);
                        wmma::load_matrix_sync(b_frag, col_data + data_offset + n*Block_M + k, Block_M);//stride怎么填，数据偏移怎么写，非常有说法
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }
                }
            }
            wmma::store_matrix_sync(store_data + m*Block_N + n, c_frag, Block_N, wmma::mem_row_major);
        }
    }

    __syncthreads();

    //将结果写回
    int store_offset = offset_m*Block_N + offset_n;
    times = (Core_N*Core_M - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        int pos_m = blockIdx.y*Core_M + (i*blockDim.x*blockDim.y + tid)/Core_N;
        int pos_n = blockIdx.x*Core_N + (i*blockDim.x*blockDim.y + tid)%Core_N;
        if(pos_m < N && pos_n < N && i*blockDim.x*blockDim.y + tid < Core_N*Core_M)
        {
            C[pos_m*N + pos_n] = \
            (store_data + store_offset)[(i*blockDim.x*blockDim.y + tid)%Core_N + (i*blockDim.x*blockDim.y + tid)/Core_N*Block_N];
        }
    }



    // __syncthreads();
    // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("\nrow_data:\n");
    //     printf("times:%d\n", times);
    //     for(int i = 0; i < (Block_M + row_pos_max - row_pos_min); ++i)
    //     {
    //         for(int j = 0; j < Block_N; ++j)
    //         {
    //             printf("%4.0f ", __half2float(row_data[i * Block_N + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // __syncthreads();
    
    // __syncthreads();
    // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("\ncol_data:\n");
    //     printf("times:%d\n", times);
    //     for(int i = 0; i < Block_N + col_pos_max - col_pos_min; ++i)
    //     {
    //         for(int j = 0; j < Block_M; ++j)
    //         {
    //             printf("%4.0f ", __half2float(col_data[i * Block_M + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // __syncthreads();

    // __syncthreads();
    // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("\nstore_data:\n");
    //     printf("times:%d\n", times);
    //     for(int i = 0; i < Block_M; ++i)
    //     {
    //         for(int j = 0; j < Block_N; ++j)
    //         {
    //             printf("%4.0f ", store_data[i * Block_N + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // __syncthreads();

    // __syncthreads();
    // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("\nstore_data_part:\n");
    //     printf("times:%d\n", times);
    //     for(int i = offset_m; i < offset_m + Core_M; ++i)
    //     {
    //         for(int j = offset_n; j < offset_n + Core_N; ++j)
    //         {
    //             printf("%4.0f ", store_data[i * Block_N + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // __syncthreads();

    // __syncthreads();
    // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("\nC_store_data:\n");
    //     printf("times:%d\n", times);
    //     for(int i = 0; i < Core_M; ++i)
    //     {
    //         for(int j = 0; j < Core_N; ++j)
    //         {
    //             printf("%4.0f ", (C + C_offset)[i * N + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // __syncthreads();

}