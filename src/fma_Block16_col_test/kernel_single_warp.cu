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
    half *matrix_col,\
    int halo_addr,\
    int S_addr,\
    int col_pos_min, int col_pos_max,\
    int stencil_part_size, int *stencil_part_type,int *stencil_part_pos, int *stencil_part_order,\
    int N, int padding)//分块边长大于16是必然的
{

    extern __shared__ half data[];
    
    half *col_data = (half *)data;
    half *halo_data = (half *)(data + halo_addr);
    float *S_data = (float *)(data + S_addr);
    float *store_data = (float *)(data);

    ////////////////////Block_N == stencil_row_size; Block_M == stencil_col_size
    int size = N + 2 * padding;
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int offset = (padding + blockIdx.y*Block_M)*size + padding + blockIdx.x*Block_N;//当前线程块对应的数据位置因为横向和纵向填充都是padding，但分块其实不是方形

    //对列拆分参数矩阵对应的网格进行加载(这里需要一个转置)，采取行读列存的形式，而且用不了伪向量化
    int col_offset = offset + col_pos_min;//col_pos_min是负数
    int Block_N_expand = Block_N + col_pos_max - col_pos_min;//用于在A_Padding中定位，要取的数据一行连续的长度为Block_N_expand
    int times = (Block_M*(Block_N + col_pos_max - col_pos_min) - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < Block_M*Block_N_expand)
        {
            col_data[(i*blockDim.x*blockDim.y + tid)%Block_N_expand*Block_M + (i*blockDim.x*blockDim.y + tid)/Block_N_expand] = \
            A_Padding[col_offset + (i*blockDim.x*blockDim.y + tid)%Block_N_expand + (i*blockDim.x*blockDim.y + tid)/Block_N_expand*size];
        }
    }

    // //加载参数矩阵
    times = (stencil_shape_M*stencil_shape_N - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < stencil_shape_M*stencil_shape_N)
        {
            S_data[(i*blockDim.x*blockDim.y + tid)/stencil_shape_N + (i*blockDim.x*blockDim.y + tid)%stencil_shape_N*stencil_shape_M] = \
            S[i*blockDim.x*blockDim.y + tid];
        }
    }


    __syncthreads();

    //调用tcu进行计算(注意本实现只有一个warp)(部分区域的计算)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    #pragma unroll
    for(int i = 0; i < stencil_part_size; ++i)
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;//参数矩阵
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;//网格 注意这里应该是col_major 我们构造的
        int data_offset = (stencil_part_pos[i] + abs(col_pos_min))*Block_M;//col_data
        int stencil_offset = stencil_part_order[i]*Block_M*Block_M;//matrix_col

        wmma::load_matrix_sync(a_frag,matrix_col + stencil_offset,Block_M);
        wmma::load_matrix_sync(b_frag, col_data + data_offset, Block_M);//stride怎么填，数据偏移怎么写，非常有说法
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(store_data , c_frag, Block_N, wmma::mem_row_major);

    __syncthreads();

    // halo区,上下左右分别为halo1234区
    int halo1_m = stencil_core_M;
    int halo1_n = Block_N + stencil_shape_N - 1;
    half *halo1 = (half *)(halo_data);
    int halo2_m = stencil_shape_M - stencil_core_M - 1;
    int halo2_n = Block_N + stencil_shape_N - 1;
    half *halo2 = (half *)(halo1 + halo1_m*halo1_n);
    int halo3_m = Block_M + stencil_shape_M - 1;
    int halo3_n = stencil_core_N;
    half *halo3 = (half *)(halo2 + halo2_m*halo2_n);
    int halo4_m = Block_M + stencil_shape_M - 1;
    int halo4_n = stencil_shape_N - stencil_core_N - 1;
    half *halo4 = (half *)(halo3 + halo3_m*halo3_n);

    // 加载halo区数据
    // halo1
    int halo_offset = offset - halo1_m*size - halo3_n;
    times = (halo1_m*halo1_n - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < halo1_m*halo1_n)
        {
            halo1[i*blockDim.x*blockDim.y + tid] = A_Padding[halo_offset + (i*blockDim.x*blockDim.y + tid)%halo1_n + (i*blockDim.x*blockDim.y + tid)/halo1_n*size];
        }
    }

    //halo2
    halo_offset = offset + Block_M*size - halo3_n;
    times = (halo2_m*halo2_n - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < halo2_m*halo2_n)
        {
            halo2[i*blockDim.x*blockDim.y + tid] = A_Padding[halo_offset + (i*blockDim.x*blockDim.y + tid)%halo2_n + (i*blockDim.x*blockDim.y + tid)/halo2_n*size];
        }
    }

    __syncthreads();
    //计算halo区
    
    if(tid < 16)
    {
        #pragma unroll
        for(int i = 0; i < stencil_part_size; ++i)
        {
            int s_index = (stencil_core_N + stencil_part_pos[i])*stencil_shape_M;
            #pragma unroll
            for(int row = 0; row < stencil_core_M; ++row)//halo1
            {
                #pragma unroll
                for(int num_per_row = 0; num_per_row <= row; ++num_per_row)
                {
                    // int a = -1;
                    store_data[num_per_row*Block_N + tid] += \
                    __half2float(halo1[stencil_core_N + tid + stencil_part_pos[i] + row*halo1_n]);
                    // S_data[s_index + row - num_per_row];
                }
            }
            #pragma unroll
            for(int row = 0; row < stencil_shape_M - stencil_core_M - 1; ++row)//halo2
            {
                #pragma unroll
                for(int num_per_row = 0; num_per_row <= row; ++num_per_row)
                {
                    // int a = 1;
                    store_data[(Block_M - 1 - num_per_row)*Block_N + tid] += \
                    __half2float(halo2[stencil_core_N + tid + stencil_part_pos[i] + (halo2_m - 1 - row)*halo2_n]);
                    // S_data[s_index + stencil_shape_M - 1 - row + num_per_row];  
                }
            }
        }

    }
    

    __syncthreads();


    //将结果写回
    int C_offset = blockIdx.y*Block_M*N + blockIdx.x*Block_N;
    int Block_N_float4 = Block_N/4;
    int N_float4 = N/4;
    times = (Block_N_float4*Block_M - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < Block_N_float4*Block_M)
        {
            ((float4 *)(C + C_offset))[(i*blockDim.x*blockDim.y + tid)%Block_N_float4 + (i*blockDim.x*blockDim.y + tid)/Block_N_float4*N_float4] = \
            ((float4 *)(store_data))[i*blockDim.x*blockDim.y + tid];
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
    //     printf("\nC_store_data:\n");
    //     printf("times:%d\n", times);
    //     for(int i = 0; i < Block_M; ++i)
    //     {
    //         for(int j = 0; j < Block_N; ++j)
    //         {
    //             printf("%4.0f ", (C + C_offset)[i * N + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // __syncthreads();

    // __syncthreads();
    // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("\nhalo1:\n");
    //     for(int i = 0; i < halo1_m; ++i)
    //     {
    //         for(int j = 0; j < halo1_n; ++j)
    //         {
    //             printf("%4.0f ", __half2float(halo1[i * halo1_n + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\nhalo2:\n");
    //     for(int i = 0; i < halo2_m; ++i)
    //     {
    //         for(int j = 0; j < halo2_n; ++j)
    //         {
    //             printf("%4.0f ", __half2float(halo2[i * halo2_n + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\nhalo3:\n");
    //     for(int i = 0; i < halo3_m; ++i)
    //     {
    //         for(int j = 0; j < halo3_n; ++j)
    //         {
    //             printf("%4.0f ", __half2float(halo3[i * halo3_n + j]));
    //         }
    //         printf("\n");
    //     }
    //     printf("\nhalo4:\n");
    //     for(int i = 0; i < halo4_m; ++i)
    //     {
    //         for(int j = 0; j < halo4_n; ++j)
    //         {
    //             printf("%4.0f ", __half2float(halo4[i * halo4_n + j]));
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

    // __syncthreads();
    // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     printf("\nS_data:\n");
    //     for(int i = 0; i < stencil_shape_M; ++i)
    //     {
    //         for(int j = 0; j < stencil_shape_N; ++j)
    //         {
    //             printf("%4.0f ", S_data[i * stencil_shape_N + j]);
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

}