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
    half *matrix_row, \
    int halo_addr,\
    int S_addr,\
    int row_pos_min , int row_pos_max,\
    int stencil_part_size, int *stencil_part_type,int *stencil_part_pos, int *stencil_part_order,\
    int N, int padding)//分块边长大于16是必然的
{
    extern __shared__ half data[];
    
    half *row_data = (half *)data;
    half *halo_data = (half *)(data + halo_addr);
    float *S_data = (float *)(data + S_addr);
    float *store_data = (float *)(data);
    
    ////////////////////Block_N == stencil_row_size; Block_M == stencil_col_size
    int size = N + 2 * padding;
    int tid = threadIdx.x;
    int offset = (padding + blockIdx.y*Block_M)*size + padding + blockIdx.x*Tile_X*Block_N;//当前线程块对应的数据位置因为横向和纵向填充都是padding，但分块其实不是方形

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    // wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[5];
    // for(int i = 0; i < stencil_part_size; ++i)
    // {
    //     wmma::load_matrix_sync(b_frag[i], matrix_row + stencil_part_order[i]*Block_N*Block_N, Block_N);
    // }
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    //加载参数矩阵
    int times = (stencil_shape_M*stencil_shape_N - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < stencil_shape_M*stencil_shape_N)
        {
            S_data[i*blockDim.x*blockDim.y + tid] = S[i*blockDim.x*blockDim.y + tid];
        }
    }

    for(int iter = 0; iter < Tile_X; ++iter)
    {
        // int iter = 0;
        //一次性加载行拆分部分的全部
        int row_offset = offset + row_pos_min*size + iter*Block_N;//row_pos_min是负数
        int Block_N_float4 = Block_N/8;
        int size_float4 = size/8;
        times = (Block_N_float4*(Block_M + row_pos_max - row_pos_min) - 1)/(blockDim.x*blockDim.y) + 1;
        #pragma unroll
        for(int i = 0; i < times; ++i)
        {
            if(i*blockDim.x*blockDim.y + tid < Block_N_float4*(Block_M + row_pos_max - row_pos_min))
            {
                ((float4 *)(row_data))[i*blockDim.x*blockDim.y + tid] = \
                ((float4 *)(A_Padding + row_offset))[(i*blockDim.x*blockDim.y + tid)%Block_N_float4 + (i*blockDim.x*blockDim.y + tid)/Block_N_float4*size_float4];    
            }
        }

        __syncthreads();

        //调用tcu进行计算(注意本实现只有一个warp)(部分区域的计算)
        wmma::fill_fragment(c_frag, 0.0f);
        #pragma unroll
        for(int i = 0; i < stencil_part_size; ++i)
        {
            int data_offset = (stencil_part_pos[i] + abs(row_pos_min))*Block_N;//row_data
            int stencil_offset = stencil_part_order[i]*Block_N*Block_N;//matrix_row
            wmma::load_matrix_sync(a_frag, row_data + data_offset, Block_N);//网格
            wmma::load_matrix_sync(b_frag, matrix_row + stencil_offset, Block_N);//系数
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(store_data , c_frag, Block_N, wmma::mem_row_major);

        __syncthreads();

        // halo区,上下左右分别为halo1234区
        int halo1_m = stencil_core_M;
        int halo1_n = Block_N + stencil_shape_N - 1;
        int halo2_m = stencil_shape_M - stencil_core_M - 1;
        int halo2_n = Block_N + stencil_shape_N - 1;

        int halo3_m = Block_M + stencil_shape_M - 1;
        int halo3_n = stencil_core_N;
        half *halo3 = (half *)(halo_data);
        int halo4_m = Block_M + stencil_shape_M - 1;
        int halo4_n = stencil_shape_N - stencil_core_N - 1;
        half *halo4 = (half *)(halo3 + halo3_m*halo3_n);

        // 加载halo区数据
        //halo3
        int halo_offset = offset - halo3_n - halo1_m*size + iter*Block_N;
        times = (halo3_m*halo3_n - 1)/(blockDim.x*blockDim.y) + 1;
        #pragma unroll
        for(int i = 0; i < times; ++i)
        {
            if(i*blockDim.x*blockDim.y + tid < halo3_m*halo3_n)
            {
                halo3[i*blockDim.x*blockDim.y + tid] = A_Padding[halo_offset + (i*blockDim.x*blockDim.y + tid)%halo3_n + (i*blockDim.x*blockDim.y + tid)/halo3_n*size];
            }
        }

        //halo4
        halo_offset = offset + Block_N - halo1_m*size + iter*Block_N;
        times = (halo4_m*halo4_n - 1)/(blockDim.x*blockDim.y) + 1;
        #pragma unroll
        for(int i = 0; i < times; ++i)
        {
            if(i*blockDim.x*blockDim.y + tid < halo4_m*halo4_n)
            {
                halo4[i*blockDim.x*blockDim.y + tid] = A_Padding[halo_offset + (i*blockDim.x*blockDim.y + tid)%halo4_n + (i*blockDim.x*blockDim.y + tid)/halo4_n*size];
            }
        }
        __syncthreads();
        
        //计算halo区
        // 行，即计算 halo3/halo4
        if(tid < 16)
        {
            int s_index;
            #pragma unroll
            for(int col = 0; col < stencil_core_N; ++col)//halo3
            {
                #pragma unroll
                for(int num_per_col = 0; num_per_col <= col; ++num_per_col)
                {
                    #pragma unroll
                    for(int i = 0; i < stencil_part_size; ++i)
                    {
                        s_index = (stencil_part_pos[i] + stencil_core_M)*stencil_shape_N;
                        store_data[tid*Block_N + num_per_col] += \
                        __half2float(halo3[col + (halo1_m + tid + stencil_part_pos[i])*halo3_n])*\
                        S_data[s_index + col - num_per_col];
                    }
                }
            }
            #pragma unroll
            for(int col = 0; col < stencil_shape_N - stencil_core_N - 1; ++col)//halo4
            {
                #pragma unroll
                for(int num_per_col = 0; num_per_col <= col; ++num_per_col)
                {
                    #pragma unroll
                    for(int i = 0; i < stencil_part_size; ++i)
                    {
                        s_index = (stencil_part_pos[i] + stencil_core_M)*stencil_shape_N;
                        store_data[tid*Block_N + Block_N - 1 - num_per_col] += \
                        __half2float(halo4[halo4_n - 1 - col + (halo1_m + tid + stencil_part_pos[i])*halo4_n])*\
                        S_data[s_index + stencil_shape_N - 1 - col + num_per_col];
                    }
                }
            }
        }

        

        __syncthreads();


        //将结果写回
        int C_offset = blockIdx.y*Block_M*N + blockIdx.x*Block_N*Tile_X + iter*Block_N;
        Block_N_float4 = Block_N/4;
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