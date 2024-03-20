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

#ifndef StencilPart
typedef struct
{
    vector<int> type;//横为0，竖为1
    vector<int> pos;//相对中心点stencil_core的位置，横着：上负下正，竖着：左负右正
    vector<int> order;//只有重新组合的时候会用到，记录当前的行是第几个行，当前的竖是第几个竖，用于快速确定对应matrix的offset
}StencilPart;
#endif

__global__ void mma_run(\
    half * A_Padding, float *S, float *C,\
    half *matrix_row, \
    half *matrix_col,\
    int col_addr,\
    int store_addr,\
    int halo_addr,\
    int S_addr,\
    int row_pos_min , int row_pos_max, int col_pos_min, int col_pos_max,\
    int stencil_part_size, int *stencil_part_type,int *stencil_part_pos, int *stencil_part_order,\
    int N, int padding)//块边长大于16是必然的
{

    extern __shared__ half data[];
    half *row_data = &data[0];
    half *col_data = &data[col_addr];

    //Block_N == stencil_row_size; Block_M == stencil_col_size
    int size = N + 2 * padding;
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int offset = (padding + blockIdx.y*Block_M)*size + padding + blockIdx.x*Block_N;//当前线程块对应的数据位置因为横向和纵向填充都是padding，但分块其实不是方形

    //对横向部分网格进行加载
    //中央的Block_M*Block_N的数据
    // int row_load_offset = Block_N*abs(row_pos_min);
    // int Block_N_float4 = Block_N/8;
    // int size_float4 = size/8;
    // int times = Block_N_float4*Block_M/(blockDim.x*blockDim.y);
    // #pragma unroll
    // for(int i = 0; i < times; ++i)
    // {
    //     ((float4 *)(row_data + row_load_offset))[i*blockDim.x*blockDim.y + tid] = \
    //     ((float4 *)(A_Padding + offset))[(i*blockDim.x*blockDim.y + tid)%Block_N_float4 + (i*blockDim.x*blockDim.y + tid)/Block_N_float4*size_float4];
    // }
    
    //上下的空隙加载
    //xxx

    //一次性加载横向部分的全部
    int row_offset = offset + row_pos_min*size;//row_pos_min是负数
    int Block_N_float4 = Block_N/8;
    int size_float4 = size/8;
    int times = (Block_N_float4*(Block_M + row_pos_max - row_pos_min) - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < Block_N_float4*(Block_M + row_pos_max - row_pos_min))
        {
            ((float4 *)(row_data))[i*blockDim.x*blockDim.y + tid] = \
            ((float4 *)(A_Padding + row_offset))[(i*blockDim.x*blockDim.y + tid)%Block_N_float4 + (i*blockDim.x*blockDim.y + tid)/Block_N_float4*size_float4];    
        }
    }


    //对纵向部分网格进行加载(这里需要一个转置)，采取行读列存的形式，而且用不了伪向量化
    //一次性加载纵向部分的全部
    int col_offset = offset + col_pos_min;//col_pos_min是负数
    int Block_N_expand = Block_N + col_pos_max - col_pos_min;//用于在A_Padding中定位，要取的数据一行连续的长度为Block_N_expand
    times = (Block_M*(Block_N + col_pos_max - col_pos_min) - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < Block_M*Block_N_expand)
        {
            col_data[(i*blockDim.x*blockDim.y + tid)%Block_N_expand*Block_M + (i*blockDim.x*blockDim.y + tid)/Block_N_expand] = \
            A_Padding[col_offset + (i*blockDim.x*blockDim.y + tid)%Block_N_expand + (i*blockDim.x*blockDim.y + tid)/Block_N_expand*size];
        }
    }

    //调用tcu
    int warp_num = blockDim.x*blockDim.y/32;
    int warp_id = tid/32;
    times = (stencil_part_mix.type.size() - 1)/(warp_num) + 1;
    for(int i = 0; i < times; ++i)
    {
        int index = i*warp_num + warp_id;//总所周知，stencil被拆分成了多个部分，每个部分都是一个参数矩阵和对应的网格一一对应，这一共有stencil_part_mix.type.size()个对应
        if(index < stencil_part_mix.type.size())//每一个warp处理一个对应，每个对应内部使用tcu进行多次运算，这里的index指的是第几个对应
        {
            if(stencil_part_mix.type[index] == 0)//横向
            {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;//网格
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;//参数矩阵
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

                int data_offset = (stencil_part_mix.pos[index] + abs(row_pos_min))*Block_N;//row_data
                int stencil_offset = stencil_part_mix.order[index]*Block_N*Block_N;//matrix_row
                //对一个矩阵乘进行拆分
                for(int m = 0; m < Block_M; m += WMMA_M)
                {
                    for(int n = 0; n < Block_N; n += WMMA_N)
                    {
                        wmma::fill_fragment(c_frag, 0.0f);
                        for(int k = 0; k < Block_N; k += WMMA_K)
                        {
                            wmma::load_matrix_sync(a_frag, row_data + data_offset + m*Block_N + k, Block_N);
                            wmma::load_matrix_sync(b_frag, matrix_row + stencil_offset + k*Block_N + n, Block_N);
                            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                        //存哪呢0，哇，这玩意是不是还得加锁啊，我哭死
                    }
                }
            }
            else
            {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;//参数矩阵
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;//网格 注意这里应该是col_major 我们构造的
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);

                int data_offset = (stencil_part_mix.pos[index] + abs(col_pos_min))*Block_M;//col_data
                int stencil_offset = stencil_part_mix.order[index]*Block_M*Block_M;//matrix_col

                for(int m = 0; m < Block_M; m += WMMA_M)
                {
                    for(int n = 0; n < Block_N; n += WMMA_N)
                    {
                        wmma::fill_fragment(c_frag, 0.0f);
                        for(int k = 0; k < Block_M; k += WMMA_K)
                        {
                            wmma::load_matrix_sync(a_frag,matrix_col + stencil_offset + m*Block_M + k,Block_M);
                            wmma::load_matrix_sync(b_frag, col_data + data_offset + n*Block_M + k, Block_M);//stride怎么填，数据偏移怎么写，非常有说法
                            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                        }
                        //store到某个地方
                    }
                }
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
    // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 3)
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


}