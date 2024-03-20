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
    int S_addr, \
    int N, int padding)//分块边长大于16是必然的
{

    extern __shared__ half data[];
    float *S_data = (float *)(data + S_addr);

    ////////////////////Block_N == stencil_row_size; Block_M == stencil_col_size
    int size = N + 2 * padding;
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int offset = (padding + blockIdx.y*Core_M)*size + padding + blockIdx.x*Core_N;//当前线程块对应的数据位置因为横向和纵向填充都是padding，但分块其实不是方形

    //加载数据
    int data_offset = offset - offset_m*size - offset_n;
    int times = (Block_N*Block_M - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < Block_N*Block_M)
        {
            data[i*blockDim.x*blockDim.y + tid] = \
            (A_Padding + data_offset)[(i*blockDim.x*blockDim.y + tid)%Block_N + (i*blockDim.x*blockDim.y + tid)/Block_N*size];    
        }
    }

    //加载参数矩阵
    times = (stencil_shape_N*stencil_shape_M - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        if(i*blockDim.x*blockDim.y + tid < stencil_shape_N*stencil_shape_M)
        {
            S_data[i*blockDim.x*blockDim.y + tid] = \
            S[i*blockDim.x*blockDim.y + tid];
        }
    }
    __syncthreads();

    //计算
    int C_offset = Core_M*blockIdx.y*N + Core_N*blockIdx.x;
    times = (Core_M*Core_N - 1)/(blockDim.x*blockDim.y) + 1;
    #pragma unroll
    for(int i = 0; i < times; ++i)
    {
        int pos_m = offset_m + (i*blockDim.x*blockDim.y + tid)/Core_N;
        int pos_n = offset_n + (i*blockDim.x*blockDim.y + tid)%Core_N;
        if(i*blockDim.x*blockDim.y + tid < Core_N*Core_M)
        {
            float sum = 0;
            #pragma unroll
            for(int j = 0; j < stencil_shape_M; ++j)
            {
                #pragma unroll
                for(int k = 0; k < stencil_shape_N; ++k)
                {
                    sum += S_data[j*stencil_shape_N + k]*\
                    __half2float(data[(pos_m - stencil_core_M + j)*Block_N + pos_n - stencil_core_N + k]);
                    
                }
            }
            (C + C_offset)[(i*blockDim.x*blockDim.y + tid)%Core_N + (i*blockDim.x*blockDim.y + tid)/Core_N*N] = sum;
        }
    }
}