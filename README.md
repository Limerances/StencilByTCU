# StencilByTCU

利用src里的compile.py文件来编译运行，目的是针对性的添加宏定义参数，修改compile.py文件里的src参数可以切换不同实现方式。

```
python3 compile.py
```

会自动编译在当前目录，执行定向输出到1.txt，然后用ncu采集运行时间（其实也就是src下的bash脚本在做）

学长可以主要看一下fma_Block16_row_tile类型，也就是固定分块大小，且固定使用行划分，增加了`tilex`选项这一类。

我始终感觉还是halo区的计算有问题，可能是bank冲突太严重了。

- MESH_SIZE = 3200   网格大小
- Padding = 16   在网格周围填充
- stencil_shape_M = 5   box型stencil大小
- stencil_shape_N = 5
- stencil_core_M = 2  更新点相对于stencil左上角的坐标（比如2,2就是stencil的中心，0,0就是左上角）
- stencil_core_N = 2
- WMMA_M = 16
- WMMA_N = 16
- WMMA_K = 16  
- Tile_X = 1    
- Block_M = 16  分块大小，在这个例子里grid网格相当于分成了3200/16=200,200*200=40000个线程块
- Block_N = 16
- Core_N = 16  在fma_Block16_row_tile中没有用到
- Core_M = 16





**面临的问题**：为了实现适应范围较广的计算（如多种分块大小，拆分方式），导致kernel中出现了过多的数据加载情况与分支循环问题，使得并行度下降，访存性能弱，



#### **具体的运行结果**：

**对于MESH_SIZE=3200，stencil=5*5，分块大小=16\*16**

|         FMA          |           366.53 us            |                                                   |
| :------------------: | :----------------------------: | :-----------------------------------------------: |
|     FMA_Block16      | 239.36 us （此处采用了列划分） |       固定分块大小，减少了shared memory分配       |
|   FMA_Block16_row    |           270.95 us            |          固定分块大小，且固定使用行划分           |
|   FMA_Block16_col    |           241.12 us            |          固定分块大小，且固定使用列划分           |
| FMA_Block16_row_tile |  242.85 us （在tile=1时最优）  | 固定分块大小，且固定使用行划分，增加了`tilex`选项 |
|     over_layout      |           213.44 us            |                                                   |
| over_layout_Block16  |           203.95 us            |                                                   |
|       baseline       |           197.15 us            |                                                   |
|      学姐的demo      |             90 us              |                                                   |

结果：仍比baseline略慢



瓶颈分析，在halo区的计算部分，耗时过长（100us）

```c
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
```

