#define TOLERANCE 0.01
#define MIN_NUMBER 1e-6
#ifndef RUN_TIMES
    #define RUN_TIMES 1
#endif
#ifndef STEP_TIMES
    #define STEP_TIMES 1
#endif
#define SKIP_TIMES 400

#ifndef TILE_SIZE
    #define TILE_SIZE 1
#endif

typedef long long int ll;

#define Warp_Size 32

#ifndef MESH_SIZE
    #define MESH_SIZE 3200
#endif

#ifndef Padding
    #define Padding 16
#endif

#ifndef stencil_shape_M
    #define stencil_shape_M 5
#endif

#ifndef stencil_shape_N
    #define stencil_shape_N 5
#endif

#ifndef stencil_core_M
    #define stencil_core_M 2
#endif

#ifndef stencil_core_N
    #define stencil_core_N 2
#endif

#ifndef Block_M
    #define Block_M 16//根据理论的推导，分块边长至少必定是16倍数（K）
#endif

#ifndef Block_N
    #define Block_N 16
#endif

#ifndef WMMA_M
    #define WMMA_M 16
#endif

#ifndef WMMA_N
    #define WMMA_N 16
#endif

#ifndef WMMA_K
    #define WMMA_K 16
#endif

#define Min(a,b) ((a) < (b) ? (a) : (b))
#define Max(a,b) ((a) > (b) ? (a) : (b))
#define iszero(a) ((a) < 1e-6 && (a) > -1e-6 ? 1 : 0)