from multiprocessing import Pool
import itertools, subprocess, argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

BASE_FLAGS = '-O3 -arch sm_70 -DRUN_TIMES=1'
# BASE_FLAGS = '-O3 -lineinfo -arch sm_70 -DRUN_TIMES=1'

class Param:
    def __init__(self, MESH_SIZE, Padding, stencil_shape_M, stencil_shape_N, stencil_core_M, stencil_core_N, Block_M, Block_N,Tile_X, Core_N, Core_M, WMMA_M, WMMA_N, WMMA_K):
        self.MESH_SIZE = MESH_SIZE
        self.Padding = Padding
        self.stencil_shape_M = stencil_shape_M
        self.stencil_shape_N = stencil_shape_N
        self.stencil_core_M = stencil_core_M
        self.stencil_core_N = stencil_core_N
        self.Block_M = Block_M
        self.Block_N = Block_N
        self.Tile_X = Tile_X
        self.Core_N = Core_N
        self.Core_M = Core_M
        self.WMMA_M = WMMA_M
        self.WMMA_N = WMMA_N
        self.WMMA_K = WMMA_K

def compile_command_gen(srcdir,type, param):
    cmd = ' '.join(['nvcc', BASE_FLAGS, f'\
-DMESH_SIZE={param.MESH_SIZE},\
Padding={param.Padding},\
stencil_shape_M={param.stencil_shape_M},\
stencil_shape_N={param.stencil_shape_N},\
stencil_core_M={param.stencil_core_M},\
stencil_core_N={param.stencil_core_N},\
Block_M={param.Block_M},\
Block_N={param.Block_N},\
Tile_X={param.Tile_X},\
Core_N={param.Core_N},\
Core_M={param.Core_M},\
WMMA_M={param.WMMA_M},\
WMMA_N={param.WMMA_N},\
WMMA_K={param.WMMA_K}'])
    
    output_file_name = f'_{type}_{param.MESH_SIZE}_{param.Tile_X}_{param.stencil_shape_M}_{param.stencil_shape_N}_{param.Block_M}_{param.Block_N}'
    cmd = ' '.join([cmd, '-o', output_file_name, srcdir + r"/host.cu", srcdir + r"/kernel_single_warp.cu"])
    # print(cmd)
    return cmd , output_file_name

def exec_cmd(cmd, wait=False):
    p = subprocess.Popen(cmd.split(' '))
    if wait:
        p.wait()

def run_file_serial(file_name):
    print(file_name)
    example_cmd = f'bash run.sh {file_name}'
    proc = subprocess.Popen(example_cmd.split(' '))
    proc.wait()


if __name__ == '__main__':
    
    MESH_SIZE = 3200
    Padding = 16
    stencil_shape_M = 5
    stencil_shape_N = 5
    stencil_core_M = 2
    stencil_core_N = 2
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    
    Tile_X = 1
    
    Block_M = 16
    Block_N = 16
    Core_N = 16
    Core_M = 16
    #根据理论的推导，分块边长至少必定是16倍数（K）
    
    type = "fma_Block16_row_tile"
    
    if(type == "baseline"):
        srcdir = r"./baseline"#
        Core_M = 16
        Core_N = 16
        Block_M = (Core_M + stencil_shape_M - 1)#
        Block_N = (Core_N + stencil_shape_N - 1)#
    elif(type == "over_layout"):
        srcdir = r"over_layout"#
        Block_M = 16
        Block_N = 16
        Core_N = (Block_N - (stencil_shape_N - 1))#
        Core_M = (Block_M - (stencil_shape_M - 1))#
    elif(type == "over_layout_Block16"):
        srcdir = r"over_layout_Block16"#
        WMMA_M = 16#
        WMMA_N = 16#
        WMMA_K = 16#
        Block_M = 16#
        Block_N = 16#
        Core_N = (Block_N - (stencil_shape_N - 1))#
        Core_M = (Block_M - (stencil_shape_M - 1))#
    elif(type == "fma"):
        srcdir = r"fma"#
    elif(type == "fma_Block16"):
        srcdir = r"fma_Block16"#
        WMMA_M = 16#
        WMMA_N = 16#
        WMMA_K = 16#
        Block_M = 16#
        Block_N = 16#
    elif(type == "fma_Block16_row"):
        srcdir = r"fma_Block16_row"#
        WMMA_M = 16#
        WMMA_N = 16#
        WMMA_K = 16#
        Block_M = 16#
        Block_N = 16#
    elif(type == "fma_Block16_col"):
        srcdir = r"fma_Block16_col"#
        WMMA_M = 16#
        WMMA_N = 16#
        WMMA_K = 16#
        Block_M = 16#
        Block_N = 16#
    elif(type == "fma_Block16_row_tile"):
        srcdir = r"fma_Block16_row_tile"
        WMMA_M = 16
        WMMA_N = 16
        WMMA_K = 16
        Block_M = 16
        Block_N = 16
        

    
    param = Param(MESH_SIZE, Padding, stencil_shape_M, stencil_shape_N, stencil_core_M, stencil_core_N, \
        Block_M, Block_N, Tile_X,Core_N, Core_M, WMMA_M, WMMA_N, WMMA_K)
    
    cmd, output_file_name = compile_command_gen(srcdir, type, param)
    
    exec_cmd(cmd, True)
    run_file_serial(output_file_name)
    
