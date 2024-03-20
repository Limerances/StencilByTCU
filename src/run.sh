
# nvcc -arch=sm_70 -o demo ./fma/host.cu ./fma/kernel_single_warp.cu
# nvcc -arch=sm_70 -o demo ./over_layout/host.cu ./over_layout/kernel_single_warp.cu
# nvcc -arch=sm_70 -o demo ./baseline/host.cu ./baseline/kernel_single_warp.cu
# $1='./demo'
# $1为可执行程序，执行$1，并且重定向到1.txt
./"$1" > 1.txt
ncu --clock-control none $1 | grep "Duration" | awk -v t="$1" '{printf "%s, %s, %s\n", $4t,$3,$2}'
# rm $1






