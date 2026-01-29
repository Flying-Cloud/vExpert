export NVSHMEM_DISABLE_P2P=0
export NVSHMEM_IB_ENABLE_IBGDA=1
nvcc -O3 -Xcompiler -O3 -rdc=true --ptxas-options=--register-usage-level=10 -arch=sm_90 \
-I/sgl-workspace/nvshmem/install/include \
-L/sgl-workspace/nvshmem/install/lib \
-lnvshmem \
test_cental_flag_sync.cu -o test_flag_sync_nvshmem