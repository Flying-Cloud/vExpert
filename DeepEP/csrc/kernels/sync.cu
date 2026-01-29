// csrc/kernels/sync.cu
#include "utils.cuh"
#include "ibgda_device.cuh"
#include "buffer.cuh"
#include <cuda_runtime.h>

namespace deep_ep {
namespace intranode {

// Device side: Rank0 writes, Rank1..7 poll and wait
__device__ __forceinline__ void rank0_central_flag_sync_nvshmem_dev(
    int rank,
    void** buffer_ptrs,
    int value) {

    if (rank == 0) {
        #pragma unroll
        for (int dst_rank = 1; dst_rank <= 7; ++dst_rank) {
            auto dst_ptr = reinterpret_cast<int*>(reinterpret_cast<void*>(buffer_ptrs[dst_rank]));
            st_na_release(dst_ptr, value);
        }
    } else if (rank <= 7) {
        auto my_ptr = reinterpret_cast<int*>(reinterpret_cast<void*>(buffer_ptrs[rank]));
        while (true) {
            int observed = ld_na_relaxed(my_ptr);
            if (observed == value) break;
        }
    }
}

extern "C" __global__ void central_flag_sync_kernel_nvshmem(
    int rank,
    void** buffer_ptrs,
    int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        rank0_central_flag_sync_nvshmem_dev(rank, buffer_ptrs, value);
    }
}

// Host API: For upper layer calls, keep it simple
void central_flag_sync_nvshmem(void** buffer_ptrs, int value, int rank, cudaStream_t stream) {
    dim3 grid(1), block(1);
    central_flag_sync_kernel_nvshmem<<<grid, block, 0, stream>>>(rank, buffer_ptrs, value);
}

} // namespace intranode
} // namespace deep_ep