// Fixed version: Use NVSHMEM P2P mechanism for cross-rank memory access
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>
#include <device_host_transport/nvshmem_common_ibgda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    // Local rank, no need for mapping
    if (rank == dst_rank)
        return ptr;
    auto peer_base = __ldg(reinterpret_cast<uint64_t*>(nvshmemi_device_state_d.peer_heap_base_p2p) + dst_rank);

    // RDMA connected
    if (peer_base == 0)
        return 0;

    // NVLink P2P is enabled
    return peer_base + (ptr - reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base));
}

// DeepEP-style memory semantics operations
__device__ __forceinline__ void st_release_sys_global(const int* ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ int ld_acquire_sys_global(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

#ifndef SYNC_TIMEOUT_CYCLES
#define SYNC_TIMEOUT_CYCLES (2000000000LL)
#endif

// Fixed rank0 central sync function using NVSHMEM P2P mechanism
__device__ __forceinline__ void rank0_central_flag_sync_nvshmem(
    int rank,
    volatile int* __restrict__ recv_flag_base,  // Base address, each rank has its own recv_flag region
    int value) {
    
    if (rank == 0) {
        // Rank 0: Write to recv_flag of rank 1-7
        #pragma unroll
        for (int dst_rank = 1; dst_rank <= 7; ++dst_rank) {
            // Calculate target rank's flag address
            auto dst_flag_ptr = reinterpret_cast<uint64_t>(recv_flag_base);
            // Get P2P mapped address
            printf("Try to get P2P mapping for rank %d\n", dst_rank);
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_flag_ptr, rank, dst_rank);
            
            if (dst_p2p_ptr == 0) {
                // P2P not available, use NVSHMEM RDMA operations (simplified version)
                // In actual implementation, should call nvshmemi_ibgda_amo_nonfetch_add etc.
                printf("P2P not available for rank %d, using RDMA fallback\n", dst_rank);
                // Full NVSHMEM RDMA implementation needed here
            } else {
                // P2P available, write directly
                printf("Try to write flag to rank %d\n", dst_rank);
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), value);
            }
        }
    } else if (rank <= 7) {
        // Rank 1-7: Poll own recv_flag
        auto my_flag_ptr = reinterpret_cast<uint64_t>(recv_flag_base);
        auto my_p2p_ptr = nvshmemi_get_p2p_ptr(my_flag_ptr, rank, rank);
        
        #if SYNC_TIMEOUT_CYCLES > 0
        unsigned long long start = clock64();
        #endif
        
        while (true) {
            int observed = ld_acquire_sys_global(reinterpret_cast<int*>(my_p2p_ptr));
            if (observed == value) break;
            
            #if SYNC_TIMEOUT_CYCLES > 0
            if (clock64() - start > static_cast<unsigned long long>(SYNC_TIMEOUT_CYCLES)) {
                printf("timeout: rank=%d observed=%d expect=%d\n", rank, observed, value);
                asm volatile("trap;");
            }
            #endif
        }
    }
}


extern "C" __global__ void central_flag_sync_kernel_nvshmem(
    int rank,
    int* __restrict__ recv_flag_base,
    int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        rank0_central_flag_sync_nvshmem(rank, recv_flag_base, value);
    }
}

// ===== Test Program =====
#define CHECK_CUDA(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s at %s:%d -> %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::exit(1); \
    } \
} while(0)

int main() {
    CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 16 * 1024 * 1024));

    printf("=== Testing rank0_central_flag_sync_nvshmem ===\n");

    // Allocate unified memory recv_flag[0..7]
    const int num_slots = 8;
    int* recv_flag = nullptr;
    CHECK_CUDA(cudaMallocManaged(&recv_flag, num_slots * sizeof(int)));
    for (int i = 0; i < num_slots; ++i) recv_flag[i] = 0;
    CHECK_CUDA(cudaDeviceSynchronize());

    // Create 8 streams
    cudaStream_t streams[num_slots];
    for (int i = 0; i < num_slots; ++i) CHECK_CUDA(cudaStreamCreate(&streams[i]));

    // Test NVSHMEM version
    printf("Testing NVSHMEM P2P version...\n");
    const int expect_value = 456;
    dim3 grid(1), block(1);
    
    // Start rank=1..7 first (enter polling wait)
    for (int r = 1; r <= 7; ++r) {
        central_flag_sync_kernel_nvshmem<<<grid, block, 0, streams[r]>>>(r, recv_flag, expect_value);
    }
    // Then start rank=0 (write flags)
    central_flag_sync_kernel_nvshmem<<<grid, block, 0, streams[0]>>>(0, recv_flag, expect_value);

    // Wait for all streams to complete
    for (int i = 0; i < num_slots; ++i) CHECK_CUDA(cudaStreamSynchronize(streams[i]));

    // Verify NVSHMEM version results
    bool nvshmem_ok = true;
    for (int i = 1; i <= 7; ++i) {
        if (recv_flag[i] != expect_value) {
            fprintf(stderr, "NVSHMEM Mismatch at rank %d: got %d expect %d\n", i, recv_flag[i], expect_value);
            nvshmem_ok = false;
        }
    }
    printf("NVSHMEM recv_flag[0..7] = ");
    for (int i = 0; i < num_slots; ++i) printf("%d ", recv_flag[i]);
    printf("\n");
    printf("NVSHMEM Test %s\n", nvshmem_ok ? "PASSED" : "FAILED");

    // Reset flags for next test
    for (int i = 0; i < num_slots; ++i) recv_flag[i] = 0;
    CHECK_CUDA(cudaDeviceSynchronize());



    // Cleanup
    for (int i = 0; i < num_slots; ++i) cudaStreamDestroy(streams[i]);
    cudaFree(recv_flag);
    
    bool overall_ok = nvshmem_ok ;
    printf("\n=== Overall Test %s ===\n", overall_ok ? "PASSED" : "FAILED");
    return overall_ok ? 0 : 1;
}