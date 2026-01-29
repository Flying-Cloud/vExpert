#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <stdexcept>
#include <string>

// CUDA kernel to allocate virtual experts across all layers in parallel
// Each block processes one layer
__global__ void allocate_vexpert_kernel(
    const int64_t* __restrict__ vir2log,           // [num_layers, max_num_vir]
    const float* __restrict__ weight_per_cnt,      // [num_layers, num_logical_experts]
    const int64_t* __restrict__ vexpert_count,     // [num_layers, num_gpus]
    const int64_t* __restrict__ vexpert_count_sum, // [num_layers]
    float* __restrict__ pack_weights,               // [num_layers, num_gpus]
    int64_t* __restrict__ pack_items,               // [num_layers, num_gpus]
    int64_t* __restrict__ vexpert_id,               // [num_layers, num_gpus, num_vexpert_per_buffer]
    int64_t* __restrict__ phy2log,                  // [num_layers, num_replicas]
    int64_t* __restrict__ log2phy,                  // [num_layers, num_logical_experts, maxlogcnt]
    int64_t* __restrict__ logcnt,                   // [num_layers, num_logical_experts]
    const int num_layers,
    const int num_gpus,
    const int num_logical_experts,
    const int num_replicas,
    const int num_vexpert_per_buffer,
    const int num_experts_per_gpu,
    const int num_replica_per_gpu,
    const int maxlogcnt,
    const int max_num_vir) {
    
    const int layer_id = blockIdx.x;
    if (layer_id >= num_layers) {
        return;
    }
    
    // Use shared memory for per-layer data to reduce global memory access
    __shared__ float s_pack_weights[256];  // Max 256 GPUs
    __shared__ int64_t s_pack_items[256];
    __shared__ int64_t s_pack_capacity[256];
    __shared__ int min_pack_shared;
    
    const int tid = threadIdx.x;
    
    // Initialize shared memory
    if (tid < num_gpus) {
        s_pack_weights[tid] = pack_weights[layer_id * num_gpus + tid];
        s_pack_items[tid] = pack_items[layer_id * num_gpus + tid];
        s_pack_capacity[tid] = vexpert_count[layer_id * num_gpus + tid] + num_experts_per_gpu;
    }
    __syncthreads();
    
    // Get total virtual experts for this layer
    const int total_virtual = static_cast<int>(vexpert_count_sum[layer_id]);
    if (total_virtual == 0) {
        // Copy back pack_weights
        if (tid < num_gpus) {
            pack_weights[layer_id * num_gpus + tid] = s_pack_weights[tid];
        }
        return;
    }
    
    // Calculate null_experts_num
    const int null_experts_num = (layer_id % 2 == 0) ? 0 : 
                                 (num_vexpert_per_buffer - static_cast<int>(vexpert_count[layer_id * num_gpus]));
    
    // Process each virtual expert sequentially (greedy algorithm)
    for (int j_idx = 0; j_idx < total_virtual; ++j_idx) {
        __syncthreads();
        
        // Check if any GPU is available - use thread 0 to check all GPUs
        // This avoids warp synchronization issues
        bool has_available = false;
        if (tid == 0) {
            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                if (s_pack_items[gpu] < s_pack_capacity[gpu]) {
                    has_available = true;
                    break;
                }
            }
            min_pack_shared = has_available ? 1 : 0;
        }
        __syncthreads();
        
        if (min_pack_shared == 0) {
            break;  // All GPUs are full
        }
        
        // Find the GPU with minimum weight among available ones
        // Use thread 0 to find minimum (simple and safe)
        if (tid == 0) {
            float min_weight = FLT_MAX;
            int min_pack = -1;
            
            for (int gpu = 0; gpu < num_gpus; ++gpu) {
                if (s_pack_items[gpu] < s_pack_capacity[gpu]) {
                    if (s_pack_weights[gpu] < min_weight) {
                        min_weight = s_pack_weights[gpu];
                        min_pack = gpu;
                    }
                }
            }
            
            min_pack_shared = min_pack;
        }
        __syncthreads();
        
        if (min_pack_shared == -1) {
            break;
        }
        
        // Thread 0 handles the assignment
        if (tid == 0) {
            // Get logical expert ID
            const int logical_expert_id = static_cast<int>(vir2log[layer_id * max_num_vir + j_idx]);
            const int pack_items_val = static_cast<int>(s_pack_items[min_pack_shared]);
            
            // Update vexpert_id
            const int slot_idx = pack_items_val - num_experts_per_gpu;
            if (slot_idx >= 0 && slot_idx < num_vexpert_per_buffer) {
                vexpert_id[layer_id * num_gpus * num_vexpert_per_buffer + min_pack_shared * num_vexpert_per_buffer + slot_idx] = logical_expert_id;
            }
            
            // Get weight value
            const float weight_val = weight_per_cnt[layer_id * num_logical_experts + logical_expert_id];
            
            // Update pack_weights
            s_pack_weights[min_pack_shared] += weight_val;
            
            // Calculate physical expert ID
            const int physical_expert_id = min_pack_shared * num_replica_per_gpu + pack_items_val + null_experts_num;
            
            // Update phy2log
            if (physical_expert_id < num_replicas) {
                phy2log[layer_id * num_replicas + physical_expert_id] = logical_expert_id;
            }
            
            // Update pack_items
            s_pack_items[min_pack_shared] += 1;
            
            // Update logcnt and log2phy
            const int logcnt_val = static_cast<int>(logcnt[layer_id * num_logical_experts + logical_expert_id]);
            if (logcnt_val < maxlogcnt) {
                log2phy[layer_id * num_logical_experts * maxlogcnt + logical_expert_id * maxlogcnt + logcnt_val] = physical_expert_id;
                logcnt[layer_id * num_logical_experts + logical_expert_id] = logcnt_val + 1;
            }
        }
        __syncthreads();
    }
    
    // Copy back pack_weights
    if (tid < num_gpus) {
        pack_weights[layer_id * num_gpus + tid] = s_pack_weights[tid];
        pack_items[layer_id * num_gpus + tid] = s_pack_items[tid];
    }
}

// Wrapper function to launch the kernel
void allocate_vexpert_cuda_launcher(
    const torch::Tensor& vir2log,
    const torch::Tensor& weight_per_cnt,
    const torch::Tensor& vexpert_count,
    const torch::Tensor& vexpert_count_sum,
    torch::Tensor& pack_weights,
    torch::Tensor& pack_items,
    torch::Tensor& vexpert_id,
    torch::Tensor& phy2log,
    torch::Tensor& log2phy,
    torch::Tensor& logcnt,
    const int num_layers,
    const int num_gpus,
    const int num_logical_experts,
    const int num_replicas,
    const int num_vexpert_per_buffer,
    const int num_experts_per_gpu,
    const int num_replica_per_gpu,
    const int maxlogcnt) {
    
    auto stream = at::cuda::getCurrentCUDAStream(vir2log.device().index());
    
    // Launch one block per layer
    dim3 grid(num_layers);
    dim3 block(256);  // Enough threads for up to 256 GPUs
    
    const int max_num_vir = vir2log.size(1);
    
    allocate_vexpert_kernel<<<grid, block, 0, stream>>>(
        vir2log.data_ptr<int64_t>(),
        weight_per_cnt.data_ptr<float>(),
        vexpert_count.data_ptr<int64_t>(),
        vexpert_count_sum.data_ptr<int64_t>(),
        pack_weights.data_ptr<float>(),
        pack_items.data_ptr<int64_t>(),
        vexpert_id.data_ptr<int64_t>(),
        phy2log.data_ptr<int64_t>(),
        log2phy.data_ptr<int64_t>(),
        logcnt.data_ptr<int64_t>(),
        num_layers,
        num_gpus,
        num_logical_experts,
        num_replicas,
        num_vexpert_per_buffer,
        num_experts_per_gpu,
        num_replica_per_gpu,
        maxlogcnt,
        max_num_vir
    );
    
    // Synchronize to ensure kernel completes before returning
    // This prevents the program from hanging when accessing output data
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel synchronization failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel execution error: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

// Torch-compatible wrapper function
void allocate_vexpert_cuda(
    torch::Tensor vir2log,
    torch::Tensor weight_per_cnt,
    torch::Tensor vexpert_count,
    torch::Tensor vexpert_count_sum,
    torch::Tensor pack_weights,
    torch::Tensor pack_items,
    torch::Tensor vexpert_id,
    torch::Tensor phy2log,
    torch::Tensor log2phy,
    torch::Tensor logcnt,
    int64_t num_layers,
    int64_t num_gpus,
    int64_t num_logical_experts,
    int64_t num_replicas,
    int64_t num_vexpert_per_buffer,
    int64_t num_experts_per_gpu,
    int64_t num_replica_per_gpu,
    int64_t maxlogcnt) {
    
    allocate_vexpert_cuda_launcher(
        vir2log,
        weight_per_cnt,
        vexpert_count,
        vexpert_count_sum,
        pack_weights,
        pack_items,
        vexpert_id,
        phy2log,
        log2phy,
        logcnt,
        static_cast<int>(num_layers),
        static_cast<int>(num_gpus),
        static_cast<int>(num_logical_experts),
        static_cast<int>(num_replicas),
        static_cast<int>(num_vexpert_per_buffer),
        static_cast<int>(num_experts_per_gpu),
        static_cast<int>(num_replica_per_gpu),
        static_cast<int>(maxlogcnt)
    );
}

