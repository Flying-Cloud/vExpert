#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <stdexcept>
#include <string>

// CUDA kernel to replicate virtual experts across all layers in parallel
// Each block processes one layer
__global__ void replicate_vexpert_kernel(
    const float* __restrict__ weight,              // [num_layers, num_logical_experts]
    const int64_t* __restrict__ pinned_to_logical_map, // [num_layers, num_pinned_experts]
    const int64_t* __restrict__ vexpert_count,     // [num_layers, num_gpus]
    int64_t* __restrict__ vir2log,                 // [num_layers, max_num_vir] (out)
    int64_t* __restrict__ rank,                    // [num_layers, max_num_vir] (out)
    int64_t* __restrict__ global_cnt,              // [num_layers, num_logical_experts] (out)
    const int num_layers,
    const int num_logical_experts,
    const int num_pinned_experts,
    const int num_gpus,
    const int max_num_vir) {
    
    const int layer_id = blockIdx.x;
    if (layer_id >= num_layers) {
        return;
    }
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Use shared memory for per-layer data
    __shared__ float s_weight_pinned[256];  // Max 256 pinned experts
    __shared__ int64_t s_pinned_to_logical[256];
    __shared__ int64_t s_pinned_cnt[256];
    __shared__ float s_current_load[256];
    __shared__ int s_total_vir;

    // Calculate total virtual experts for this layer
    if (tid == 0) {
        s_total_vir = 0;
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            s_total_vir += static_cast<int>(vexpert_count[layer_id * num_gpus + gpu]);
        }
    }
    __syncthreads();
    int total_vir = s_total_vir;
    
    if (total_vir == 0) {
        return;
    }
    
    // Initialize pinned data in shared memory
    // Each thread handles some pinned experts
    for (int i = tid; i < num_pinned_experts; i += num_threads) {
        int logical_id = static_cast<int>(pinned_to_logical_map[layer_id * num_pinned_experts + i]);
        s_weight_pinned[i] = weight[layer_id * num_logical_experts + logical_id];
        s_pinned_to_logical[i] = logical_id;
        s_pinned_cnt[i] = 1;  // Start with count 1
    }
    __syncthreads();
    
    // Initialize global_cnt and vir2log (each thread handles some elements)
    for (int i = tid; i < num_logical_experts; i += num_threads) {
        global_cnt[layer_id * num_logical_experts + i] = 1;
    }
    // Initialize vir2log to -1 (unused entries)
    for (int i = tid; i < max_num_vir; i += num_threads) {
        vir2log[layer_id * max_num_vir + i] = -1;
    }
    __syncthreads();
    
    // Process each virtual expert sequentially (greedy algorithm)
    // Use shared memory for max_load_idx to ensure proper synchronization
    __shared__ int s_max_load_idx;
    
    for (int copy_round = 0; copy_round < total_vir && copy_round < max_num_vir; ++copy_round) {
        __syncthreads();
        
        // Find the pinned expert with maximum load (using thread 0)
        // Note: Python's max() returns the first index when multiple values are equal
        if (tid == 0) {
            float max_load = -FLT_MAX;  // Use -FLT_MAX to handle any float values
            s_max_load_idx = -1;
            
            // Update current_load
            for (int i = 0; i < num_pinned_experts; ++i) {
                s_current_load[i] = s_weight_pinned[i] / static_cast<float>(s_pinned_cnt[i]);
            }
            
            // Find maximum (use > to match Python's behavior: returns first index when equal)
            // Python's max() returns the first index when multiple values are equal
            // Using > ensures we only update when we find a strictly larger value,
            // which means when values are equal, we keep the first one
            for (int i = 0; i < num_pinned_experts; ++i) {
                if (s_current_load[i] > max_load) {
                    max_load = s_current_load[i];
                    s_max_load_idx = i;
                }
            }
        }
        __syncthreads();
        
        if (s_max_load_idx == -1) {
            break;
        }
        
        // Thread 0 handles the update
        if (tid == 0) {
            int logical_expert_id = static_cast<int>(s_pinned_to_logical[s_max_load_idx]);
            int current_rank = static_cast<int>(s_pinned_cnt[s_max_load_idx]);
            
            vir2log[layer_id * max_num_vir + copy_round] = logical_expert_id;
            rank[layer_id * max_num_vir + copy_round] = current_rank;
            
            // Update counts - this will change the load for next iteration
            s_pinned_cnt[s_max_load_idx] += 1;
            global_cnt[layer_id * num_logical_experts + logical_expert_id] += 1;
        }
        __syncthreads();
    }
}

// Wrapper function to launch the kernel
void replicate_vexpert_cuda_launcher(
    const torch::Tensor& weight,
    const torch::Tensor& pinned_to_logical_map,
    const torch::Tensor& vexpert_count,
    torch::Tensor& vir2log,
    torch::Tensor& rank,
    torch::Tensor& global_cnt,
    const int num_layers,
    const int num_logical_experts,
    const int num_pinned_experts,
    const int num_gpus,
    const int max_num_vir) {
    
    auto stream = at::cuda::getCurrentCUDAStream(weight.device().index());
    
    // Launch one block per layer
    dim3 grid(num_layers);
    dim3 block(256);  // Enough threads for processing
    
    replicate_vexpert_kernel<<<grid, block, 0, stream>>>(
        weight.data_ptr<float>(),
        pinned_to_logical_map.data_ptr<int64_t>(),
        vexpert_count.data_ptr<int64_t>(),
        vir2log.data_ptr<int64_t>(),
        rank.data_ptr<int64_t>(),
        global_cnt.data_ptr<int64_t>(),
        num_layers,
        num_logical_experts,
        num_pinned_experts,
        num_gpus,
        max_num_vir
    );
    
    // Synchronize to ensure kernel completes before returning
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
void replicate_vexpert_cuda(
    torch::Tensor weight,
    torch::Tensor pinned_to_logical_map,
    torch::Tensor vexpert_count,
    torch::Tensor vir2log,
    torch::Tensor rank,
    torch::Tensor global_cnt,
    int64_t num_layers,
    int64_t num_logical_experts,
    int64_t num_pinned_experts,
    int64_t num_gpus,
    int64_t max_num_vir) {
    
    replicate_vexpert_cuda_launcher(
        weight,
        pinned_to_logical_map,
        vexpert_count,
        vir2log,
        rank,
        global_cnt,
        static_cast<int>(num_layers),
        static_cast<int>(num_logical_experts),
        static_cast<int>(num_pinned_experts),
        static_cast<int>(num_gpus),
        static_cast<int>(max_num_vir)
    );
}

