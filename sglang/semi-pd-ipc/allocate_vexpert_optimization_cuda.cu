#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <stdexcept>
#include <string>

// Constants (should match Python values)
// BASE_A2A = 7168 * 8 * 3 / (15 * 1024 * 1024 * 1024) / 0.8
// BASE_H2D = (7168 * 2048 * 3) / (26 * 1024 * 1024 * 1024)
// These will be passed as parameters to avoid hardcoding

// CUDA kernel to compute balanceness
// Each block processes one layer pair (2 layers)
__global__ void compute_balanceness_kernel(
    const float* __restrict__ balance_init,  // [layers//2, 2, num_gpus]
    float* __restrict__ balanceness,         // [layers//2, 2]
    const int num_layer_pairs,
    const int num_gpus) {
    
    const int layer_pair_idx = blockIdx.x;
    const int buffer_idx = threadIdx.x;  // 0 or 1
    
    if (layer_pair_idx >= num_layer_pairs || buffer_idx >= 2) {
        return;
    }
    
    // Compute mean and max for this layer pair and buffer
    float sum = 0.0f;
    float max_val = -FLT_MAX;
    
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        const float val = balance_init[layer_pair_idx * 2 * num_gpus + buffer_idx * num_gpus + gpu];
        sum += val;
        max_val = fmaxf(max_val, val);
    }
    
    const float mean_val = sum / num_gpus;
    
    // Store balanceness = mean / max
    // Match PyTorch behavior: if max_val is 0, result is NaN or inf, but we handle it as 0
    // For numerical stability, use a small epsilon
    const float epsilon = 1e-8f;
    if (max_val > epsilon) {
        balanceness[layer_pair_idx * 2 + buffer_idx] = mean_val / max_val;
    } else {
        // If max_val is 0 or very small, set to 0 (matching PyTorch behavior when all values are 0)
        balanceness[layer_pair_idx * 2 + buffer_idx] = 0.0f;
    }
}

// CUDA kernel to set vexpert_count_max based on masks
__global__ void set_vexpert_count_max_kernel(
    int64_t* __restrict__ vexpert_count_max,  // [layers//2, 2, num_gpus]
    const float* __restrict__ balanceness,     // [layers//2, 2]
    const int num_layer_pairs,
    const int num_gpus,
    const int v_m,
    const float thre1) {
    
    const int layer_pair_idx = blockIdx.x;
    const int gpu_idx = threadIdx.x;
    
    if (layer_pair_idx >= num_layer_pairs || gpu_idx >= num_gpus) {
        return;
    }
    
    const float b0 = balanceness[layer_pair_idx * 2 + 0];
    const float b1 = balanceness[layer_pair_idx * 2 + 1];
    
    const bool mask0 = (b0 > thre1) && (b1 < thre1);
    const bool mask1 = (b1 > thre1) && (b0 < thre1);
    const bool mask2 = (b0 > thre1) && (b1 > thre1);
    
    int64_t val0 = v_m;
    int64_t val1 = v_m;
    
    if (mask0) {
        val0 = 0;
        val1 = v_m * 2;
    } else if (mask1) {
        val0 = v_m * 2;
        val1 = 0;
    } else if (mask2) {
        val0 = 0;
        val1 = 0;
    }
    
    vexpert_count_max[layer_pair_idx * 2 * num_gpus + 0 * num_gpus + gpu_idx] = val0;
    vexpert_count_max[layer_pair_idx * 2 * num_gpus + 1 * num_gpus + gpu_idx] = val1;
}

// CUDA kernel to find optimal v_opt for each layer
__global__ void find_optimal_vexpert_kernel(
    const int64_t* __restrict__ vexpert_count_max,  // [layers, num_gpus]
    const float* __restrict__ balanceness,           // [layers]
    int64_t* __restrict__ vexpert_count,            // [layers, num_gpus]
    const int num_layers,
    const int num_gpus,
    const int num_vexpert_per_buffer,
    const float base_a2a,
    const float base_h2d,
    const int bs,
    const float eta1) {
    
    const int layer_idx = blockIdx.x;
    const int gpu_idx = threadIdx.x;
    
    if (layer_idx >= num_layers || gpu_idx >= num_gpus) {
        return;
    }
    
    // All GPUs in the same layer should get the same v_opt
    // So we only compute it once per layer (using thread 0)
    __shared__ int64_t shared_v_opt;
    
    if (gpu_idx == 0) {
        const int64_t v_max = vexpert_count_max[layer_idx * num_gpus + 0];
        const float b_i = balanceness[layer_idx];
        
        float t_min = FLT_MAX;
        int64_t v_opt = 0;
        
        for (int64_t v = 0; v <= v_max; ++v) {
            float t_h2d, t_a2a, t_moe, delta_l, eta1_local, eta2, t;
            
            if (v == 0) {
                t_h2d = 0.0f;
                t_a2a = 0.0f;
                t_moe = 0.0f;
                delta_l = 0.0f;
                eta1_local = 0.0f;
                eta2 = 0.0f;
            } else {
                t_h2d = v * base_h2d;
                t_a2a = base_a2a * bs;
                
                if (t_h2d > 1.5f * t_a2a) {
                    continue;
                }
                
                delta_l = -0.75f * b_i + 0.53f + v * 0.1f;
                delta_l = fminf(0.8f - b_i, delta_l);
                eta1_local = eta1;  // Use the original eta1 for v > 0
                eta2 = delta_l / (delta_l + b_i);
                t_moe = 0.0f;
            }
            t = t_h2d * eta1_local - t_a2a * eta2 - t_moe;
            if (t < t_min) {
                t_min = t;
                v_opt = v;
            }
        }
        
        // Store v_opt in shared memory
        shared_v_opt = v_opt;
    }
    __syncthreads();
    
    // All threads set vexpert_count for their GPU
    vexpert_count[layer_idx * num_gpus + gpu_idx] = shared_v_opt;
}

// Wrapper function to launch all kernels
void allocate_vexpert_optimization_cuda_launcher(
    const torch::Tensor& weight_gpu_sum,      // [layers, num_gpus]
    const torch::Tensor& balance_init,        // [layers//2, 2, num_gpus]
    torch::Tensor& vexpert_count,             // [layers, num_gpus]
    const int num_vexpert_per_buffer,
    const int num_gpus,
    const int bs,
    const float base_a2a,
    const float base_h2d,
    const int v_m) {
    
    auto stream = at::cuda::getCurrentCUDAStream(weight_gpu_sum.device().index());
    
    const int num_layers = weight_gpu_sum.size(0);
    const int num_layer_pairs = num_layers / 2;
    
    // Step 1: Compute balanceness
    torch::Tensor balanceness = torch::zeros({num_layer_pairs, 2}, 
                                              torch::TensorOptions()
                                              .dtype(torch::kFloat32)
                                              .device(weight_gpu_sum.device()));
    
    dim3 grid1(num_layer_pairs);
    dim3 block1(2);  // 2 buffers per layer pair
    
    compute_balanceness_kernel<<<grid1, block1, 0, stream>>>(
        balance_init.data_ptr<float>(),
        balanceness.data_ptr<float>(),
        num_layer_pairs,
        num_gpus
    );
    
    // Step 2: Set vexpert_count_max
    // v_m is now passed from Python to ensure consistency
    const float thre1 = 0.7f;
    
    torch::Tensor vexpert_count_max = torch::ones({num_layer_pairs, 2, num_gpus},
                                                    torch::TensorOptions()
                                                    .dtype(torch::kInt64)
                                                    .device(weight_gpu_sum.device())) * v_m;
    
    dim3 grid2(num_layer_pairs);
    dim3 block2(num_gpus);
    
    set_vexpert_count_max_kernel<<<grid2, block2, 0, stream>>>(
        vexpert_count_max.data_ptr<int64_t>(),
        balanceness.data_ptr<float>(),
        num_layer_pairs,
        num_gpus,
        v_m,
        thre1
    );
    
    // Reshape vexpert_count_max to [layers, num_gpus]
    vexpert_count_max = vexpert_count_max.view({num_layers, num_gpus});
    
    // Reshape balanceness to [layers]
    balanceness = balanceness.view({num_layers});
    
    // Step 3: Find optimal v_opt for each layer
    const float eta1 = 0.19f;
    
    dim3 grid3(num_layers);
    dim3 block3(num_gpus);
    
    find_optimal_vexpert_kernel<<<grid3, block3, 0, stream>>>(
        vexpert_count_max.data_ptr<int64_t>(),
        balanceness.data_ptr<float>(),
        vexpert_count.data_ptr<int64_t>(),
        num_layers,
        num_gpus,
        num_vexpert_per_buffer,
        base_a2a,
        base_h2d,
        bs,
        eta1
    );
    
    // Synchronize to ensure kernels complete
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
void allocate_vexpert_optimization_cuda(
    torch::Tensor weight_gpu_sum,
    torch::Tensor balance_init,
    torch::Tensor vexpert_count,
    int64_t num_vexpert_per_buffer,
    int64_t num_gpus,
    int64_t bs,
    double base_a2a,
    double base_h2d,
    int64_t v_m) {
    
    allocate_vexpert_optimization_cuda_launcher(
        weight_gpu_sum,
        balance_init,
        vexpert_count,
        static_cast<int>(num_vexpert_per_buffer),
        static_cast<int>(num_gpus),
        static_cast<int>(bs),
        static_cast<float>(base_a2a),
        static_cast<float>(base_h2d),
        static_cast<int>(v_m)
    );
}

