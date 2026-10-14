# vExpert: Virtualizing Expert Storage for Adaptive Load Balancing in Distributed MoE Inference

This repository provides the official implementation of the paper **vExpert: Virtualizing Expert Storage for Adaptive Load Balancing in Distributed MoE Inference**, corresponding to NeurIPS 2026.

- **Authors**: Wenxun Wang, Xiuhong Li, Yida Wang, Chen Tang, Zongle Huang, Ke Hong, Yu Wang, and Yongpan Liu.
- **Venue**: NeurIPS 2026.
- **Paper**: Public paper link forthcoming.

**vExpert** is a virtualized expert storage system that addresses dynamic load imbalance in distributed Mixture-of-Experts (MoE) model deployment. This codebase is built on top of [SGLang](https://github.com/sgl-project/sglang) and implements the vExpert system.

## Overview

**vExpert** solves this problem through a virtualized expert storage system that decouples expert redundancy from device memory constraints via a tiered storage hierarchy. The system treats host DRAM as a globally visible redundant expert pool and dynamically streams experts into device HBM on demand, enabling fine-grained and adaptive load balance without blocking inference.

## Installation

### Prerequisites

- PyTorch (consistent with SGLang requirements)
- CUDA toolchain 
- Sufficient host DRAM for expert redundancy pool 

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Flying-Cloud/vExpert.git
   cd vExpert
   ```
   Run the following steps from this repository root. Parentheses in the commands below run each installation in a subshell, so the working directory remains `vExpert/`.

2. **Install SGLang base dependencies**
   Prepare a Python 3.10 or newer environment with CUDA-enabled PyTorch and the CUDA toolkit. This repository is based on SGLang v0.5.1; refer to the bundled [package configuration](sglang/python/pyproject.toml) for dependencies and the [SGLang installation guide](https://docs.sglang.ai/get_started/install.html) for environment setup.

3. **Install the semi-pd-ipc module**
   This module provides inter-process communication and CUDA kernels for virtual expert memory allocation:
   ```bash
   (cd sglang/semi-pd-ipc && python setup.py install)
   ```

4. **Install the customized DeepGEMM module**
   Install the bundled implementation, which includes vExpert support:
   ```bash
   (cd DeepGEMM && python setup.py install)
   ```

5. **Install the customized DeepEP module**
   Install NVSHMEM first and set `NVSHMEM_DIR` to its installation directory. The bundled DeepEP implementation includes the IPC features required by vExpert:
   ```bash
   export NVSHMEM_DIR=/path/to/nvshmem/install
   (cd DeepEP && python setup.py install)
   ```

6. **Install the SGLang Python package**
   Install the bundled SGLang implementation in the same Python environment:
   ```bash
   python -m pip install -e ./sglang/python
   ```

## Usage

### Basic Usage: Enabling vExpert

To enable the vExpert system, use the following parameters when launching the SGLang server:

```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --enable-cpu-pinned \
    --pinned-expert-num 128 \
    --num-vexpert-buffer 2 \
    --ep-dispatch-algorithm virtual \
    --init-expert-location /path/to/expert_distribution.pt \
    --expert-distribution-recorder-mode stat
```
Other parameters are the same as the original SGLang server launch command. We also provide a complete example script in `sglang/example.sh`.
### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable-cpu-pinned` | bool | False | Enable vExpert virtualized expert storage system |
| `--pinned-expert-num` | int | 64 | Number of experts to store in host DRAM pool (redundancy capacity), if sufficient, recommended to be set as 256 |
| `--num-vexpert-buffer` | int | 2 | Number of maximum virtual experts for dynamic streaming |
| `--numa-aware` | bool | False | Enable NUMA-aware memory allocation for multi-node systems |
| `--ep-dispatch-algorithm` | str | - | Expert dispatch algorithm; use `virtual` for vExpert |
| `--init-expert-location` | str | - | Path to expert distribution statistics file (.pt format) for initialization |


## Implementation Details

### Core Components

1. **semi-pd-ipc Module** (`sglang/semi-pd-ipc/`):
   - `allocate_vexpert_cuda.cu`: CUDA kernels for virtual expert memory allocation
   - `allocate_vexpert_optimization_cuda.cu`: Optimized allocation implementation guided by performance model
   - `replicate_vexpert_cuda.cu`: CUDA kernels for expert data replication and streaming
   - `ipc.cpp`: Inter-process communication interface

2. **Expert Location Management**:
   - `expert_location.py`: Expert location tracking and virtual-to-physical mapping
   - `expert_location_dispatch.py`: Dynamic expert dispatch with load balancing
   - `expert_distribution.py`: Expert distribution algorithms and statistics

3. **MoE Layer Integration**:
   - `ep_moe/layer.py`: Expert Parallelism MoE layer with vExpert support
   - `fused_moe_triton/layer.py`: Fused MoE layer with virtual expert integration, supports pinned expert weight loading for virtual expert.
   - **Note**: Currently implemented specifically for DeepSeek-V3/R1 models

4. **Host-to-Device Manager**:
   - `distributed/device_communicators/custom_h2d_async_cpy.py`: Customized H2D async copy implementation for virtual expert.
   - `moe/expert_host_registry.py`: Expert host registry for virtual expert, supports pinned expert weight loading for virtual expert.


5. **Allocator Strategy** (`eplb_algorithms/deepseek.py`):
   - `replicate_experts_virtual`: Performance-model-guided allocator for virtual expert allocation
   - **Important**: Performance model parameters are hardcoded for the experimental hardware platform (8-node H100 cluster with 4x400Gbps InfiniBand interconnect)
   - Users with different hardware configurations should recalibrate these parameters.

### Performance Model

The allocator uses a performance model that considers:
- **Load balancing gains**: Expected improvement from expert redundancy
- **Host-to-device overhead**: PCIe transfer costs for expert streaming
- **PCIe contention**: Bandwidth competition between concurrent transfers

The model guides dynamic allocation decisions to optimize the trade-off between load balance and transfer overhead.

**Note**: The performance model parameters are currently **hardcoded** in the `replicate_experts_virtual` function in `python/sglang/srt/eplb/eplb_algorithms/deepseek.py`. These parameters are calibrated for the experimental hardware platform (8-node H100 cluster). For different hardware configurations, users may need to adjust these parameters to match their specific platform characteristics.

## Performance Tuning
### Recommended Settings

1. **pinned_expert_num**:
   - Start with 128 for initial experiments
   - Increase to 256 for better load balance (requires more host DRAM)
   - Balance between redundancy capacity and memory usage

2. **num_vexpert_buffer**:
   - Default value of 2 is sufficient for most cases

3. **NUMA Configuration**:
   - Enable `--numa-aware` for multi-node systems
   - Can slightly improve performance, but not necessary since it doubles DRAM usage.

## Example Scripts

Refer to `sglang/example.sh` for a complete example of deploying vExpert in a Prefill-Decode Disaggregation scenario with DeepSeek-V3.1.

## Experimental Setup

- **Model**: DeepSeek-V3.1
- **Hardware**: 8-node H100 cluster with 4x400Gbps InfiniBand interconnect

## Compatibility

- **SGLang Version**: Based on SGLang v0.5.1, compatible with SGLang's main features
- **Model Support**: **Currently only adapted for DeepSeek-V3 and DeepSeek-R1 models**. The implementation is model-specific and may require modifications for other MoE models.
- **Hardware Requirements**: 
  - CUDA-capable GPU (H100 recommended for best performance)
  - Sufficient host DRAM for expert redundancy pool (700GB for DeepSeek-V3.1)

## Known Limitations

1. **Model Support**: The current implementation is **only adapted for DeepSeek-V3 and DeepSeek-R1 models**. Adapting to other MoE models requires model-specific modifications.

2. **Hardcoded Performance Model Parameters**: The performance model parameters in the allocator strategy (`replicate_experts_virtual` function in `deepseek.py`) are hardcoded for the experimental hardware platform (8-node H100 cluster). For different hardware configurations, users need to recalibrate these parameters based on their platform's characteristics (PCIe bandwidth, host-to-device transfer latency, etc.).

3. Requires sufficient host DRAM for expert redundancy pool

## Citation

If you use vExpert in your research, please cite the accompanying paper. The citation below includes the paper title and authors; it will be updated with the public paper link and final proceedings metadata when available:

```bibtex
@misc{wang2026vexpert,
  title     = {vExpert: Virtualizing Expert Storage for Adaptive Load Balancing in Distributed {MoE} Inference},
  author    = {Wang, Wenxun and Li, Xiuhong and Wang, Yida and Tang, Chen and Huang, Zongle and Hong, Ke and Wang, Yu and Liu, Yongpan},
  year      = {2026}
}
```

## License

The vExpert-specific contributions are licensed under the **Apache License 2.0**; see [LICENSE](LICENSE).

The repository also includes third-party components with their own license files:

| Component | License | License file |
|-----------|---------|--------------|
| SGLang | Apache License 2.0 | [sglang/LICENSE](sglang/LICENSE) |
| DeepEP | MIT License | [DeepEP/LICENSE](DeepEP/LICENSE) |
| DeepGEMM | MIT License | [DeepGEMM/LICENSE](DeepGEMM/LICENSE) |

Bundled dependencies retain the license and copyright notices provided in their respective directories.

## Acknowledgments

vExpert builds on [SGLang](https://github.com/sgl-project/sglang) and includes customized versions of [DeepEP](https://github.com/deepseek-ai/DeepEP) and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM). We thank the maintainers and contributors of these projects.
