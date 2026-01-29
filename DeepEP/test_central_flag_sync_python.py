#!/usr/bin/env python3
"""
Test script for DeepEP central_flag_sync_nvshmem function.

This script tests the rank0-centered synchronization mechanism where:
- Rank 0 writes flags to ranks 1-7 using NVSHMEM P2P
- Ranks 1-7 poll their flags until they receive the expected value

Usage:
    python test_central_flag_sync.py
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional
from tests.utils import init_dist

# Add the current directory to Python path to import deep_ep
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import deep_ep
    import deep_ep_cpp
except ImportError as e:
    print(f"Error importing deep_ep: {e}")
    print("Please make sure DeepEP is properly installed and NVSHMEM is available")
    sys.exit(1)


def init_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set CUDA device
    torch.cuda.set_device(rank)
    
    print(f"Rank {rank} initialized on device {torch.cuda.current_device()}")


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def test_central_flag_sync(rank: int, world_size: int):
    """Test the central flag synchronization mechanism."""
    print(f"[Rank {rank}] Starting central flag sync test...")
    
    # Initialize distributed
    rank,num_ranks, group = init_dist(rank, world_size)
    
    try:
        # Create DeepEP buffer (minimal configuration for testing)
        buffer = deep_ep.Buffer(
            group=group,  # Will use the default process group
            num_nvl_bytes=1024 * 1024,  # 1MB NVLink buffer
            num_rdma_bytes=1024 * 1024,  # 1MB RDMA buffer
            low_latency_mode=True,
            explicitly_destroy=True
        )
        
        # Wait for all ranks to initialize
        dist.barrier()
        
        # Create a symmetric flag tensor using NVSHMEM
        # Each rank gets its own flag that can be accessed by other ranks
        recv_flag = torch.zeros(1, dtype=torch.int32, device='cuda')
        
        # Initialize the flag to 0
        recv_flag.zero_()
        
        # Synchronize to ensure all ranks have initialized their flags
        dist.barrier()
        
        print(f"[Rank {rank}] Flag initialized, value: {recv_flag.item()}")
        
        # Test parameters
        expect_value = 12345

        # Start timing
        start_time = time.time()
        for _ in range(100):
            if rank == 0:
                buffer.runtime.central_flag_sync_nvshmem(expect_value, rank)
            else:
                buffer.runtime.central_flag_sync_nvshmem(expect_value, rank)
            
        # Wait for kernel completion
        torch.cuda.synchronize()
        
        end_time = time.time()
        elapsed = (end_time - start_time) / 100
        
        print(f"[Rank {rank}] Sync completed in {elapsed*1e6:.3f} us")
        print(f"[Rank {rank}] Final flag value: {recv_flag.item()}")
        
        # Verify the result
        if rank == 0:
            # Rank 0's flag should remain 0 (it doesn't write to itself)
            expected_final_value = 0
        else:
            # Ranks 1-7 should have received the expected value
            expected_final_value = expect_value
            
        if recv_flag.item() == expected_final_value:
            print(f"[Rank {rank}] ✓ Test PASSED: flag value {recv_flag.item()} matches expected {expected_final_value}")
        else:
            print(f"[Rank {rank}] ✗ Test FAILED: flag value {recv_flag.item()} does not match expected {expected_final_value}")
            
        # Clean up
        buffer.destroy()
        
    except Exception as e:
        print(f"[Rank {rank}] Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        cleanup_distributed()


def main():
    """Main function to run the test."""
    print("=== DeepEP Central Flag Sync Test ===")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)
        
    # Check number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} CUDA devices")
    
    if num_gpus < 2:
        print("Warning: This test requires at least 2 GPUs for meaningful results")
        print("Running with single GPU (simulating multiple ranks)...")
        world_size = 2  # Simulate 2 ranks on single GPU
    else:
        world_size = min(8, num_gpus)  # Use up to 8 GPUs
        
    print(f"Running test with {world_size} ranks")
    
        
    # Run the test
    if world_size == 1:
        # Single process test (for debugging)
        print("Running single-process test...")
        test_central_flag_sync(0, 1)
    else:
        # Multi-process test
        print(f"Spawning {world_size} processes...")
        mp.spawn(
            test_central_flag_sync,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        
    print("=== Test completed ===")


if __name__ == "__main__":
    main()
