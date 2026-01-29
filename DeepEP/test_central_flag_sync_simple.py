#!/usr/bin/env python3
"""
Simple test script for DeepEP central_flag_sync_nvshmem function on single GPU.

This script tests the rank0-centered synchronization mechanism using multiple CUDA streams
to simulate different ranks on the same GPU.

Usage:
    python test_central_flag_sync_simple.py
"""

import os
import sys
import time
import torch
import threading
from typing import List

# Add the current directory to Python path to import deep_ep
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import deep_ep
    import deep_ep_cpp
except ImportError as e:
    print(f"Error importing deep_ep: {e}")
    print("Please make sure DeepEP is properly installed")
    sys.exit(1)


def test_central_flag_sync_single_gpu():
    """Test central flag sync on single GPU using multiple streams."""
    print("=== DeepEP Central Flag Sync Test (Single GPU) ===")
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return False
        
    device = torch.cuda.current_device()
    print(f"Using CUDA device {device}")
    
    # Create a shared flag tensor
    recv_flag = torch.zeros(1, dtype=torch.int32, device='cuda')
    expect_value = 12345
    
    print(f"Initial flag value: {recv_flag.item()}")
    
    # Create multiple streams to simulate different ranks
    num_streams = 8
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    
    # Create DeepEP buffers for each "rank"
    buffers = []
    for i in range(num_streams):
        # Create a minimal buffer for testing
        buffer = deep_ep.Buffer(
            group=None,
            num_nvl_bytes=1024 * 1024,  # 1MB
            num_rdma_bytes=0,  # No RDMA for single GPU test
            low_latency_mode=False,
            explicitly_destroy=True
        )
        buffers.append(buffer)
    
    print(f"Created {num_streams} streams and buffers")
    
    # Test results
    results = [False] * num_streams
    start_time = time.time()
    
    def run_rank(rank: int):
        """Run the sync test for a specific rank."""
        stream = streams[rank]
        buffer = buffers[rank]
        
        with torch.cuda.stream(stream):
            try:
                if rank == 0:
                    # Rank 0: Wait a bit, then write flags
                    print(f"[Rank {rank}] Waiting 1 second before writing flags...")
                    time.sleep(1)
                    
                    print(f"[Rank {rank}] Writing flags to ranks 1-7 with value {expect_value}")
                    buffer.central_flag_sync_nvshmem(recv_flag, expect_value)
                    
                else:
                    # Ranks 1-7: Start polling immediately
                    print(f"[Rank {rank}] Starting to poll flag, expecting value {expect_value}")
                    buffer.central_flag_sync_nvshmem(recv_flag, expect_value)
                
                # Wait for completion
                stream.synchronize()
                
                # Check result
                final_value = recv_flag.item()
                if rank == 0:
                    expected = 0  # Rank 0 doesn't write to itself
                else:
                    expected = expect_value
                    
                if final_value == expected:
                    print(f"[Rank {rank}] ✓ PASSED: flag value {final_value} matches expected {expected}")
                    results[rank] = True
                else:
                    print(f"[Rank {rank}] ✗ FAILED: flag value {final_value} does not match expected {expected}")
                    results[rank] = False
                    
            except Exception as e:
                print(f"[Rank {rank}] Error: {e}")
                results[rank] = False
    
    # Start all ranks
    threads = []
    for rank in range(num_streams):
        thread = threading.Thread(target=run_rank, args=(rank,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nTest completed in {elapsed:.3f} seconds")
    print(f"Final flag value: {recv_flag.item()}")
    
    # Check overall results
    passed = sum(results)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} ranks passed")
    
    if passed == total:
        print("✓ Overall test PASSED")
        return True
    else:
        print("✗ Overall test FAILED")
        return False
    
    # Clean up
    for buffer in buffers:
        buffer.destroy()


def test_basic_functionality():
    """Test basic DeepEP functionality without sync."""
    print("\n=== Testing Basic DeepEP Functionality ===")
    
    try:
        # Create a simple buffer
        buffer = deep_ep.Buffer(
            group=None,
            num_nvl_bytes=1024 * 1024,
            num_rdma_bytes=0,
            low_latency_mode=False,
            explicitly_destroy=True
        )
        
        print("✓ DeepEP Buffer created successfully")
        
        # Test tensor operations
        test_tensor = torch.zeros(1, dtype=torch.int32, device='cuda')
        print(f"✓ Test tensor created: {test_tensor.item()}")
        
        # Clean up
        buffer.destroy()
        print("✓ Buffer destroyed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("DeepEP Central Flag Sync Test")
    print("=" * 40)
    
    # Test basic functionality first
    if not test_basic_functionality():
        print("Basic functionality test failed, skipping sync test")
        return
    
    # Test central flag sync
    success = test_central_flag_sync_single_gpu()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()


