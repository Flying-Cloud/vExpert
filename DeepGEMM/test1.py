import torch
import time

def test_tensor_copy():
    """Test the time required to copy a 7168 * 2048 * 3 * 8 tensor on GPU"""
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
    
    device = 'cuda'
    
    # Tensor shape: 7168 * 2048 * 3 * 8
    shape = (7168, 2048, 3, 8)
    
    # Calculate tensor size (bytes)
    num_elements = 7168 * 2048 * 3 * 8
    dtype = torch.float32
    element_size = 4  # float32 is 4 bytes
    total_bytes = num_elements * element_size
    
    print(f"Tensor shape: {shape}")
    print(f"Total elements: {num_elements:,}")
    print(f"Total size: {total_bytes / (1024**3):.4f} GB ({total_bytes / (1024**2):.2f} MB)")
    print(f"Data type: {dtype}")
    print(f"Device: {device}")
    print("-" * 60)
    
    # Create source and destination tensors
    src_tensor = torch.randn(shape, dtype=dtype, device=device)
    dst_tensor = torch.empty_like(src_tensor)
    
    # Synchronize to ensure initialization is complete
    torch.cuda.synchronize()
    
    # Warmup
    num_warmups = 10
    print(f"Warming up {num_warmups} times...")
    for _ in range(num_warmups):
        dst_tensor.copy_(src_tensor)
    torch.cuda.synchronize()
    
    # Test copy time
    num_tests = 100
    print(f"Running {num_tests} tests...")
    
    # Use CUDA Event for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_tests):
        dst_tensor.copy_(src_tensor)
    end_event.record()
    torch.cuda.synchronize()
    
    # Calculate average time
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_tests
    avg_time_us = avg_time_ms * 1000
    
    # Calculate bandwidth
    bandwidth_gb_per_s = (total_bytes / (1024**3)) / (avg_time_ms / 1000)
    bandwidth_mb_per_s = (total_bytes / (1024**2)) / (avg_time_ms / 1000)
    
    print("-" * 60)
    print("Test results:")
    print(f"  Average copy time: {avg_time_ms:.4f} ms ({avg_time_us:.2f} μs)")
    print(f"  Total test time: {elapsed_time_ms:.2f} ms")
    print(f"  Bandwidth: {bandwidth_gb_per_s:.2f} GB/s ({bandwidth_mb_per_s:.2f} MB/s)")
    print("-" * 60)
    
    # Additional tests: using different copy methods
    print("\nTesting different copy methods:")
    
    # Method 1: copy_
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_tests):
        dst_tensor.copy_(src_tensor)
    end_event.record()
    torch.cuda.synchronize()
    time_copy = start_event.elapsed_time(end_event) / num_tests
    print(f"  copy_(): {time_copy:.4f} ms")
    
    # Method 2: direct assignment
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_tests):
        dst_tensor = src_tensor.clone()
    end_event.record()
    torch.cuda.synchronize()
    time_clone = start_event.elapsed_time(end_event) / num_tests
    print(f"  clone(): {time_clone:.4f} ms")
    
    # Method 3: using index copy
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_tests):
        dst_tensor[:] = src_tensor
    end_event.record()
    torch.cuda.synchronize()
    time_slice = start_event.elapsed_time(end_event) / num_tests
    print(f"  [:] assignment: {time_slice:.4f} ms")
    
    return avg_time_ms, bandwidth_gb_per_s


if __name__ == '__main__':
    print("=" * 60)
    print("GPU Tensor Copy Performance Test")
    print("=" * 60)
    test_tensor_copy()

