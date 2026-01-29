import logging
import os
from dataclasses import dataclass
import ctypes
from ctypes import c_int, c_ulong, c_void_p
from multiprocessing import shared_memory
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.distributed.naive_distributed import get_naive_distributed
from sglang.srt.utils import check_cuda_result

logger = logging.getLogger(__name__)


class HostSharedMemoryManager:
    def __init__(self, base_name: str):
        self._base_name = Path(base_name)
        self._operation_index = 0
        self._records: List[_Record] = []
        self._libnuma = None
        try:
            self._libnuma = ctypes.CDLL("libnuma.so.1")
            self._libnuma.numa_available.restype = c_int
            self._libnuma.numa_set_preferred.argtypes = [c_int]
            self._libnuma.numa_set_strict.argtypes = [c_int]
        except OSError:
            logger.info(f"Fail to access libNuma !")
            self._libnuma = None

    def malloc(self, *, shape, dtype):
        meta_tensor = torch.empty(size=shape, dtype=dtype, device="meta")
        raw = self._malloc_raw(num_bytes=meta_tensor.nbytes)
        return raw.view(dtype).view(*shape)

    def _malloc_raw(self, *, num_bytes: int) -> torch.Tensor:
        import cuda.bindings.runtime as cuda_rt

        self._operation_index += 1
        shm_name = f"{self._base_name}_op{self._operation_index}"

        # TODO handle dispose
        if get_naive_distributed().get_rank() == 0:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=num_bytes)

        get_naive_distributed().barrier()

        if get_naive_distributed().get_rank() != 0:
            shm = shared_memory.SharedMemory(name=shm_name)

        np_array = np.ndarray((num_bytes,), dtype=np.uint8, buffer=shm.buf)
        tensor = torch.from_numpy(np_array)

        check_cuda_result(
            cuda_rt.cudaHostRegister(
                tensor.data_ptr(), num_bytes, cuda_rt.cudaHostRegisterPortable
            )
        )

        get_naive_distributed().barrier()

        self._records.append(
            _Record(
                shm=shm,
                np_array=np_array,
                tensor=tensor,
            )
        )
        return tensor

    def _maybe_bind_numa(self, numa_node: int | None):
        if self._libnuma is None or numa_node is None:
            return None
        if self._libnuma.numa_available() == -1:
            return None
        current_numa_node = self._libnuma.numa_preferred()
        # Use preferred policy to bias allocation to the node
        self._libnuma.numa_set_preferred(c_int(numa_node))
        self._libnuma.numa_set_strict(c_int(1))
        return current_numa_node

    def _touch_pages(self, np_array: "np.ndarray"):
        page_size = os.sysconf("SC_PAGESIZE")
        n = len(np_array)
        # step by page to enforce physical allocation on current NUMA policy
        for offset in range(0, n, page_size):
            np_array[offset] = np_array[offset]

    def malloc_named_create(self, *, name: str, num_bytes: int, numa_node: int | None = None, w13: bool = False, w13_idx: int = 0) -> torch.Tensor:
        import cuda.bindings.runtime as cuda_rt

        shm_name = name
        prev_policy_active = self._maybe_bind_numa(numa_node)
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
        except FileNotFoundError:
            shm = shared_memory.SharedMemory(name=shm_name, create= True, size = num_bytes)
        
        # Method 1: Cost 1GB-2GB VRAM
        np_array = np.ndarray((num_bytes,), dtype=np.uint8, buffer=shm.buf)
        
        # Touch pages to fault-in on current NUMA node
        try:
            self._touch_pages(np_array)
        except Exception:
            pass
        
        tensor = torch.from_numpy(np_array)
        
        check_cuda_result(
            cuda_rt.cudaHostRegister(
                tensor.data_ptr(), num_bytes, cuda_rt.cudaHostRegisterPortable
            )
        )
        
        if prev_policy_active is not None and self._libnuma is not None:
            self._libnuma.numa_set_preferred(c_int(prev_policy_active))
            self._libnuma.numa_set_strict(c_int(0))
        return tensor

    def malloc_named_init(self, *, name: str, num_bytes: int, numa_node: int | None = None, w13: bool = False, w13_idx: int = 0) -> torch.Tensor:
        import cuda.bindings.runtime as cuda_rt

        shm_name = name
        prev_policy_active = self._maybe_bind_numa(numa_node)

        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=num_bytes)

        # Method 1: Cost 1GB-2GB VRAM
        np_array = np.ndarray((num_bytes,), dtype=np.uint8, buffer=shm.buf)
        # Touch pages to fault-in on current NUMA node
        
        try:
            self._touch_pages(np_array)
        except Exception:
            pass
        
        tensor = torch.from_numpy(np_array)
        
        check_cuda_result(
            cuda_rt.cudaHostRegister(
                tensor.data_ptr(), num_bytes, cuda_rt.cudaHostRegisterPortable
            )
        )
        
        if prev_policy_active is not None and self._libnuma is not None:
            self._libnuma.numa_set_preferred(c_int(prev_policy_active))
            self._libnuma.numa_set_strict(c_int(0))
        return tensor

    def malloc_named_virtual(self, *, name: str, num_bytes: int, numa_node: int | None = None) -> torch.Tensor:
        import cuda.bindings.runtime as cuda_rt

        shm_name = name
        prev_policy_active = self._maybe_bind_numa(numa_node)

        shm = shared_memory.SharedMemory(name=shm_name,create=False)
        
        np_array = np.ndarray((num_bytes,), dtype=np.uint8, buffer=shm.buf)
        tensor = torch.from_numpy(np_array)
        check_cuda_result(
            cuda_rt.cudaHostRegister(
                tensor.data_ptr(), num_bytes, cuda_rt.cudaHostRegisterPortable
            )
        )
        if prev_policy_active is not None and self._libnuma is not None:
            self._libnuma.numa_set_preferred(c_int(prev_policy_active))
            self._libnuma.numa_set_strict(c_int(0))
        return tensor
    

    def malloc_on_numa_create(self, *, shape, dtype, name: str, numa_node: int | None,w13: bool = False, w13_idx: int = 0, pinned: bool = False) -> torch.Tensor:
        if w13:
            shape = (shape[0] * 2, shape[1])
        meta_tensor = torch.empty(size=shape, dtype=dtype, device="meta")
        raw = self.malloc_named_create(name=name, num_bytes=meta_tensor.nbytes, numa_node=numa_node, w13 = w13, w13_idx = w13_idx, pinned = pinned)
        return raw.view(dtype).view(*shape)

    def malloc_on_numa_init(self, *, shape, dtype, name: str, numa_node: int | None,w13: bool = False, w13_idx: int = 0) -> torch.Tensor:
        if w13:
            shape = (shape[0] * 2, shape[1])
        meta_tensor = torch.empty(size=shape, dtype=dtype, device="meta")
        raw = self.malloc_named_init(name=name, num_bytes=meta_tensor.nbytes, numa_node=numa_node, w13 = w13, w13_idx = w13_idx)
        return raw.view(dtype).view(*shape)

    def malloc_on_numa_virtual(self, *, shape, dtype, name: str, numa_node: int | None) -> torch.Tensor:
        meta_tensor = torch.empty(size=shape, dtype=dtype, device="meta")
        raw = self.malloc_named_virtual(name=name, num_bytes=meta_tensor.nbytes, numa_node=numa_node)
        return raw.view(dtype).view(*shape)


@dataclass
class _Record:
    shm: shared_memory.SharedMemory
    np_array: np.ndarray
    tensor: torch.Tensor


# Can have multi instances if needed
_instance: Optional[HostSharedMemoryManager] = None


def get_host_shared_memory_manager():
    assert _instance is not None
    return _instance


def set_host_shared_memory_manager(instance: HostSharedMemoryManager):
    global _instance
    assert _instance is None
    _instance = instance
