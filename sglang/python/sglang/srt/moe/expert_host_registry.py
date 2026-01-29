import hashlib
from typing import Dict, Tuple, List
import threading

import torch
import torch.distributed as dist

from sglang.srt.distributed import (
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
)
from ctypes import c_int, c_ulong, c_void_p
from sglang.srt.host_shared_memory import HostSharedMemoryManager
from sglang.srt.distributed.naive_distributed import get_naive_distributed
import logging
import os
import psutil
import numa

logger = logging.getLogger(__name__)
class ExpertHostRegistry:
    def __init__(self, layer_num: int = 58, pinned_expert_num: int = 128, numa_aware: bool = False):
        # key: (layer_id, expert_idx, param_idx) -> (tensor_numa)
        # self._store: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._store: Dict[Tuple[int, int, int],  torch.Tensor] = {}
        #self._create:Dict[int, Tuple[torch.Tensor, torch.Tensor, List]] = {}
        self._create = {}
        self._pinned:Dict[Tuple[int, int, int,int], bool] = {}
        # Thread lock to protect buffer creation from race conditions in multi-threaded weight loading
        self._lock = threading.Lock()
        self._layer_locks: Dict[int, threading.Lock] = {}
        # Use a stable base to avoid naming collision across runs in the same tmp namespace
        world = get_world_group()
        # HardCode
        self._numel = 7168 * 2048  * 3
        self._offset = 7168 * 2048
        self.w13_shape = (4096 * 7168,)
        self.w2_shape  = (7168 * 2048,)
    
    def _get_layer_lock(self, layer_id: int) -> threading.Lock:
        with self._lock:
            if layer_id not in self._layer_locks:
                self._layer_locks[layer_id] = threading.Lock()
            return self._layer_locks[layer_id]
    # One Pinned Memory for W13,W2
    def init_pinned_copies(
        self, layer_id: int, expert_idx: int, param_idx: int, src: torch.Tensor, numa_node: int = 0, pinned_expert_num: int = 128, numa_aware: bool = False, cpu_expert_map: torch.Tensor = None,
    ):
        #expert_idx = expert_idx % 256
        #numa_node = expert_idx % 2
        key = (layer_id, expert_idx, numa_node)
        dtype = src.dtype # torch.float8_e4m3fn
        layer_lock = self._get_layer_lock(layer_id)
        chunk_size = 24 if pinned_expert_num == 256 else 12
        chunk_size = chunk_size * 2 if numa_aware else chunk_size
        with layer_lock:
            if layer_id not in self._create.keys():
                logger.info(f"Create Buffer for Layer {layer_id}") 
                if not numa_aware:
                    experts_per_buffer = pinned_expert_num // 2
                    num_chunks = (experts_per_buffer + chunk_size - 1) // chunk_size  # Round up
                    
                    numa.set_membind([0])
                    buf0_chunks = []
                    for i in range(num_chunks):
                        chunk_experts = min(chunk_size, experts_per_buffer - i * chunk_size)
                        if chunk_experts > 0:
                            chunk = torch.empty((chunk_experts, self._numel), dtype=dtype, device="cpu", pin_memory=True)
                            buf0_chunks.append(chunk)
                    buf0 = buf0_chunks
                    
                    numa.set_membind([1])
                    buf1_chunks = []
                    for i in range(num_chunks):
                        chunk_experts = min(chunk_size, experts_per_buffer - i * chunk_size)
                        if chunk_experts > 0:
                            chunk = torch.empty((chunk_experts, self._numel), dtype=dtype, device="cpu", pin_memory=True)
                            buf1_chunks.append(chunk)
                    buf1 = buf1_chunks
                    numa.set_localalloc()
                    expert_ids = []
                    self._create[layer_id] = (buf0, buf1, expert_ids)
                else:
                    num_chunks = (pinned_expert_num + chunk_size - 1) // chunk_size  # Round up
                    
                    numa.set_membind([0])
                    buf0_chunks = []
                    for i in range(num_chunks):
                        chunk_experts = min(chunk_size, pinned_expert_num - i * chunk_size)
                        if chunk_experts > 0:
                            chunk = torch.empty((chunk_experts, self._numel), dtype=dtype, device="cpu", pin_memory=True)
                            buf0_chunks.append(chunk)
                    buf0 =  buf0_chunks
                    
                    numa.set_membind([1])
                    buf1_chunks = []
                    for i in range(num_chunks):
                        chunk_experts = min(chunk_size, pinned_expert_num - i * chunk_size)
                        if chunk_experts > 0:
                            chunk = torch.empty((chunk_experts, self._numel), dtype=dtype, device="cpu", pin_memory=True)
                            buf1_chunks.append(chunk)
                    buf1 = buf1_chunks
                    numa.set_localalloc()
                    expert_ids = []
                    self._create[layer_id] = (buf0, buf1, expert_ids)
            else:
                buf0,buf1,expert_ids = self._create[layer_id]        
        

        cpu_expert_map_list = cpu_expert_map.tolist()
        expert_local_idx = cpu_expert_map_list.index(expert_idx)
        if not numa_aware:
            # 0-255 -> 0-127
            t_idx = expert_local_idx // 2
            chunk_idx = t_idx // chunk_size
            local_idx = t_idx % chunk_size
            if expert_local_idx % 2 == 0:
                buf = buf0[chunk_idx][local_idx]
            else:
                buf = buf1[chunk_idx][local_idx]
        else:
            t_idx = expert_local_idx
            chunk_idx = t_idx // chunk_size
            local_idx = t_idx % chunk_size
            if numa_node == 0:
                buf = buf0[chunk_idx][local_idx]
            else:
                buf = buf1[chunk_idx][local_idx]
        if param_idx == 0:
            buf[:self._offset].copy_(src.view(-1).detach())
        elif param_idx == 1:
            buf[self._offset: self._offset * 2].copy_(src.view(-1).detach())
        elif param_idx == 2:
            buf[self._offset * 2 :].copy_(src.view(-1).detach())
        self._store[key] = buf

                
    def get_pinned_tensor(self,layer_id: int, expert_idx: int, numa: int = 0):
        #expert_idx = expert_idx % 128
        key = (layer_id, expert_idx, numa)
        if key not in self._store.keys():
            logger.info(f"ExpertHostRegistry: {self._store.keys()}")
            raise ValueError(f"ExpertHostRegistry: No pinned memory for layer {layer_id} expert {expert_idx} numa {numa}")
        return self._store[key]


_GLOBAL_REGISTRY: ExpertHostRegistry | None = None


def get_expert_host_registry() -> ExpertHostRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ExpertHostRegistry()
    return _GLOBAL_REGISTRY


