import ctypes
import logging
from contextlib import contextmanager
from typing import List, Optional, Sequence
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.utils import is_cuda, get_available_gpu_memory
from sglang.srt.moe.expert_host_registry import get_expert_host_registry, ExpertHostRegistry
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.managers.schedule_batch import global_server_args_dict

from deep_ep import Buffer, Config
import semi_pd_ipc
logger = logging.getLogger(__name__)
class MultiMemCpyNodeManager:
    """
    Manages source address updates for multiple MemCpyNodes in CUDA graph
    
    Used for CustomH2DAsyncCpy scenario:
    - Each layer executes num_devices * vexpert_num_per_buffer * 2 MemCpyNodes
    - Each expert has w13 and w2 copies
    - Need to dynamically update source addresses of all nodes based on id_pairs
    """
    def __init__(self, graph: Optional[torch.cuda.CUDAGraph] = None, expert_tensors: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None, 
                 num_devices: Optional[int] = None, vexpert_num_per_buffer: Optional[int] = None, registry: Optional[ExpertHostRegistry] = None,
                 rank: Optional[int] = None):
        """
        Args:
            graph: Captured CUDA graph
            expert_tensors: expert_tensors[pingpong][offset] = (w13_tensor, w2_tensor)
            num_devices: Number of devices
            vexpert_num_per_buffer: Number of experts per buffer
            registry: expert host registry, used to get host tensor
        """
        self.graphs = {}
        self.expert_tensors = expert_tensors
        self.num_devices = num_devices
        self.vexpert_num_per_buffer = vexpert_num_per_buffer
        self.registry = registry
        self.rank = rank
        self.lib = CudaRTLibrary()
        
        # Get graph handles
        self.graph_handles = {}
        self.graph_execs = {}
    
        # Extract all H2D memcpy nodes and build mapping
        self.copy_nodes = {}  # All node handles (in execution order)
        self.copy_params_list = {}  # All node parameters
        self.dst_ptr_to_node_list = {}  # dstPtr -> List[(node_idx, device_id, offset, is_w2, src_ptr_captured, exec_order)]
        # For fast lookup, also keep a mapping: exec_order -> (node_idx, device_id, offset, is_w2, dst_ptr, src_ptr_captured)
        self.exec_order_to_node_info = {}  # exec_order -> (node_idx, device_id, offset, is_w2, dst_ptr, src_ptr_captured)
        self.layer_counts = {}
        self.numa_aware = global_server_args_dict.get("numa_aware", False)
    
    def set_graph(self, graphs):
        self.graphs = graphs
        self.graph_handles = {bs: graph.raw_cuda_graph() for bs, graph in graphs.items()}
    
    def _extract_and_map_nodes(self):
        """
        Extract all H2D memcpy nodes and build mapping
        Handle cases where multiple nodes correspond to the same dst_ptr (distinguished by execution order)
        """
        if self.rank == 0:
            for bs, graph in self.graphs.items():
                self.graph_execs[bs] = graph.raw_cuda_graph_exec()
            for bs, graph_handle in self.graph_handles.items():
                # Get all nodes (in execution order)
                nodes = self.lib.cudaGraphGetNodes(graph_handle)
                # Traverse all nodes, find H2D memcpy nodes, record execution order
                h2d_nodes = []
                exec_order = 0
                layer_order = 0
                for node in nodes:
                    layer_order = exec_order // (self.num_devices * self.vexpert_num_per_buffer * 2)
                    offset      = exec_order %  (self.num_devices * self.vexpert_num_per_buffer * 2)
                    device_id   = offset // (2 * self.vexpert_num_per_buffer)
                    vexpert_id  = (offset // 2) % self.vexpert_num_per_buffer
                    is_w13       = offset % 2 == 0
                    node_type = self.lib.cudaGraphNodeGetType(node)
                    if node_type == 1:  # CUDA_GRAPH_NODE_TYPE_MEMCPY
                        params = self.lib.cudaGraphMemcpyNodeGetParams(node)
                        if params.kind == 1:  # cudaMemcpyHostToDevice
                            dst_ptr_value = params.dstPtr.ptr.value if hasattr(params.dstPtr.ptr, 'value') else int(params.dstPtr.ptr)
                            src_ptr_value = params.srcPtr.ptr.value if hasattr(params.srcPtr.ptr, 'value') else int(params.srcPtr.ptr)
                            h2d_nodes.append((node, params, dst_ptr_value, src_ptr_value, layer_order, device_id, vexpert_id, is_w13, offset))
                            exec_order += 1
                
                # Initialize dst_ptr_to_node_list (each dst_ptr corresponds to a node list)
                dst_ptr_to_node_list = {}
                copy_params_list = []
                copy_nodes = []
                exec_order_to_node_info = {}
                layer_counts = 0
                for pingpong in range(2):
                    for offset in range(len(self.expert_tensors[pingpong])):
                        w13_tensor, w2_tensor = self.expert_tensors[pingpong][offset]
                        w13_dst_ptr = ctypes.c_void_p(w13_tensor.data_ptr()).value
                        w2_dst_ptr = ctypes.c_void_p(w2_tensor.data_ptr()).value
                        if w13_dst_ptr not in dst_ptr_to_node_list:
                            dst_ptr_to_node_list[w13_dst_ptr] = []
                        if w2_dst_ptr not in dst_ptr_to_node_list:
                            dst_ptr_to_node_list[w2_dst_ptr] = []
                
                # Match nodes: for each h2d_node, find corresponding (device_id, offset, is_w2)
                matched_nodes = set()
                for node, params, dst_ptr_value, src_ptr_value, layer_order, device_id0, vexpert_id, is_w13, offset0 in h2d_nodes:
                    if node in matched_nodes:
                        continue
                    
                    # Find matching (device_id, offset, is_w2)
                    matched = False
                    for pingpong in range(2):
                        for offset in range(len(self.expert_tensors[pingpong])):
                            w13_tensor, w2_tensor = self.expert_tensors[pingpong][offset]
                            device_id  = offset // self.vexpert_num_per_buffer
                            vexpert_id = offset % self.vexpert_num_per_buffer
                            
                            w13_dst_ptr = ctypes.c_void_p(w13_tensor.data_ptr()).value
                            w2_dst_ptr = ctypes.c_void_p(w2_tensor.data_ptr()).value
                            
                            if dst_ptr_value == w13_dst_ptr:
                                node_idx = len(copy_nodes)
                                copy_nodes.append(node)
                                params_copy = self.lib.cudaGraphMemcpyNodeGetParams(node)
                                copy_params_list.append(params_copy)
                                
                                # Add to dst_ptr_to_node_list (support multiple nodes corresponding to same dst_ptr)
                                node_info = (node_idx, device_id, vexpert_id, False, src_ptr_value, layer_order)
                                dst_ptr_to_node_list[dst_ptr_value].append(node_info)
                                exec_order_to_node_info[layer_order] = (node_idx, device_id, vexpert_id, False, dst_ptr_value, src_ptr_value)
                                matched_nodes.add(node)
                                matched = True
                                break
                            elif dst_ptr_value == w2_dst_ptr:
                                node_idx = len(copy_nodes)
                                copy_nodes.append(node)
                                params_copy = self.lib.cudaGraphMemcpyNodeGetParams(node)
                                copy_params_list.append(params_copy)
                                
                                node_info = (node_idx, device_id, vexpert_id, True, src_ptr_value, layer_order)
                                dst_ptr_to_node_list[dst_ptr_value].append(node_info)
                                exec_order_to_node_info[layer_order] = (node_idx, device_id, vexpert_id, True, dst_ptr_value, src_ptr_value)
                                matched_nodes.add(node)
                                matched = True
                                break
                        
                    if not matched:
                        logger.info(f"Warning: Unmatched node (dst_ptr={hex(dst_ptr_value)}, src_ptr={hex(src_ptr_value)}, exec_order={exec_order})")
                
                # Sort node list for each dst_ptr by execution order
                for dst_ptr in dst_ptr_to_node_list:
                    dst_ptr_to_node_list[dst_ptr].sort(key=lambda x: x[5])  # Sort by layer_order
                
                logger.info(f"Extracted {len(copy_nodes)} H2D memcpy nodes")
                # Count number of nodes corresponding to each dst_ptr
                layer_counts = exec_order // (self.num_devices * self.vexpert_num_per_buffer * 2)
                self.dst_ptr_to_node_list[bs] = dst_ptr_to_node_list
                self.copy_params_list[bs] = copy_params_list
                self.copy_nodes[bs] = copy_nodes
                self.exec_order_to_node_info[bs] = exec_order_to_node_info
                self.layer_counts[bs] = layer_counts
    def update_src_ptrs_from_id_pairs(self, layers_id_pairs: List[List[Tuple[int, int]]], bs: int, layers_pairs_valid: List[Tuple[int, int]]):
        """
        Update source addresses of all MemCpyNodes based on id_pairs
        Handle cases where multiple nodes correspond to the same dst_ptr (distinguished by layer_exec_order)
        
        Args:
            id_pairs: List[Tuple[int, int]], length is num_devices * vexpert_num_per_buffer
                     Each element is (layer_id, expert_id)
            pingpong: pingpong buffer index
            layer_exec_order: Current layer's execution order in graph (used to distinguish multiple nodes with same dst_ptr)
                            Default is 0, if multiple layers use same expert_tensors, need to pass correct order
        """
        
        # Update source address for each node
        if self.rank == 0:
            updated_count = 0
            layer_counts = self.layer_counts[bs]
            dst_ptr_to_node_list = self.dst_ptr_to_node_list[bs]
            copy_params_list = self.copy_params_list[bs]
            copy_nodes = self.copy_nodes[bs]
            exec_order_to_node_info = self.exec_order_to_node_info[bs]
            len_id_pairs = self.vexpert_num_per_buffer
            for layer_order in range(layer_counts):
                # 0~29
                pingpong = layer_order % 2
                id_pairs= layers_id_pairs[layer_order]
                vexpert_count1, vexpert_count2 = layers_pairs_valid[layer_order]
                valid_count = vexpert_count1 + vexpert_count2
                unvalid_range = list(range(vexpert_count1, len_id_pairs - vexpert_count2))
                for d in range(self.num_devices):
                    offset = d * self.vexpert_num_per_buffer
                    for i in range(valid_count):
                        if i < vexpert_count1:
                            layer_id, expert_id = id_pairs[d * len_id_pairs + i]
                        else:
                            ii = i - vexpert_count1
                            layer_id, expert_id = id_pairs[d * len_id_pairs + len_id_pairs - vexpert_count2 + ii]
                        numa_id = 0 if d <= 3 else 1
                        numa_id = numa_id if self.numa_aware else 0
                        w13_tensor, w2_tensor = self.expert_tensors[pingpong][offset]
                        w13_dst_ptr = ctypes.c_void_p(w13_tensor.data_ptr()).value
                        w2_dst_ptr = ctypes.c_void_p(w2_tensor.data_ptr()).value    
                        element_size = w13_tensor.element_size()
                        w13_size_bytes = self.registry._offset * 2 * element_size
                        w2_size_bytes = self.registry._offset * element_size
                        
                        node_list = dst_ptr_to_node_list[w13_dst_ptr]
                        node_idx,_ ,_,_,_,_ = node_list[(layer_order - pingpong) // 2]
                        params = copy_params_list[node_idx]
                        if expert_id != -1:
                            host_pinned_tensor = self.registry.get_pinned_tensor(layer_id, expert_id, numa_id)
                            host_ptr = host_pinned_tensor.data_ptr()
                            params.srcPtr.ptr = ctypes.c_void_p(host_ptr)
                            params.extent.width = w13_size_bytes
                        else:
                            #logger.info(f"Expert_Id -1, Before Update: {params.extent.width}, {params.extent.height}, {params.extent.depth}")
                            params.extent.width = 8
                        self.lib.cudaGraphMemcpyNodeSetParams(copy_nodes[node_idx], params)
                        updated_count += 1
                        
                        node_list = dst_ptr_to_node_list[w2_dst_ptr]
                        node_idx,_ ,_,_,_,_ = node_list[(layer_order - pingpong) // 2]
                        params = copy_params_list[node_idx]
                        if expert_id != -1:
                            host_pinned_tensor = self.registry.get_pinned_tensor(layer_id, expert_id, numa_id)
                            host_ptr = host_pinned_tensor.data_ptr()
                            params.srcPtr.ptr = ctypes.c_void_p(host_ptr + w13_size_bytes)
                            params.extent.width = w2_size_bytes
                        else:
                            #logger.info(f"Expert_Id -1, Before Update: {params.extent.width}, {params.extent.height}, {params.extent.depth}")
                            params.extent.width = 8
                        self.lib.cudaGraphMemcpyNodeSetParams(copy_nodes[node_idx], params)
                        updated_count += 1
                        offset += 1
                    for i in range(self.vexpert_num_per_buffer - valid_count):
                        w13_tensor, w2_tensor = self.expert_tensors[pingpong][offset]
                        w13_dst_ptr = ctypes.c_void_p(w13_tensor.data_ptr()).value
                        w2_dst_ptr = ctypes.c_void_p(w2_tensor.data_ptr()).value    
                        element_size = w13_tensor.element_size()
                        w13_size_bytes = self.registry._offset * 2 * element_size
                        w2_size_bytes = self.registry._offset * element_size
                        
                        node_list = dst_ptr_to_node_list[w13_dst_ptr]
                        node_idx,_ ,_,_,_,_ = node_list[(layer_order - pingpong) // 2]
                        params = copy_params_list[node_idx]
                        params.extent.width = 8
                        self.lib.cudaGraphMemcpyNodeSetParams(copy_nodes[node_idx], params)
                        updated_count += 1
                        
                        node_list = dst_ptr_to_node_list[w2_dst_ptr]
                        node_idx,_ ,_,_,_,_ = node_list[(layer_order - pingpong) // 2]
                        params = copy_params_list[node_idx]
                        params.extent.width = 8
                        self.lib.cudaGraphMemcpyNodeSetParams(copy_nodes[node_idx], params)
                        updated_count += 1
                        offset += 1
            if updated_count > 0:
                self.lib.cudaGraphExecUpdate(self.graph_execs[bs], self.graph_handles[bs])
            else:
                logger.info(f"Warning: No nodes were updated!")
        
DTYPE_TO_ATEN = {
    torch.float32: "at::kFloat",
    torch.float64: "at::kDouble",
    torch.float16: "at::kHalf",
    torch.int64: "at::kLong",
    torch.int32: "at::kInt",
    torch.int16: "at::kShort",
    torch.int8: "at::kChar",
    torch.uint64: "at::kUInt64",
    torch.uint32: "at::kUInt32",
    torch.uint16: "at::kUInt16",
    torch.uint8: "at::kByte",
    torch.uint32: "at::kUInt32",
    torch.uint64: "at::kUInt64",
    torch.bool: "at::kBool",
    torch.bfloat16: "at::kBFloat16",
    torch.complex32: "at::kComplexHalf",
    torch.complex64: "at::kComplexFloat",
    torch.complex128: "at::kComplexDouble",
    torch.float8_e4m3fn: "at::kFloat8_e4m3fn",
    torch.float8_e5m2: "at::kFloat8_e5m2",
    torch.float8_e4m3fnuz: "at::kFloat8_e4m3fnuz",
    torch.float8_e5m2fnuz: "at::kFloat8_e5m2fnuz",
}
def create_shared_buffer(
    size_in_bytes: int, group: Optional[ProcessGroup] = None
) -> List[int]:
    """
    Creates a shared buffer and returns a list of pointers
    representing the buffer on all processes in the group.
    """
    lib = CudaRTLibrary()
    pointer = lib.cudaMalloc(size_in_bytes)
    handle = lib.cudaIpcGetMemHandle(pointer)
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    handles = [None] * world_size
    dist.all_gather_object(handles, handle, group=group)

    pointers: List[int] = []
    for i, h in enumerate(handles):
        if i == rank:
            pointers.append(pointer.value)  # type: ignore
        else:
            pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore

    return pointers
class CustomH2DAsyncCpy:
    """
    Rank0 process is responsible for:
    - Allocating target device tensors on local cards 0..N-1 (or reusing existing tensors), and initiating async H2D copy from pinned memory;
    - Broadcasting dst device pointers to other processes via CUDA IPC, making them visible to other processes;

    Provides two interfaces:
    - start_async_copies(host_tensors): Start async copy on each card, using independent stream and recording events;
    - wait_all(): Wait for all copies to complete via events or stream (Graph compatible, does not call torch.cuda.synchronize).

    Notes:
    - Target device buffers use PyTorch tensors allocated on corresponding devices, and get handles via libcudart's cudaIpcGetMemHandle.
      This relies on PyTorch using cudaMalloc, satisfying CUDA IPC requirements;
    - Non-Rank0 processes open via cudaIpcOpenMemHandle, visible to same physical memory (no need to wrap pointers as tensors in this class).
    """

    def __init__(self, 
                 group: ProcessGroup = None, 
                 device: Union[int, str, torch.device] = None,
                 num_devices: int = None) -> None:
        assert group is not None
        assert device is not None
        assert num_devices is not None
        self.group = group
        self.num_devices = num_devices
        self.vexpert_num_per_buffer = global_server_args_dict["num_vexpert_buffer"]
        self.numa_aware = global_server_args_dict.get("numa_aware", False)
        self.layer_num_per_buffer = 2
        self.registry = get_expert_host_registry()
        self.buffer_size = self.vexpert_num_per_buffer * self.registry._numel # expert * (w13 + w2)
        self.rank = dist.get_rank(group=group) % 8
        self.local_rank_group = [i for i in range(dist.get_rank(group=group) // 8 * 8, dist.get_rank(group=group) // 8 * 8 + 8)]
        self.world_size = dist.get_world_size(group=group)
        self.pingpong = 0
        #logger.info(f"rank:{self.rank}, world_size:{self.world_size}")
        if not is_cuda():
            logger.warning("CustomH2DAsyncCpy disabled: CUDA not available.")
            return

        self.device = torch.device(f"cuda:{self.rank}")
        
        self.lib = CudaRTLibrary()      
        # PingPong Buffer, register as pytorch tensor
        # (layer_id - 3) // 2 % 2
        if global_server_args_dict.get("enable_cpu_pinned", False):
            self.storage_tensors0, self.expert_tensors0 = self.create_shared_buffer_v0(
                size_in_bytes=self.buffer_size, group=self.group)
            self.storage_tensors1, self.expert_tensors1 = self.create_shared_buffer_v0(
                size_in_bytes=self.buffer_size, group=self.group)
            self.storage_tensors = [self.storage_tensors0, self.storage_tensors1]
            self.expert_tensors = [self.expert_tensors0, self.expert_tensors1]
            self.memcpy_manager = MultiMemCpyNodeManager(None, self.expert_tensors, self.num_devices, self.vexpert_num_per_buffer, self.registry, self.rank)
            self.empty_pinned = torch.empty((128,), dtype=fp8_dtype, device="cpu", pin_memory=True)
        self.events: List[Optional[torch.cuda.Event]] = [None] * self.num_devices
        if self.rank == 0:
            for i in range(self.num_devices):
                self.events[i] = torch.cuda.Event(blocking=False)

        self.streams: List[Optional[torch.cuda.Stream]] = [None] * self.num_devices
        if self.rank == 0:
            for i in range(self.num_devices):
                # Use high priority stream for GPU0 to reduce resource contention
                # High priority stream may get better DMA engine and PCIe bandwidth allocation
                self.streams[i] = torch.cuda.Stream(device=i)

        self._is_capturing = False
    def set_deepep_buffer(self, buffer: Buffer = None):
        assert buffer is not None
        self.buffer = buffer
    def set_graph(self, graph: torch.cuda.CUDAGraph):
        self.memcpy_manager.set_graph(graph)
    # Rank0 Create Buffer and make it visiable to other Rank

    
    def create_shared_buffer_v0(
        self, size_in_bytes: int, group: Optional[ProcessGroup] = None
    ) -> List[int]:
        """
        Creates a shared buffer and returns a list of pointers and handles
        representing the buffer on all processes in the group.
        """
        world_size = dist.get_world_size(group=group)
        handles = [None] * world_size
        # Implement using Pytorch API
        # storage = torch.UntypedStorage(size_in_bytes).cuda()
        #storage_tensor = torch.empty(0, dtype=fp8_dtype, device=self.device
        #                ).set_(storage, storage_offset=0, size=(size_in_bytes,))
        storage_tensor_local = torch.empty(size_in_bytes, dtype=fp8_dtype, device=self.device)

        offs = storage_tensor_local.untyped_storage()._share_cuda_()[3]
        handle = (semi_pd_ipc.get_ipc_handle(storage_tensor_local), offs)
        
        #handle = storage_tensor_local.untyped_storage()._share_cuda_()
        dist.all_gather_object(handles, handle, group=group)
        storage_tensors = []
        expert_tensors = []
        for i,h in enumerate(handles):
            if i not in self.local_rank_group:
                continue
            local_i = i % 8
            target_device = torch.device(f"cuda:{local_i}")
            # SharedStorage: layer_num_per_buffer * vexpert_num_per_layer * expert_size
            # (int device, bytes handle, int storage_size_bytes, int storage_offset_bytes, bytes _ref_counter, int _ref_counter_offset, bytes event_handle, bool event_sync_required)
            #shared_storage = torch.UntypedStorage._new_shared_cuda(*h)
            if self.rank == 0:
                torch.cuda.set_device(target_device)
                if local_i == self.rank:
                    storage_tensor_all = storage_tensor_local
                else:
                    dtype_str = DTYPE_TO_ATEN[fp8_dtype]
                    storage_tensor_all = semi_pd_ipc.convert_ipc_handle_to_tensor(h, size_in_bytes, dtype_str, target_device)       
                
                shared_storage = storage_tensor_all.untyped_storage()
                #logger.info("Create Shared Buffer Success")
                storage_tensor = shared_storage
                for j in range(self.vexpert_num_per_buffer):
                    # Load j-th virtual expert
                    # expert_offset = j * self.registry._numel
                    w13_offset    = (self.registry._numel // 3) * 2 * j
                    w2_offset     = (self.registry._numel // 3) * 2 * self.vexpert_num_per_buffer + (self.registry._numel // 3) * j
                    #storage_tensor = torch.empty(
                    #    0, dtype=fp8_dtype, device = target_device
                    #).set_(shared_storage, storage_offset=expert_offset, size=(self.registry._numel,))
                    w13_tensor    = torch.empty(
                        0, dtype=fp8_dtype, device = target_device
                    ).set_(shared_storage, storage_offset=w13_offset, size=self.registry.w13_shape)
                    w2_tensor     = torch.empty(
                        0, dtype=fp8_dtype, device = target_device
                    ).set_(shared_storage, storage_offset=w2_offset, size=self.registry.w2_shape)
                    
                    # storage_tensor.shape:torch.Size([44040192]), w13_tensor.shape:torch.Size([4096, 7168]), w2_tensor.shape:torch.Size([7168, 2048]) 
                    #logger.info(f"storage_tensor.shape:{storage_tensor.shape}, w13_tensor.shape:{w13_tensor.shape}, w2_tensor.shape:{w2_tensor.shape}")
                    expert_tensors.append((w13_tensor, w2_tensor))
                storage_tensors.append(storage_tensor)
            else:
                if local_i == self.rank:
                    storage_tensor_all = storage_tensor_local
                    shared_storage = storage_tensor_all.untyped_storage()
                    storage_tensor = shared_storage
                    for j in range(self.vexpert_num_per_buffer):
                        #expert_offset = j * self.registry._numel
                        w13_offset    = (self.registry._numel // 3) * 2 * j
                        w2_offset     = (self.registry._numel // 3) * 2 * self.vexpert_num_per_buffer + (self.registry._numel // 3) * j
                        #storage_tensor s= torch.empty(
                        #    0, dtype=fp8_dtype, device = target_device
                        #).set_(shared_storage, storage_offset=expert_offset, size=(self.registry._numel,))
                        w13_tensor    = torch.empty(
                            0, dtype=fp8_dtype, device = target_device
                        ).set_(shared_storage, storage_offset=w13_offset, size=self.registry.w13_shape)
                        w2_tensor     = torch.empty(
                            0, dtype=fp8_dtype, device = target_device
                        ).set_(shared_storage, storage_offset=w2_offset, size=self.registry.w2_shape)
                        
                        # storage_tensor.shape:torch.Size([44040192]), w13_tensor.shape:torch.Size([4096, 7168]), w2_tensor.shape:torch.Size([7168, 2048]) 
                        #logger.info(f"storage_tensor.shape:{storage_tensor.shape}, w13_tensor.shape:{w13_tensor.shape}, w2_tensor.shape:{w2_tensor.shape}")
                        expert_tensors.append((w13_tensor, w2_tensor))
                    storage_tensors.append(storage_tensor)
                else:
                    # Each rank except rank0 only sees its own buffer
                    for j in range(self.vexpert_num_per_buffer):
                        storage_tensor = None
                        w13_tensor,w2_tensor = None,None
                        expert_tensors.append((w13_tensor, w2_tensor))
                    storage_tensors.append(storage_tensor)
        torch.cuda.set_device(self.device)
        # Cleanup
        torch.cuda.empty_cache()
        return storage_tensors, expert_tensors
    
        
    def create_shared_event(
         self, group: Optional[ProcessGroup] = None
    ) -> List[int]:
        """
        Makes AsynCpy Event Visiable for All Process
        """
        lib = CudaRTLibrary()
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        if self.rank == 0:
            handles = []
            for i in range(len(self.events)):
                handles.append(self.events[i].ipc_handle())
        else:
            handles = [None] * world_size
        dist.broadcast_object_list(handles, src=0, group=group)

        events = []
        for i, h in enumerate(handles):
            if i == rank:
                events.append(self.events[i])
            else:
                events.append(lib.cudaIpcOpenEventHandle(h))
        self.events = events
        return events

    def start_async_copies(self, main_stream, layer_id, id_pairs, pairs_valid) -> None:
        """
        layer_id: int, Id of Layer that start
        id_pairs: Tuple[int, int], Len(id_pairs) = vexpert_num_per_buffer
        """
        registry = get_expert_host_registry()
        self.pingpong = ((layer_id - 3) // 2) % 2
        vexpert_count1, vexpert_count2 = pairs_valid
        valid_count = vexpert_count1 + vexpert_count2
        # Copy Start
        if self.rank == 0:
            # Optimization: Use different strategy for GPU0 to avoid device context switch and wait_stream overhead
            # Process GPU1-7 first, using device context switch
            for d in range(self.num_devices):
                stream = self.streams[d]
                with torch.cuda.stream(stream):
                    stream.wait_stream(main_stream)
                    for i in range(self.vexpert_num_per_buffer):
                        if i < vexpert_count1:
                            layer_id = id_pairs[d * self.vexpert_num_per_buffer + i][0]
                            expert_id= id_pairs[d * self.vexpert_num_per_buffer + i][1]
                        elif i >= vexpert_count1 and i < valid_count:
                            ii = i - vexpert_count1
                            layer_id = id_pairs[(d+1) * self.vexpert_num_per_buffer - vexpert_count2 + ii][0]
                            expert_id= id_pairs[(d+1) * self.vexpert_num_per_buffer - vexpert_count2 + ii][1]
                        else:
                            layer_id = id_pairs[d * self.vexpert_num_per_buffer][0]
                            expert_id = -1
                        numa_id = 0 if d <= 3 else 1
                        numa_id = numa_id if self.numa_aware else 0
                        offset = i + d * self.vexpert_num_per_buffer
                        w13_tensor, w2_tensor = self.expert_tensors[self.pingpong][offset]
                        if expert_id != -1:
                            host_pinned_tensor = registry.get_pinned_tensor(layer_id,expert_id,numa_id)
                            host_ptr = host_pinned_tensor.data_ptr()
                            element_size = host_pinned_tensor.element_size()
                            w13_size_bytes = registry._offset * 2 * element_size
                            w2_size_bytes = w2_tensor.numel() * w2_tensor.element_size()
                            w13_src_ptr = host_ptr
                            w2_src_ptr = host_ptr + w13_size_bytes
                        else:
                            host_pinned_tensor = None
                            host_ptr = self.empty_pinned.data_ptr()
                            w13_size_bytes = 8
                            w2_size_bytes = 8
                            w13_src_ptr = host_ptr
                            w2_src_ptr = host_ptr
                        #device_tensor = self.storage_tensors[self.pingpong][offset]
                        '''
                        w13_tensor.copy_(host_pinned_tensor[:registry._offset * 2], non_blocking=True)
                        w2_tensor.copy_(host_pinned_tensor[registry._offset * 2:], non_blocking=True)
                        '''
                        # Start Copy using CUDA API
                        cuda_stream_handle = ctypes.c_void_p(stream.cuda_stream)
                        # Copy w13 (first 2 offsets)
                        self.lib.cudaMemcpyAsync(
                            ctypes.c_void_p(w13_tensor.data_ptr()),
                            ctypes.c_void_p(w13_src_ptr),
                            w13_size_bytes,
                            cuda_stream_handle,
                            kind=1
                        )
                        # Copy w2 (from offset * 2 to end)
                        self.lib.cudaMemcpyAsync(
                            ctypes.c_void_p(w2_tensor.data_ptr()),
                            ctypes.c_void_p(w2_src_ptr),
                            w2_size_bytes,
                            cuda_stream_handle,
                            kind=1
                        )
                    self.events[d].record(stream)
            # After using context manager, stream will automatically restore to previous stream (main_stream)
    
    def ensure_copy_done(self,main_stream, layer_id) -> None:
        # Wait For Copy Finish
        if self.rank == 0:
            # Ensure waiting for events on main_stream
            # During CUDA graph capture, torch.cuda.set_stream() will not be recorded
            # But wait_event operations will be recorded, so no need to explicitly set stream here
            # Because caller should already be on main_stream
            with torch.cuda.stream(main_stream):
                for d in range(self.num_devices):
                    main_stream.wait_event(self.events[d])
        # Rank0 sends sync signal to all Ranks via NVSHMEM after completing all Stream copies
        # Other Ranks proceed with MoE computation after receiving sync signal, otherwise data may not be ready when computation starts in tight timing, generally this won't happen        
        self.buffer.runtime.central_flag_sync_nvshmem(layer_id, self.rank, main_stream.cuda_stream)
    
    def get_weight_tensor(self, offset, layer_id=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the weight tensor from the buffer
        """
        if layer_id is not None:
            pingpong = (layer_id - 3) // 2 % 2
        else:
            pingpong = self.pingpong
        return self.expert_tensors[1 - pingpong][offset]
    
    def get_storage(self, offset ,layer_id=None):
        if layer_id is not None:
            pingpong = (layer_id - 3) // 2 % 2
        else:
            pingpong = self.pingpong
        return self.storage_tensors[1 - pingpong][offset]
                    

    @contextmanager
    def capture(self):
        try:
            self._is_capturing = True
            yield
        finally:
            self._is_capturing = False

_GLOBAL_H2DCPY: CustomH2DAsyncCpy | None = None

def get_custom_h2dcpy(group=None,device=None,num_devices=None) -> CustomH2DAsyncCpy:
    global _GLOBAL_H2DCPY
    if _GLOBAL_H2DCPY is None and group is not None:
        _GLOBAL_H2DCPY = CustomH2DAsyncCpy(group = group, device = device, num_devices=num_devices)
    return _GLOBAL_H2DCPY