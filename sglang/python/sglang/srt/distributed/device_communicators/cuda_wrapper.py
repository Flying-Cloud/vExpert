# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/cuda_wrapper.py

"""This file is a pure Python wrapper for the cudart library.
It avoids the need to compile a separate shared library, and is
convenient for use when we just need to call a few functions.
"""

import ctypes
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# this line makes it possible to directly load `libcudart.so` using `ctypes`
import torch  # noqa

logger = logging.getLogger(__name__)

# === export types and functions from cudart to Python ===
# for the original cudart definition, please check
# https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int

# CUDA opaque handle types
cudaEvent_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p
cudaGraph_t = ctypes.c_void_p
cudaGraphExec_t = ctypes.c_void_p
cudaGraphNode_t = ctypes.c_void_p

# CUDA Graph node types enum
cudaGraphNodeType = ctypes.c_int
# Common node type values (from cuda_runtime_api.h)
cudaGraphNodeTypeKernel = 0x00
cudaGraphNodeTypeMemcpy = 0x01
cudaGraphNodeTypeMemset = 0x02
cudaGraphNodeTypeHost = 0x03
cudaGraphNodeTypeGraph = 0x04
cudaGraphNodeTypeEmpty = 0x05
cudaGraphNodeTypeWaitEvent = 0x06
cudaGraphNodeTypeEventRecord = 0x07
cudaGraphNodeTypeExtSemaphoreSignal = 0x08
cudaGraphNodeTypeExtSemaphoreWait = 0x09
cudaGraphNodeTypeMemAlloc = 0x0a
cudaGraphNodeTypeMemFree = 0x0b

# CUDA Graph Exec Update Result enum
cudaGraphExecUpdateResult = ctypes.c_int
# Update result values (from cuda_runtime_api.h)
cudaGraphExecUpdateSuccess = 0x00
cudaGraphExecUpdateError = 0x01
cudaGraphExecUpdateErrorTopologyChanged = 0x02
cudaGraphExecUpdateErrorNodeTypeChanged = 0x03
cudaGraphExecUpdateErrorFunctionChanged = 0x04
cudaGraphExecUpdateErrorParametersChanged = 0x05
cudaGraphExecUpdateErrorNotSupported = 0x06
cudaGraphExecUpdateErrorUnsupportedFunctionChange = 0x07


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]

class cudaIpcEventHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 64)]


# CUDA Graph types
class cudaPos(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_size_t),
        ("y", ctypes.c_size_t),
        ("z", ctypes.c_size_t),
    ]


class cudaExtent(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
        ("depth", ctypes.c_size_t),
    ]


class cudaPitchedPtr(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("pitch", ctypes.c_size_t),
        ("xsize", ctypes.c_size_t),
        ("ysize", ctypes.c_size_t),
    ]


# cudaArray_t is an opaque handle type
cudaArray_t = ctypes.c_void_p


class cudaMemcpy3DParms(ctypes.Structure):
    _fields_ = [
        ("srcArray", ctypes.POINTER(cudaArray_t)),
        ("srcPos", cudaPos),
        ("srcPtr", cudaPitchedPtr),
        ("dstArray", ctypes.POINTER(cudaArray_t)),
        ("dstPos", cudaPos),
        ("dstPtr", cudaPitchedPtr),
        ("extent", cudaExtent),
        ("kind", cudaMemcpyKind),
    ]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """  # noqa
    found = False
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found = True
                break
    if not found:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(
        lib_name
    ), f"Unexpected filename: {filename} for library {lib_name}"
    return path


class CudaRTLibrary:
    exported_functions = [
        # ​cudaError_t cudaSetDevice ( int  device )
        Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        # cudaError_t 	cudaDeviceSynchronize ( void )
        Function("cudaDeviceSynchronize", cudaError_t, []),
        # ​cudaError_t cudaDeviceReset ( void )
        Function("cudaDeviceReset", cudaError_t, []),
        # const char* 	cudaGetErrorString ( cudaError_t error )
        Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        # ​cudaError_t 	cudaMalloc ( void** devPtr, size_t size )
        Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        # ​cudaError_t 	cudaFree ( void* devPtr )
        Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        # ​cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )
        Function(
            "cudaMemset", cudaError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        ),
        # ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) # noqa
        Function(
            "cudaMemcpy",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind],
        ),
        # cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream ) # noqa
        Function(
            "cudaMemcpyAsync",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind, cudaStream_t],
        ),
        # cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr ) # noqa
        Function(
            "cudaIpcGetMemHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p],
        ),
        # ​cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags ) # noqa
        Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
        # cudaError_t cudaIpcOpenEventHandle ( cudaEvent_t* event, cudaIpcEventHandle_t handle ) # noqa
        Function(
            "cudaIpcOpenEventHandle",
            cudaError_t,
            [ctypes.POINTER(cudaEvent_t), cudaIpcEventHandle_t],
        ),
        # cudaError_t cudaGraphMemcpyNodeGetParams ( cudaGraphNode_t node, cudaMemcpy3DParms* nodeParams ) # noqa
        Function(
            "cudaGraphMemcpyNodeGetParams",
            cudaError_t,
            [cudaGraphNode_t, ctypes.POINTER(cudaMemcpy3DParms)],
        ),
        # cudaError_t cudaGraphGetNodes ( cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes ) # noqa
        Function(
            "cudaGraphGetNodes",
            cudaError_t,
            [cudaGraph_t, ctypes.POINTER(cudaGraphNode_t), ctypes.POINTER(ctypes.c_size_t)],
        ),
        # cudaError_t cudaGraphNodeGetType ( cudaGraphNode_t node, cudaGraphNodeType* type ) # noqa
        Function(
            "cudaGraphNodeGetType",
            cudaError_t,
            [cudaGraphNode_t, ctypes.POINTER(cudaGraphNodeType)],
        ),
        # cudaError_t cudaGraphMemcpyNodeSetParams ( cudaGraphNode_t node, const cudaMemcpy3DParms* nodeParams ) # noqa
        Function(
            "cudaGraphMemcpyNodeSetParams",
            cudaError_t,
            [cudaGraphNode_t, ctypes.POINTER(cudaMemcpy3DParms)],
        ),
        # cudaError_t cudaGraphExecUpdate ( cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t* hErrorNode_out, cudaGraphExecUpdateResult* updateResult_out ) # noqa
        Function(
            "cudaGraphExecUpdate",
            cudaError_t,
            [
                cudaGraphExec_t,
                cudaGraph_t,
                ctypes.POINTER(cudaGraphNode_t),
                ctypes.POINTER(cudaGraphExecUpdateResult),
            ],
        ),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = find_loaded_library("libcudart")
            assert so_file is not None, "libcudart is not loaded in the current process"
        if so_file not in CudaRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            CudaRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = CudaRTLibrary.path_to_library_cache[so_file]

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in CudaRTLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            raise RuntimeError(f"CUDART error: {error_str}")

    def cudaGetErrorString(self, error: cudaError_t) -> str:
        return self.funcs["cudaGetErrorString"](error).decode("utf-8")

    def cudaSetDevice(self, device: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaSetDevice"](device))

    def cudaDeviceSynchronize(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceSynchronize"]())

    def cudaDeviceReset(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceReset"]())

    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(self.funcs["cudaMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def cudaFree(self, devPtr: ctypes.c_void_p) -> None:
        self.CUDART_CHECK(self.funcs["cudaFree"](devPtr))

    def cudaMemset(self, devPtr: ctypes.c_void_p, value: int, count: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaMemset"](devPtr, value, count))

    def cudaMemcpy(
        self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int
    ) -> None:
        cudaMemcpyDefault = 4
        kind = cudaMemcpyDefault
        self.CUDART_CHECK(self.funcs["cudaMemcpy"](dst, src, count, kind))

    def cudaMemcpyAsync(
        self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int, stream: cudaStream_t, kind: int = 4
    ) -> None:
        """Asynchronous memory copy.
        
        Args:
            dst: Destination memory address
            src: Source memory address
            count: Number of bytes to copy
            stream: CUDA stream to execute the copy on
        """
        self.CUDART_CHECK(self.funcs["cudaMemcpyAsync"](dst, src, count, kind, stream))


    def cudaIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.CUDART_CHECK(
            self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), devPtr)
        )
        return handle

    def cudaIpcOpenEventHandle(self, handle: cudaIpcEventHandle_t) -> cudaEvent_t:
        """Open an IPC event handle from another process.
        
        Args:
            handle: The IPC event handle from another process
            
        Returns:
            A cudaEvent_t handle that can be used for synchronization
        """
        event = cudaEvent_t()
        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenEventHandle"](ctypes.byref(event), handle)
        )
        return event

    def cudaIpcOpenMemHandle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cudaIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess
            )
        )
        return devPtr

    def cudaGraphMemcpyNodeGetParams(self, node: cudaGraphNode_t) -> cudaMemcpy3DParms:
        """Get the parameters of a memcpy node in a CUDA graph.
        
        Args:
            node: The graph node handle (cudaGraphNode_t)
            
        Returns:
            A cudaMemcpy3DParms structure containing the node parameters
        """
        params = cudaMemcpy3DParms()
        self.CUDART_CHECK(
            self.funcs["cudaGraphMemcpyNodeGetParams"](node, ctypes.byref(params))
        )
        return params

    def cudaGraphGetNodes(self, graph: cudaGraph_t) -> List[cudaGraphNode_t]:
        """Get all nodes in a CUDA graph.
        
        Args:
            graph: The graph handle (cudaGraph_t)
            
        Returns:
            A list of graph node handles
        """
        # First call to get the number of nodes (pass NULL for nodes array)
        num_nodes = ctypes.c_size_t()
        null_ptr = ctypes.cast(None, ctypes.POINTER(cudaGraphNode_t))
        self.CUDART_CHECK(
            self.funcs["cudaGraphGetNodes"](graph, null_ptr, ctypes.byref(num_nodes))
        )
        
        if num_nodes.value == 0:
            return []
        
        # Second call to get the actual nodes
        nodes_array = (cudaGraphNode_t * num_nodes.value)()
        self.CUDART_CHECK(
            self.funcs["cudaGraphGetNodes"](graph, nodes_array, ctypes.byref(num_nodes))
        )
        
        return list(nodes_array)

    def cudaGraphNodeGetType(self, node: cudaGraphNode_t) -> int:
        """Get the type of a CUDA graph node.
        
        Args:
            node: The graph node handle (cudaGraphNode_t)
            
        Returns:
            The node type (cudaGraphNodeType enum value)
        """
        node_type = cudaGraphNodeType()
        self.CUDART_CHECK(
            self.funcs["cudaGraphNodeGetType"](node, ctypes.byref(node_type))
        )
        return node_type.value

    def cudaGraphMemcpyNodeSetParams(
        self, node: cudaGraphNode_t, params: cudaMemcpy3DParms
    ) -> None:
        """Set the parameters of a memcpy node in a CUDA graph.
        
        Args:
            node: The graph node handle (cudaGraphNode_t)
            params: The memcpy parameters to set (cudaMemcpy3DParms)
        """
        self.CUDART_CHECK(
            self.funcs["cudaGraphMemcpyNodeSetParams"](node, ctypes.byref(params))
        )

    def cudaGraphExecUpdate(
        self,
        graphExec: cudaGraphExec_t,
        graph: cudaGraph_t,
    ) -> Tuple[int, Optional[cudaGraphNode_t]]:
        """Update an instantiated graph execution instance with an updated graph.
        
        This function attempts to update an existing cudaGraphExec_t with a new graph
        that has the same topology. If successful, the graph execution instance is
        updated in-place without needing to re-instantiate.
        
        Args:
            graphExec: The graph execution instance to update (cudaGraphExec_t)
            graph: The updated graph with the same topology (cudaGraph_t)
            
        Returns:
            A tuple of (update_result, error_node) where:
            - update_result: The update result code (cudaGraphExecUpdateResult enum value)
            - error_node: The node that caused the error, if any (None if successful)
        """
        error_node = cudaGraphNode_t()
        update_result = cudaGraphExecUpdateResult()
        
        # Call the function - note: this may return an error code even if update fails
        # The update_result will contain the actual result
        result = self.funcs["cudaGraphExecUpdate"](
            graphExec,
            graph,
            ctypes.byref(error_node),
            ctypes.byref(update_result),
        )
        
        # Check for CUDA runtime errors
        self.CUDART_CHECK(result)
        
        # Return the update result and error node (if any)
        if update_result.value == cudaGraphExecUpdateSuccess:
            return (update_result.value, None)
        else:
            return (update_result.value, error_node)
