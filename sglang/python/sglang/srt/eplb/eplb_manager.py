import logging
import time
import threading
import multiprocessing
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch.cuda
from concurrent.futures import ProcessPoolExecutor, Future
import atexit,os
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import compute_virtual, ExpertLocationMetadata, get_global_expert_location_metadata
from torch.distributed import Backend, ProcessGroup
from sglang.srt.distributed.device_communicators.custom_h2d_async_cpy import get_custom_h2dcpy
from sglang.srt.eplb.expert_location import get_id_pairs
if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)



class EPLBManager:
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        self._enable_cpu_pinned = self._server_args.enable_cpu_pinned
        self._src_rank = 0
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations
        self._rebalance_num_iterations_virtual = 10

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )
        # Create process pool, use initializer to set process priority
        # initializer will be called when each worker process starts, used to set process priority
        self._main_generator = self._entrypoint()
        self._virtual_generator = self._entrypoint_virtual()
        self._virtual_bs = 256
        if self._enable_cpu_pinned:
            self._h2d_cpy = get_custom_h2dcpy()
        self.initial_rebalance()
        self._obj = get_global_expert_location_metadata()
        self._rebalance_finish = False
    def on_forward_pass_end(self):
        if self._enable_cpu_pinned:
            next(self._virtual_generator)
        else:
            next(self._main_generator)
        self._step_counter += 1
    def initial_rebalance(self):
        self._step_counter = 0
        if self._enable_cpu_pinned:
            self._compute_ongoing = False
            self._tp_sync_ongoing = False
            self._update_ongoing = False
            self._begin_transfer_step = None
            self._rebalance_finish = False
    # can be more complex if needed
    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield
            yield from self.rebalance()
    def _entrypoint_virtual(self):
        while True:
            for _ in range(self._rebalance_num_iterations_virtual):
                yield
            self.initial_rebalance()
            self.rebalance_virtual()
    def warmup(self,graphs):
        if self._enable_cpu_pinned:
            get_global_expert_distribution_recorder().dump_record(
                output_mode="object",cpu_pinned = True
            )
            self._model_runner.set_layer_ids()
            self._model_runner.set_layer_vinfo()
            for bs,_ in graphs.items():
                self._h2d_cpy.memcpy_manager.update_src_ptrs_from_id_pairs(self._model_runner.layer_id_pairs, bs,self._model_runner.layers_pairs_valid)
    def rebalance(self):
        logger.info("[EPLBManager] rebalance start")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.cuda.synchronize()
            time_start = time.time()

        logical_count = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )["logical_count"]
        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count
        )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.cuda.synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)
    
    def rebalance_virtual(self):
        logger.info("[EPLBManager] rebalance virtual start")
        self._rebalance_finish = False
        enable_timing = True
        if enable_timing:
            torch.cuda.synchronize()
            time_start = time.time()
        if self._enable_cpu_pinned:
            # logical count Gather and compute Vinfo 10ms for EP16
            self.compute(bs = self._virtual_bs)
        msg = f"[EPLBManager] rebalance virtual end"
        if enable_timing:
            torch.cuda.synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)
        self._rebalance_finish = True
            
    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))

    def _update_vinfo_graph(self):
        self._obj.update_virtual_direct(self._virtual_results)
        self._model_runner.set_layer_ids()
    def update(self):
        time1 = time.time()
        self._update_vinfo_graph()
        time2 = time.time()
        logger.info(f"update time: {time2 - time1}s")
    def compute(self,bs = 256):
        #time1 = time.time()
        logical_count = get_global_expert_distribution_recorder().dump_record(
            output_mode="object", cpu_pinned=True
        )["logical_count"]
        logical_count_sum = logical_count
        #time2 = time.time()
        if self._obj is not None:
            self._virtual_results = self._obj.compute_virtual(logical_count=logical_count_sum, num_vexpert_per_buffer=self._server_args.num_vexpert_buffer, num_gpus=self._server_args.ep_size, bs = bs)
        #logger.info(f"compute time: {time3 - time2}s, dump record time: {time2 - time1}s")

    def _compute_vexpert_metadata(self, logical_count):
        local_rank = self._model_runner.tp_rank
        num_gpu_per_node = 8
        torch.cuda.set_device(local_rank % num_gpu_per_node)
        obj = get_global_expert_location_metadata()
        if obj is not None:
            self._virtual_results = obj.compute_virtual(logical_count=logical_count, num_vexpert_per_buffer=self._server_args.num_vexpert_buffer, num_gpus=self._server_args.ep_size)
        self._compute_ongoing = False
    def tp_sync(self):
        """Barrier-like handshake among all TP ranks.

        Each rank sends a step counter to every other rank and
        waits for all peers' counters.  When all messages are completed,
        we know that the new expert mapping is globally visible and can
        be applied safely.
        """
        self._send_signal_step = self._step_counter
        send_works, recv_works, buffer_kept = self._gen_rec_send_works()
        self._tp_sync_ongoing = True
        self._tp_sync_thread = threading.Thread(
                target=self._wait_for_tp_sync_signals,
                args=(send_works, recv_works,),
                daemon=True
            )
        self._tp_sync_thread.start()
        yield
        while self._tp_sync_ongoing:
            yield 
        # the largest counter among all ranks indicates the earliest
        # forward-pass index after which the new mapping becomes valid
        max_tensor = torch.stack(buffer_kept).max()
        # Add a buffer of 2 steps to ensure all ranks see the new step before acting on it.
        self._begin_transfer_step = max(max_tensor.item(),self._send_signal_step) + 2
    def _gen_rec_send_works(self):
        world_size = self._model_runner.tp_group.world_size
        local_rank = self._model_runner.tp_group.rank_in_group
        # Using the dedicated CPU(Gloo) process-group avoids touching NCCL streams and
        # therefore never blocks the GPU-side decoding kernels that are running in parallel.
        group = self._model_runner.tp_group.cpu_group
        send_works = []
        recv_buffer_kept = []
        recv_works = []
        for src in range(world_size):
            if src == local_rank:
                continue
            w, t = self._recv_single_signal(src, group)
            recv_works.append(w)
            recv_buffer_kept.append(t)
        for dst in range(world_size):
            if dst == local_rank:
                continue
            w, t = self._send_single_signal(dst, self._send_signal_step, group)
            send_works.append(w)
        return send_works, recv_works, recv_buffer_kept
    
    def _send_single_signal(self, dst_rank_in_group: int, value: int, group: Optional[ProcessGroup] = None,
                            ) -> Tuple[Optional[torch.distributed.Work], torch.Tensor]:
        signal_tensor = torch.tensor([value], dtype=torch.long)
        w = torch.distributed.isend(signal_tensor, dst=dst_rank_in_group, group=group)
        return w, signal_tensor
    
    def _recv_single_signal(self, src_rank_in_group: int, group: Optional[ProcessGroup] = None, 
                            ) -> Tuple[Optional[torch.distributed.Work], torch.Tensor]:
        signal_tensor = torch.empty(1, dtype=torch.long)
        w = torch.distributed.irecv(signal_tensor, src=src_rank_in_group, group=group)
        return w, signal_tensor
    
    def _wait_for_tp_sync_signals(self, send_works,recv_works):
        works = send_works + recv_works
        for w in works:
            w.wait()
        self._tp_sync_ongoing = False


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
