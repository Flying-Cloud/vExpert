# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
from math import log2
from typing import Tuple

import torch
import time
from sglang.srt.utils import get_bool_env_var

try:
    import semi_pd_ipc
    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False
    semi_pd_ipc = None

# HardCode for performance optimization
EXPERT_SIZE = 7168 * 2048 * 3
BW_H2D = 26 * 1024 * 1024 * 1024 # 26GB
BW_A2A = 15 * 1024 * 1024 * 1024 # 15GB
BASE_A2A = 7168 * 8 * 3 / BW_A2A / 0.8
BASE_H2D = EXPERT_SIZE / BW_H2D
def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy
    )  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """

    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if enable_hierarchical:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    return phy2log, log2phy, logcnt

def replicate_experts_virtual(
    weight: torch.Tensor, 
    vexpert_count: torch.Tensor,
    pinned_to_logical_map: torch.Tensor,
    num_gpus: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    weight: [layers, num_logical_experts]f
    vexpert_count: [layers, num_gpus], number of virtual expert each layer per GPU
    pinned_to_logical_map: [layers, num_pinned_experts], logical expert id for pinned expert
    """
    n, num_log = weight.shape
    num_pinned = pinned_to_logical_map.shape[1]
    device = weight.device
    
    max_num_vir = vexpert_count.max().item() * num_gpus
    
    if HAS_CUDA_KERNEL and weight.is_cuda:
        vir2log = torch.full((n, max_num_vir), -1, dtype=torch.int64, device=device)
        rank = torch.zeros(n, max_num_vir, dtype=torch.int64, device=device)
        global_cnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
        
        # Call CUDA kernel
        semi_pd_ipc.replicate_vexpert_cuda(
            weight.float(),
            pinned_to_logical_map,
            vexpert_count,
            vir2log,
            rank,
            global_cnt,
            n,
            num_log,
            num_pinned,
            num_gpus,
            max_num_vir
        )
        
        return vir2log, rank, global_cnt
    
    weight_pinned = weight.gather(-1, pinned_to_logical_map)
    vir2log = torch.zeros(n, max_num_vir, dtype=torch.int64, device=device)
    rank = torch.zeros(n, max_num_vir, dtype=torch.int64, device=device)
    pinned_cnt = torch.ones(n, num_pinned, dtype=torch.int64, device=device)
    global_cnt = torch.ones(n, num_log, dtype=torch.int64, device=device)

    vir2log = torch.full((n, max_num_vir), -1, dtype=torch.int64, device=device)
    
    num_v = vexpert_count.sum(dim=-1).view(-1)
    for copy_round in range(max_num_vir):
        mask = (copy_round) < num_v
        
        if not mask.any():
            break
            
        current_load = weight_pinned / pinned_cnt
        vir_idx = current_load.max(dim=-1).indices
        redundant_indices = torch.gather(pinned_to_logical_map, 1, vir_idx.unsqueeze(dim=1)).view(-1)

        target_col = copy_round
        mask_indices = torch.where(mask)[0]  
        if len(mask_indices) > 0:
            rows_to_update = vir2log[mask_indices].clone()  
            col_idx = torch.full((len(mask_indices), 1), target_col, dtype=torch.int64, device=device)
            rows_to_update.scatter_(1, col_idx, redundant_indices[mask_indices].unsqueeze(1))
            vir2log[mask_indices] = rows_to_update  
            
            rank_rows_to_update = rank[mask_indices].clone()
            rank_rows_to_update.scatter_(1, col_idx, 
                                       pinned_cnt[mask_indices, vir_idx[mask_indices]].unsqueeze(1))
            rank[mask_indices] = rank_rows_to_update
            
            pinned_cnt[mask_indices, vir_idx[mask_indices]] += 1
            global_cnt[mask_indices, redundant_indices[mask_indices]] += 1
    
    return vir2log, rank, global_cnt
def inverse(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv.scatter_(
        1,
        perm,
        torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
            perm.shape
        ),
    )
    return inv
def allocate_vexpert(
    weight_gpu_sum: torch.Tensor, # [layers, num_gpus]
    balance_init: torch.Tensor,   # [layers//2, 2, num_gpus]
    num_vexpert_per_buffer: int,
    num_gpus: int,
    bs: int = 256):
    
    # Try CUDA implementation first if available
    
    
    if HAS_CUDA_KERNEL and weight_gpu_sum.is_cuda and balance_init.is_cuda:
        try:
            # Pre-allocate output tensor (initialized to default value for consistency)
            layer_num = weight_gpu_sum.shape[0]
            vexpert_count = torch.ones((layer_num, num_gpus), dtype=torch.int64, device=weight_gpu_sum.device) * (num_vexpert_per_buffer // 2)
            # Calculate v_m in Python to ensure consistency with CPU version
            v_m = int(BASE_A2A * bs / BASE_H2D)
            v_m = 1
            semi_pd_ipc.allocate_vexpert_optimization_cuda(
                weight_gpu_sum, balance_init, vexpert_count, num_vexpert_per_buffer, num_gpus, bs, BASE_A2A, BASE_H2D, v_m
            )
            return vexpert_count
        except Exception as e:
            print(f"CUDA implementation failed, falling back to Uniform: {e}")
    
    balanceness = balance_init.mean(dim=-1) / balance_init.max(dim=-1).values
    vexpert_count = torch.ones((weight_gpu_sum.shape[0] // 2,2,num_gpus), dtype=torch.int64, device=weight_gpu_sum.device) * num_vexpert_per_buffer // 2
    vexpert_count_max = torch.ones((weight_gpu_sum.shape[0] // 2,2,num_gpus), dtype=torch.int64, device=weight_gpu_sum.device) * num_vexpert_per_buffer // 2
    vexpert_count = torch.ones_like(vexpert_count_max)
    return vexpert_count


    


def rebalance_virtual_experts(
    weight: torch.Tensor,
    num_vexpert_per_buffer: int,
    num_replicas: int,
    num_gpus: int,
    logical_to_pinned_map: torch.Tensor, # [layers, num_pinned_experts]
    physical_to_logical_map_non_pinned: torch.Tensor, # [layers, num_logical_experts]
    logical_to_physical_map_non_pinned: torch.Tensor, 
    expert_count_non_pinned: torch.Tensor,
    bs: int = 256
):
    time1 = time.time()

    num_nodes = num_gpus // 8
    weight = weight[3:].float()
    num_layers, num_logical_experts = weight.shape
    assert num_replicas == (num_vexpert_per_buffer * num_gpus + num_logical_experts)
    # logical expert id in each rank
    phy2log_nonpinned = physical_to_logical_map_non_pinned[3:].view(-1, num_logical_experts)
    log2phy_nonpined = logical_to_physical_map_non_pinned[3:]
    expert_count_non_pinned = expert_count_non_pinned[3:]
    # Step1: 
    weight_gpu_sum = weight.gather(-1,phy2log_nonpinned)\
                .view(-1, num_gpus, num_logical_experts // num_gpus).sum(dim=-1)

    balance_init = weight_gpu_sum.view(-1,2,num_gpus)
    balanceness_init = balance_init.mean(dim=-1) / balance_init.max(dim=-1).values
    # 4ms
    vexpert_count = allocate_vexpert(weight_gpu_sum, balance_init, num_vexpert_per_buffer, num_gpus, bs = bs)


    # Step2: 
    log2phy_pinned = logical_to_pinned_map[3:]
    vir2log, rank, global_cnt = replicate_experts_virtual(
        weight, vexpert_count, log2phy_pinned, num_gpus
    )
    # Step3: Pack virtual Expert to GPUs
    num_experts_per_gpu = num_logical_experts // num_gpus
    vexpert_id = torch.full(
        (num_layers, num_gpus, num_vexpert_per_buffer),
        -1,
        dtype=torch.int64,
        device=vir2log.device,
    )
    weight_new = torch.zeros(num_layers, num_gpus, dtype=torch.float32, device=vir2log.device)
    maxlogcnt = 40
    phy2log = torch.full(
        (num_layers, num_replicas),
        -1,
        dtype=torch.int64,
        device=vir2log.device,
    )
    log2phy= torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=global_cnt.device,
    )
    log2phy_nodes = torch.full(
        (num_layers, num_logical_experts, num_nodes, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=global_cnt.device,
    )
    log2phy_null = torch.full(
        (num_layers, num_logical_experts, 1),
        -1,
        dtype=torch.int64,
        device=global_cnt.device,
    )
    

    num_replica_per_gpu = num_replicas // num_gpus
    log2phy_null[:,:,0] = log2phy_nonpined[:,:,0] // num_experts_per_gpu * num_replica_per_gpu + log2phy_nonpined[:,:,0] % num_experts_per_gpu
    log2phy[:,:,0] = log2phy_null[:,:,0]

    logcnt_nodes = torch.ones(num_layers, num_logical_experts, num_nodes, dtype=torch.int64, device=global_cnt.device)
    logcnt = torch.ones(num_layers, num_logical_experts, dtype=torch.int64, device=global_cnt.device)
    phy2log_mask = torch.arange(num_logical_experts, dtype=torch.int64, device=global_cnt.device)
    phy2log_mask = phy2log_mask // num_experts_per_gpu * num_replica_per_gpu + phy2log_mask % num_experts_per_gpu
    phy2log_mask = phy2log_mask.unsqueeze(0).expand(num_layers, -1)
    phy2log.scatter_(
        dim = 1,
        index = phy2log_mask,
        src = phy2log_nonpinned
    )

    vexpert_count_sum = vexpert_count.sum(dim=-1)  # [num_layers]
    weight_c = weight / global_cnt
    

    pack_weights_all = torch.zeros(num_layers, num_gpus, dtype=torch.float32, device=weight.device)
    pack_items_all = torch.ones(num_layers, num_gpus, dtype=torch.int64, device=weight.device) * num_experts_per_gpu
    temp_pack_weights = torch.empty(num_gpus, dtype=torch.float32, device=weight.device)
    

    phy_indices_all = log2phy_nonpined[:, :, 0]  # [num_layers, num_logical_experts]
    packs_normal_all = phy_indices_all // num_experts_per_gpu  # [num_layers, num_logical_experts]

    for i in range(num_layers):
        pack_weights_all[i].scatter_add_(0, packs_normal_all[i], weight_c[i])
    # 0.2ms

    if HAS_CUDA_KERNEL:
        semi_pd_ipc.allocate_vexpert_cuda(
            vir2log,
            weight_c,
            vexpert_count,
            vexpert_count_sum,
            pack_weights_all,
            pack_items_all,
            vexpert_id,
            phy2log,
            log2phy,
            logcnt,
            num_layers,
            num_gpus,
            num_logical_experts,
            num_replicas,
            num_vexpert_per_buffer,
            num_experts_per_gpu,
            num_replica_per_gpu,
            maxlogcnt
        )
        weight_new = pack_weights_all
    else:

        for i in range(num_layers):

            vexpert_count_sum_i = vexpert_count_sum[i]
            log2phy_nonpined_i = log2phy_nonpined[i]  # [num_logical_experts, maxlogcnt]
            vexpert_count_i = vexpert_count[i]  # [num_gpus]
            weight_per_cnt = weight_c[i]  # [num_logical_experts]
            pack_capacity = vexpert_count_i + num_experts_per_gpu  # [num_gpus]
            

            pack_weights = pack_weights_all[i]
            pack_items = pack_items_all[i]
            

            total_virtual = int(vexpert_count_sum_i.item())
            if total_virtual == 0:
                weight_new[i] = pack_weights
                continue
            

            vir_logical_ids = vir2log[i, :total_virtual]  # [total_virtual]
            

            null_experts_num = 0 if i % 2 == 0 else (num_vexpert_per_buffer - int(vexpert_count_i[0].item()))

            vir_weights = weight_per_cnt[vir_logical_ids]  # [total_virtual]
            

            for j_idx in range(total_virtual):

                available_mask = pack_items < pack_capacity
                if not available_mask.any():
                    break
                
                temp_pack_weights.copy_(pack_weights)
                temp_pack_weights.masked_fill_(~available_mask, float('inf'))
                pack = int(temp_pack_weights.argmin().item())
                
                logical_expert_id = int(vir_logical_ids[j_idx].item())
                pack_items_val = int(pack_items[pack].item())
                
                slot_idx = pack_items_val - num_experts_per_gpu
                vexpert_id[i, pack, slot_idx] = logical_expert_id
                
                weight_val = vir_weights[j_idx].item()
                pack_weights[pack] += weight_val
                
                physical_expert_id = pack * num_replica_per_gpu + pack_items_val + null_experts_num
                phy2log[i, physical_expert_id] = logical_expert_id
                
                pack_items[pack] += 1
                
                logcnt_val = int(logcnt[i, logical_expert_id].item())
                log2phy[i, logical_expert_id, logcnt_val] = physical_expert_id
                logcnt[i, logical_expert_id] += 1
            
            weight_new[i] = pack_weights
    return phy2log.to(torch.int32), log2phy.to(torch.int32), logcnt.to(torch.int32), vexpert_id.view(-1,num_gpus,num_vexpert_per_buffer).to(torch.int32), vexpert_count.view(-1,num_gpus).to(torch.int32), log2phy_nodes.to(torch.int32), logcnt_nodes.to(torch.int32), log2phy_null.to(torch.int32)

    


__all__ = ["rebalance_experts", "rebalance_virtual_experts"]
