from enum import Enum, auto
from typing import Optional

import torch

from sglang.srt.eplb.eplb_algorithms import deepseek, deepseek_vec


class EplbAlgorithm(Enum):
    deepseek = auto()
    deepseek_hierarchical = auto()
    deepseek_vec = auto()
    deepseek_vec_hierarchical = auto()
    # TODO may have more algorithm later


def rebalance_experts_virtual(
    tokens_per_expert: torch.Tensor,
    num_vexpert_per_buffer: int,
    num_replicas: int,
    num_gpus: int,
    logical_to_pinned_map: torch.Tensor, # [layers, num_pinned_experts]
    physical_to_logical_map_non_pinned: torch.Tensor, # [layers, num_logical_experts]
    logical_to_physical_map_non_pinned: torch.Tensor, 
    expert_count_non_pinned: torch.Tensor,
    bs: int = 256,
):
    return deepseek.rebalance_virtual_experts(
        weight=tokens_per_expert.sum(dim=0),
        num_vexpert_per_buffer=num_vexpert_per_buffer,
        num_replicas=num_replicas,
        num_gpus=num_gpus,
        logical_to_pinned_map=logical_to_pinned_map,
        physical_to_logical_map_non_pinned=physical_to_logical_map_non_pinned,
        logical_to_physical_map_non_pinned=logical_to_physical_map_non_pinned,
        expert_count_non_pinned=expert_count_non_pinned,
        bs = bs
    )

def rebalance_experts_virtual_v1(
    tokens_per_expert: torch.Tensor,
    num_vexpert_per_buffer: int,
    num_replicas: int,
    num_gpus: int,
    logical_to_pinned_map: torch.Tensor, # [layers, num_pinned_experts]
    physical_to_logical_map_non_pinned: torch.Tensor, # [layers, num_logical_experts]
    logical_to_physical_map_non_pinned: torch.Tensor, 
    expert_count_non_pinned: torch.Tensor,
    logical_node_count: torch.Tensor, # [num_nodes, num_layers, num_logical_experts]
    rank_ratio: torch.Tensor, # [num_layers,num_gpus]
):
    return deepseek.rebalance_virtual_experts_v1(
        weight=tokens_per_expert.sum(dim=0),
        num_vexpert_per_buffer=num_vexpert_per_buffer,
        num_replicas=num_replicas,
        num_gpus=num_gpus,
        logical_to_pinned_map=logical_to_pinned_map,
        physical_to_logical_map_non_pinned=physical_to_logical_map_non_pinned,
        logical_to_physical_map_non_pinned=logical_to_physical_map_non_pinned,
        expert_count_non_pinned=expert_count_non_pinned,
        logical_node_count=logical_node_count,
        rank_ratio=rank_ratio,
    )

def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],
    num_nodes: int,
    algorithm: EplbAlgorithm,
):
    if algorithm in [EplbAlgorithm.deepseek, EplbAlgorithm.deepseek_hierarchical]:
        return deepseek.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_hierarchical,
        )

    if algorithm in [
        EplbAlgorithm.deepseek_vec,
        EplbAlgorithm.deepseek_vec_hierarchical,
    ]:
        return deepseek_vec.rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_vec_hierarchical,
        )

    raise NotImplementedError


def compute_algorithm(
    raw_algorithm: str,
    num_groups: Optional[int],
    num_nodes: int,
) -> EplbAlgorithm:
    if raw_algorithm != "auto":
        return EplbAlgorithm[raw_algorithm]

    # TODO test on real scenarios and know which ones perform better
    if (num_groups is not None) and (num_groups % num_nodes == 0):
        return EplbAlgorithm.deepseek_hierarchical
    else:
        return EplbAlgorithm.deepseek
