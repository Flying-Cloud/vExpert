# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import time
import torch
import torch.distributed
import torch.nn.functional as F

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.eplb import eplb_algorithms
from sglang.srt.model_loader import get_model_architecture
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class ExpertLocationMetadata:
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    physical_to_logical_map_cpu: torch.Tensor
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)
    logical_to_all_physical_map_cpu: torch.Tensor  # CPU copy for performance
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)
    # (layers, num_logical_experts)
    logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    cpu_pinned: bool = False
    logical_to_all_pinned_map: torch.Tensor = None # (layers, pinned_expert_num)
    logical_to_all_pinned_map_cpu: torch.Tensor = None
    logical_to_all_physical_plus_virtual_map: torch.Tensor = None
    logical_to_all_physical_plus_virtual_map_cpu: torch.Tensor = None
    logical_to_all_physical_plus_virtual_map_num_valid: torch.Tensor = None
    logical_to_all_physical_plus_virtual_map_num_valid_cpu: torch.Tensor = None
    physical_to_logical_plus_virtual_map_cpu: torch.Tensor = None
    vexpert_id: torch.Tensor = None # (layers, num_gpus, num_vexpert_per_buffer)
    vexpert_count: torch.Tensor = None # (layers, num_gpus)
    logical_to_all_physical_plus_virtual_map_nodes: torch.Tensor = None # (layers, num_logical_experts, num_nodes, X)
    logical_to_all_physical_plus_virtual_map_nodes_num_valid: torch.Tensor = None # (layers, num_logical_experts, num_nodes)
    logical_to_all_physical_map_with_null: torch.Tensor = None # (layers, num_logical_experts, X)
    logical_to_all_physical_map_with_null_cpu: torch.Tensor = None
    logical_to_all_physical_map_with_null_num_valid: torch.Tensor = None # (layers, num_logical_experts)
    # -------------------------------- properties ------------------------------------

    @property
    def num_layers(self) -> int:
        return self.physical_to_logical_map.shape[0]

    @property
    def num_physical_experts(self) -> int:
        if self.cpu_pinned:
            return self.physical_to_logical_map.shape[1] + self.vexpert_id.shape[2] * self.ep_size 
        return self.physical_to_logical_map.shape[1]

    @property
    def num_local_physical_experts(self) -> int:
        ans, remainder = divmod(self.num_physical_experts, self.ep_size)
        assert remainder == 0
        return ans

    @property
    def num_logical_experts(self) -> int:
        return self.logical_to_all_physical_map.shape[1]

    @property
    def ep_size(self):
        # TODO change when EP size != world size
        return torch.distributed.get_world_size()

    def __post_init__(self):
        num_layers_0, num_physical_experts_0 = self.physical_to_logical_map.shape
        num_layers_1, num_logical_experts_0, num_physical_experts_1 = (
            self.logical_to_all_physical_map.shape
        )
        num_layers_2, num_logical_experts_1 = (
            self.logical_to_all_physical_map_num_valid.shape
        )
        assert num_layers_0 == num_layers_1 == num_layers_2
        assert num_logical_experts_0 == num_logical_experts_1
        assert num_physical_experts_0 == num_physical_experts_1

    # -------------------------------- construction ------------------------------------

    @staticmethod
    def init_trivial(server_args: ServerArgs, model_config: ModelConfig):
        """Trivial location - logical expert i corresponds to physical expert i"""
        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        num_physical_experts = common["num_physical_experts"]
        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_layers = model_config_for_expert_location.num_layers
        num_logical_experts = model_config_for_expert_location.num_logical_experts

        physical_to_logical_map = (
            torch.arange(0, num_physical_experts).repeat(num_layers, 1)
            % num_logical_experts
        )

        return ExpertLocationMetadata.init_by_mapping(
            server_args,
            model_config,
            physical_to_logical_map=physical_to_logical_map,
        )

    @staticmethod
    def init_by_mapping(
        server_args: ServerArgs,
        model_config: ModelConfig,
        physical_to_logical_map,
    ):
        if not isinstance(physical_to_logical_map, torch.Tensor):
            physical_to_logical_map = torch.tensor(physical_to_logical_map)
        physical_to_logical_map = physical_to_logical_map.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        logical_to_all_physical_map = _compute_logical_to_all_physical_map(
            physical_to_logical_map,
            num_logical_experts=model_config_for_expert_location.num_logical_experts,
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
        )

    @staticmethod
    def init_by_eplb(
        server_args: ServerArgs, model_config: ModelConfig, logical_count: torch.Tensor
    ):
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.to(server_args.device)

        common = ExpertLocationMetadata._init_common(server_args, model_config)

        if common is None:
            return None

        model_config_for_expert_location = common["model_config_for_expert_location"]
        num_physical_experts = common["num_physical_experts"]
        num_groups = model_config_for_expert_location.num_groups
        num_nodes = server_args.nnodes

        physical_to_logical_map, logical_to_all_physical_map, expert_count = (
            eplb_algorithms.rebalance_experts(
                tokens_per_expert=logical_count,
                num_physical_experts=num_physical_experts,
                num_local_physical_experts=num_physical_experts // common["ep_size"],
                num_groups=num_groups,
                num_nodes=num_nodes,
                algorithm=eplb_algorithms.compute_algorithm(
                    raw_algorithm=server_args.eplb_algorithm,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                ),
            )
        )

        return ExpertLocationMetadata._init_raw(
            server_args=server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=physical_to_logical_map.to(server_args.device),
            logical_to_all_physical_map=logical_to_all_physical_map.to(
                server_args.device
            ),
        )

    @staticmethod
    def _init_common(server_args: ServerArgs, model_config: ModelConfig):
        model_config_for_expert_location = (
            ModelConfigForExpertLocation.from_model_config(model_config)
        )

        if model_config_for_expert_location is None:
            return None

        num_physical_experts = (
            model_config_for_expert_location.num_logical_experts
            + server_args.ep_num_redundant_experts
        )
        ep_size = server_args.ep_size
        assert num_physical_experts % ep_size == 0
        num_local_physical_experts = num_physical_experts // ep_size

        return dict(
            model_config_for_expert_location=model_config_for_expert_location,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            ep_size=ep_size,
        )

    @staticmethod
    def _init_raw(
        server_args: ServerArgs,
        ep_size: int,
        physical_to_logical_map: torch.Tensor,
        logical_to_all_physical_map: torch.Tensor,
    ):
        _, num_physical_experts = physical_to_logical_map.shape

        logical_to_all_physical_map_padded = F.pad(
            logical_to_all_physical_map,
            (0, num_physical_experts - logical_to_all_physical_map.shape[-1]),
            value=-1,
        )

        logical_to_all_physical_map_num_valid = torch.count_nonzero(
            logical_to_all_physical_map != -1, dim=-1
        )

        return ExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            physical_to_logical_map_cpu=physical_to_logical_map.cpu(),
            logical_to_all_physical_map=logical_to_all_physical_map_padded,
            logical_to_all_physical_map_cpu=logical_to_all_physical_map_padded.cpu(),
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            logical_to_rank_dispatch_physical_map=(
                compute_logical_to_rank_dispatch_physical_map(
                    logical_to_all_physical_map=logical_to_all_physical_map,
                    num_gpus=ep_size,
                    num_physical_experts=num_physical_experts,
                    # TODO improve when we have real EP rank
                    ep_rank=torch.distributed.get_rank() % ep_size,
                )
                if server_args.ep_dispatch_algorithm == "static"
                else None
            ),
        )

    # -------------------------------- mutation ------------------------------------

    def update(
        self,
        other: "ExpertLocationMetadata",
        update_layer_ids: List[int],
    ):
        for field in [
            "ep_size",
        ]:
            assert getattr(self, field) == getattr(other, field)

        for field in [
            "physical_to_logical_map",
            "physical_to_logical_map_cpu",
            "logical_to_all_physical_map",
            "logical_to_all_physical_map_cpu",
            "logical_to_all_physical_map_num_valid",
            "logical_to_rank_dispatch_physical_map",
        ]:
            other_field = getattr(other, field)
            self_field = getattr(self, field)
            assert (other_field is not None) == (self_field is not None)
            if self_field is not None:
                mask_update = torch.tensor(
                    [i in update_layer_ids for i in range(self.num_layers)]
                )
                mask_update = mask_update.view(*([-1] + [1] * (self_field.dim() - 1)))
                mask_update = mask_update.to(self_field.device, non_blocking=True)
                self_field[...] = torch.where(mask_update, other_field, self_field)

    # -------------------------------- usage ------------------------------------

    def logical_to_all_physical(
        self, layer_id: int, logical_expert_id: int
    ) -> List[int]:
        # Use CPU copy to avoid GPU→CPU sync on every call, which is expensive in update weights scenario
        return [
            physical_expert_id
            for physical_expert_id in self.logical_to_all_physical_map_cpu[
                layer_id, logical_expert_id
            ].tolist()
            if physical_expert_id != -1
        ]

    # -------------------------------- virtual ------------------------------------
    def init_virtual(
        self, server_args: ServerArgs, model_config: ModelConfig, logical_count: torch.Tensor,
    ):
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.to(server_args.device)
        logical_count = logical_count.sum(dim=0).cpu()
        layers = self.logical_to_all_physical_map.shape[0]
        num_logical_experts = self.logical_to_all_physical_map.shape[1]
        pinned_expert_num = server_args.pinned_expert_num
        
        logical_to_all_pinned_map = torch.zeros((layers, pinned_expert_num), dtype=torch.long, device=server_args.device)
        '''
        # Select pinned_expert_num experts from existing experts to store in pinned memory
        # Current logic: pin the top experts on each rank, TODO: needs optimization
        for layer_id in range(layers):
            cpu_expert_map = []
            for r in range(torch.distributed.get_world_size()):
                num_local_experts = num_logical_experts // torch.distributed.get_world_size()
                cpu_expert_per_rank = pinned_expert_num // torch.distributed.get_world_size()
                # Expert IDs on rank R
                current_expert_ids = self.physical_to_logical_map_cpu \
                                [layer_id] \
                                [r * num_local_experts : (r + 1) * num_local_experts].cpu()
                expert_count = logical_count[layer_id][current_expert_ids]
                t_ids = torch.sort(expert_count,descending = True).indices
                max_expert_id= current_expert_ids[t_ids][:cpu_expert_per_rank]
                cpu_expert_map.extend(max_expert_id.tolist())
            cpu_expert_map = torch.tensor(cpu_expert_map)
            # test
            # cpu_expert_map = torch.arange(0,32,dtype=torch.int32)
            logical_to_all_pinned_map[layer_id, :] = cpu_expert_map
        '''
        for layer_id in range(layers):
            # Top pinned_expert_num experts from logical_count
            top_experts = logical_count[layer_id].topk(pinned_expert_num, dim=-1)[1]
            logical_to_all_pinned_map[layer_id, :] = top_experts

        self.logical_to_all_pinned_map = logical_to_all_pinned_map
        self.logical_to_all_pinned_map_cpu = logical_to_all_pinned_map.cpu()
        
            
    def logical_to_all_pinned(
        self, layer_id: int, logical_expert_id: int
    ) -> List[int]:
        expert_ids = self.logical_to_all_pinned_map[layer_id]
        return True if logical_expert_id in expert_ids else False
    
    def compute_virtual(
        self,
        logical_count: torch.Tensor,
        num_vexpert_per_buffer: int,
        num_gpus: int,
        bs: int = 256
    ):
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        num_replicas = self.logical_to_all_physical_map.shape[1] + num_vexpert_per_buffer * num_gpus
        rank = torch.distributed.get_rank() % 8
        device = torch.device(f"cuda:{rank}")
        phy2log, log2phy, logcnt, vexpert_id, vexpert_count, log2phy_nodes, logcnt_nodes, log2phy_null= eplb_algorithms.rebalance_experts_virtual(
            tokens_per_expert=logical_count.to(device)  ,
            num_vexpert_per_buffer=num_vexpert_per_buffer,
            num_replicas=num_replicas,
            num_gpus=num_gpus,
            logical_to_pinned_map=self.logical_to_all_pinned_map,
            physical_to_logical_map_non_pinned=self.physical_to_logical_map,
            logical_to_physical_map_non_pinned=self.logical_to_all_physical_map,
            expert_count_non_pinned=self.logical_to_all_physical_map_num_valid.to(device),
            bs = bs
        )
        # Return in cpu for async copy
        logical_to_all_physical_plus_virtual_map = log2phy.cpu()
        logical_to_all_physical_plus_virtual_map_cpu = logical_to_all_physical_plus_virtual_map.cpu()
        logical_to_all_physical_plus_virtual_map_num_valid = logcnt.cpu()
        physical_to_logical_plus_virtual_map = phy2log.cpu()
        vexpert_id = vexpert_id.cpu()
        vexpert_count = vexpert_count.cpu()
        logical_to_all_physical_plus_virtual_map_nodes = log2phy_nodes.cpu()
        logical_to_all_physical_plus_virtual_map_nodes_num_valid = logcnt_nodes.cpu()
        logical_to_all_physical_map_with_null = log2phy_null.cpu()
        return logical_to_all_physical_plus_virtual_map, logical_to_all_physical_plus_virtual_map_cpu, logical_to_all_physical_plus_virtual_map_num_valid, physical_to_logical_plus_virtual_map, vexpert_id, vexpert_count, logical_to_all_physical_plus_virtual_map_nodes, logical_to_all_physical_plus_virtual_map_nodes_num_valid, logical_to_all_physical_map_with_null
    def update_virtual(
        self,
        logical_count: torch.Tensor,
        num_vexpert_per_buffer: int,
        num_gpus: int,
        logical_node_count: Optional[torch.Tensor] = None,
        rank_ratio: Optional[torch.Tensor] = None,
        bs: int = 256
    ):
        if not isinstance(logical_count, torch.Tensor):
            logical_count = torch.tensor(logical_count)
        if len(logical_count.shape) == 2:
            logical_count = logical_count.unsqueeze(0)
        logical_count = logical_count.cpu()
        num_replicas = self.logical_to_all_physical_map.shape[1] + num_vexpert_per_buffer * num_gpus
        rank = torch.distributed.get_rank() % 8
        device = torch.device(f"cuda:{rank}")
        phy2log, log2phy, logcnt, vexpert_id, vexpert_count, log2phy_nodes, logcnt_nodes, log2phy_null= eplb_algorithms.rebalance_experts_virtual(
            tokens_per_expert=logical_count.to(device),
            num_vexpert_per_buffer=num_vexpert_per_buffer,
            num_replicas=num_replicas,
            num_gpus=num_gpus,
            logical_to_pinned_map=self.logical_to_all_pinned_map_cpu.to(device),
            physical_to_logical_map_non_pinned=self.physical_to_logical_map_cpu.to(device),
            logical_to_physical_map_non_pinned=self.logical_to_all_physical_map_cpu.to(device),
            expert_count_non_pinned=self.logical_to_all_physical_map_num_valid.to(device),
            bs = bs
        )
        phy2log, log2phy, logcnt, vexpert_id, vexpert_count, log2phy_nodes, logcnt_nodes, log2phy_null = phy2log.to(device), log2phy.to(device), logcnt.to(device), vexpert_id.to(device), vexpert_count.to(device), log2phy_nodes.to(device), logcnt_nodes.to(device), log2phy_null.to(device)
        self.cpu_pinned = True
        # 3-layer concatenation - use in-place operations to support CUDA graph capture
        # Ensure new tensor is on the correct device
        rank = torch.distributed.get_rank() % 8
        device = torch.device(f"cuda:{rank}")
        
        temp = torch.full((3,log2phy.shape[1],log2phy.shape[2]),-1,dtype=torch.int32,device=device)
        new_logical_to_all_physical_plus_virtual_map = torch.cat((temp,log2phy),dim=0).to(device)
        if not new_logical_to_all_physical_plus_virtual_map.is_cuda:
            new_logical_to_all_physical_plus_virtual_map = new_logical_to_all_physical_plus_virtual_map.cuda()
        
        if self.logical_to_all_physical_plus_virtual_map is not None and \
           self.logical_to_all_physical_plus_virtual_map.shape == new_logical_to_all_physical_plus_virtual_map.shape and \
           self.logical_to_all_physical_plus_virtual_map.device == new_logical_to_all_physical_plus_virtual_map.device:
            self.logical_to_all_physical_plus_virtual_map.copy_(new_logical_to_all_physical_plus_virtual_map, non_blocking=False)
        else:
            self.logical_to_all_physical_plus_virtual_map = new_logical_to_all_physical_plus_virtual_map
        self.logical_to_all_physical_plus_virtual_map_cpu = self.logical_to_all_physical_plus_virtual_map.cpu()
        
        temp_num_valid = torch.full((3,logcnt.shape[1]),0,dtype=torch.int32,device=device)
        new_logical_to_all_physical_plus_virtual_map_num_valid = torch.cat((temp_num_valid,logcnt),dim=0).to(device)
        if not new_logical_to_all_physical_plus_virtual_map_num_valid.is_cuda:
            new_logical_to_all_physical_plus_virtual_map_num_valid = new_logical_to_all_physical_plus_virtual_map_num_valid.cuda()
        
        if self.logical_to_all_physical_plus_virtual_map_num_valid is not None and \
           self.logical_to_all_physical_plus_virtual_map_num_valid.shape == new_logical_to_all_physical_plus_virtual_map_num_valid.shape and \
           self.logical_to_all_physical_plus_virtual_map_num_valid.device == new_logical_to_all_physical_plus_virtual_map_num_valid.device:
            self.logical_to_all_physical_plus_virtual_map_num_valid.copy_(new_logical_to_all_physical_plus_virtual_map_num_valid, non_blocking=False)
        else:
            self.logical_to_all_physical_plus_virtual_map_num_valid = new_logical_to_all_physical_plus_virtual_map_num_valid
        
        if self.logical_to_all_physical_plus_virtual_map_num_valid_cpu is not None:
            self.logical_to_all_physical_plus_virtual_map_num_valid_cpu.copy_(new_logical_to_all_physical_plus_virtual_map_num_valid.cpu(), non_blocking=False)
        else:
            self.logical_to_all_physical_plus_virtual_map_num_valid_cpu = new_logical_to_all_physical_plus_virtual_map_num_valid.cpu()
        
        temp = torch.full((3,phy2log.shape[1]),-1,dtype=torch.int32,device=device)
        new_physical_to_logical_plus_virtual_map = torch.cat((temp,phy2log),dim=0).to(device)
        if not new_physical_to_logical_plus_virtual_map.is_cuda:
            new_physical_to_logical_plus_virtual_map = new_physical_to_logical_plus_virtual_map.cuda()
        
        if self.physical_to_logical_plus_virtual_map_cpu is not None:
            self.physical_to_logical_plus_virtual_map_cpu.copy_(new_physical_to_logical_plus_virtual_map.cpu(), non_blocking=False)
        else:
            self.physical_to_logical_plus_virtual_map_cpu = new_physical_to_logical_plus_virtual_map.cpu()
        
        temp_vexpert_id = torch.full((3,vexpert_id.shape[1],vexpert_id.shape[2]),-1,dtype=torch.int32,device=device)
        new_vexpert_id = torch.cat((temp_vexpert_id,vexpert_id),dim=0).to(device)
        if not new_vexpert_id.is_cuda:
            new_vexpert_id = new_vexpert_id.cuda()
        
        if self.vexpert_id is not None:
            self.vexpert_id.copy_(new_vexpert_id.cpu(), non_blocking=False)
        else:
            self.vexpert_id = new_vexpert_id.cpu()
        
        temp_vexpert_count = torch.full((3,vexpert_count.shape[1]),0,dtype=torch.int32,device="cpu")
        new_vexpert_count = torch.cat((temp_vexpert_count,vexpert_count.cpu()),dim=0)
        
        if self.vexpert_count is not None:
            self.vexpert_count.copy_(new_vexpert_count.cpu(), non_blocking=False)
        else:
            self.vexpert_count = new_vexpert_count.cpu()
        
        temp_nodes = torch.full((3,log2phy_nodes.shape[1],log2phy_nodes.shape[2],log2phy_nodes.shape[3]),-1,dtype=torch.int32,device=device)
        new_logical_to_all_physical_plus_virtual_map_nodes = torch.cat((temp_nodes,log2phy_nodes),dim=0).to(device)
        if not new_logical_to_all_physical_plus_virtual_map_nodes.is_cuda:
            new_logical_to_all_physical_plus_virtual_map_nodes = new_logical_to_all_physical_plus_virtual_map_nodes.cuda()
        
        if self.logical_to_all_physical_plus_virtual_map_nodes is not None:
            self.logical_to_all_physical_plus_virtual_map_nodes.copy_(new_logical_to_all_physical_plus_virtual_map_nodes, non_blocking=False)
        else:
            self.logical_to_all_physical_plus_virtual_map_nodes = new_logical_to_all_physical_plus_virtual_map_nodes
        
        temp_num_valid_nodes = torch.full((3,logcnt_nodes.shape[1],logcnt_nodes.shape[2]),0,dtype=torch.int32,device=device)
        new_logical_to_all_physical_plus_virtual_map_nodes_num_valid = torch.cat((temp_num_valid_nodes,logcnt_nodes),dim=0).to(device)
        if not new_logical_to_all_physical_plus_virtual_map_nodes_num_valid.is_cuda:
            new_logical_to_all_physical_plus_virtual_map_nodes_num_valid = new_logical_to_all_physical_plus_virtual_map_nodes_num_valid.cuda()
        
        if self.logical_to_all_physical_plus_virtual_map_nodes_num_valid is not None:
            self.logical_to_all_physical_plus_virtual_map_nodes_num_valid.copy_(new_logical_to_all_physical_plus_virtual_map_nodes_num_valid, non_blocking=False)
        else:
            self.logical_to_all_physical_plus_virtual_map_nodes_num_valid = new_logical_to_all_physical_plus_virtual_map_nodes_num_valid
        
        temp_null = torch.full((3,log2phy_null.shape[1],log2phy_null.shape[2]),-1,dtype=torch.int32,device=device)
        new_logical_to_all_physical_map_with_null = torch.cat((temp_null,log2phy_null),dim=0).to(device)
        
        if self.logical_to_all_physical_map_with_null is not None:
            self.logical_to_all_physical_map_with_null.copy_(new_logical_to_all_physical_map_with_null, non_blocking=False)
        else:
            self.logical_to_all_physical_map_with_null = new_logical_to_all_physical_map_with_null
    
        if self.logical_to_all_physical_map_with_null_cpu is not None:
            self.logical_to_all_physical_map_with_null_cpu.copy_(new_logical_to_all_physical_map_with_null.cpu(), non_blocking=False)
        else:
            self.logical_to_all_physical_map_with_null_cpu = new_logical_to_all_physical_map_with_null.cpu()
        
    def update_virtual_direct(self,
        args: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        #self.logical_to_all_physical_plus_virtual_map[3:,:,:].copy_(args[0], non_blocking=False)
        self.logical_to_all_physical_plus_virtual_map_cpu[3:,:,:].copy_(args[1], non_blocking=False)
        self.logical_to_all_physical_plus_virtual_map_num_valid_cpu[3:,:].copy_(args[2], non_blocking=False)
        self.physical_to_logical_plus_virtual_map_cpu[3:,:].copy_(args[3], non_blocking=False)
        self.vexpert_id[3:,:,:].copy_(args[4].cpu(), non_blocking=False)
        self.vexpert_count[3:,:].copy_(args[5].cpu(), non_blocking=False)
        #self.logical_to_all_physical_plus_virtual_map_nodes[3:,:,:,:].copy_(args[6], non_blocking=False)
        #self.logical_to_all_physical_plus_virtual_map_nodes_num_valid[3:,:,:].copy_(args[7], non_blocking=False)
        self.logical_to_all_physical_map_with_null_cpu[3:,:,:].copy_(args[8], non_blocking=False)
_global_expert_location_metadata: Optional[ExpertLocationMetadata] = None

def compute_virtual(
    logical_to_all_physical_map: torch.Tensor,
    logical_to_all_pinned_map: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    logical_to_all_physical_map_num_valid: torch.Tensor,
    logical_count: torch.Tensor,
    num_vexpert_per_buffer: int,
    num_gpus: int,
    bs: int = 256
):
    if not isinstance(logical_count, torch.Tensor):
        logical_count = torch.tensor(logical_count)
    if len(logical_count.shape) == 2:
        logical_count = logical_count.unsqueeze(0)
    logical_count = logical_count.cpu()
    num_replicas = logical_to_all_physical_map.shape[1] + num_vexpert_per_buffer * num_gpus
    phy2log, log2phy, logcnt, vexpert_id, vexpert_count, log2phy_nodes, logcnt_nodes, log2phy_null= eplb_algorithms.rebalance_experts_virtual(
        tokens_per_expert=logical_count,
        num_vexpert_per_buffer=num_vexpert_per_buffer,
        num_replicas=num_replicas,
        num_gpus=num_gpus,
        logical_to_pinned_map=logical_to_all_pinned_map,
        physical_to_logical_map_non_pinned=physical_to_logical_map,
        logical_to_physical_map_non_pinned=logical_to_all_physical_map,
        expert_count_non_pinned=logical_to_all_physical_map_num_valid,
        bs = bs
    )
    
    logical_to_all_physical_plus_virtual_map = log2phy
    logical_to_all_physical_plus_virtual_map_cpu = logical_to_all_physical_plus_virtual_map.cpu()
    logical_to_all_physical_plus_virtual_map_num_valid = logcnt
    physical_to_logical_plus_virtual_map = phy2log
    vexpert_id = vexpert_id
    vexpert_count = vexpert_count
    logical_to_all_physical_plus_virtual_map_nodes = log2phy_nodes
    logical_to_all_physical_plus_virtual_map_nodes_num_valid = logcnt_nodes
    logical_to_all_physical_map_with_null = log2phy_null
    
    return logical_to_all_physical_plus_virtual_map, logical_to_all_physical_plus_virtual_map_cpu, logical_to_all_physical_plus_virtual_map_num_valid, physical_to_logical_plus_virtual_map, vexpert_id, vexpert_count, logical_to_all_physical_plus_virtual_map_nodes, logical_to_all_physical_plus_virtual_map_nodes_num_valid, logical_to_all_physical_map_with_null
def get_global_expert_location_metadata():
    return _global_expert_location_metadata


def set_global_expert_location_metadata(value):
    global _global_expert_location_metadata
    assert _global_expert_location_metadata is None
    _global_expert_location_metadata = value


def _compute_logical_to_all_physical_map(
    physical_to_logical_map: torch.Tensor, num_logical_experts: int
):
    # This is rarely called, so we use for loops for maximum clarity

    num_layers, num_physical_experts = physical_to_logical_map.shape

    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]
    for layer_id in range(num_layers):
        for physical_expert_id in range(num_physical_experts):
            logical_expert_id = physical_to_logical_map[
                layer_id, physical_expert_id
            ].item()
            logical_to_all_physical_map[layer_id][logical_expert_id].append(
                physical_expert_id
            )

    logical_to_all_physical_map = _pad_nested_array(
        logical_to_all_physical_map, pad_value=-1
    )

    return torch.tensor(
        logical_to_all_physical_map, device=physical_to_logical_map.device
    )


def _pad_nested_array(arr, pad_value):
    max_len = max(len(inner) for outer in arr for inner in outer)
    padded = [
        [inner + [pad_value] * (max_len - len(inner)) for inner in outer]
        for outer in arr
    ]
    return padded


# TODO optimize performance (rewrite and/or run in separate process with overlap)
def compute_logical_to_rank_dispatch_physical_map(
    logical_to_all_physical_map: torch.Tensor,
    num_gpus: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):
    r = random.Random(seed)

    num_local_physical_experts = num_physical_experts // num_gpus
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    logical_to_rank_dispatch_physical_map = torch.full(
        size=(num_gpus, num_layers, num_logical_experts),
        fill_value=-1,
        dtype=dtype,
    )

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            candidate_physical_expert_ids = _logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )
            output_partial = logical_to_rank_dispatch_physical_map[
                :, layer_id, logical_expert_id
            ]

            for gpu_id in range(num_gpus):
                same_gpu_physical_expert_ids = [
                    physical_expert_id
                    for physical_expert_id in candidate_physical_expert_ids
                    if _compute_gpu_id_of_physical_expert(
                        physical_expert_id, num_local_physical_experts
                    )
                    == gpu_id
                ]
                if len(same_gpu_physical_expert_ids) > 0:
                    output_partial[gpu_id] = same_gpu_physical_expert_ids[0]

            num_remain = torch.sum(output_partial == -1).item()
            output_partial[output_partial == -1] = torch.tensor(
                _fair_choices(candidate_physical_expert_ids, k=num_remain, r=r),
                dtype=dtype,
            )

    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    device = logical_to_all_physical_map.device
    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)


def _logical_to_all_physical_raw(
    logical_to_all_physical_map, layer_id: int, logical_expert_id: int
) -> List[int]:
    return [
        physical_expert_id
        for physical_expert_id in logical_to_all_physical_map[
            layer_id, logical_expert_id
        ].tolist()
        if physical_expert_id != -1
    ]


def _compute_gpu_id_of_physical_expert(
    physical_expert_id: int, num_local_physical_experts: int
) -> int:
    return physical_expert_id // num_local_physical_experts


def _fair_choices(arr: List, k: int, r: random.Random) -> List:
    quotient, remainder = divmod(k, len(arr))
    ans = arr * quotient + r.sample(arr, k=remainder)
    r.shuffle(ans)
    return ans


@dataclass
class ModelConfigForExpertLocation:
    num_layers: int
    num_logical_experts: int
    num_groups: Optional[int] = None

    @staticmethod
    def from_model_config(model_config: ModelConfig):
        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_model_config_for_expert_location"):
            return model_class.get_model_config_for_expert_location(
                model_config.hf_config
            )
        else:
            return None


def compute_initial_expert_location_metadata(
    server_args: ServerArgs, model_config: ModelConfig
) -> Optional[ExpertLocationMetadata]:
    data = server_args.init_expert_location
    if data == "trivial":
        return ExpertLocationMetadata.init_trivial(server_args, model_config)

    # TODO unify with the utils function
    if data.endswith(".pt"):
        data_dict = torch.load(data, weights_only=True)
    elif data.endswith(".json"):
        data_dict = json.loads(Path(data).read_text())
    else:
        data_dict = json.loads(data)

    if "physical_to_logical_map" in data_dict:
        logger.info(
            "init_expert_location from init_by_mapping using ServerArgs.init_expert_location"
        )
        return ExpertLocationMetadata.init_by_mapping(
            server_args, model_config, **data_dict
        )
    elif "logical_count" in data_dict:
        logger.info(
            "init_expert_location from init_by_eplb using ServerArgs.init_expert_location"
        )
        logical_count = data_dict["logical_count"]
        obj = ExpertLocationMetadata.init_by_eplb(
            server_args, model_config, logical_count=logical_count
        )
        bss = server_args.cuda_graph_bs
        bs_list = [bs for bs in bss]
        bs = max(bs_list)
        if server_args.enable_cpu_pinned:
            obj.init_virtual(server_args, model_config, logical_count=logical_count)
            obj.update_virtual(logical_count=logical_count, num_vexpert_per_buffer=server_args.num_vexpert_buffer, num_gpus=server_args.ep_size, bs = bs)
        return obj
    else:
        raise NotImplementedError(
            f"Unknown init_expert_location format ({list(data_dict.keys())=})"
        )

def get_id_pairs(rank, num_vexpert_buffer):
    expert_location_metadata = get_global_expert_location_metadata()
    vexpert_id = expert_location_metadata.vexpert_id.clone() # [layers, num_gpus, num_vexpert_buffer]
    vexpert_count= expert_location_metadata.vexpert_count.clone() # [layers, num_gpus]
    vexpert_id = vexpert_id.view(-1,vexpert_id.shape[1] // 8, 8, vexpert_id.shape[2])
    vexpert_count = vexpert_count.view(-1, vexpert_count.shape[1] // 8, 8)
    node_id = rank // 8
    layers_id_pairs = []
    for layer_id in range(0,vexpert_id.shape[0] - 3,2):
        start_layer_id = layer_id + 3
        id_pairs = []
        for d in range(8):
            id_pairs_t = [(-1,-1)] * num_vexpert_buffer
            vcount1 = vexpert_count[start_layer_id,node_id,d]
            vcount2 = vexpert_count[start_layer_id+1,node_id,d]
            for k in range(vcount1):
                id_pairs_t[k] = (start_layer_id,vexpert_id[start_layer_id,node_id,d,k].item())
            for k in range(vcount2):
                id_pairs_t[num_vexpert_buffer - vcount2 + k] = (start_layer_id+1,vexpert_id[start_layer_id+1,node_id,d,k].item())
            id_pairs.extend(id_pairs_t)
        layers_id_pairs.append(id_pairs)
    return layers_id_pairs