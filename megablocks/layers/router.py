# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch

from megablocks.layers import common
from megablocks.layers.arguments import Arguments


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, num_experts: int):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)


_uniform_expert_assignment = _UniformExpertAssignment.apply


class LossFreeRouter(torch.nn.Module):
    """
    Router implementing Loss-Free Balancing for MoE routing.
    Automatically adjusts per-expert biases during training to balance load
    without requiring auxiliary losses.
    """

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device)
        args.init_method(self.layer.weight)

        # Per-expert biases for load balancing
        self.register_buffer(
            'expert_biases',
            torch.zeros(args.moe_num_experts, dtype=common.dtype(args), device=args.device)
        )
        self.bias_update_rate = 0.01  # Can be made configurable via args

    def _sync_biases(self):
        """Synchronize expert biases across DDP processes."""
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.expert_biases)
            self.expert_biases.div_(torch.distributed.get_world_size())

    def _update_biases(self, token_counts, batch_size):
        """Update expert biases based on load violations."""
        if not self.training:
            return

        # Calculate target tokens per expert (average load)
        tokens_per_expert = batch_size * self.args.moe_top_k / self.args.moe_num_experts
        
        # Calculate load violations
        load_violations = token_counts.float() - tokens_per_expert
        
        # Update biases using sign of load violations
        self.expert_biases.add_(
            self.bias_update_rate * load_violations.sign()
        )
        
        # Sync updated biases across processes
        self._sync_biases()

    def _top_k(self, scores):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x, attention_mask=None):
        batch_shape = x.shape[:-1]
        flat_x = x.view(-1, x.shape[-1])
        batch_size = flat_x.shape[0]

        # Get raw routing scores
        raw_scores = self.layer(flat_x)
        
        # Add expert biases to scores
        scores = raw_scores + self.expert_biases
        scores = scores.softmax(dim=-1)

        # Get top-k routing assignments
        expert_weights, expert_indices = self._top_k(scores)

        # Count tokens per expert
        token_counts = torch.bincount(
            expert_indices.view(-1),
            minlength=self.args.moe_num_experts
        )

        # Update biases based on load violations
        self._update_biases(token_counts, batch_size)

        # Reshape outputs to match input batch shape
        scores = scores.view(*batch_shape, -1)
        expert_weights = expert_weights.view(*batch_shape, -1)
        expert_indices = expert_indices.view(*batch_shape, -1)

        return scores, expert_weights, expert_indices


class LearnedRouter(torch.nn.Module):

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device,
        )
        args.init_method(self.layer.weight)

    def jitter(self, x: torch.Tensor):
        assert isinstance(self.args.moe_jitter_eps, float)
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores: torch.Tensor):
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(self, x: torch.Tensor):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)
        scores = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1, dtype=x.dtype)
        expert_weights, expert_indices = self._top_k(scores)
        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights,
                p=self.args.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            )

        expert_indices = (
            _uniform_expert_assignment(
                expert_indices,
                self.args.moe_num_experts,
            ) if self.args.uniform_expert_assignment else expert_indices
        )
        return scores, expert_weights, expert_indices

        
class ExpertChoiceRouter(LearnedRouter):
    def expert_capacity(self, tokens):
        world_size = mpu.get_expert_parallel_world_size(self.args)
        # divide equally to the nearsest integer rounded up
        tokens_per_expert = (tokens * world_size // self.args.moe_num_experts)
        tokens_per_expert += (tokens * world_size % self.args.moe_num_experts) > 0
        return int(self.args.moe_capacity_factor * tokens_per_expert)

    def _top_k(self, scores):
        # use first index since we transpose before passing in 
        tokens = scores.shape[-1]

        top_k = self.expert_capacity(tokens)

        return torch.topk(scores, top_k, dim=-1) 

    def forward(self, x, attention_mask=None):
        # output is shape (sl * bs, num_experts)
        # why do we take softmax then top_k(scores.T) instead of softmax(layer.T)?
        # (bs, sl, hs) -> (bs, sl, num_experts)
        scores = self.layer(x).softmax(dim=-1)

        if attention_mask is not None:
            scores = scores * attention_mask.unsqueeze(-1)

        # (bs, sl, num_experts) -> (bs, num_experts, expert_capacity)
        expert_weights, expert_indices = self._top_k(scores.transpose(1, 2))
        if self.args.moe_normalize_expert_weights:
            batch_size, _, _ = expert_weights.shape
            seq_len = x.shape[1]
            # pseudocode of what's happening below
            # for b in range(batch_size):
            #     row_weights = expert_weights[b, :, :]
            #     row_tokens = expert_indices[b, :, :].unique()
            #     for token in row_tokens:
            #         denom = row_weights[expert_indices[b, :, :] == token].sum()
            #         slow_expert_weights[b, :, :][expert_indices[b, :, :] == token] /= denom

            # denom = torch.ones_like(expert_weights)

            input_shape = expert_indices.shape
            # offset indices by bs * sl
            batch_offset = torch.arange(batch_size, device=expert_indices.device).unsqueeze(1).unsqueeze(2) * seq_len
            offset_indices = expert_indices + batch_offset
            offset_indices_flat = offset_indices.flatten()

            weights = expert_weights.flatten()

            max_index = offset_indices_flat.max()
            output = torch.zeros(max_index + 1, dtype=weights.dtype, device=expert_indices.device, requires_grad=True)
            output = output.scatter_add(0, offset_indices_flat, weights)
            # clamp to avoid nans when we mask out scores for the padded tokens
            output = torch.clamp(output, min=1e-6)

            normalized_weight = weights / output[offset_indices_flat]

            expert_weights = normalized_weight.view(input_shape)

        return scores, expert_weights, expert_indices