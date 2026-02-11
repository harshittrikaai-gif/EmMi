"""
Mixture of Experts layer.

Components:
  • MoERouter  — soft top-k routing with load-balancing + z-loss
  • SwiGLUExpert — single SwiGLU FFN expert
  • MoEFeedForward — sparse MoE layer replacing the dense FFN
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class MoERouter(nn.Module):
    """
    Learned router that selects top-k experts per token.

    Produces two auxiliary losses:
      * **load_balancing_loss** — encourages uniform expert utilisation
      * **z_loss** — penalises large router logits to prevent routing collapse
    """

    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_token: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_token

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            routing_weights : [num_tokens, top_k]  (renormalised)
            selected_experts: [num_tokens, top_k]   (expert indices)
            aux_loss        : dict with ``load_balancing_loss`` and ``z_loss``
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)  # [N, H]

        # Router logits
        router_logits = self.gate(hidden_flat)  # [N, E]

        # Softmax routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)

        # Top-k expert selection
        routing_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )

        # Renormalise selected weights to sum to 1
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Auxiliary losses
        aux_loss = self._compute_auxiliary_loss(router_logits, routing_probs)

        return routing_weights, selected_experts, aux_loss

    # ---------------------------------------------------------------

    def _compute_auxiliary_loss(
        self,
        router_logits: torch.Tensor,
        routing_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute load-balancing and z-loss."""

        # --- Load-balancing loss (Switch Transformer style) ---
        # f_i = fraction of tokens dispatched to expert i
        # P_i = mean routing probability for expert i
        # loss = N * sum(f_i * P_i)
        num_tokens = router_logits.shape[0]

        # One-hot of top-1 expert per token → dispatch fraction
        top1_expert = routing_probs.argmax(dim=-1)
        expert_mask = F.one_hot(top1_expert, self.num_experts).float()  # [N, E]
        tokens_per_expert = expert_mask.mean(dim=0)  # [E]
        prob_per_expert = routing_probs.mean(dim=0)   # [E]

        load_balancing_loss = (
            self.num_experts * (tokens_per_expert * prob_per_expert).sum()
        )

        # --- Z-loss: penalise large logits ---
        z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()

        # --- Expert Utilization (for monitoring) ---
        # count how many tokens chose each expert as top-1
        counts = torch.bincount(top1_expert, minlength=self.num_experts).float()
        utilization = counts / num_tokens

        return {
            "load_balancing_loss": load_balancing_loss,
            "z_loss": z_loss,
            "expert_utilization": utilization
        }


# ---------------------------------------------------------------------------
# Single Expert (SwiGLU FFN)
# ---------------------------------------------------------------------------

class SwiGLUExpert(nn.Module):
    """Feed-forward expert with SwiGLU activation (Shazeer, 2020)."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)  # gate proj
        self.w2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)  # down proj
        self.w3 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)  # up proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down( silu(gate(x)) ⊙ up(x) )
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Sparse MoE Feed-Forward
# ---------------------------------------------------------------------------

class MoEFeedForward(nn.Module):
    """
    Sparse Mixture-of-Experts FFN layer.

    Each token is routed to ``top_k`` experts; outputs are combined by
    the router-assigned weights.  Uses *grouped dispatch* (loop over
    experts, not tokens) for better GPU utilisation.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.capacity_factor = config.expert_capacity_factor

        self.router = MoERouter(
            config.hidden_size,
            config.num_experts,
            config.num_experts_per_token,
        )

        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(config.hidden_size, config.ffn_hidden_size)
                for _ in range(config.num_experts)
            ]
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output   : [batch, seq_len, hidden_size]
            aux_loss : dict with auxiliary losses
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)  # [N, H]

        # Route tokens
        routing_weights, selected_experts, aux_loss = self.router(hidden_states)
        # routing_weights:  [N, top_k]
        # selected_experts: [N, top_k]

        output = torch.zeros_like(hidden_flat)

        # Grouped dispatch: iterate over experts
        for expert_id in range(self.num_experts):
            # Mask: [N, top_k] boolean where selected_experts == expert_id
            expert_mask = selected_experts == expert_id  # [N, top_k]

            # Find (token_idx, k_idx) pairs for this expert
            token_indices, k_indices = expert_mask.nonzero(as_tuple=True)

            if token_indices.numel() == 0:
                continue

            # Gather input tokens and corresponding weights
            expert_input = hidden_flat[token_indices]  # [T_e, H]
            expert_weights = routing_weights[token_indices, k_indices].unsqueeze(-1)  # [T_e, 1]

            # Forward through expert
            expert_output = self.experts[expert_id](expert_input)  # [T_e, H]

            # Weighted scatter-add back
            output.index_add_(0, token_indices, expert_weights * expert_output)

        output = output.view(batch_size, seq_len, hidden_size)
        return output, aux_loss
