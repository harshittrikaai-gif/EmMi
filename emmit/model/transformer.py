"""
Emmit transformer block and full model.

Assembles attention, MoE FFN, and RMSNorm into a complete
auto-regressive language model with optional vision inputs.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from emmit.model.attention import GroupedQueryAttention
from emmit.model.config import EmmitConfig
from emmit.model.moe import MoEFeedForward
from emmit.model.norms import RMSNorm
from emmit.model.vision import VisionEncoder


class EmmitTransformerBlock(nn.Module):
    """Single transformer block: LN → Attention → residual → LN → MoE FFN → residual."""

    def __init__(self, config: EmmitConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_checkpointing = config.gradient_checkpointing

        self.ln1 = RMSNorm(config.hidden_size)
        self.attention = GroupedQueryAttention(config)
        self.ln2 = RMSNorm(config.hidden_size)
        self.moe_ffn = MoEFeedForward(config)

    # ------------------------------------------------------------------

    def _attention_forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.attention(self.ln1(hidden_states), attention_mask)

    def _ffn_forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.moe_ffn(self.ln2(hidden_states))

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # --- Attention ---
        if self.use_checkpointing and self.training:
            attn_out = checkpoint.checkpoint(
                self._attention_forward,
                hidden_states,
                attention_mask,
                use_reentrant=False,
            )
        else:
            attn_out = self._attention_forward(hidden_states, attention_mask)

        hidden_states = hidden_states + attn_out

        # --- MoE FFN ---
        if self.use_checkpointing and self.training:
            # checkpoint doesn't natively return aux dicts, so we wrap
            ffn_out, aux_loss = checkpoint.checkpoint(
                self._ffn_forward,
                hidden_states,
                use_reentrant=False,
            )
        else:
            ffn_out, aux_loss = self._ffn_forward(hidden_states)

        hidden_states = hidden_states + ffn_out

        return hidden_states, aux_loss


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class EmmitModel(nn.Module):
    """
    Complete Emmit MoE language model.

    Pipeline:
      token ids  →  embedding  →  N × TransformerBlock  →  RMSNorm  →  lm_head
    """

    def __init__(self, config: EmmitConfig):
        super().__init__()
        self.config = config

        # Token + position embedding (RoPE is inside attention, so no pos emb here)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [EmmitTransformerBlock(config, i) for i in range(config.num_layers)]
        )

        # Final norm
        self.norm = RMSNorm(config.hidden_size)

        # Vision Encoder (optional)
        self.vision_encoder = None
        if config.vision_enabled:
            self.vision_encoder = VisionEncoder(
                image_size=config.image_size,
                patch_size=config.patch_size,
                hidden_size=config.vision_hidden_size,
                num_layers=config.vision_num_layers,
                num_heads=config.num_attention_heads, # Use same heads for simplicity or add vision_num_heads
                projection_dim=config.hidden_size,
            )

        # Language model head (weight-tied with embedding)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

        # Auxiliary loss coefficients
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef

        # Init weights
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      [batch, seq_len]
            attention_mask: optional [batch, 1, seq_len, seq_len] or None
            labels:         [batch, seq_len]  (shifted inside this method)
            pixel_values:   optional [batch, 3, H, W]
        """
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)  # [B, S, H]

        # --- Vision Integration (Multimodal) ---
        if pixel_values is not None and self.vision_encoder is not None:
            vision_tokens = self.vision_encoder(pixel_values) # [B, num_v, H]
            # Simple prepend: [vision tokens, text tokens]
            hidden_states = torch.cat([vision_tokens, hidden_states], dim=1)
            
            # Update attention mask to include vision prefix
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    v_mask = torch.ones(
                        (batch_size, vision_tokens.shape[1]), 
                        device=attention_mask.device, 
                        dtype=attention_mask.dtype
                    )
                    attention_mask = torch.cat([v_mask, attention_mask], dim=1)

        # Accumulate auxiliary losses across layers
        total_load_loss = torch.tensor(0.0, device=hidden_states.device)
        total_z_loss = torch.tensor(0.0, device=hidden_states.device)
        total_utilization = torch.zeros(self.config.num_experts, device=hidden_states.device)

        for layer in self.layers:
            hidden_states, aux = layer(hidden_states, attention_mask)
            total_load_loss = total_load_loss + aux["load_balancing_loss"]
            total_z_loss = total_z_loss + aux["z_loss"]
            total_utilization = total_utilization + aux["expert_utilization"]

        hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)  # [B, S, V]

        # --- Combine auxiliary losses ---
        num_layers = len(self.layers)
        aux_loss = {
            "load_balancing_loss": total_load_loss / num_layers,
            "z_loss": total_z_loss / num_layers,
            "expert_utilization": total_utilization / num_layers,
        }

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": aux_loss,
        }

        # --- Language-modelling loss (causal) ---
        if labels is not None:
            # Shift so that token i predicts token i+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Total loss with weighted aux
            total_loss = (
                loss
                + self.router_aux_loss_coef * aux_loss["load_balancing_loss"]
                + self.router_z_loss_coef * aux_loss["z_loss"]
            )

            result["loss"] = loss
            result["total_loss"] = total_loss

        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count parameters."""
        return sum(
            p.numel() for p in self.parameters() if (not only_trainable or p.requires_grad)
        )

    def num_active_parameters(self) -> int:
        """Approximate *active* parameters per forward pass (top-k experts only)."""
        total = self.num_parameters()
        # Expert parameters are only partially active
        expert_params = sum(
            p.numel()
            for layer in self.layers
            for expert in layer.moe_ffn.experts
            for p in expert.parameters()
        )
        # Only top_k / num_experts fraction of expert params are active
        k = self.config.num_experts_per_token
        e = self.config.num_experts
        active_expert_params = expert_params * k / e
        non_expert_params = total - expert_params
        return int(non_expert_params + active_expert_params)
