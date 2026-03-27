# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Implementation of Hyper-Connections from "Hyper-Connections" (ICLR 2025)
# Paper: https://arxiv.org/abs/2409.19606

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperConnection(nn.Module):
    """
    Hyper-Connections module that can serve as an alternative to residual connections.
    
    This implementation supports both Static Hyper-Connections (SHC) and 
    Dynamic Hyper-Connections (DHC).
    
    The hyper-connection matrix HC is structured as:
        HC = [[0,      B    ],
              [Am,     Ar   ]]
    
    where:
        - B (1 x n): weights for the layer output
        - Am (n x 1): weights for computing layer input from hyper hidden matrix
        - Ar (n x n): weights for width connections between hyper hidden vectors
    
    Args:
        hidden_size: Dimension of the hidden states
        expansion_rate: Number of hyper hidden vectors (n in the paper)
        layer_idx: Index of the current layer (used for initialization)
        dynamic: Whether to use dynamic hyper-connections
        use_tanh: Whether to apply tanh to dynamic weights
        norm_type: Type of normalization for dynamic weights ('layer' or 'rms')
    """
    
    def __init__(
        self,
        hidden_size: int,
        expansion_rate: int = 4,
        layer_idx: int = 0,
        dynamic: bool = True,
        use_tanh: bool = True,
        norm_type: str = 'rms',
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.expansion_rate = expansion_rate
        self.layer_idx = layer_idx
        self.dynamic = dynamic
        self.use_tanh = use_tanh
        
        n = expansion_rate
        
        # Static parameters (B, Am, Ar)
        # Initialize B as ones (n,)
        self.static_beta = nn.Parameter(torch.ones(n))
        
        # Initialize Am and Ar according to paper's initialization (Eq. 14)
        # Am is initialized as e_{k mod n} (one-hot based on layer index)
        # Ar is initialized as identity matrix
        init_alpha_m = torch.zeros(n, 1)
        init_alpha_m[layer_idx % n, 0] = 1.0
        
        # Combine Am and Ar into a single parameter for efficiency
        # alpha has shape (n, n+1) where first column is Am and rest is Ar
        init_alpha = torch.cat([init_alpha_m, torch.eye(n)], dim=1)
        self.static_alpha = nn.Parameter(init_alpha)
        
        if dynamic:
            # Dynamic parameters
            # Normalization layer (before computing dynamic weights)
            if norm_type == 'layer':
                self.norm = nn.LayerNorm(hidden_size)
            else:
                self.norm = nn.RMSNorm(hidden_size)
            
            # Linear projections for dynamic weights
            # W_beta: (hidden_size,) -> scalar per hyper hidden
            self.dynamic_beta_proj = nn.Parameter(torch.zeros(hidden_size))
            
            # W_alpha: (hidden_size,) -> (n+1) weights for Am and Ar combined
            self.dynamic_alpha_proj = nn.Parameter(torch.zeros(hidden_size, n + 1))
            
            # Small initial scales for dynamic components (s_beta, s_alpha in paper)
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
    
    def _compute_weights(
        self,
        hyper_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the connection weights (alpha and beta).
        
        Args:
            hyper_hidden: Hyper hidden matrix of shape (batch, seq_len, n, hidden_size)
            
        Returns:
            alpha: Shape (batch, seq_len, n, n+1) or (1, 1, n, n+1) for static
            beta: Shape (batch, seq_len, n) or (1, 1, n) for static
        """
        if self.dynamic:
            # Normalize the hyper hidden states
            # hyper_hidden: (B, L, N, D)
            norm_h = self.norm(hyper_hidden)
            
            # Compute dynamic alpha weights
            # norm_h @ dynamic_alpha_proj: (B, L, N, D) @ (D, N+1) -> (B, L, N, N+1)
            dynamic_alpha = torch.matmul(norm_h, self.dynamic_alpha_proj)
            
            if self.use_tanh:
                dynamic_alpha = torch.tanh(dynamic_alpha)
            
            dynamic_alpha = dynamic_alpha * self.dynamic_alpha_scale
            
            # Add static component: (B, L, N, N+1) + (N, N+1)
            alpha = dynamic_alpha + self.static_alpha.unsqueeze(0).unsqueeze(0)
            
            # Compute dynamic beta weights
            # norm_h @ dynamic_beta_proj: (B, L, N, D) @ (D,) -> (B, L, N)
            dynamic_beta = torch.matmul(norm_h, self.dynamic_beta_proj)
            
            if self.use_tanh:
                dynamic_beta = torch.tanh(dynamic_beta)
            
            dynamic_beta = dynamic_beta * self.dynamic_beta_scale
            
            # Add static component: (B, L, N) + (N,)
            beta = dynamic_beta + self.static_beta.unsqueeze(0).unsqueeze(0)
        else:
            # Static weights only
            alpha = self.static_alpha.unsqueeze(0).unsqueeze(0)
            beta = self.static_beta.unsqueeze(0).unsqueeze(0)
        
        return alpha, beta
    
    def width_connection(
        self,
        hyper_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform width connections to compute layer input and intermediate states.
        
        This implements: mix_h = alpha^T @ H
        where mix_h[:, :, 0, :] is the layer input (h_0)
        and mix_h[:, :, 1:, :] are the intermediate states (H')
        
        Args:
            hyper_hidden: Hyper hidden matrix of shape (batch, seq_len, n, hidden_size)
            
        Returns:
            mix_h: Mixed hidden states of shape (batch, seq_len, n+1, hidden_size)
            beta: Connection weights for depth connections
        """
        # Get connection weights
        alpha, beta = self._compute_weights(hyper_hidden)
        
        # Width connection: mix_h = alpha^T @ H
        # alpha: (B, L, N, N+1) or (1, 1, N, N+1)
        # hyper_hidden: (B, L, N, D)
        # We want: (N+1, N) @ (N, D) -> (N+1, D) for each batch and seq position
        # Using einsum: 'blnm,blnd->blmd' where m = n+1
        alpha_t = alpha.transpose(-2, -1)  # (B, L, N+1, N) or (1, 1, N+1, N)
        mix_h = torch.matmul(alpha_t, hyper_hidden)  # (B, L, N+1, D)
        
        return mix_h, beta
    
    def depth_connection(
        self,
        mix_h: torch.Tensor,
        layer_output: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform depth connections to combine layer output with intermediate states.
        
        This implements: H_out = beta^T * layer_output + H'
        where H' = mix_h[:, :, 1:, :] (the intermediate states from width connection)
        
        Args:
            mix_h: Mixed hidden states from width_connection, shape (batch, seq_len, n+1, hidden_size)
            layer_output: Output from the transformer layer, shape (batch, seq_len, hidden_size)
            beta: Connection weights, shape (batch, seq_len, n) or (1, 1, n)
            
        Returns:
            hyper_hidden: Updated hyper hidden matrix of shape (batch, seq_len, n, hidden_size)
        """
        # H' = mix_h[:, :, 1:, :], shape (B, L, N, D)
        h_prime = mix_h[:, :, 1:, :]
        
        # beta^T * layer_output: broadcast beta across hidden dimension
        # beta: (B, L, N) or (1, 1, N)
        # layer_output: (B, L, D)
        # Result: (B, L, N, D)
        beta_expanded = beta.unsqueeze(-1)  # (B, L, N, 1)
        layer_output_expanded = layer_output.unsqueeze(-2)  # (B, L, 1, D)
        weighted_output = beta_expanded * layer_output_expanded  # (B, L, N, D)
        
        # Depth connection: H_out = beta^T * layer_output + H'
        hyper_hidden = weighted_output + h_prime
        
        return hyper_hidden
    
    def forward(
        self,
        hyper_hidden: torch.Tensor,
        layer_fn: callable,
        **layer_kwargs,
    ) -> Tuple[torch.Tensor, any]:
        """
        Full hyper-connection forward pass.
        
        Args:
            hyper_hidden: Hyper hidden matrix of shape (batch, seq_len, n, hidden_size)
            layer_fn: The layer function to apply (attention or FFN)
            **layer_kwargs: Additional arguments to pass to the layer
            
        Returns:
            hyper_hidden: Updated hyper hidden matrix
            layer_outputs: Additional outputs from the layer (e.g., attention weights, cache)
        """
        # Width connection
        mix_h, beta = self.width_connection(hyper_hidden)
        
        # Extract layer input (h_0)
        layer_input = mix_h[:, :, 0, :]  # (B, L, D)
        
        # Apply layer
        layer_outputs = layer_fn(layer_input, **layer_kwargs)
        
        if isinstance(layer_outputs, tuple):
            layer_output = layer_outputs[0]
            extra_outputs = layer_outputs[1:]
        else:
            layer_output = layer_outputs
            extra_outputs = None
        
        # Depth connection
        hyper_hidden = self.depth_connection(mix_h, layer_output, beta)
        
        if extra_outputs is not None:
            return hyper_hidden, extra_outputs
        return hyper_hidden, None


def expand_to_hyper_hidden(
    hidden_states: torch.Tensor,
    expansion_rate: int,
) -> torch.Tensor:
    """
    Expand hidden states to hyper hidden matrix by replicating n times.
    
    Args:
        hidden_states: Shape (batch, seq_len, hidden_size)
        expansion_rate: Number of copies (n)
        
    Returns:
        hyper_hidden: Shape (batch, seq_len, n, hidden_size)
    """
    # Replicate along a new dimension
    # (B, L, D) -> (B, L, 1, D) -> (B, L, N, D)
    return hidden_states.unsqueeze(-2).expand(-1, -1, expansion_rate, -1).clone()


def collapse_hyper_hidden(
    hyper_hidden: torch.Tensor,
) -> torch.Tensor:
    """
    Collapse hyper hidden matrix to single hidden state by summing.
    
    Args:
        hyper_hidden: Shape (batch, seq_len, n, hidden_size)
        
    Returns:
        hidden_states: Shape (batch, seq_len, hidden_size)
    """
    return hyper_hidden.sum(dim=-2)