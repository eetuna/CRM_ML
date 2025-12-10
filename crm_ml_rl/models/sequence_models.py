"""
Sequence-aware dynamics models.

Contains transformer-based and diffusion-based dynamics models for use in
model-based RL or MPC. These models provide richer temporal context and
stochastic predictions compared to the standard MLP/Hybrid variants.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDynamicsModel(nn.Module):
    """
    Transformer-based dynamics model that reasons over a short history
    of state-action pairs before predicting the next state.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        context_len: int = 5,
        dropout: float = 0.1,
        predict_delta: bool = True
    ):
        """
        Args:
            state_dim: State dimension (position + velocity).
            action_dim: Action dimension (currents).
            d_model: Transformer model width.
            nhead: Number of attention heads.
            num_layers: Transformer encoder layers.
            dim_feedforward: Feedforward width inside encoder.
            context_len: Maximum number of historical steps to attend to.
            dropout: Dropout probability.
            predict_delta: Predict state delta vs absolute state.
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.predict_delta = predict_delta

        self.input_proj = nn.Linear(state_dim + action_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, state_dim)

        # Learned positional encoding for up to context_len tokens.
        self.pos_encoding = nn.Parameter(torch.randn(1, context_len, d_model) * 0.02)

    def _build_sequence(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Build sequence tensor of shape (batch, seq_len, state_dim + action_dim).

        Args:
            state: Current state (batch, state_dim).
            action: Current action (batch, action_dim).
            history: Optional history tensor with shape
                (batch, hist_len, state_dim + action_dim).
        """
        token = torch.cat([state, action], dim=-1).unsqueeze(1)
        if history is None or history.numel() == 0:
            seq = token
        else:
            if history.size(1) >= self.context_len - 1:
                history = history[:, -self.context_len + 1 :, :]
            seq = torch.cat([history, token], dim=1)
        return seq

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict next state.

        Args:
            state: Current state (batch, state_dim).
            action: Current action (batch, action_dim).
            history: Optional tensor of past concatenated tokens
                (batch, hist_len, state_dim + action_dim).
        """
        seq = self._build_sequence(state, action, history)
        seq_len = seq.size(1)

        x = self.input_proj(seq)
        pos = self.pos_encoding[:, :seq_len, :]
        x = x + pos

        x = self.transformer(x)
        x = self.norm(x[:, -1, :])
        delta = self.output_proj(x)

        if self.predict_delta:
            return state + delta
        return delta

    def predict_sequence(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next states for an entire sequence.

        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
        """
        seq_len = states.size(1)
        outputs = []
        history = None
        for t in range(seq_len):
            state_t = states[:, t, :]
            action_t = actions[:, t, :]
            next_state = self.forward(state_t, action_t, history)
            outputs.append(next_state.unsqueeze(1))

            token = torch.cat([state_t, action_t], dim=-1).unsqueeze(1)
            history = token if history is None else torch.cat([history, token], dim=1)
            if history.size(1) > self.context_len - 1:
                history = history[:, -self.context_len + 1 :, :]

        return torch.cat(outputs, dim=1)


def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    device = timesteps.device
    half_dim = dim // 2
    emb = torch.exp(
        torch.arange(half_dim, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0)) / (half_dim - 1))
    )
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


@dataclass
class DiffusionDynamicsConfig:
    """Configuration for diffusion-based dynamics."""

    state_dim: int = 6
    action_dim: int = 3
    hidden_dim: int = 256
    num_layers: int = 4
    timestep_dim: int = 128
    num_diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    predict_delta: bool = True
    dropout: float = 0.0


class DiffusionDynamicsModel(nn.Module):
    """
    Conditional denoising diffusion model that learns the distribution
    over next-state deltas given current state and action.
    """

    def __init__(self, config: Optional[DiffusionDynamicsConfig] = None):
        super().__init__()
        self.config = config or DiffusionDynamicsConfig()

        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.predict_delta = self.config.predict_delta

        input_dim = self.state_dim + self.action_dim + self.state_dim
        dims = [input_dim + self.config.timestep_dim]
        for _ in range(self.config.num_layers):
            dims.append(self.config.hidden_dim)
        dims.append(self.state_dim)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                if self.config.dropout > 0:
                    layers.append(nn.Dropout(self.config.dropout))
        self.network = nn.Sequential(*layers)

        # Diffusion schedule
        betas = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.num_diffusion_steps,
            dtype=torch.float32
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def _model_input(
        self,
        noisy_delta: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        cond = torch.cat([state, action], dim=-1)
        time_emb = _sinusoidal_embedding(timesteps, self.config.timestep_dim)
        return torch.cat([noisy_delta, cond, time_emb], dim=-1)

    def _predict_noise(
        self,
        noisy_delta: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        model_in = self._model_input(noisy_delta, state, action, timesteps)
        return self.network(model_in)

    def diffusion_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard DDPM training loss (MSE on noise prediction).
        """
        delta = next_state - state if self.predict_delta else next_state

        batch_size = state.size(0)
        device = state.device
        t = torch.randint(
            0,
            self.config.num_diffusion_steps,
            (batch_size,),
            device=device
        )

        noise = torch.randn_like(delta)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noisy_delta = sqrt_alpha * delta + sqrt_one_minus_alpha * noise

        pred_noise = self._predict_noise(noisy_delta, state, action, t)
        return F.mse_loss(pred_noise, noise)

    def sample(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample next state via reverse diffusion.
        """
        batch_size = state.size(0)
        device = state.device
        if deterministic:
            delta = torch.zeros(batch_size, self.state_dim, device=device)
        else:
            delta = torch.randn(batch_size, self.state_dim, device=device)

        for step in reversed(range(self.config.num_diffusion_steps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            pred_noise = self._predict_noise(delta, state, action, t)

            alpha = self.alphas[step]
            alpha_cumprod = self.alphas_cumprod[step]
            beta = self.betas[step]

            delta = (1 / torch.sqrt(alpha)) * (
                delta - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            )
            if step > 0 and not deterministic:
                noise = torch.randn_like(delta)
                delta = delta + torch.sqrt(self.posterior_variance[step]) * noise

        if self.predict_delta:
            return state + delta
        return delta

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Convenience method returning either deterministic prediction
        (using DDIM-like single step) or sampled next state.
        """
        return self.sample(state, action, deterministic=deterministic)

    def sample_multiple(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        num_samples: int = 10
    ) -> torch.Tensor:
        """Draw multiple stochastic next-state samples."""
        samples = []
        for _ in range(num_samples):
            samples.append(self.sample(state, action, deterministic=False).unsqueeze(0))
        return torch.cat(samples, dim=0)
