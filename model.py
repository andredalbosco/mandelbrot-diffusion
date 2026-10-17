"""Pixel-space JiT with clean-image rectified flow. See LICENSE-JiT."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def time_features(t, width=256):
    half = width // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * 1000 * frequencies[None]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def position_features(grid_size, width):
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
    frequencies = torch.exp(-math.log(10000) * torch.arange(width // 4) / (width // 4))
    features = []
    for coordinate in (x, y):
        angles = coordinate.flatten().float()[:, None] * frequencies[None]
        features.extend((angles.sin(), angles.cos()))
    return torch.cat(features, dim=-1).unsqueeze(0)


def modulate(tokens, shift, scale):
    return tokens * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)

    def forward(self, tokens):
        batch, count, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, count, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.output(attended.transpose(1, 2).reshape(batch, count, width))


class RotaryAttention(Attention):
    """JiT attention: Q/K RMS normalization and two-dimensional RoPE."""

    def __init__(self, width, heads, grid_size):
        super().__init__(width, heads)
        head_width = width // heads
        self.q_norm = nn.RMSNorm(head_width, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_width, eps=1e-6)
        frequencies = 10000 ** (-torch.arange(0, head_width // 2, 2).float() / (head_width // 2))
        y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
        angles = torch.cat([coordinate.flatten()[:, None] * frequencies for coordinate in (y, x)], dim=-1)
        angles = angles.repeat_interleave(2, dim=-1)
        self.register_buffer("cosines", angles.cos())
        self.register_buffer("sines", angles.sin())

    def rotate(self, x):
        pairs = x.unflatten(-1, (-1, 2))
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return x * self.cosines + rotated * self.sines

    def forward(self, tokens):
        batch, count, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, count, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rotate(self.q_norm(q)), self.rotate(self.k_norm(k))
        attended = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v)
        return self.output(attended.transpose(1, 2).reshape(batch, count, width))


class SwiGLU(nn.Module):
    def __init__(self, width, mlp_ratio, hidden_size=None):
        super().__init__()
        # Explicit widths permit aligned small models; None preserves old checkpoints.
        hidden = int(int(width * mlp_ratio) * 2 / 3) if hidden_size is None else hidden_size
        if type(hidden) is not int or hidden < 1:
            raise ValueError("SwiGLU hidden size must be a positive integer")
        self.input = nn.Linear(width, hidden * 2)
        self.output = nn.Linear(hidden, width)

    def forward(self, x):
        gate, value = self.input(x).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


class JiTBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio, grid_size, mlp_hidden_size=None):
        super().__init__()
        self.norm_attention = nn.RMSNorm(width, eps=1e-6)
        self.norm_mlp = nn.RMSNorm(width, eps=1e-6)
        self.attention = RotaryAttention(width, heads, grid_size)
        self.mlp = SwiGLU(width, mlp_ratio, mlp_hidden_size)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))

    def forward(self, tokens, condition):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.condition(condition).chunk(6, dim=-1)
        tokens = tokens + gate_a[:, None] * self.attention(modulate(self.norm_attention(tokens), shift_a, scale_a))
        return tokens + gate_m[:, None] * self.mlp(modulate(self.norm_mlp(tokens), shift_m, scale_m))


class JiT(nn.Module):
    """Conditional clean-image predictor; small JiT without class-context tokens."""

    prediction = "x"

    def __init__(self, size=128, patch_size=8, hidden_size=256, depth=8,
                 num_heads=8, bottleneck_dim=128, mlp_ratio=4.0, channels=3, mask_channels=0,
                 mlp_hidden_size=None):
        super().__init__()
        dimensions = (size, patch_size, hidden_size, depth, num_heads, bottleneck_dim, channels)
        if any(type(v) is not int or v <= 0 for v in dimensions):
            raise ValueError("JiT dimensions must be positive integers")
        if size % patch_size or hidden_size % num_heads or (hidden_size // num_heads) % 4:
            raise ValueError("JiT size must divide into patches; head width must divide by four")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(int(hidden_size * mlp_ratio) * 2 / 3) < 1:
            raise ValueError("JiT MLP ratio must produce a positive hidden dimension")
        self.size, self.patch_size, self.channels = size, patch_size, channels
        if mask_channels not in (0, 1):
            raise ValueError("JiT supports zero or one coverage channel")
        self.mask_channels = mask_channels
        self.patch_embed = nn.Sequential(
            nn.Conv2d(channels * 2 + mask_channels, bottleneck_dim, patch_size, stride=patch_size, bias=False),
            nn.Conv2d(bottleneck_dim, hidden_size, 1))
        self.register_buffer("positions", position_features(size // patch_size, hidden_size))
        self.time_embed = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.blocks = nn.ModuleList(JiTBlock(hidden_size, num_heads, mlp_ratio, size // patch_size, mlp_hidden_size) for _ in range(depth))
        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.final_condition = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.final_projection = nn.Linear(hidden_size, patch_size * patch_size * channels)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight.flatten(1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for layer in (self.time_embed[0], self.time_embed[2]):
            nn.init.normal_(layer.weight, std=.02)
        for layer in [block.condition[-1] for block in self.blocks] + [self.final_condition[-1], self.final_projection]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t, y, mask=None):
        if x.shape != y.shape or x.shape[1:] != (self.channels, self.size, self.size) or t.shape != (len(x),):
            raise ValueError("Need matching state/condition images and one time per image")
        parts = (x, y)
        if self.mask_channels:
            if mask is None or mask.shape != (len(x), 1, self.size, self.size):
                raise ValueError("Masked JiT requires [N,1,H,W] coverage")
            # Missing evidence is neutral zero in normalized RGB, regardless of
            # caller padding. Fractional masks describe interpolation support.
            parts = (x, torch.where(mask > 0, y, 0), mask)
        elif mask is not None:
            raise ValueError("This checkpoint has no mask input")
        tokens = self.patch_embed(torch.cat(parts, dim=1)).flatten(2).transpose(1, 2) + self.positions
        condition = self.time_embed(time_features(t / 1000))
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = self.final_condition(condition).chunk(2, dim=-1)
        patches = self.final_projection(modulate(self.final_norm(tokens), shift, scale))
        batch, p = len(patches), self.patch_size
        side = self.size // p
        patches = patches.reshape(batch, side, side, p, p, self.channels)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(batch, self.channels, self.size, self.size)


class ConditionalFlow(nn.Module):
    """Noise-to-image rectified flow with fixed RGB evidence and coverage.

    JiT predicts clean RGB. Its error is weighted by 1/(1-t)^2, equivalent
    to velocity MSE away from the clamped endpoint. One Euler step at t=0
    returns that clean-image prediction directly; good one-step quality is
    evaluated, not assumed or guaranteed by this objective.
    """

    def __init__(self, time_mean=-.8, time_std=.8, minimum_remaining=.05):
        super().__init__()
        if time_std <= 0 or not 0 < minimum_remaining < 1:
            raise ValueError("Invalid flow time distribution")
        self.time_mean, self.time_std = time_mean, time_std
        self.minimum_remaining = minimum_remaining

    def loss(self, model, clean, condition, mask, generator=None, t=None, noise=None):
        if t is None:
            t = (torch.randn(len(clean), device=clean.device, generator=generator)*self.time_std+self.time_mean).sigmoid()
        if noise is None:
            noise = torch.randn(clean.shape, device=clean.device, generator=generator)
        time = t[:, None, None, None]
        state = (1-time)*noise+time*clean
        prediction = model(state, t, condition, mask).float()
        return ((prediction-clean)/(1-time).clamp_min(self.minimum_remaining)).square().mean()

    @torch.no_grad()
    def sample(self, model, condition, mask, steps=16, seed=1234, solver="euler"):
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        if solver not in ("euler", "heun"):
            raise ValueError("Solver must be euler or heun")
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        state = torch.randn(condition.shape, device=condition.device, generator=generator)
        was_training = model.training
        model.eval()
        def velocity(value, time):
            t = torch.full((len(value),), time, device=value.device)
            prediction = model(value, t, condition, mask).float()
            return (prediction-value)/max(1-time, self.minimum_remaining)

        try:
            for index in range(steps):
                time, next_time = index/steps, (index+1)/steps
                dt = next_time-time
                first = velocity(state, time)
                proposal = state+dt*first
                # Match JiT: Heun corrections except for the final Euler step.
                if solver == "heun" and index < steps-1:
                    second = velocity(proposal, next_time)
                    state = state+dt*.5*(first+second)
                else:
                    state = proposal
        finally:
            model.train(was_training)
        return (state.clamp(-1, 1)+1)/2


def build_model(config):
    if config["architecture"] != "jit":
        raise ValueError("This branch supports JiT checkpoints only")
    return JiT(size=config["size"], **config["model"])
