"""Pixel-space DiT, JiT and U-Net image flows.

Architecture: Peebles & Xie, Scalable Diffusion Models with Transformers.
Includes unconditional x-prediction and deterministic paired rectified flows.
JiT follows Li & He (2025), https://github.com/LTH14/JiT (MIT).
See docs/licenses/jit-LICENSE.txt for upstream attribution.
"""

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


class DiTBlock(nn.Module):
    def __init__(self, width, heads, mlp_ratio):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, int(width * mlp_ratio)),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(int(width * mlp_ratio), width))
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))

    def forward(self, tokens, condition):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.condition(condition).chunk(6, dim=-1)
        tokens = tokens + gate_a[:, None] * self.attention(modulate(self.norm_attention(tokens), shift_a, scale_a))
        tokens = tokens + gate_m[:, None] * self.mlp(modulate(self.norm_mlp(tokens), shift_m, scale_m))
        return tokens


class DiT(nn.Module):
    def __init__(self, size=64, patch_size=4, hidden_size=192, depth=6,
                 num_heads=6, mlp_ratio=4.0, channels=1, condition_channels=0):
        super().__init__()
        integers = (size, patch_size, hidden_size, depth, num_heads, channels)
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("DiT dimensions must be positive integers")
        if type(condition_channels) is not int or condition_channels < 0:
            raise ValueError("Condition channels must be a nonnegative integer")
        if size % patch_size or hidden_size % 4 or hidden_size % num_heads:
            raise ValueError("Size must divide into patches; width must divide by 4 and head count")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hidden_size * mlp_ratio) < 1:
            raise ValueError("MLP ratio must produce a positive hidden dimension")
        self.size, self.patch_size = size, patch_size
        self.channels = channels
        self.condition_channels = condition_channels
        self.patch_embed = nn.Conv2d(channels + condition_channels, hidden_size, patch_size, stride=patch_size)
        self.register_buffer("positions", position_features(size // patch_size, hidden_size))
        self.time_embed = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.blocks = nn.ModuleList(DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth))
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_condition = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.final_projection = nn.Linear(hidden_size, patch_size * patch_size * channels)
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.bias)
        for layer in (self.time_embed[0], self.time_embed[2]):
            nn.init.normal_(layer.weight, std=0.02)
        # Each transformer block begins as the identity; network output begins at zero.
        for block in self.blocks:
            nn.init.zeros_(block.condition[-1].weight)
            nn.init.zeros_(block.condition[-1].bias)
        for layer in (self.final_condition[-1], self.final_projection):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def unpatchify(self, patches):
        batch = len(patches)
        side, p = self.size // self.patch_size, self.patch_size
        if patches.shape[1:] != (side * side, p * p * self.channels):
            raise ValueError("Patch tensor shape disagrees with image dimensions")
        patches = patches.reshape(batch, side, side, p, p, self.channels)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(batch, self.channels, self.size, self.size)

    def forward(self, x, t, y=None):
        if x.ndim != 4 or x.shape[1:] != (self.channels, self.size, self.size):
            raise ValueError(f"DiT expects [N,{self.channels},{self.size},{self.size}]")
        if t.shape != (len(x),):
            raise ValueError("Need one scalar noise level per image")
        if self.condition_channels:
            if y is None or y.shape != (len(x), self.condition_channels, self.size, self.size):
                raise ValueError("Condition must match configured channels and image dimensions")
            x = torch.cat((x, y), dim=1)
        elif y is not None:
            raise ValueError("This DiT was configured without image conditioning")
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2) + self.positions
        condition = self.time_embed(time_features(t))
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = self.final_condition(condition).chunk(2, dim=-1)
        patches = self.final_projection(modulate(self.final_norm(tokens), shift, scale))
        return self.unpatchify(patches)


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


class JiTBlock(DiTBlock):
    def __init__(self, width, heads, mlp_ratio, grid_size, mlp_hidden_size=None):
        super().__init__(width, heads, mlp_ratio)
        self.norm_attention = nn.RMSNorm(width, eps=1e-6)
        self.norm_mlp = nn.RMSNorm(width, eps=1e-6)
        self.attention = RotaryAttention(width, heads, grid_size)
        self.mlp = SwiGLU(width, mlp_ratio, mlp_hidden_size)


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
        return DiT.unpatchify(self, patches)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_width):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_width, out_channels * 2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(time)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(h))


class SpatialAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.attention = Attention(width, 4)

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        return x + self.attention(tokens).transpose(1, 2).reshape(batch, channels, height, width)


class UNet(nn.Module):
    """Conditional pixel-space U-Net; attention at the two lowest resolutions."""

    def __init__(self, size=128, channels=3, widths=(64, 128, 256, 256)):
        super().__init__()
        if len(widths) != 4 or any(w <= 0 or w % 8 for w in widths) or size % 8:
            raise ValueError("Need four widths divisible by 8 and an image size divisible by 8")
        self.size, self.channels = size, channels
        time_width = widths[0] * 4
        self.time_embed = nn.Sequential(nn.Linear(256, time_width), nn.SiLU(), nn.Linear(time_width, time_width))
        self.input = nn.Conv2d(channels * 2, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, width in enumerate(widths):
            self.encoders.append(nn.ModuleList([
                ResidualBlock(width, width, time_width),
                ResidualBlock(width, width, time_width),
                SpatialAttention(width) if i >= 2 else nn.Identity(),
            ]))
            if i < 3:
                self.downsamples.append(nn.Conv2d(width, widths[i+1], 3, stride=2, padding=1))
        self.middle = ResidualBlock(widths[-1], widths[-1], time_width)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(2, -1, -1):
            self.upsamples.append(nn.Conv2d(widths[i+1], widths[i], 3, padding=1))
            self.decoders.append(nn.ModuleList([
                ResidualBlock(widths[i] * 2, widths[i], time_width),
                ResidualBlock(widths[i], widths[i], time_width),
                SpatialAttention(widths[i]) if i >= 2 else nn.Identity(),
            ]))
        self.output = nn.Sequential(nn.GroupNorm(8, widths[0]), nn.SiLU(), nn.Conv2d(widths[0], channels, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x, t, y):
        if x.shape != y.shape or x.shape[1:] != (self.channels, self.size, self.size) or t.shape != (len(x),):
            raise ValueError("Need matching state/condition images and one time per image")
        time = self.time_embed(time_features(t))
        h = self.input(torch.cat((x, y), dim=1))
        skips = []
        for i, (first, second, attention) in enumerate(self.encoders):
            h = attention(second(first(h, time), time))
            skips.append(h)
            if i < 3:
                h = self.downsamples[i](h)
        h = self.middle(h, time)
        for upsample, (first, second, attention), skip in zip(self.upsamples, self.decoders, reversed(skips[:-1])):
            h = upsample(F.interpolate(h, size=skip.shape[-2:], mode="nearest"))
            h = attention(second(first(torch.cat((h, skip), dim=1), time), time))
        return self.output(h)


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


class RectifiedFlow(nn.Module):
    """Deterministic paired flow: t=0 is the coarse input; t=1 is the target."""

    def loss(self, model, clean, condition, t=None):
        if clean.shape != condition.shape:
            raise ValueError("Target and condition must have matching shapes")
        if t is None:
            t = torch.rand(len(clean), device=clean.device)
        if t.shape != (len(clean),) or not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any():
            raise ValueError("Need one finite time in [0,1] per image")
        time = t[:, None, None, None]
        state = (1 - time) * condition + time * clean
        prediction = model(state, t, condition)
        if getattr(model, "prediction", "v") == "x":
            # JiT predicts clean pixels; train with the equivalent velocity loss.
            scale = (1 - time).clamp_min(.05)
            return ((prediction.float() - clean) / scale).square().mean()
        return F.mse_loss(prediction, clean - condition)

    @torch.no_grad()
    def sample(self, model, condition, steps=16):
        if steps < 1:
            raise ValueError("Integration steps must be positive")
        was_training = model.training
        model.eval()
        state = condition.clone()
        try:
            for index in range(steps):
                t = torch.full((len(state),), index / steps, device=state.device)
                prediction = model(state, t, condition)
                velocity = (prediction - state) / (1 - index / steps) if getattr(model, "prediction", "v") == "x" else prediction
                state = state + velocity / steps
        finally:
            model.train(was_training)
        return (state.clamp(-1, 1) + 1) / 2


def build_model(config):
    architectures = {"dit": DiT, "jit": JiT, "unet": UNet}
    return architectures[config["architecture"]](size=config["size"], **config["model"])


class XPredictionFlow(nn.Module):
    """One-modality adaptation of Modality Forcing, equations (5) and (7).

    t=0 is clean and t=1 is noise. The network predicts clean x0, while
    the loss compares (prediction-xt)/t with x0-noise. No reflow is used.
    """

    def __init__(self, minimum_time=0.05, time_shift=1.1):
        super().__init__()
        if not 0 < minimum_time < 1 or time_shift <= 0:
            raise ValueError("Minimum time must be in (0,1); shift must be positive")
        self.minimum_time = minimum_time
        self.time_shift = time_shift

    def times(self, count, device, generator=None):
        t = torch.randn((count,), device=device, generator=generator).sigmoid()
        t = self.time_shift * t / (1 + (self.time_shift-1) * t)
        return t.clamp(min=self.minimum_time)

    @staticmethod
    def interpolate(clean, t, noise):
        t = t[:, None, None, None]
        return (1-t) * clean + t * noise

    def loss(self, model, clean, t=None, noise=None):
        if t is None:
            t = self.times(len(clean), clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        if (t <= 0).any() or (t > 1).any():
            raise ValueError("Loss times must be in (0,1]")
        noisy = self.interpolate(clean, t, noise)
        predicted_clean = model(noisy, t)
        predicted_velocity = (predicted_clean-noisy) / t[:, None, None, None]
        return F.mse_loss(predicted_velocity, clean-noise)

    @torch.no_grad()
    def sample(self, model, count, size, steps=50, seed=1234, device=None, channels=1):
        if steps < 1 or count < 1:
            raise ValueError("Sampling steps and count must be positive")
        device = next(model.parameters()).device if device is None else torch.device(device)
        was_training = model.training
        model.eval()
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn((count, channels, size, size), device=device, generator=generator)
        # Backward-time Euler: dx/dt=(x-predicted_clean)/t, t runs 1 -> 0.
        for index in range(steps):
            current = 1 - index / steps
            t = torch.full((count,), current, device=device)
            predicted_clean = model(x, t)
            x = x + (predicted_clean-x) / (current * steps)
        model.train(was_training)
        return (x.clamp(-1, 1) + 1) / 2
