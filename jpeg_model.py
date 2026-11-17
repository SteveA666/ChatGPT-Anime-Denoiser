import math

import torch
from torch import nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timestep, dim, max_period=10000):
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=timestep.device)
            / half
        )
        args = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timestep):
        embedding = self.timestep_embedding(timestep, self.frequency_embedding_size)
        return self.mlp(embedding.to(next(self.parameters()).dtype))


class LiteLA(nn.Module):
    """SANA lightweight linear attention with optional rank enhancement.

    ``rank_enhance_kernel`` adds the depth-wise ``W_d V`` branch proposed by
    RELA.  It keeps the global attention linear in the token count while
    restoring a full-rank local path.  A value of zero preserves the original
    SANA implementation and its checkpoint layout.
    """

    def __init__(
        self,
        hidden_size,
        linear_head_dim=32,
        eps=1e-8,
        qk_norm=False,
        rank_enhance_kernel=0,
    ):
        super().__init__()
        if rank_enhance_kernel and rank_enhance_kernel % 2 == 0:
            raise ValueError("rank_enhance_kernel must be zero or an odd integer")
        self.num_heads = hidden_size // linear_head_dim
        self.head_dim = hidden_size // self.num_heads
        self.eps = eps
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.rank_enhance = (
            nn.Conv2d(
                hidden_size,
                hidden_size,
                rank_enhance_kernel,
                padding=rank_enhance_kernel // 2,
                groups=hidden_size,
                bias=False,
            )
            if rank_enhance_kernel
            else None
        )
        if qk_norm:
            self.q_norm = nn.RMSNorm(hidden_size, eps=1e-5)
            self.k_norm = nn.RMSNorm(hidden_size, eps=1e-5)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, height=None, width=None):
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
        local = None
        if self.rank_enhance is not None:
            if height is None or width is None or tokens != height * width:
                raise ValueError(
                    "height and width matching the token count are required "
                    "for rank-enhanced linear attention"
                )
            local = self.rank_enhance(
                v.transpose(1, 2).reshape(batch, channels, height, width)
            ).flatten(2).transpose(1, 2)
        dtype = q.dtype
        q = self.q_norm(q).transpose(-1, -2)
        k = self.k_norm(k).transpose(-1, -2)
        v = v.transpose(-1, -2)

        q = q.reshape(batch, self.num_heads, self.head_dim, tokens)
        k = k.reshape(batch, self.num_heads, self.head_dim, tokens)
        v = v.reshape(batch, self.num_heads, self.head_dim, tokens)
        q = F.relu(q)
        k = F.relu(k)

        v = F.pad(v, (0, 0, 0, 1), value=1)
        vk = torch.matmul(v, k.transpose(-1, -2))
        out = torch.matmul(vk, q)
        if out.dtype in (torch.float16, torch.bfloat16):
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        out = out.to(dtype).reshape(batch, channels, tokens).permute(0, 2, 1)
        if local is not None:
            out = out + local
        return self.proj(out)


class ShiftedWindowAttention(nn.Module):
    """Swin-style local softmax attention for arbitrary rectangular grids."""

    def __init__(self, hidden_size, num_heads, window_size=8, shift_size=None):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if window_size < 2:
            raise ValueError("window_size must be at least two")
        shift_size = window_size // 2 if shift_size is None else shift_size
        if not 0 < shift_size < window_size:
            raise ValueError("shift_size must be between zero and window_size")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)

        relative_size = 2 * window_size - 1
        self.relative_position_bias = nn.Parameter(
            torch.zeros(num_heads, relative_size, relative_size)
        )
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        ).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates += window_size - 1
        self.register_buffer(
            "relative_position_index",
            relative_coordinates[0] * relative_size + relative_coordinates[1],
            persistent=False,
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def _attention_bias(self, height, width, device, dtype):
        window = self.window_size
        shift_height = self.shift_size if height > window else 0
        shift_width = self.shift_size if width > window else 0
        padded_height = math.ceil(height / window) * window
        padded_width = math.ceil(width / window) * window

        region = torch.zeros((1, padded_height, padded_width, 1), device=device)
        height_slices = (
            (slice(0, -window), slice(-window, -shift_height), slice(-shift_height, None))
            if shift_height
            else (slice(0, None),)
        )
        width_slices = (
            (slice(0, -window), slice(-window, -shift_width), slice(-shift_width, None))
            if shift_width
            else (slice(0, None),)
        )
        region_id = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                region[:, height_slice, width_slice] = region_id
                region_id += 1
        region = region.view(
            1,
            padded_height // window,
            window,
            padded_width // window,
            window,
            1,
        )
        region = region.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window)
        attention_mask = region[:, :, None] != region[:, None, :]

        valid = torch.zeros((1, padded_height, padded_width, 1), device=device)
        valid[:, :height, :width] = 1
        valid = torch.roll(
            valid, shifts=(-shift_height, -shift_width), dims=(1, 2)
        )
        valid = valid.view(
            1,
            padded_height // window,
            window,
            padded_width // window,
            window,
            1,
        )
        valid = valid.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window).bool()
        attention_mask = attention_mask | ~valid[:, None, :]

        relative_bias = self.relative_position_bias.flatten(1)[
            :, self.relative_position_index.flatten()
        ].view(self.num_heads, window * window, window * window)
        bias = relative_bias[None].expand(attention_mask.shape[0], -1, -1, -1).to(dtype)
        return bias.masked_fill(attention_mask[:, None], torch.finfo(dtype).min)

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError("token count does not match the supplied spatial grid")

        window = self.window_size
        padded_height = math.ceil(height / window) * window
        padded_width = math.ceil(width / window) * window
        shift_height = self.shift_size if height > window else 0
        shift_width = self.shift_size if width > window else 0
        x = x.view(batch, height, width, channels)
        x = F.pad(x, (0, 0, 0, padded_width - width, 0, padded_height - height))
        x = torch.roll(x, shifts=(-shift_height, -shift_width), dims=(1, 2))
        x = x.view(
            batch,
            padded_height // window,
            window,
            padded_width // window,
            window,
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)

        q, k, v = self.qkv(x).view(
            x.shape[0], window * window, 3, self.num_heads, self.head_dim
        ).unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        bias = self._attention_bias(height, width, x.device, x.dtype)
        bias = bias.repeat(batch, 1, 1, 1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(-1, window * window, channels)
        x = self.proj(x)

        x = x.view(
            batch,
            padded_height // window,
            padded_width // window,
            window,
            window,
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, padded_height, padded_width, channels
        )
        x = torch.roll(x, shifts=(shift_height, shift_width), dims=(1, 2))
        return x[:, :height, :width].reshape(batch, tokens, channels)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qk_norm=False):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.kv_linear = nn.Linear(hidden_size, hidden_size * 2)
        self.proj = nn.Linear(hidden_size, hidden_size)
        if qk_norm:
            self.q_norm = nn.RMSNorm(hidden_size, eps=1e-6)
            self.k_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, condition):
        batch, _, channels = x.shape
        q = self.q_norm(self.q_linear(x))
        k, v = self.kv_linear(condition).view(batch, -1, 2, channels).unbind(2)
        k = self.k_norm(k)
        q = q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch, -1, channels)
        return self.proj(out)


class GLUMBConv(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=2.5):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.hidden_features = hidden_features
        self.inverted_conv = nn.Conv2d(hidden_size, hidden_features * 2, 1)
        self.depth_conv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            3,
            padding=1,
            groups=hidden_features * 2,
        )
        self.point_conv = nn.Conv2d(hidden_features, hidden_size, 1, bias=False)

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        x = x.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        x = F.silu(self.inverted_conv(x))
        x, gate = self.depth_conv(x).chunk(2, dim=1)
        x = x * F.silu(gate)
        x = self.point_conv(x)
        return x.reshape(batch, channels, tokens).permute(0, 2, 1)


class SanaBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=2.5,
        linear_head_dim=32,
        qk_norm=False,
        cross_norm=False,
        use_cross_attention=True,
        use_window_attention=False,
        window_size=8,
        rank_enhance_kernel=0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = (
            ShiftedWindowAttention(hidden_size, num_heads, window_size)
            if use_window_attention
            else LiteLA(
                hidden_size,
                linear_head_dim,
                eps=1e-8,
                qk_norm=qk_norm,
                rank_enhance_kernel=rank_enhance_kernel,
            )
        )
        self.cross_attn = (
            MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm)
            if use_cross_attention
            else None
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = GLUMBConv(hidden_size, mlp_ratio)
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, condition, timestep, height, width):
        batch = x.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch, 6, -1)
        ).chunk(6, dim=1)
        attention_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attention_output = self.attn(attention_input, height, width)
        x = x + gate_msa * attention_output
        if self.cross_attn is not None:
            x = x + self.cross_attn(x, condition)
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp), height, width
        )
        return x


class T2IFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)

    def forward(self, x, timestep):
        shift, scale = (self.scale_shift_table[None] + timestep[:, None]).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class RearrangedPatchEmbed(nn.Module):
    """Expose every patch coefficient before a 1x1 channel projection."""

    def __init__(self, in_channels, hidden_size, patch_size):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(patch_size)
        self.projection = nn.Conv2d(
            in_channels * patch_size**2, hidden_size, 1
        )
        nn.init.dirac_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, image):
        return self.projection(self.unshuffle(image))


class SanaLinearDiT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        patch_size=8,
        hidden_size=384,
        depth=12,
        num_heads=12,
        mlp_ratio=2.5,
        linear_head_dim=32,
        caption_channels=384,
        condition_tokens=1,
        qk_norm=False,
        cross_norm=False,
        use_cross_attention=True,
        window_size=8,
        window_block_interval=0,
        rank_enhance_kernel=0,
        rearranged_patch_embed=False,
        quality_split=None,
        positional_reference_size=None,
    ):
        super().__init__()
        if quality_split is not None and not 0 < quality_split < depth:
            raise ValueError("quality_split must be in (0, depth)")
        if positional_reference_size is not None and positional_reference_size < 2:
            raise ValueError("positional_reference_size must be at least two")
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.quality_split = quality_split
        self.positional_reference_size = positional_reference_size
        self.x_embedder = (
            RearrangedPatchEmbed(in_channels, hidden_size, patch_size)
            if rearranged_patch_embed
            else nn.Conv2d(
                in_channels,
                hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
            )
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.register_buffer(
                "condition",
                torch.randn(1, condition_tokens, caption_channels) / caption_channels**0.5,
            )
            self.condition_embedder = nn.Sequential(
                nn.Linear(caption_channels, hidden_size),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            self.condition = None
            self.condition_embedder = None
        if quality_split is not None:
            self.quality_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )
            self.quality_embedder = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            self.quality_head = None
            self.quality_embedder = None
        self.blocks = nn.ModuleList(
            [
                SanaBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    linear_head_dim,
                    qk_norm,
                    cross_norm,
                    use_cross_attention,
                    use_window_attention=(
                        window_block_interval > 0 and (index + 1) % window_block_interval == 0
                    ),
                    window_size=window_size,
                    rank_enhance_kernel=rank_enhance_kernel,
                )
                for index in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        if isinstance(self.x_embedder, nn.Conv2d):
            nn.init.xavier_uniform_(
                self.x_embedder.weight.view(self.x_embedder.weight.shape[0], -1)
            )
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        if self.condition_embedder is not None:
            nn.init.normal_(self.condition_embedder[0].weight, std=0.02)
            nn.init.normal_(self.condition_embedder[2].weight, std=0.02)
        if self.quality_head is not None:
            nn.init.zeros_(self.quality_head[2].weight)
            nn.init.zeros_(self.quality_head[2].bias)
            nn.init.normal_(self.quality_embedder[0].weight, std=0.02)
            nn.init.zeros_(self.quality_embedder[2].weight)
            nn.init.zeros_(self.quality_embedder[2].bias)

    @staticmethod
    def positional_embedding(
        height, width, dim, device, dtype, reference_size=None
    ):
        assert dim % 4 == 0
        if reference_size is None:
            y_coordinates = torch.arange(
                height, device=device, dtype=torch.float32
            )
            x_coordinates = torch.arange(
                width, device=device, dtype=torch.float32
            )
        else:
            # Interpolate the pretraining coordinate domain instead of
            # extrapolating sin/cos positions never seen on the 16x16 grid.
            y_coordinates = torch.linspace(
                0, reference_size - 1, height, device=device
            )
            x_coordinates = torch.linspace(
                0, reference_size - 1, width, device=device
            )
        y, x = torch.meshgrid(
            y_coordinates,
            x_coordinates,
            indexing="ij",
        )
        omega = torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4)
        omega = 1.0 / 10000**omega
        y = y.flatten()[:, None] * omega[None]
        x = x.flatten()[:, None] * omega[None]
        embedding = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
        return embedding.unsqueeze(0).to(dtype)

    def unpatchify(self, x, height, width):
        batch = x.shape[0]
        patch = self.patch_size
        x = x.reshape(batch, height, width, patch, patch, self.out_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(batch, self.out_channels, height * patch, width * patch)

    def forward(self, image, timestep):
        batch = image.shape[0]
        x = self.x_embedder(image)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.positional_embedding(
            height,
            width,
            self.hidden_size,
            x.device,
            x.dtype,
            self.positional_reference_size,
        )

        t = self.t_embedder(timestep)
        t_block = self.t_block(t)
        condition = None
        if self.condition_embedder is not None:
            condition = self.condition_embedder(self.condition.expand(batch, -1, -1))
        split = self.quality_split or len(self.blocks)
        for block in self.blocks[:split]:
            x = block(x, condition, t_block, height, width)
        quality = None
        if self.quality_head is not None:
            quality = self.quality_head(x.mean(dim=1))
            t = t + self.quality_embedder(quality)
            t_block = self.t_block(t)
        for block in self.blocks[split:]:
            x = block(x, condition, t_block, height, width)
        output = self.unpatchify(self.final_layer(x, t), height, width)
        return (output, quality) if quality is not None else output


class LinearDiTRestoration(SanaLinearDiT):
    """SANA Linear DiT adapted to the local restoration interface."""

    def __init__(self, timestep=0, **kwargs):
        super().__init__(**kwargs)
        self.timestep = timestep

    def forward(self, image):
        h, w = image.shape[-2:]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        timestep = torch.full(
            (image.shape[0],), self.timestep, device=image.device, dtype=image.dtype
        )
        output = super().forward(image, timestep)

import torch.nn.functional as F


def pad_to_multiple(image, multiple):
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    return F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
import torch
import torch.nn.functional as F
from torch import nn



def haar_encode(image):
    """Apply an orthonormal one-level 2D Haar transform."""
    top_left = image[..., 0::2, 0::2]
    top_right = image[..., 0::2, 1::2]
    bottom_left = image[..., 1::2, 0::2]
    bottom_right = image[..., 1::2, 1::2]

    low = (top_left + top_right + bottom_left + bottom_right) * 0.5
    # Match PyWavelets' (LL, LH, HL, HH) channel convention, which is also the
    # convention used by the original wavelet implementation. The distinction
    # matters here because Mamba is sensitive to the order of the subbands.
    vertical = (top_left + top_right - bottom_left - bottom_right) * 0.5
    horizontal = (top_left - top_right + bottom_left - bottom_right) * 0.5
    diagonal = (top_left - top_right - bottom_left + bottom_right) * 0.5
    return torch.cat((low, vertical, horizontal, diagonal), dim=1)


def haar_decode(coefficients):
    """Invert coefficients produced by :func:`haar_encode`."""
    low, vertical, horizontal, diagonal = coefficients.chunk(4, dim=1)
    top_left = (low + horizontal + vertical + diagonal) * 0.5
    top_right = (low - horizontal + vertical - diagonal) * 0.5
    bottom_left = (low + horizontal - vertical - diagonal) * 0.5
    bottom_right = (low - horizontal - vertical + diagonal) * 0.5

    batch, channels, height, width = low.shape
    image = coefficients.new_empty(batch, channels, height * 2, width * 2)
    image[..., 0::2, 0::2] = top_left
    image[..., 0::2, 1::2] = top_right
    image[..., 1::2, 0::2] = bottom_left
    image[..., 1::2, 1::2] = bottom_right
    return image


    """Run a restoration model after one or more levels of Haar encoding."""

    def __init__(self, model, levels=1):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        self.model = model
        self.levels = levels

    def forward(self, image):
        height, width = image.shape[-2:]
        multiple = 2**self.levels
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple
        padded = image
        if pad_height or pad_width:
            padded = F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")
        encoded = padded
        for _ in range(self.levels):
            encoded = haar_encode(encoded)
        for _ in range(self.levels):
            restored = haar_decode(restored)
import torch
from torch import nn



class Palette(nn.Module):
    """Condition a backbone on a degraded image in the same representation."""

    backbone = None

    def __init__(self, wavelet_levels=0, **kwargs):
        super().__init__()
        if wavelet_levels < 0:
            raise ValueError("wavelet_levels must be non-negative")
        self.wavelet_levels = wavelet_levels
        image_channels = kwargs.pop("out_channels", 3)
        latent_channels = image_channels * 4**wavelet_levels
        self.network = self.backbone(
            in_channels=latent_channels * 2,
            out_channels=latent_channels,
            **kwargs,
        )

    @property
    def pixel_multiple(self):
        return 2**self.wavelet_levels * self.network.patch_size

    def pad(self, image):
        return pad_to_multiple(image, self.pixel_multiple)

    def encode(self, image):
        for _ in range(self.wavelet_levels):
            image = haar_encode(image)
        return image

    def decode(self, state):
        for _ in range(self.wavelet_levels):
            state = haar_decode(state)
        return state

    def forward(self, state, timestep, condition):
        return self.network(torch.cat((state, condition), dim=1), timestep)


class LinearDiTPalette(Palette):
    backbone = SanaLinearDiT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        nn.init.normal_(self.network.final_layer.linear.weight, std=1e-3)
        nn.init.zeros_(self.network.final_layer.linear.bias)
import torch
from torch import nn


class ConditionalFlow(nn.Module):
    """Shared representation adapter for conditional flow objectives."""

    def __init__(self, model, timestep_scale=1000.0, residual_scale=16.0):
        super().__init__()
        if timestep_scale <= 0 or residual_scale <= 0:
            raise ValueError("flow scales must be positive")
        self.model = model
        self.timestep_scale = timestep_scale
        self.residual_scale = residual_scale

    def predict(self, state, timestep, condition):
        output = self.model(
            state, timestep * self.timestep_scale, condition
        )
        return output if isinstance(output, tuple) else (output, None)

    def encode(self, image):
        return self.model.encode(image)

    def decode(self, state):
        return self.model.decode(state)

    def target(self, clean, degraded):
        return (self.encode(clean) - self.encode(degraded)) * self.residual_scale

    def restore(self, condition, residual):
        return self.decode(condition + residual / self.residual_scale)

    def prepare_noise(self, noise, reference, image, image_size, generator):
        if noise is None:
            return torch.randn(
                reference.shape, device=reference.device, dtype=reference.dtype,
                generator=generator,
            )

        noise = noise.to(reference)
        if noise.shape[-2:] == image_size:
            noise = self.model.pad(noise)
        if noise.shape == image.shape:
            noise = self.encode(noise)
        if noise.shape != reference.shape:
            raise ValueError("noise must match the image or encoded state")
        return noise
import torch
import torch.nn.functional as F



def expand(timestep, reference):
    return timestep.view((-1,) + (1,) * (reference.ndim - 1))


class MeanFlow(ConditionalFlow):
    """Pixel MeanFlow with the improved velocity-space objective."""

    def __init__(
        self,
        model,
        sample_steps=1,
        timestep_scale=1000.0,
        min_time=0.05,
        time_logit_mean=0.8,
        time_logit_std=0.8,
        instantaneous_ratio=0.5,
        adaptive_power=1.0,
        adaptive_eps=0.01,
        lpips_weight=0.0,
        lpips_max_time=0.8,
        lpips_backbone="vgg",
        quality_loss_weight=0.0,
        residual_scale=16.0,
        derivative_method="jvp",
        finite_difference_eps=1e-3,
    ):
        super().__init__(model, timestep_scale, residual_scale)
        if sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        if min_time <= 0 or time_logit_std <= 0 or adaptive_eps <= 0:
            raise ValueError("mean flow scales must be positive")
        if not 0 <= instantaneous_ratio <= 1:
            raise ValueError("instantaneous_ratio must be in [0, 1]")
        if adaptive_power < 0:
            raise ValueError("adaptive_power must be non-negative")
        if lpips_weight < 0 or not 0 < lpips_max_time <= 1:
            raise ValueError("invalid LPIPS configuration")
        if lpips_backbone not in ("alex", "vgg", "squeeze"):
            raise ValueError("invalid LPIPS backbone")
        if quality_loss_weight < 0:
            raise ValueError("quality_loss_weight must be non-negative")
        if derivative_method not in ("jvp", "rcm", "forward_difference"):
            raise ValueError(
                "derivative_method must be 'jvp', 'rcm', or "
                "'forward_difference'"
            )
        if finite_difference_eps <= 0:
            raise ValueError("finite_difference_eps must be positive")

        self.sample_steps = sample_steps
        self.min_time = min_time
        self.time_logit_mean = time_logit_mean
        self.time_logit_std = time_logit_std
        self.instantaneous_ratio = instantaneous_ratio
        self.adaptive_power = adaptive_power
        self.adaptive_eps = adaptive_eps
        self.lpips_weight = lpips_weight
        self.lpips_max_time = lpips_max_time
        self.lpips_backbone = lpips_backbone
        self.quality_loss_weight = quality_loss_weight
        self.derivative_method = derivative_method
        self.finite_difference_eps = finite_difference_eps

    def _adaptive_loss(self, loss):
        weight = (loss.detach() + self.adaptive_eps).pow(
            -self.adaptive_power
        )
        return weight * loss

    def sample_times(self, reference, generator=None):
        logits = torch.randn(
            reference.shape[0],
            2,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        times = torch.sigmoid(
            self.time_logit_mean + self.time_logit_std * logits
        ).sort(dim=1).values
        r, t = times.unbind(1)
        instantaneous = torch.rand(
            r.shape, device=r.device, generator=generator
        ) < self.instantaneous_ratio
        r = torch.where(instantaneous, t, r)
        return r, t

    @staticmethod
    def interpolate(x, e, t):
        t = expand(t, x)
        return (1 - t) * x + t * e

    def u(self, z, r, t, condition, return_quality=False):
        x, quality = self.predict(z, t - r, condition)
        average = (z - x) / expand(t.clamp_min(self.min_time), z)
        return (average, quality) if return_quality else average

    @staticmethod
    def _jvp_segment(function, primals, tangents):
        """Run one exact JVP segment and discard its tangent graph."""
        output, tangent = torch.func.jvp(function, primals, tangents)
        if isinstance(tangent, tuple):
            tangent = tuple(value.detach() for value in tangent)
        else:
            tangent = tangent.detach()
        return output, tangent

    def _rcm_jvp(self, z, r, t, v, condition):
        """Exact, memory-efficient JVP segmented like the rCM network."""
        network = getattr(self.model, "network", None)
        required = (
            "x_embedder",
            "t_embedder",
            "t_block",
            "blocks",
            "final_layer",
            "unpatchify",
            "positional_embedding",
        )
        if network is None or any(
            not hasattr(network, name) for name in required
        ):
            raise TypeError(
                "the rcm JVP currently requires a Palette wrapper around "
                "SanaLinearDiT"
            )

        image = torch.cat((z, condition), dim=1)
        image_tangent = torch.cat((v, torch.zeros_like(condition)), dim=1)
        timestep = (t - r) * self.timestep_scale
        timestep_tangent = torch.ones_like(t) * self.timestep_scale

        def embed_fn(image, timestep):
            x = network.x_embedder(image)
            height, width = x.shape[-2:]
            x = x.flatten(2).transpose(1, 2)
            x = x + network.positional_embedding(
                height,
                width,
                network.hidden_size,
                x.device,
                x.dtype,
                network.positional_reference_size,
            )
            embedded_timestep = network.t_embedder(timestep)
            block_timestep = network.t_block(embedded_timestep)
            return x, embedded_timestep, block_timestep

        (x, embedded_timestep, block_timestep), (
            x_tangent,
            embedded_timestep_tangent,
            block_timestep_tangent,
        ) = self._jvp_segment(
            embed_fn,
            (image, timestep),
            (image_tangent, timestep_tangent),
        )

        block_condition = None
        if network.condition_embedder is not None:
            block_condition = network.condition_embedder(
                network.condition.expand(z.shape[0], -1, -1)
            )

        height = image.shape[-2] // network.patch_size
        width = image.shape[-1] // network.patch_size
        split = network.quality_split or len(network.blocks)

        def run_block(block, x, x_tangent):
            def block_fn(x, block_timestep):
                return block(
                    x,
                    block_condition,
                    block_timestep,
                    height,
                    width,
                )

            return self._jvp_segment(
                block_fn,
                (x, block_timestep),
                (x_tangent, block_timestep_tangent),
            )

        for block in network.blocks[:split]:
            x, x_tangent = run_block(block, x, x_tangent)

        quality = None
        if network.quality_head is not None:

            def quality_fn(x, embedded_timestep):
                quality = network.quality_head(x.mean(dim=1))
                embedded_timestep = (
                    embedded_timestep
                    + network.quality_embedder(quality)
                )
                return (
                    embedded_timestep,
                    network.t_block(embedded_timestep),
                    quality,
                )

            (embedded_timestep, block_timestep, quality), (
                embedded_timestep_tangent,
                block_timestep_tangent,
                _,
            ) = self._jvp_segment(
                quality_fn,
                (x, embedded_timestep),
                (x_tangent, embedded_timestep_tangent),
            )

        for block in network.blocks[split:]:
            x, x_tangent = run_block(block, x, x_tangent)

        def output_fn(x, embedded_timestep, z, t):
            prediction = network.unpatchify(
                network.final_layer(x, embedded_timestep),
                height,
                width,
            )
            return (z - prediction) / expand(
                t.clamp_min(self.min_time), z
            )

        u, du_dt = self._jvp_segment(
            output_fn,
            (x, embedded_timestep, z, t),
            (x_tangent, embedded_timestep_tangent, v, torch.ones_like(t)),
        )
        return u, du_dt, quality

    def compute_loss(
        self,
        degraded,
        clean,
        quality=None,
        *,
        generator=None,
        r=None,
        t=None,
        noise=None,
        perceptual_loss=None,
        include_perceptual=True,
        reduction="mean",
    ):
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")
        if (r is None) != (t is None):
            raise ValueError("r and t must be provided together")

        image_size = degraded.shape[-2:]
        degraded = self.model.pad(degraded)
        clean = self.model.pad(clean)
        condition = self.encode(degraded)
        x = self.target(clean, degraded)
        e = self.prepare_noise(noise, x, degraded, image_size, generator)

        if r is None:
            r, t = self.sample_times(x, generator)
        else:
            r, t = r.to(x), t.to(x)
        if r.shape != (x.shape[0],) or t.shape != r.shape:
            raise ValueError("r and t must have one value per image")
        if torch.any((r < 0) | (r > t) | (t > 1)):
            raise ValueError("times must satisfy 0 <= r <= t <= 1")

        z = self.interpolate(x, e, t)

        def u_fn(z, r, t):
            return self.u(z, r, t, condition)

        with torch.no_grad():
            v = u_fn(z, t, t)
        tangents = (v, torch.zeros_like(r), torch.ones_like(t))
        predicted_quality = None
        if self.derivative_method == "jvp":
            if self.quality_loss_weight:

                def u_with_quality(z, r, t):
                    return self.u(
                        z, r, t, condition, return_quality=True
                    )

                u, du_dt, predicted_quality = torch.func.jvp(
                    u_with_quality, (z, r, t), tangents, has_aux=True
                )
            else:
                u, du_dt = torch.func.jvp(
                    u_fn, (z, r, t), tangents
                )
        elif self.derivative_method == "rcm":
            u, du_dt, predicted_quality = self._rcm_jvp(
                z, r, t, v, condition
            )
        else:
            if self.quality_loss_weight:
                u, predicted_quality = self.u(
                    z, r, t, condition, return_quality=True
                )
            else:
                u = u_fn(z, r, t)
            epsilon = self.finite_difference_eps
            with torch.no_grad():
                shifted = u_fn(z + epsilon * v, r, t + epsilon)
                du_dt = (shifted - u.detach()) / epsilon
        V = u + expand(t - r, u) * du_dt.detach()
        dx = V - (e - x)

        mean_flow = dx.square().flatten(1).mean(1)
        loss = self._adaptive_loss(mean_flow)

        quality_loss = None
        if self.quality_loss_weight:
            if quality is None or predicted_quality is None:
                raise ValueError("quality prediction is required")
            quality_loss = F.l1_loss(
                predicted_quality,
                quality.to(predicted_quality),
                reduction="none",
            ).flatten(1).mean(1)
            loss = loss + self.quality_loss_weight * quality_loss

        lpips = mean_flow.new_zeros(mean_flow.shape)
        if self.lpips_weight and include_perceptual:
            if perceptual_loss is None:
                raise ValueError(
                    "perceptual_loss is required when LPIPS is enabled"
                )
            selected = t < self.lpips_max_time
            if selected.any():
                predicted_x = z - expand(t.clamp_min(self.min_time), z) * u
                predicted = self.restore(condition, predicted_x)
                height, width = image_size
                scores = perceptual_loss(
                    predicted[selected, :, :height, :width].clamp(0, 1),
                    clean[selected, :, :height, :width],
                ).reshape(-1)
                perceptual_loss.reset()
                indices = selected.nonzero(as_tuple=False).flatten()
                lpips = lpips.index_copy(0, indices, scores)
            loss = loss + self.lpips_weight * self._adaptive_loss(lpips)

        if reduction == "mean":
            loss = loss.mean()
            mean_flow = mean_flow.mean()
            if quality_loss is not None:
                quality_loss = quality_loss.mean()
            selected_count = (t < self.lpips_max_time).sum().clamp_min(1)
            lpips = lpips.sum() / selected_count
        losses = {"loss": loss, "mean_flow": mean_flow.detach()}
        if quality_loss is not None:
            losses["quality"] = quality_loss.detach()
        if self.lpips_weight and include_perceptual:
            losses["lpips"] = lpips.detach()
        return losses

    def sample(self, degraded, *, generator=None, noise=None, sample_steps=None):
        steps = self.sample_steps if sample_steps is None else sample_steps
        if steps < 1:
            raise ValueError("sample_steps must be positive")

        height, width = degraded.shape[-2:]
        degraded = self.model.pad(degraded)
        condition = self.encode(degraded)
        z = self.prepare_noise(
            noise, condition, degraded, (height, width), generator
        )

        batch = degraded.shape[0]
        for step in range(steps, 0, -1):
            t = condition.new_full((batch,), step / steps)
            r = t - 1 / steps
            u = self.u(z, r, t, condition)
            z = z - expand(t - r, z) * u

        restored = self.restore(condition, z)[..., :height, :width]
        return restored.clamp(0, 1), None

    def forward(self, degraded):
        return self.sample(degraded)

DEFAULT_PALETTE_CONFIG = {
    "wavelet_levels": 2,
    "out_channels": 3,
    "patch_size": 2,
    "hidden_size": 576,
    "rearranged_patch_embed": True,
    "positional_reference_size": 16,
    "depth": 16,
    "num_heads": 18,
    "mlp_ratio": 2.5,
    "linear_head_dim": 32,
    "rank_enhance_kernel": 5,
    "window_block_interval": 0,
    "use_cross_attention": False,
    "quality_split": 3,
}

DEFAULT_FLOW_CONFIG = {
    "sample_steps": 2,
    "timestep_scale": 1000.0,
    "min_time": 0.05,
    "residual_scale": 16.0,
    "derivative_method": "rcm",
}


def load_jpeg_model(
    checkpoint,
    device="cpu",
    palette_config=DEFAULT_PALETTE_CONFIG,
    flow_config=DEFAULT_FLOW_CONFIG,
):
    """Load a trained MeanFlow JPEG-restoration checkpoint."""
    backbone = LinearDiTPalette(**palette_config)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {key.removeprefix("model."): value for key, value in state.items()}
    backbone.load_state_dict(state)
    flow = MeanFlow(backbone, **flow_config)
    return flow.to(device).eval()


def restore(flow, image, sample_steps=None):
    """Restore a (B, 3, H, W) float tensor of degraded JPEG pixels in [0, 1]."""
    with torch.inference_mode():
        restored, _ = flow.sample(image, sample_steps=sample_steps)
    return restored
