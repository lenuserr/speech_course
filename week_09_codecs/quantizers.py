import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def uniform_init(*shape: int) -> torch.Tensor:
    tensor = torch.empty(*shape)
    nn.init.kaiming_uniform_(tensor)
    return tensor


class Perplexity(nn.Module):
    EPS = 1e-8

    def __init__(self, codebook_size: int):
        super().__init__()
        self.codebook_size = codebook_size

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(indices.flatten(), minlength=self.codebook_size).float()
        probs = counts / counts.sum()
        return torch.exp(-torch.sum(probs * torch.log(probs + self.EPS)))


class VectorQuantizationLoss(nn.Module):
    def __init__(self, commitment_cost: float = 1.0):
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        return codebook_loss + self.commitment_cost * commitment_loss


class CommitmentLoss(nn.Module):
    def __init__(self, commitment_cost: float = 1.0):
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        return self.commitment_cost * F.mse_loss(z_q.detach(), z)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.copy_(uniform_init(codebook_size, embedding_dim))

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 3
        B, D, T = z.shape

        flat_z = z.permute(0, 2, 1).reshape(B * T, D)
        cb = self.codebook.weight

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + cb.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * flat_z @ cb.t()
        )

        indices = distances.argmin(dim=1)
        return indices.view(B, T)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.codebook(indices)
        return z_q.permute(0, 2, 1).contiguous()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(z))


class VectorQuantizerEMA(VectorQuantizer):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__(codebook_size=codebook_size, embedding_dim=embedding_dim)
        self.decay = decay
        self.eps = eps

        self.register_buffer("ema_counts", torch.zeros(codebook_size))
        self.register_buffer("ema_embedding_sum", self.codebook.weight.data.clone())

        self.codebook.weight.requires_grad_(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, D, T = z.shape
        flat_z = z.permute(0, 2, 1).reshape(B * T, D)

        with torch.no_grad():
            indices = self.encode(z).reshape(-1)

            if self.training:
                one_hot = F.one_hot(indices, self.codebook_size).type(flat_z.dtype)

                counts_batch = one_hot.sum(dim=0)
                embedding_sum_batch = one_hot.t() @ flat_z

                self.ema_counts.mul_(self.decay).add_(counts_batch, alpha=1.0 - self.decay)
                self.ema_embedding_sum.mul_(self.decay).add_(
                    embedding_sum_batch, alpha=1.0 - self.decay
                )

                new_codebook = self.ema_embedding_sum / (
                    self.ema_counts.unsqueeze(1) + self.eps
                )
                self.codebook.weight.data.copy_(new_codebook)

        return self.decode(indices.view(B, T))


class VectorQuantizerRestart(VectorQuantizer):
    def __init__(self, codebook_size: int, embedding_dim: int, threshold: int = 20):
        super().__init__(codebook_size=codebook_size, embedding_dim=embedding_dim)
        self.threshold = threshold

        self.register_buffer("unused_steps", torch.zeros(codebook_size, dtype=torch.long))

    def _maybe_restart(self, flat_z: torch.Tensor, indices: torch.Tensor):
        device = flat_z.device

        used = torch.zeros(self.codebook_size, dtype=torch.bool, device=device)
        used.scatter_(0, indices.long().reshape(-1), True)

        self.unused_steps[used] = 0
        self.unused_steps[~used] += 1

        dead_mask = self.unused_steps >= self.threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return

        n_pool = flat_z.shape[0]
        rand_idx = torch.randint(0, n_pool, (n_dead,), device=device)
        replacements = flat_z[rand_idx].detach()
        self.codebook.weight.data[dead_mask] = replacements
        self.unused_steps[dead_mask] = 0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        flat_z = z.permute(0, 2, 1).reshape(-1, z.shape[1])
        with torch.no_grad():
            indices = self.encode(z).reshape(-1)
            if self.training:
                self._maybe_restart(flat_z, indices)
        return self.decode(indices.view(z.shape[0], z.shape[2]))


class FiniteScalarQuantizer(nn.Module):
    uses_straight_through_estimator = False

    def __init__(self, levels: list[int], embedding_dim: int):
        super().__init__()
        assert all(l % 2 == 1 for l in levels), "All levels must be odd integers"
        self.levels = levels
        self.fsq_dim = len(levels)
        self.embedding_dim = embedding_dim
        self.codebook_size = 1
        for l in levels:
            self.codebook_size *= l

        self.register_buffer("half_range", torch.tensor([l // 2 for l in levels], dtype=torch.float))
        self.register_buffer("levels_tensor", torch.tensor(levels, dtype=torch.long))
        basis = [1]
        for level in levels[:-1]:
            basis.append(basis[-1] * level)
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))

        self.project_down = nn.Conv1d(embedding_dim, self.fsq_dim, kernel_size=1)
        self.project_up = nn.Conv1d(self.fsq_dim, embedding_dim, kernel_size=1)

    def _quantize(self, h: torch.Tensor) -> torch.Tensor:
        half = self.half_range.view(1, -1, 1)
        squashed = torch.tanh(h) * half
        rounded = squashed.round()
        return squashed + (rounded - squashed).detach()

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.project_down(z)
        half = self.half_range.view(1, -1, 1)
        codes = (torch.tanh(h) * half).round()
        return codes.long()

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.project_up(codes.float())

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() != 3:
            raise ValueError(f"Expected FSQ codes with shape [B, fsq_dim, T], got {tuple(codes.shape)}")

        shifted = codes.long() + self.half_range.long()[None, :, None]
        return (shifted * self.basis[None, :, None]).sum(dim=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.project_down(z)
        z_q_low = self._quantize(h)
        return self.project_up(z_q_low)


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        n_codebooks: int = 8,
        quantizer_cls: type = VectorQuantizer,
        quantizer_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.n_codebooks = n_codebooks

        kw = quantizer_kwargs or {}
        self.codebooks = nn.ModuleList([
            quantizer_cls(codebook_size=codebook_size, embedding_dim=embedding_dim, **kw)
            for _ in range(n_codebooks)
        ])

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        indices_per_layer = []
        residual = z
        for q in self.codebooks:
            idx = q.encode(residual)
            z_k = q.decode(idx)
            indices_per_layer.append(idx)
            residual = residual - z_k
        return torch.stack(indices_per_layer, dim=1)

    def quantize_with_loss(
        self,
        z: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices_per_layer = []
        z_q = torch.zeros_like(z)
        residual = z
        vq_loss = z.new_zeros(())

        for q in self.codebooks:
            idx = q.encode(residual)
            z_k = q.decode(idx)

            indices_per_layer.append(idx)
            if loss_fn is not None:
                vq_loss = vq_loss + loss_fn(residual, z_k)

            z_q = z_q + z_k
            residual = residual - z_k.detach()

        indices = torch.stack(indices_per_layer, dim=1)
        if loss_fn is not None and self.n_codebooks > 0:
            vq_loss = vq_loss / self.n_codebooks
        return z_q, indices, vq_loss

    def decode(self, indices: torch.Tensor, n_layers: Optional[int] = None) -> torch.Tensor:
        if n_layers is None:
            n_layers = self.n_codebooks
        n_layers = min(n_layers, self.n_codebooks)

        out = None
        for k in range(n_layers):
            z_k = self.codebooks[k].decode(indices[:, k])
            out = z_k if out is None else out + z_k
        if out is None:
            B, _, T = indices.shape
            out = indices.new_zeros((B, self.embedding_dim, T), dtype=torch.float32)
        return out

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(z))
