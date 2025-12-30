import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, repeat
from contextlib import contextmanager


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


@contextmanager
def null_context():
    yield


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

def orthogonal_ema_inplace(matrix, decay):
    # C = U * S * V.T  => C_ortho = U * V.T
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    # v = vh.mH # vh is V.T, so v is V.T.T = V. For real matrices, .mH is .T
    # v_t = v.transpose(-2, -1)
    orthogonal_target = u @ vh

    update_direction = orthogonal_target - matrix
    matrix.data.add_(update_direction, alpha=(1 - decay))

def gram_schmidt(v):
    
    def projection(u, v):
        denominator = (u * u).sum()

        if denominator < 1e-8:
            return (v * u).sum() * u / (denominator + 1e-8)
        return (v * u).sum() * u / denominator

    ortho = torch.zeros_like(v, device=v.device)
    ortho[0] = v[0].clone()

    for k in range(1, v.shape[0]):
        vk = v[k].clone()
        uk = 0
        for j in range(k):
            uj = ortho[j].clone()
            uk = uk + projection(uj, vk)
        ortho[k] = vk - uk
    return ortho

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

# regularization losses


def orthgonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = einsum('i d, j d -> i j', normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)

def info_nce(code, embeds, pos_indices, dist):
    # assume embeds are B, T, D
    # codes are B, T, D with N quantized values

    B, T, D = code.shape
    # calculate dist matrix B, T, N
    dist = dist.reshape(B, T, -1)
    # might need to accomodate time

    # print(code.shape, embeds.shape, dist.shape)

    # pos_loss = 1/2 * (code - embeds).pow(2).sum(dim=-1, keepdim=True)  # B, T, 1
    # contrast_loss = torch.logsumexp(dist/2, dim=0, keepdim=True)  # sum over batch dim to generate negatie samples
    # loss = pos_loss + contrast_loss
    p = torch.log_softmax(dist/2, dim=-1)
    # loss = -torch.log_softmax(p, dim=0)
    loss = -p
    # pos_mask = dist.max(dim=-1).indices
    loss = loss[pos_indices]

    # print(pos_loss.shape, contrast_loss.shape)
    return loss.mean()


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2
    ):
        super().__init__()
        self.decay = decay
        self.orthogonal_decay = 0.9 * decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    def forward(self, x, soft_counts=None):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        embed = self.embed.t()

        self.init_embed_(flatten)

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            if soft_counts is not None:
                batch_usage_counts = soft_counts
            else:
                batch_usage_counts = embed_onehot

            ema_inplace(self.cluster_size, batch_usage_counts.sum(0), self.decay)
            embed_sum = flatten.t() @ batch_usage_counts
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            # orthogonal_ema_inplace(self.embed_avg, self.orthogonal_decay)
            # self.embed_avg = gram_schmidt(self.embed_avg)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

            # orthogonal_ema_inplace(self.embed, self.decay)
            self.expire_codes_(x)

        return quantize, embed_ind, dist


# class CosineSimCodebook(nn.Module):
#     def __init__(
#         self,
#         dim,
#         codebook_size,
#         kmeans_init=False,
#         kmeans_iters=10,
#         decay=0.8,
#         eps=1e-5,
#         threshold_ema_dead_code=2
#     ):
#         super().__init__()
#         self.decay = decay

#         if not kmeans_init:
#             embed = l2norm(torch.randn(codebook_size, dim))
#         else:
#             embed = torch.zeros(codebook_size, dim)

#         self.codebook_size = codebook_size
#         self.kmeans_iters = kmeans_iters
#         self.eps = eps
#         self.threshold_ema_dead_code = threshold_ema_dead_code

#         self.register_buffer('initted', torch.Tensor([not kmeans_init]))
#         self.register_buffer('cluster_size', torch.zeros(codebook_size))
#         self.register_buffer('embed', embed)

#     @torch.jit.ignore
#     def init_embed_(self, data):
#         if self.initted:
#             return

#         embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters,
#                                      use_cosine_sim=True)
#         self.embed.data.copy_(embed)
#         self.cluster_size.data.copy_(cluster_size)
#         self.initted.data.copy_(torch.Tensor([True]))

#     def replace(self, samples, mask):
#         samples = l2norm(samples)
#         modified_codebook = torch.where(
#             mask[..., None],
#             sample_vectors(samples, self.codebook_size),
#             self.embed
#         )
#         self.embed.data.copy_(modified_codebook)

#     def expire_codes_(self, batch_samples):
#         if self.threshold_ema_dead_code == 0:
#             return

#         expired_codes = self.cluster_size < self.threshold_ema_dead_code
#         if not torch.any(expired_codes):
#             return
#         batch_samples = rearrange(batch_samples, '... d -> (...) d')
#         self.replace(batch_samples, mask=expired_codes)

#     def forward(self, x):
#         shape, dtype = x.shape, x.dtype
#         flatten = rearrange(x, '... d -> (...) d')
#         flatten = l2norm(flatten)

#         self.init_embed_(flatten)

#         embed = l2norm(self.embed)
#         dist = flatten @ embed.t()
#         embed_ind = dist.max(dim=-1).indices
#         embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
#         embed_ind = embed_ind.view(*shape[:-1])

#         quantize = F.embedding(embed_ind, self.embed)

#         if self.training:
#             bins = embed_onehot.sum(0)
#             ema_inplace(self.cluster_size, bins, self.decay)

#             zero_mask = (bins == 0)
#             bins = bins.masked_fill(zero_mask, 1.)

#             embed_sum = flatten.t() @ embed_onehot
#             embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
#             embed_normalized = l2norm(embed_normalized)
#             embed_normalized = torch.where(zero_mask[..., None], embed,
#                                            embed_normalized)
#             ema_inplace(self.embed, embed_normalized, self.decay)
#             self.expire_codes_(x)

#         return quantize, embed_ind, dist

# # main class

class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        # 新增：随机化相关超参（默认关闭）
        code_dropout_p: float = 0.0,       # 训练时随机屏蔽码字的概率
        per_sample_dropout: bool = True,   # True: 每个样本独立屏蔽；False: 整个 batch 共享同一遮罩
        gumbel_noise: float = 0.0,         # 往相似度加的 Gumbel 噪声强度（0 为关闭）
        sample_top_k: int | None = None,   # 若 >1：在 top-k 中随机选一个
        ema_bins_from_greedy: bool = False # 用贪心最近邻统计 EMA，减少统计偏差
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(torch.randn(codebook_size, dim))
        else:
            embed = torch.zeros(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # 新增超参保存
        self.code_dropout_p = float(code_dropout_p)
        self.per_sample_dropout = bool(per_sample_dropout)
        self.gumbel_noise = float(gumbel_noise)
        self.sample_top_k = sample_top_k
        self.ema_bins_from_greedy = bool(ema_bins_from_greedy)

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        embed, cluster_size = kmeans(
            data, self.codebook_size, self.kmeans_iters,
            use_cosine_sim=True
        )
        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        samples = l2norm(samples)
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    def _apply_code_dropout(self, dist: torch.Tensor, greedy_idx: torch.Tensor | None):
        """
        对相似度矩阵 dist [N, C] 进行随机屏蔽与/或加噪，仅在训练时调用。
        返回 masked_dist，用于后续选码字。
        """
        masked_dist = dist

        # 1) 随机屏蔽部分码字（codebook dropout）
        if self.code_dropout_p > 0.0:
            if self.per_sample_dropout:
                drop_mask = (torch.rand_like(dist) < self.code_dropout_p)
            else:
                drop_vec = (torch.rand(dist.size(-1), device=dist.device) < self.code_dropout_p)
                drop_mask = drop_vec.unsqueeze(0).expand_as(dist)

            masked_dist = dist.masked_fill(drop_mask, -1e9)

            # 保护：若某行被全屏蔽，则回退为未屏蔽（或至少放开贪心 idx）
            all_dropped = drop_mask.all(dim=-1)
            if all_dropped.any():
                if greedy_idx is None:
                    # 直接完全回退该行
                    masked_dist[all_dropped] = dist[all_dropped]
                else:
                    # 仅解锁贪心位置，确保至少有一个可选
                    rows = torch.where(all_dropped)[0]
                    masked_dist[rows] = masked_dist[rows]  # no-op 为了清晰
                    masked_dist[rows, greedy_idx[rows]] = dist[rows, greedy_idx[rows]]

        # 2) Gumbel 噪声（Gumbel-Max 采样）
        if self.gumbel_noise > 0.0:
            # 经典 Gumbel(0,1)
            u = torch.rand_like(masked_dist).clamp_(min=1e-9, max=1 - 1e-9)
            g = -torch.log(-torch.log(u))
            masked_dist = masked_dist + self.gumbel_noise * g

        return masked_dist

    def _choose_indices(self, masked_dist: torch.Tensor):
        """
        从 masked_dist 选择码字索引。
        支持 top-k 均匀随机；否则用 argmax。
        """
        if (self.sample_top_k is not None) and (self.sample_top_k > 1):
            k = min(self.sample_top_k, masked_dist.size(-1))
            topk = masked_dist.topk(k, dim=-1)
            topk_idx = topk.indices  # [N, k]
            # 在每行 top-k 里均匀随机一个
            n = masked_dist.size(0)
            r = torch.randint(k, (n,), device=masked_dist.device)
            chosen = topk_idx[torch.arange(n, device=masked_dist.device), r]
            return chosen
        else:
            return masked_dist.max(dim=-1).indices

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        flatten = l2norm(flatten)

        self.init_embed_(flatten)

        embed = l2norm(self.embed)
        # 原始相似度（作为返回 & 可选用于 EMA 统计的“贪心”）
        dist = flatten @ embed.t()  # [N, C]

        # 贪心最近邻（用于可选的稳定 EMA 统计）
        greedy_idx = dist.max(dim=-1).indices

        # 训练期可选：对相似度进行随机屏蔽/加噪 => 采样式选码字
        if self.training and (self.code_dropout_p > 0.0 or self.gumbel_noise > 0.0 or (self.sample_top_k is not None and self.sample_top_k > 1)):
            masked_dist = self._apply_code_dropout(dist, greedy_idx=greedy_idx)
            chosen_idx = self._choose_indices(masked_dist)
        else:
            masked_dist = dist
            chosen_idx = greedy_idx

        embed_ind = chosen_idx.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            # 统计用的 one-hot：可选使用贪心或实际选中的索引
            bins_idx = greedy_idx if self.ema_bins_from_greedy else chosen_idx
            embed_onehot = F.one_hot(bins_idx, self.codebook_size).type(dtype)

            bins = embed_onehot.sum(0)
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = flatten.t() @ embed_onehot  # [D, C]
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()  # [C, D]
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(zero_mask[..., None], embed, embed_normalized)

            ema_inplace(self.embed, embed_normalized, self.decay)

            # 注意：若使用较大的 dropout/随机化，可能导致“死亡码字”偏多
            self.expire_codes_(x)

        # 返回原始 dist（未加噪/未遮罩），与原版保持一致
        return quantize, embed_ind, dist

class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        n_embed=None,
        codebook_dim=None,
        decay=0.8,
        eps=1e-5,
        kmeans_init=False,
        kmeans_iters=10,
        use_cosine_sim=False,
        threshold_ema_dead_code=0,
        channel_last=True,
        accept_image_fmap=False,
        commitment_weight=None,
        commitment=1.,  # deprecate in next version, turn off by default
        orthogonal_reg_weight=0.,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        quantize_full_precision=False,
        cpc=False,
        **args
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        codebook_dim = default(codebook_dim, dim)
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection \
            else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection \
            else nn.Identity()

        self.eps = eps
        self.commitment_weight = default(commitment_weight, commitment)

        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_class = EuclideanCodebook if not use_cosine_sim \
            else CosineSimCodebook

        self._codebook = codebook_class(
            dim=codebook_dim,
            codebook_size=n_embed,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            **args
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

        # when quantizing, whether to turn off mixed precision
        self.quantize_full_precision = quantize_full_precision

        # use contrastive-predictive-coding for commitment loss
        self.cpc = cpc

    @property
    def codebook(self):
        return self._codebook.embed

    def forward(self, x):
        shape, device, codebook_size = x.shape, x.device, self.codebook_size

        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)

        context = null_context() if not self.quantize_full_precision else autocast(enabled=False)

        with context:
            quantize, embed_ind, dist = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                if self.cpc:
                    commit_loss = info_nce(quantize.detach(), x, embed_ind, dist)
                else:
                    commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthgonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(embed_ind, 'b (h w) -> b h w', h=height, w=width)

        return quantize, embed_ind, loss
