"""CPU reference for ancestry-local RDD birth; NOT a distributed runtime.

Input is a recovered CLEAN endpoint, not the noisy end-of-stage state.
The RNG is partition invariant but does not reproduce torch.randn's sequence.
Only integer-repeat transitions and Wan patch (1,2,2) are supported.
FP32 mixing / optional BF16 storage do not emulate native BF16 noise and
intermediate multiply/add rounding. This is not trajectory equivalence.
"""
from dataclasses import dataclass
from itertools import product
import math
import numpy as np

MASK = (1 << 32) - 1


def philox(counter, key):
    if len(counter) != 4 or len(key) != 2:
        raise ValueError('Philox requires four counter and two key words')
    c = [int(x) & MASK for x in counter]
    k = [int(x) & MASK for x in key]
    for _ in range(10):
        p0, p1 = 0xD2511F53 * c[0], 0xCD9E8D57 * c[2]
        c = [(p1 >> 32) ^ c[1] ^ k[0], p1 & MASK,
             (p0 >> 32) ^ c[3] ^ k[1], p0 & MASK]
        k = [(k[0] + 0x9E3779B9) & MASK,
             (k[1] + 0xBB67AE85) & MASK]
    return tuple(c)


def normal_pair(pair_id, seed, stage, stream=0):
    if not (0 <= pair_id < 2**64 and 0 <= seed < 2**64
            and 0 <= stage < 2**32 and 0 <= stream < 2**32):
        raise ValueError('RNG address out of range')
    a, b, _, _ = philox((pair_id & MASK, pair_id >> 32, stage, stream),
                        (seed & MASK, seed >> 32))
    # FP32 open-at-zero uniforms. Rounding may produce 1, never 0.
    u = np.float32(np.float32(a) * np.float32(2**-32) + np.float32(2**-32))
    v = np.float32(np.float32(b) * np.float32(2**-32) + np.float32(2**-32))
    radius = np.float32(np.sqrt(np.float32(-2) * np.log(u)))
    angle = np.float32(np.float32(2 * math.pi) * v)
    return np.float32(radius * np.cos(angle)), np.float32(radius * np.sin(angle))


def shard_range(tokens, degree, rank):
    if tokens < 1 or degree < 1 or not 0 <= rank < degree or degree > tokens:
        raise ValueError('Invalid or empty sequence shard')
    return tokens * rank // degree, tokens * (rank + 1) // degree


def validate_shapes(parent, fine):
    if len(parent) != 3 or len(fine) != 3:
        raise ValueError('Shapes must be (T,H,W)')
    if any(not isinstance(n, int) or n < 1 for n in (*parent, *fine)):
        raise ValueError('Positive integer dimensions required')
    if any(n % 2 for n in (*parent[1:], *fine[1:])):
        raise ValueError('Wan spatial patch size is 2')
    if any(f % p or f < p for p, f in zip(parent, fine)):
        raise ValueError('Only integer repeat-upsample supported')
    return tuple(f // p for p, f in zip(parent, fine))


@dataclass
class BirthPacket:
    parent_shape: tuple
    fine_shape: tuple
    source_degree: int
    destination_degree: int
    rank: int
    token_start: int
    token_end: int
    voxel_ids: np.ndarray
    parent_map: np.ndarray
    source_owners: np.ndarray
    source_physical_ranks: tuple
    destination_physical_rank: int

    def byte_counts(self, batch, channels, value_bytes=2):
        remote = np.asarray(self.source_physical_ranks)[self.source_owners] != self.destination_physical_rank
        return dict(value_bytes=batch * channels * self.voxel_ids.size * value_bytes,
                    remote_value_bytes=batch * channels * int(remote.sum()) * value_bytes,
                    local_map_bytes=self.parent_map.nbytes,
                    # Planner IDs can be regenerated/cached; count them separately.
                    planner_id_bytes=self.voxel_ids.nbytes + self.source_owners.nbytes)


def packet_plan(parent_shape, fine_shape, source_degree, destination_degree, rank,
                source_ranks=None, destination_ranks=None):
    ratios = validate_shapes(parent_shape, fine_shape)
    tc, hc, wc = parent_shape
    tf, hf, wf = fine_shape
    parent_tokens = tc * (hc // 2) * (wc // 2)
    shard_range(parent_tokens, source_degree, 0)
    start, end = shard_range(tf * (hf // 2) * (wf // 2), destination_degree, rank)
    source_ranks = tuple(range(source_degree)) if source_ranks is None else tuple(source_ranks)
    destination_ranks = tuple(range(destination_degree)) if destination_ranks is None else tuple(destination_ranks)
    for members,degree in ((source_ranks,source_degree),(destination_ranks,destination_degree)):
        if len(members)!=degree or len(set(members))!=degree or any(not isinstance(x,int) or x<0 for x in members):
            raise ValueError('Distinct nonnegative physical rank IDs required')
    token = np.arange(start, end, dtype=np.int64)
    t = token // ((hf // 2) * (wf // 2))
    y = token // (wf // 2) % (hf // 2)
    x = token % (wf // 2)
    ids = np.stack([((t // ratios[0]) * hc + (2*y + dy) // ratios[1]) * wc
                    + (2*x + dx) // ratios[2] for dy, dx in product(range(2), repeat=2)], axis=1)
    unique, inverse = np.unique(ids, return_inverse=True)
    pt = unique // (hc * wc)
    py, px = unique // wc % hc, unique % wc
    parent_token = (pt * (hc // 2) + py // 2) * (wc // 2) + px // 2
    # Search actual uneven-shard boundaries; token*P/N is not always correct.
    ends = np.array([parent_tokens * (r+1) // source_degree for r in range(source_degree)])
    owners = np.searchsorted(ends, parent_token, side='right').astype(np.int32)
    return BirthPacket(tuple(parent_shape), tuple(fine_shape), source_degree,
                       destination_degree, rank, start, end, unique,
                       inverse.reshape(end-start, 4).astype(np.int64), owners,
                       source_ranks,destination_ranks[rank])


def bf16_rne(values):
    """CPU emulation of finite FP32 -> BF16 RNE -> FP32 (not a BF16 GEMM)."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('BF16 conversion requires finite input')
    words = values.view(np.uint32)
    rounded = ((words + np.uint32(0x7fff) + ((words >> 16) & 1))
               & np.uint32(0xffff0000)).view(np.float32)
    if not np.isfinite(rounded).all():
        raise ValueError('BF16 conversion overflow')
    return rounded


def birth_patch(clean_parent, plan, data_time, seed, stage, stream=0,
                emulate_bf16=False, arithmetic='fp32_fused', sigma=None):
    """Produce [B, local patch tokens, C, 4] BEFORE original patch projection."""
    if clean_parent.ndim != 5 or tuple(clean_parent.shape[2:]) != plan.parent_shape:
        raise ValueError('Expected clean parent BCTHW with matching geometry')
    if not 0 <= data_time <= 1 or not np.isfinite(data_time):
        raise ValueError('data_time must be finite in [0,1]')
    if not np.isfinite(clean_parent).all():
        raise ValueError('Nonfinite clean parent')
    if arithmetic not in ('fp32_fused', 'bf16_sigma', 'bf16_scalar'):
        raise ValueError('Unknown birth arithmetic mode')
    if sigma is not None and (arithmetic=='fp32_fused' or not np.isfinite(sigma) or not 0<=sigma<=1):
        raise ValueError('Explicit sigma requires a BF16 mode and value in [0,1]')
    batch, channels = clean_parent.shape[:2]
    tf, hf, wf = plan.fine_shape
    if batch * channels * tf * hf * wf >= 2**63:
        raise ValueError('Address exceeds CUDA signed int64 indexing')
    packet = clean_parent.astype(np.float32).reshape(batch, channels, -1)[:, :, plan.voxel_ids]
    if not np.isfinite(packet).all():
        raise ValueError('Clean parent overflows FP32')
    if emulate_bf16 or arithmetic != 'fp32_fused':
        packet = bf16_rne(packet)
    alpha = np.float32(data_time)
    beta = np.float32(1) - alpha
    if arithmetic != 'fp32_fused':
        # Do not reconstruct scheduler sigma from ALREADY rounded FP32 alpha.
        # Use original sigma directly when available; otherwise Python data-time.
        beta = np.float32(1-data_time if sigma is None else sigma)
    if arithmetic == 'bf16_sigma':
        # sampler.py: sigma explicitly stored in BF16, then BF16 1-sigma.
        beta = bf16_rne(np.array(beta))
        alpha = bf16_rne(np.array(np.float32(1) - beta))
    n = plan.token_end-plan.token_start
    result = np.empty((batch, n, channels, 4), np.float32)
    for j, token in enumerate(range(plan.token_start, plan.token_end)):
        t = token // ((hf//2)*(wf//2))
        y, x = token // (wf//2) % (hf//2), token % (wf//2)
        for b, c, dy in product(range(batch), range(channels), range(2)):
            address = ((b*channels+c)*tf+t)*hf*wf + (2*y+dy)*wf+2*x
            noise = normal_pair(address//2, seed, stage, stream)
            for dx in range(2):
                parent = packet[b,c,plan.parent_map[j,2*dy+dx]]
                if arithmetic == 'fp32_fused':
                    result[b,j,c,2*dy+dx] = alpha*parent + beta*noise[dx]
                else:
                    stored_noise = bf16_rne(np.array(noise[dx]))
                    result[b,j,c,2*dy+dx] = bf16_rne(
                        bf16_rne(np.array(alpha*parent)) + bf16_rne(np.array(beta*stored_noise)))
    return bf16_rne(result) if emulate_bf16 else result


def patch_project_reference(patches, conv_weight, bias=None):
    """CPU FP32 adapter: original Conv3d [D,C,1,2,2] -> patch GEMM.

    Verifies weight/order algebra only; no BF16 GEMM or GPU accuracy claim.
    """
    if patches.ndim != 4 or patches.shape[-1] != 4:
        raise ValueError('Patches must be [B,L,C,4]')
    if conv_weight.ndim != 5 or tuple(conv_weight.shape[2:]) != (1,2,2):
        raise ValueError('Expected Wan Conv3d weight [D,C,1,2,2]')
    if conv_weight.shape[1] != patches.shape[2]:
        raise ValueError('Input channel mismatch')
    if bias is not None and np.asarray(bias).shape != (conv_weight.shape[0],):
        raise ValueError('Bias shape mismatch')
    x = np.asarray(patches,dtype=np.float32).reshape(*patches.shape[:2],-1)
    w = np.asarray(conv_weight,dtype=np.float32).reshape(conv_weight.shape[0],-1)
    y = x @ w.T
    return y if bias is None else y + np.asarray(bias,dtype=np.float32)


def best_sp_schedule(stage_seconds, transition_seconds, vae_seconds=0,
                     startup_seconds=None, final_assembly_seconds=None, max_gpus=4):
    """Use MEASURED stage/edge times. No built-in speed guesses or profiles.

    transition_seconds[(boundary, source_degree, destination_degree)] must
    include birth/reshard/synchronization cost, including same-degree birth.
    Keys may be degree integers (prefix groups) or explicit physical-rank
    tuples. Use measured profiles of LEGAL groups, including SP3 if supported.
    Startup/final dictionaries are required for nonzero costs; omitted means
    an explicitly already-resident / zero-assembly-cost synthetic scope.
    All candidates use the same GPU budget, batching, RNG, kernels and VAE.
    """
    if not stage_seconds or vae_seconds < 0 or not math.isfinite(vae_seconds):
        raise ValueError('Invalid timing profile')
    def members(group):
        if isinstance(group,int) and group>0:
            return tuple(range(group))
        if isinstance(group,tuple) and group and len(set(group))==len(group) and all(isinstance(r,int) and r>=0 for r in group):
            return group
        raise ValueError('Invalid SP physical group')
    if not isinstance(max_gpus,int) or max_gpus<1:
        raise ValueError('Invalid GPU budget')
    for row in stage_seconds:
        if not row or any(not math.isfinite(v) or v <= 0 for v in row.values()):
            raise ValueError('Positive stage times required')
        for group in row:
            if len(members(group))>max_gpus: raise ValueError('Group exceeds GPU budget')
    def cost(schedule):
        edges = [transition_seconds[(i, a, b)] for i, (a,b) in enumerate(zip(schedule, schedule[1:]))]
        start=0. if startup_seconds is None else startup_seconds[schedule[0]]
        finish=0. if final_assembly_seconds is None else final_assembly_seconds[schedule[-1]]
        edges += [start,finish]
        if any(not math.isfinite(v) or v < 0 for v in edges):
            raise ValueError('Invalid boundary time')
        return vae_seconds + sum(stage_seconds[i][p] for i,p in enumerate(schedule)) + sum(edges)
    candidates = [(cost(s), s) for s in product(*(tuple(row) for row in stage_seconds))
                  if len(set(r for group in s for r in members(group)))<=max_gpus]
    if not candidates:
        raise ValueError('No schedule within physical GPU budget')
    degrees = set.intersection(*(set(row) for row in stage_seconds))
    if not degrees:
        raise ValueError('No fixed-degree control available')
    fixed = min(((cost((d,)*len(stage_seconds)), (d,)*len(stage_seconds)) for d in degrees),
                key=lambda x:(x[0],repr(x[1])))
    best = min(candidates,key=lambda x:(x[0],repr(x[1])))
    return dict(schedule=list(best[1]), seconds=best[0], fixed_schedule=list(fixed[1]),
                fixed_seconds=fixed[0], reduction_percent=100*(1-best[0]/fixed[0]))
