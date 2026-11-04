"""CPU-only contracts for C-SP and stage-bucket serving research.

No distributed runtime, NCCL, quantized kernel or measured timing is provided.
Payload estimates count remote SENT bytes per rank, not sent+received bytes,
fabric hops, synchronization or critical-path latency. All geometry is exact,
unpadded, balanced, B=1, and all heads are resident on window-owning ranks.
"""
from dataclasses import dataclass, asdict
import math


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class CSPBudget:
    tokens: int = 61440
    hidden: int = 1536
    ranks: int = 4
    layers: int = 30
    root_slots: int = 3840
    summary_size: int = 64
    element_bytes: int = 2
    # Native FastVideo VSA sends Q/K/V/GATE forward, O backward.
    # A structurally absent gate is a DIFFERENT checkpoint contract.
    native_arrays: int = 5

    def calculate(self):
        for value in asdict(self).values():
            if not positive_int(value):
                raise ValueError('Positive integer dimensions required')
        n, d, p, w = self.tokens, self.hidden, self.ranks, self.element_bytes
        if n % (self.summary_size*p) or self.root_slots > n:
            raise ValueError('Unpadded balanced summaries and ROOT<=N required')
        summary_tokens = n//self.summary_size
        # Each all-to-all sends 1/P of each local shard to each remote peer.
        native = self.native_arrays*n*d*w*(p-1)/(p*p)
        # Local queries/gates/outputs stay local; all-gather CURRENT Kc,Vc.
        current = 2*summary_tokens*d*w*(p-1)/p
        # This ordinary control pools already-projected current K/V to R.
        # It uses the SAME window-owned local-global graph and C kernel.
        ordinary = current + 2*self.root_slots*d*w*(p-1)/p
        root_feature = self.root_slots*d*w
        return dict(
            scope='Analytical payload only; not measured time or novelness proof',
            native_remote_sent_bytes_per_rank_layer=native,
            c_current_summary_remote_sent_bytes_per_rank_layer=current,
            ordinary_current_bank_remote_sent_bytes_per_rank_layer=ordinary,
            c_per_layer_payload_reduction=0 if native == 0 else 1-current/native,
            native_remote_sent_bytes_per_rank_fine_stage=native*self.layers,
            c_current_summary_remote_sent_bytes_per_rank_fine_stage=current*self.layers,
            # Logical delivery from one producer to P-1 consumers. Tree broadcast
            # topology/leader bottleneck and producer distribution are not modeled.
            root_feature_logical_delivery_bytes_once=root_feature*(p-1),
            root_feature_bytes_per_rank=root_feature,
            root_layer_kv_scratch_bytes=2*root_feature,
            root_projection_flops_all_ranks=4*p*self.root_slots*d*d*self.layers,
            ordinary_n64_reuse_summary_extra_bank_bytes=0,
            ignored=['collective startup/overlap/topology/padding',
                     'root norm/time/RoPE and redundant projections',
                     'halo/shift communication if windows change',
                     'common text and tensor parallel communication',
                     'all model compute, root/middle stages, VAE'],
            measured_speedup=None)


def balanced_window_owners(shape=(16,48,80), window=8, ranks=4):
    """Complete full-time spatial windows; each VSA cube stays with one rank."""
    if len(shape) != 3 or any(not positive_int(x) for x in shape):
        raise ValueError('Positive T/H/W required')
    t,h,w=shape
    if not positive_int(window) or not positive_int(ranks):
        raise ValueError('Positive window/ranks required')
    if t%4 or window%4 or h%window or w%window:
        raise ValueError('Windows must align with original 4x4x4 cubes')
    count=(h//window)*(w//window)
    if count%ranks:
        raise ValueError('This prototype excludes unbalanced/padded ownership')
    owners=[[] for _ in range(ranks)]
    for iy in range(h//window):
        for ix in range(w//window):
            rank=(iy*(w//window)+ix)%ranks
            owners[rank].extend((z*h+y)*w+x for z in range(t)
                               for y in range(iy*window,(iy+1)*window)
                               for x in range(ix*window,(ix+1)*window))
    return owners


def batch_capacity(tokens, pairs, token_budget, pair_budget, memory_limit_batch):
    """Capacity envelope, NOT performance-optimal batch or a latency forecast.

    Batch attention costs scale with B*N², so token budgets alone are unsafe.
    memory_limit_batch must come from actual peak-memory profiling eventually.
    """
    vals=(tokens,pairs,token_budget,pair_budget,memory_limit_batch)
    if any(not positive_int(x) for x in vals):
        raise ValueError('Positive integer capacities required')
    return min(token_budget//tokens,pair_budget//pairs,memory_limit_batch)


@dataclass(frozen=True)
class BankVersion:
    """Cache identity; layer K/V changes with consumer timestep and weights."""
    producer_handle: str
    model_revision: str
    lora_revision: str
    layer: int
    timestep: float
    rope_layout: str
    precision_layout: str

    def __post_init__(self):
        if any(not isinstance(x,str) or not x for x in
               (self.producer_handle,self.model_revision,self.lora_revision,
                self.rope_layout,self.precision_layout)):
            raise ValueError('Explicit producer/model/LoRA/RoPE/precision identity required')
        if not isinstance(self.layer,int) or isinstance(self.layer,bool) or self.layer<0:
            raise ValueError('Nonnegative layer required')
        if isinstance(self.timestep,bool) or not isinstance(self.timestep,(int,float)) or not math.isfinite(self.timestep):
            raise ValueError('Finite timestep required')


def research_budget():
    cases={str(r):CSPBudget(root_slots=r).calculate() for r in (3840,960)}
    n=61440
    return dict(gpu_run=False,measured_speedup=None,communication=cases,
                stage_capacity_envelope=[dict(tokens=x,batch=batch_capacity(
                    x,x*x,n,n*n,16)) for x in (n//16,n//4,n)],
                warning='16/4/1 is a capacity example, not a measured batch optimum. No GPU/NCCL runtime.')
