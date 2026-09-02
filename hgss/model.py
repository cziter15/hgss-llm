import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

try:
    from .triton_scan import (
        triton_factorized_quaternion_read,
        triton_quaternion_scan,
    )
except (ImportError, ModuleNotFoundError):
    # Keep the model importable as a standalone file.  The optimized
    # factorized backend below does not require Triton.
    triton_quaternion_scan = None
    triton_factorized_quaternion_read = None

Q = Tuple[torch.Tensor, torch.Tensor]
HGSS_ARCHITECTURE_VERSION = 4


def qmul(a: Q, b: Q) -> Q:
    a1, a2 = a; b1, b2 = b
    return a1*b1 - a2*torch.conj(b2), a1*b2 + a2*torch.conj(b1)


def qconj(q: Q) -> Q:
    return torch.conj(q[0]), -q[1]


def compose(a: Q, b: Q, c: Q, d: Q):
    ac = qmul(c, a); cb = qmul(c, b)
    return ac, (cb[0] + d[0], cb[1] + d[1])


def recurrent_scan(a: Q, b: Q) -> Q:
    s = (torch.zeros_like(b[0][:, 0]), torch.zeros_like(b[1][:, 0]))
    out1, out2 = [], []
    for t in range(b[0].size(1)):
        x = qmul((a[0][:, t], a[1][:, t]), s)
        s = (x[0] + b[0][:, t], x[1] + b[1][:, t])
        out1.append(s[0]); out2.append(s[1])
    return torch.stack(out1, 1), torch.stack(out2, 1)


def blelloch_scan(a: Q, b: Q, return_a: bool = False):
    T = b[0].size(1)
    if T == 1:
        return (a, b) if return_a else b
    P = 1 << (T - 1).bit_length(); pad = P - T
    oa, ob = a, b
    if pad:
        sa, sb = list(a[0].shape), list(b[0].shape); sa[1] = sb[1] = pad
        a = (torch.cat((a[0], torch.ones(sa, device=a[0].device, dtype=a[0].dtype)), 1),
             torch.cat((a[1], torch.zeros(sa, device=a[1].device, dtype=a[1].dtype)), 1))
        b = (torch.cat((b[0], torch.zeros(sb, device=b[0].device, dtype=b[0].dtype)), 1),
             torch.cat((b[1], torch.zeros(sb, device=b[1].device, dtype=b[1].dtype)), 1))

    tree = [(a, b)]
    while tree[-1][0][0].size(1) > 1:
        x, y = tree[-1]
        tree.append(compose((x[0][:,::2], x[1][:,::2]), (y[0][:,::2], y[1][:,::2]),
                            (x[0][:,1::2], x[1][:,1::2]), (y[0][:,1::2], y[1][:,1::2])))

    ra, rb = tree[-1]
    pa = (torch.ones_like(ra[0]), torch.zeros_like(ra[1]))
    pb = (torch.zeros_like(rb[0]), torch.zeros_like(rb[1]))
    inter = lambda x, y: torch.stack((x, y), 2).flatten(1, 2)
    for x, y in reversed(tree[:-1]):
        la, lb = (x[0][:,::2], x[1][:,::2]), (y[0][:,::2], y[1][:,::2])
        qa, qb = compose(pa, pb, la, lb)
        pa = (inter(pa[0], qa[0]), inter(pa[1], qa[1]))
        pb = (inter(pb[0], qb[0]), inter(pb[1], qb[1]))
    za, zb = compose((pa[0][:,:T], pa[1][:,:T]), (pb[0][:,:T], pb[1][:,:T]), oa, ob)
    return (za, zb) if return_a else zb


def _scan_chunk(a1, a2, b1, b2):
    za, zb = blelloch_scan((a1, a2), (b1, b2), return_a=True)
    return za[0], za[1], zb[0], zb[1]


_compiled_scan_chunk = None


def _get_compiled_scan_chunk():
    global _compiled_scan_chunk
    if _compiled_scan_chunk is None:
        _compiled_scan_chunk = (
            torch.compile(
                _scan_chunk,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
            if hasattr(torch, "compile") else _scan_chunk
        )
    return _compiled_scan_chunk


def chunked_scan(
    a: Q,
    b: Q,
    chunk_size: int = 128,
    checkpoint_scan: bool = False,
    compile_scan: bool = False,
) -> Q:
    """Memory-bounded scan with exact recurrent carry between chunks."""
    T = b[0].size(1)
    state = (torch.zeros_like(b[0][:, 0]), torch.zeros_like(b[1][:, 0]))
    out1, out2 = [], []
    for start in range(0, T, chunk_size):
        stop = min(start + chunk_size, T)
        ca = (a[0][:, start:stop], a[1][:, start:stop])
        cb = (b[0][:, start:stop], b[1][:, start:stop])
        scan_fn = _get_compiled_scan_chunk() if compile_scan and ca[0].is_cuda else _scan_chunk
        if checkpoint_scan and torch.is_grad_enabled():
            za1, za2, zb1, zb2 = _checkpoint(
                scan_fn, ca[0], ca[1], cb[0], cb[1],
                use_reentrant=False, preserve_rng_state=False,
            )
            prefix_a, z = (za1, za2), (zb1, zb2)
        else:
            za1, za2, zb1, zb2 = scan_fn(ca[0], ca[1], cb[0], cb[1])
            prefix_a, z = (za1, za2), (zb1, zb2)
        # Prefix result is from zero state; inject the incoming state exactly.
        # Recompute the affine prefix transition using the chunk's transitions.
        if start == 0:
            zz = z
        else:
            hstate = qmul(
                (prefix_a[0], prefix_a[1]),
                (state[0].unsqueeze(1), state[1].unsqueeze(1)),
            )
            zz = (z[0] + hstate[0], z[1] + hstate[1])
        out1.append(zz[0]); out2.append(zz[1])
        state = (zz[0][:, -1], zz[1][:, -1])
    return torch.cat(out1, 1), torch.cat(out2, 1)


@dataclass
class HGSSConfig:
    vocab_size: int
    dim: int = 128
    layers: int = 4
    heads: int = 4
    d_k: int = 8
    d_v: int = 16
    d_conv: int = 4
    memory_min: float = 4.0
    memory_max: float = 32768.0
    # Compatibility aliases used by the shared BPE trainer.
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    memory_min_horizon: Optional[float] = None
    memory_max_horizon: Optional[float] = None
    boundary_token_id: Optional[int] = None
    # ``auto`` selects the factorized FlashHGSS-style path.  The legacy
    # materializing backends remain available for regression benchmarks.
    scan_backend: str = "auto"
    # 0 chooses a power-of-two chunk near sqrt(sequence_length).
    scan_chunk_size: int = 0
    scan_checkpoint: bool = True
    scan_compile: bool = False
    scan_triton: bool = False
    # Preserve the complete quaternion-valued scan read instead of retaining
    # only its first real component.  The scalar mode remains available for
    # loading and benchmarking architecture-v4 checkpoints.
    full_quaternion_read: bool = True

    def __post_init__(self):
        if self.num_layers is not None:
            self.layers = int(self.num_layers)
        if self.num_heads is not None:
            self.heads = int(self.num_heads)
        if self.memory_min_horizon is not None:
            self.memory_min = float(self.memory_min_horizon)
        if self.memory_max_horizon is not None:
            self.memory_max = float(self.memory_max_horizon)
        if self.vocab_size <= 0 or self.dim <= 0 or self.layers <= 0:
            raise ValueError("vocab_size, dim, and layers must be positive")
        if self.heads <= 0 or self.d_k <= 0 or self.d_v <= 0:
            raise ValueError("heads, d_k, and d_v must be positive")
        if self.d_conv <= 0:
            raise ValueError("d_conv must be positive")
        if not 0 < self.memory_min <= self.memory_max:
            raise ValueError("expected 0 < memory_min <= memory_max")
        if self.scan_chunk_size < 0:
            raise ValueError("scan_chunk_size must be non-negative")
        allowed = {
            "auto", "flash", "triton", "legacy_triton",
            "chunked", "blelloch", "recurrent",
        }
        if self.scan_backend not in allowed:
            raise ValueError(f"scan_backend must be one of {sorted(allowed)}")


class HGSSBlock(nn.Module):
    def __init__(self, c: HGSSConfig):
        super().__init__()
        self.H, self.K, self.V, self.C = c.heads, c.d_k, c.d_v, c.d_conv
        self.inner = self.H * self.V
        self.scan_backend = c.scan_backend
        self.scan_chunk_size = int(c.scan_chunk_size)
        self.scan_checkpoint = bool(c.scan_checkpoint)
        self.scan_compile = bool(c.scan_compile)
        self.scan_triton = bool(c.scan_triton)
        self.full_quaternion_read = bool(c.full_quaternion_read)
        self.qn, self.kn = 4*self.H*self.K, 4*self.H*self.K
        self.vn, self.pn, self.dn = 4*self.H*self.V, 3*self.H*self.K, self.H*self.K
        self.norm = nn.RMSNorm(c.dim)
        self.proj = nn.Linear(c.dim, self.qn+self.kn+self.vn+self.pn+self.dn, bias=False)
        self.decay = nn.Parameter(torch.empty(self.H, self.K))
        self.key_scale = nn.Parameter(torch.ones(self.H))
        self.conv = nn.Conv1d(c.dim, c.dim, self.C, groups=c.dim, padding=self.C-1)
        self.gate = nn.Linear(c.dim, self.inner, bias=False)
        self.read_width = self.inner * (4 if self.full_quaternion_read else 1)
        self.out = nn.Linear(self.read_width, c.dim, bias=False)
        self._reset(c.memory_min, c.memory_max)

    def _reset(self, lo, hi):
        with torch.no_grad():
            h = torch.logspace(math.log10(lo), math.log10(hi), self.K, device=self.decay.device)
            h = torch.stack([h.roll(i % self.K) for i in range(self.H)])
            self.decay.copy_(torch.log(torch.expm1(h.reciprocal())))
            p0, p1 = self.qn+self.kn+self.vn, self.qn+self.kn+self.vn+self.pn
            self.proj.weight[p0:p1].uniform_(-.01, .01)
            self.proj.weight[p1:].zero_()

    def _q(self, x, n):
        B,T,_ = x.shape; x = x.float().view(B,T,self.H,n,4)
        return torch.complex(x[...,0], x[...,1]), torch.complex(x[...,2], x[...,3])

    def _prepare(self, x):
        qf,kf,vf,pf,df = torch.split(self.proj(x), [self.qn,self.kn,self.vn,self.pn,self.dn], -1)
        q, k, v = self._q(qf,self.K), self._q(kf,self.K), self._q(vf,self.V)
        n = torch.sqrt((k[0].abs().square()+k[1].abs().square()).sum(-1,keepdim=True)+1e-6)
        s = self.key_scale[None,None,:,None].float(); k = (k[0]/n*s, k[1]/n*s)

        B,T,_ = pf.shape
        p = (math.pi/math.sqrt(3))*torch.tanh(pf.float().view(B,T,self.H,3,self.K))
        dd = 2*torch.tanh(df.float().view(B,T,self.H,self.K)/2)
        rho = torch.exp(-F.softplus(self.decay.float()[None,None] + dd))
        p = p * torch.sqrt(torch.clamp(1-rho, min=1e-6)).unsqueeze(3)
        x,y,z = p[:,:, :,0], p[:,:,:,1], p[:,:,:,2]
        th = torch.sqrt(x*x+y*y+z*z+1e-8); g = torch.sin(th)/th
        a = (torch.complex(rho*torch.cos(th), rho*g*x).unsqueeze(3),
             torch.complex(rho*g*y, rho*g*z).unsqueeze(3))

        w = torch.sqrt(torch.clamp(1-rho*rho, min=1e-8))
        # Return low-rank factors.  Building b here would allocate the full
        # [B,T,H,V,K] quaternion tensor and defeat memory-efficient training.
        return q, k, v, a, w

    @staticmethod
    def _make_b(k, v, w):
        """Materialize an injection tensor only for a requested slice."""
        kc = qconj(k)
        return qmul(
            (v[0].unsqueeze(-1), v[1].unsqueeze(-1)),
            ((kc[0]*w).unsqueeze(-2), (kc[1]*w).unsqueeze(-2)),
        )

    def _effective_chunk_size(self, length):
        if self.scan_chunk_size > 0:
            return min(self.scan_chunk_size, length)
        # A power of two close to sqrt(T) minimizes C + ceil(T/C), while
        # bounds keep compilation and launch behavior reasonable.
        target = max(1, int(math.sqrt(length)))
        lower = 1 << max(0, target.bit_length() - 1)
        upper = lower << 1
        chunk = lower if target - lower <= upper - target else upper
        return min(length, max(16, min(512, chunk)))

    def _scan_read_gate_chunk(
        self,
        a0, a1,
        q0, q1,
        k0, k1,
        v0, v1,
        w,
        x,
        state0, state1,
    ):
        """Replay one chunk without materializing its recurrent-state history."""
        B, C = x.shape[:2]
        gate = F.silu(self.gate(x)).view(B, C, self.H, self.V)
        outputs = []
        state = (state0, state1)

        for t in range(C):
            aa = (a0[:,t], a1[:,t])
            kk = (k0[:,t], k1[:,t])
            vv = (v0[:,t], v1[:,t])
            bb = self._make_b(kk, vv, w[:,t])
            transition = qmul(aa, state)
            state = (transition[0] + bb[0], transition[1] + bb[1])

            qq0 = q0[:,t].unsqueeze(-2)
            qq1 = q1[:,t].unsqueeze(-2)
            r0 = (state[0] * qq0 - state[1] * torch.conj(qq1)).sum(-1)
            if self.full_quaternion_read:
                r1 = (state[0] * qq1 + state[1] * torch.conj(qq0)).sum(-1)
                read = torch.stack((r0.real, r0.imag, r1.real, r1.imag), dim=-1)
                mixed = read.to(x.dtype) * gate[:,t].unsqueeze(-1)
                outputs.append(mixed.flatten(1))
            else:
                mixed = r0.real.to(x.dtype) * gate[:,t]
                outputs.append(mixed.flatten(1))

        return torch.stack(outputs, 1), state[0], state[1]

    def _factorized_scan_read_gate(self, a, q, k, v, w, x):
        """FlashHGSS-style scan: factorized injection and sparse state checkpoints."""
        B, T = x.shape[:2]
        state = (
            torch.zeros((B,self.H,self.V,self.K), dtype=a[0].dtype, device=x.device),
            torch.zeros((B,self.H,self.V,self.K), dtype=a[1].dtype, device=x.device),
        )
        chunk_size = self._effective_chunk_size(T)
        checkpoint_chunks = self.scan_checkpoint and self.training and torch.is_grad_enabled()
        outputs = []

        for start in range(0, T, chunk_size):
            stop = min(start + chunk_size, T)
            args = (
                a[0][:,start:stop], a[1][:,start:stop],
                q[0][:,start:stop], q[1][:,start:stop],
                k[0][:,start:stop], k[1][:,start:stop],
                v[0][:,start:stop], v[1][:,start:stop],
                w[:,start:stop], x[:,start:stop],
                state[0], state[1],
            )
            if checkpoint_chunks:
                mixed, state0, state1 = _checkpoint(
                    self._scan_read_gate_chunk,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                mixed, state0, state1 = self._scan_read_gate_chunk(*args)
            outputs.append(mixed)
            state = (state0, state1)
        return torch.cat(outputs, 1), state

    def _selected_backend(self):
        if self.scan_triton:
            return "triton"
        return "flash" if self.scan_backend == "auto" else self.scan_backend

    def _read(self, z, q):
        q0 = q[0].unsqueeze(-2)
        q1 = q[1].unsqueeze(-2)
        r0 = (z[0] * q0 - z[1] * torch.conj(q1)).sum(-1)
        if not self.full_quaternion_read:
            return r0.real
        r1 = (z[0] * q1 + z[1] * torch.conj(q0)).sum(-1)
        return torch.stack((r0.real, r0.imag, r1.real, r1.imag), dim=-1)

    def _read_and_gate(self, z, q, x):
        read = self._read(z, q)
        gate = F.silu(self.gate(x)).view(*x.shape[:-1], self.H, self.V)
        if self.full_quaternion_read:
            gate = gate.unsqueeze(-1)
        return (read.to(x.dtype) * gate).flatten(-3 if self.full_quaternion_read else -2)

    def _causal_conv(self, x, reset_mask=None):
        T = x.size(1)
        if reset_mask is None:
            return F.silu(self.conv(x.transpose(1,2))[:,:,:T].transpose(1,2))
        segments = reset_mask.to(torch.long).cumsum(1)
        y = torch.zeros_like(x)
        weight = self.conv.weight[:, 0]
        for lag in range(min(self.C, T)):
            if lag == 0:
                shifted = x
                valid = torch.ones_like(reset_mask)
            else:
                shifted = F.pad(x[:, :-lag], (0, 0, lag, 0))
                valid = F.pad(segments[:, lag:] == segments[:, :-lag], (lag, 0), value=False)
            y = y + shifted * valid.unsqueeze(-1) * weight[:, -1-lag]
        if self.conv.bias is not None:
            y = y + self.conv.bias
        return F.silu(y)

    def forward(self, x, cache=False, reset_mask=None):
        r = x; x = self.norm(x); T = x.size(1)
        c = self._causal_conv(x, reset_mask)
        q,k,v,a,w = self._prepare(c)
        if reset_mask is not None:
            reset = reset_mask[:, :, None, None, None]
            a = (a[0].masked_fill(reset, 0.0), a[1].masked_fill(reset, 0.0))
        backend = self._selected_backend()
        if backend in ("flash", "triton"):
            if backend == "flash":
                mixed, final_state = self._factorized_scan_read_gate(a,q,k,v,w,x)
            else:
                if not x.is_cuda:
                    raise RuntimeError("the factorized Triton backend requires CUDA")
                if triton_factorized_quaternion_read is None:
                    raise RuntimeError(
                        "the factorized Triton backend requires the sibling triton_scan module"
                    )
                a4 = (a[0].real, a[0].imag, a[1].real, a[1].imag)
                q4 = (q[0].real, q[0].imag, q[1].real, q[1].imag)
                k4 = (k[0].real, k[0].imag, k[1].real, k[1].imag)
                v4 = (v[0].real, v[0].imag, v[1].real, v[1].imag)
                gate = F.silu(self.gate(x)).view(x.size(0), T, self.H, self.V)
                mixed, state4 = triton_factorized_quaternion_read(
                    a4, q4, k4, v4, w, gate,
                    full_quaternion_read=self.full_quaternion_read,
                    chunk_size=self.scan_chunk_size,
                )
                mixed = mixed.to(x.dtype).flatten(
                    -3 if self.full_quaternion_read else -2
                )
                final_state = (
                    torch.complex(state4[0], state4[1]),
                    torch.complex(state4[2], state4[3]),
                )
            y = r + self.out(mixed)
        else:
            # Legacy paths are intentionally kept for numerical/performance
            # comparisons.  They materialize b and the complete z history.
            b = self._make_b(k,v,w)
            if backend == "legacy_triton":
                if not x.is_cuda:
                    raise RuntimeError("the Triton backend requires CUDA")
                if triton_quaternion_scan is None:
                    raise RuntimeError("the Triton backend requires the sibling triton_scan module")
                a4 = (a[0].real, a[0].imag, a[1].real, a[1].imag)
                b4 = (b[0].real, b[0].imag, b[1].real, b[1].imag)
                z4 = triton_quaternion_scan(a4, b4)
                z = (torch.complex(z4[0], z4[1]), torch.complex(z4[2], z4[3]))
            elif backend == "chunked":
                z = chunked_scan(
                    a, b, self._effective_chunk_size(T), self.scan_checkpoint, self.scan_compile
                )
            elif backend == "blelloch":
                z = blelloch_scan(a,b)
            elif backend == "recurrent":
                z = recurrent_scan(a,b)
            else:
                raise RuntimeError(f"unsupported scan backend: {backend}")
            y = r + self.out(self._read_and_gate(z, q, x))
            final_state = (z[0][:,-1], z[1][:,-1])
        if not cache: return y
        n = self.C-1; cc = x[:,-n:] if n else x[:,:0]
        if n and cc.size(1)<n: cc = F.pad(cc,(0,0,n-cc.size(1),0))
        return y, ((final_state[0].detach(), final_state[1].detach()), cc.detach())

    def step(self, x, cache):
        r=x; x=self.norm(x); z,cc=cache
        if self.C > 1:
            win=torch.cat((cc,x[:,None]),1); cc=win[:,1:].detach()
        else: win=x[:,None]
        c=F.silu(torch.einsum('bkd,dk->bd',win,self.conv.weight[:,0]) + self.conv.bias)
        q,k,v,a,w=self._prepare(c[:,None]); aa=(a[0][:,0],a[1][:,0])
        b=self._make_b(k,v,w); bb=(b[0][:,0],b[1][:,0])
        t=qmul(aa,z); z=(t[0]+bb[0],t[1]+bb[1]); qq=(q[0][:,0],q[1][:,0])
        y=r+self.out(self._read_and_gate(z,qq,x))
        return y, ((z[0].detach(),z[1].detach()),cc)


class HGSS(nn.Module):
    def __init__(self, c: HGSSConfig):
        super().__init__(); self.cfg=c
        self.emb=nn.Embedding(c.vocab_size,c.dim)
        self.blocks=nn.ModuleList([HGSSBlock(c) for _ in range(c.layers)])
        self.norm=nn.RMSNorm(c.dim); self.head=nn.Linear(c.dim,c.vocab_size,bias=False)
        self.apply(self._init); [b._reset(c.memory_min,c.memory_max) for b in self.blocks]
        self.head.weight=self.emb.weight

    # Names expected by the shared BPE trainer; properties avoid duplicate
    # state-dict entries while retaining the compact HGSS4 names internally.
    @property
    def lm_head(self):
        return self.head

    @property
    def token_embedding(self):
        return self.emb

    @property
    def final_norm(self):
        return self.norm

    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,std=.02)

    def forward(self, ids, targets=None):
        x=self.emb(ids)
        reset_mask = (ids == self.cfg.boundary_token_id) if self.cfg.boundary_token_id is not None else None
        for b in self.blocks: x=b(x, reset_mask=reset_mask)
        logits=self.head(self.norm(x))
        if targets is None: return logits
        return logits, F.cross_entropy(logits.flatten(0,1),targets.flatten())

    @torch.no_grad()
    def prefill(self, ids):
        x=self.emb(ids); caches=[]
        reset_mask = (ids == self.cfg.boundary_token_id) if self.cfg.boundary_token_id is not None else None
        for b in self.blocks: x,c=b(x,True,reset_mask=reset_mask); caches.append(c)
        return self.head(self.norm(x[:,-1])),caches

    @torch.no_grad()
    def step(self, token, caches):
        if token.ndim != 1:
            token = token.reshape(token.size(0))
        if self.cfg.boundary_token_id is not None:
            reset = token == self.cfg.boundary_token_id
            # Do this branchlessly. ``if torch.any(reset)`` forces a GPU->CPU
            # synchronization on every decoded token, even though BOS is rare.
            shaped = reset[:, None, None, None]
            caches = [
                (
                    (z[0].masked_fill(shaped, 0.0), z[1].masked_fill(shaped, 0.0)),
                    conv.masked_fill(reset[:, None, None], 0.0),
                )
                for (z, conv) in caches
            ]
        x=self.emb(token); out=[]
        for b,c in zip(self.blocks,caches): x,c=b.step(x,c); out.append(c)
        return self.head(self.norm(x)),out

    @torch.no_grad()
    def generate(
        self,
        ids,
        n=300,
        temperature=.8,
        top_k=40,
        max_new_tokens=None,
        use_cuda_graph=True,
    ):
        if max_new_tokens is not None:
            n = int(max_new_tokens)
        if n <= 0:
            return ids
        if use_cuda_graph and ids.is_cuda and ids.size(0) == 1:
            return self.generate_cuda_graph(
                ids,
                n=n,
                temperature=temperature,
                top_k=top_k,
            )
        logits,c=self.prefill(ids); generated=[]
        for _ in range(n):
            if temperature <= 0:
                t = logits.argmax(dim=-1)
            else:
                s=logits/max(float(temperature),1e-5)
                if top_k > 0:
                    cut=torch.topk(s,min(int(top_k),s.size(-1))).values[:,-1:]
                    s=s.masked_fill(s<cut,float('-inf'))
                t=torch.multinomial(F.softmax(s.float(),-1),1).squeeze(1)
            generated.append(t[:,None]); logits,c=self.step(t,c)
        return torch.cat((ids,*generated),1)

    @torch.no_grad()
    def generate_cuda_graph(
        self,
        ids,
        n=300,
        temperature=.8,
        top_k=40,
        banned_token_ids=None,
        precision="bf16",
    ):
        """Fast batch-1 CUDA decoding with recurrent state kept on device.

        The graph includes sampling, all recurrent blocks, the LM head, cache
        updates, and writing generated token IDs. It deliberately executes
        exactly ``n`` steps without a per-token CPU EOS synchronization; callers
        can truncate at EOS after the single final device-to-host transfer.
        """
        if not ids.is_cuda or ids.size(0) != 1:
            raise ValueError("CUDA graph generation requires CUDA batch_size=1")
        if n <= 0:
            return ids
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")

        def amp():
            return (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if precision == "bf16"
                else nullcontext()
            )

        with amp():
            initial_logits, initial_caches = self.prefill(ids)
        static_logits = initial_logits.clone()
        static_caches = [
            ((z[0].clone(), z[1].clone()), conv.clone())
            for z, conv in initial_caches
        ]
        generated = torch.empty((1, n), dtype=torch.long, device=ids.device)
        position = torch.zeros((1,), dtype=torch.long, device=ids.device)
        banned = None
        if banned_token_ids:
            banned = torch.tensor(
                sorted(set(int(token) for token in banned_token_ids)),
                dtype=torch.long,
                device=ids.device,
            )

        def one_step():
            scores = static_logits
            if banned is not None:
                scores = scores.clone()
                scores.index_fill_(1, banned, -torch.inf)
            if temperature <= 0:
                token = scores.argmax(dim=-1)
            else:
                scores = scores / max(float(temperature), 1e-5)
                if top_k > 0:
                    values, indices = torch.topk(
                        scores, min(int(top_k), scores.size(-1)), dim=-1,
                    )
                    draw = torch.multinomial(F.softmax(values.float(), dim=-1), 1)
                    token = indices.gather(-1, draw).squeeze(-1)
                else:
                    token = torch.multinomial(F.softmax(scores.float(), dim=-1), 1).squeeze(-1)
            generated.scatter_(1, position.view(1, 1), token.view(1, 1))
            position.add_(1)
            new_logits, new_caches = self.step(token, static_caches)
            static_logits.copy_(new_logits)
            for (old_z, old_conv), (new_z, new_conv) in zip(static_caches, new_caches):
                old_z[0].copy_(new_z[0])
                old_z[1].copy_(new_z[1])
                old_conv.copy_(new_conv)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), amp():
            for _ in range(3):
                # Sampling writes through a dynamic device-side position.  Keep
                # warmup inside the allocated [1,n] output even when n == 1.
                position.zero_()
                one_step()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        position.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), amp():
            one_step()

        # Capture and warmup advance the static state. Restore the true prompt
        # state before the first replay without reallocating graph addresses.
        static_logits.copy_(initial_logits)
        for (static_z, static_conv), (initial_z, initial_conv) in zip(
            static_caches, initial_caches,
        ):
            static_z[0].copy_(initial_z[0])
            static_z[1].copy_(initial_z[1])
            static_conv.copy_(initial_conv)
        position.zero_()

        for _ in range(n):
            graph.replay()
        torch.cuda.synchronize()
        return torch.cat((ids, generated), dim=1)


HGSSLanguageModel = HGSS


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
