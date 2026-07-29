#!/usr/bin/env python3
"""
train_elya_moe.py — Lock-On MoE trainer (see docs/LOCKON_MOE.md).

Trains N topic experts that SHARE one embedding table plus a small learned
router, and exports a single SGTM blob.

Why shared embedding: it is needed for every token regardless of which
expert answers, so it lives in the fixed ROM region and is never banked
out. Training it jointly across all experts also gives every expert the
same view of the alphabet, which matters when they must produce the same
character set.

Each expert is a full 2L/64d ternary transformer trained on its own corpus
shard (~30 QA pairs instead of 122), which is the whole point: capacity
per task goes up 4x without the working set growing at all.

Usage: train_elya_moe.py [steps_per_expert]
"""
import json, math, struct, sys, time, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experts import shard, NAMES, N_EXPERTS

N_LAYERS, N_EMBED, N_HEADS, VOCAB, CTX = 2, 64, 4, 256, 64
HEAD_DIM = N_EMBED // N_HEADS
FFN_DIM = N_EMBED * 4
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
LR = float(os.environ.get("EG_LR", "5e-4"))
PIN_W = float(os.environ.get("EG_PIN", "1e-2"))
REPEAT = int(os.environ.get("EG_QA_REPEAT", "1200"))

HERE = Path(__file__).resolve().parent
OUT_BIN = HERE.parent / "res" / "elya_moe.bin"
OUT_PT = HERE / "elya_moe.pt"
OUT_VEC = HERE / "moe_vectors.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ── data: one corpus per expert ───────────────────────────────────────
shards, router_examples, _generic = shard()
data = []
for i, lines in enumerate(shards):
    expanded = []
    for _ in range(REPEAT):
        l = lines[:]
        np.random.shuffle(l)
        expanded.extend(l)
    b = ("\n".join(expanded) + "\n").encode("ascii", errors="replace")
    data.append(torch.tensor(list(b), dtype=torch.long))
    print(f"expert {i} {NAMES[i]:10s}: {len(b):,} bytes")

# ── fake-quant (identical to the single-model trainer) ────────────────
def ste(x, q): return x + (q - x).detach()
def fq_act(x): return ste(x, (x * 4096).round().clamp(-32768, 32767) / 4096)
def fq_kv(x):  return ste(x, (x * 16).round().clamp(-128, 127) / 16)
def fq_emb(w): return ste(w, (w * 64).round().clamp(-128, 127) / 64)

class TernaryLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, std=0.5 / math.sqrt(in_f))
    def quantized(self):
        w = self.weight
        s = w.abs().mean().clamp(min=1e-5)
        return (w / s).round().clamp(-1, 1), s
    def forward(self, x):
        wq, s = self.quantized()
        return fq_act(F.linear(x, ste(self.weight, wq * s)))

class RMSNorm(nn.Module):
    def forward(self, x):
        return fq_act(x / x.pow(2).mean(-1, keepdim=True).add(1e-8).sqrt())

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq, self.wk = TernaryLinear(N_EMBED, N_EMBED), TernaryLinear(N_EMBED, N_EMBED)
        self.wv, self.wo = TernaryLinear(N_EMBED, N_EMBED), TernaryLinear(N_EMBED, N_EMBED)
        self.register_buffer("mask", torch.tril(torch.ones(CTX, CTX)).view(1, 1, CTX, CTX))
    def forward(self, x):
        B, T, C = x.shape
        pr = lambda l, z: l(z).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        q, k, v = pr(self.wq, x), fq_kv(pr(self.wk, x)), fq_kv(pr(self.wv, x))
        a = (q @ k.transpose(-2, -1)) * (HEAD_DIM ** -0.5)
        a = a.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        o = fq_act((F.softmax(a, -1) @ v).transpose(1, 2).contiguous().view(B, T, C))
        return self.wo(o)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(), RMSNorm()
        self.attn = Attn()
        self.wff1, self.wff2 = TernaryLinear(N_EMBED, FFN_DIM), TernaryLinear(FFN_DIM, N_EMBED)
    def forward(self, x):
        x = fq_act(x + self.attn(self.ln1(x)))
        return fq_act(x + self.wff2(F.relu(self.wff1(self.ln2(x)))))

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.ln_f = RMSNorm()

class ElyaMoE(nn.Module):
    """Shared embedding + shared positional encoding + N experts + router.

    The PE is SHARED like the embedding: every expert needs it, it is
    tiny (4KB), and it lives in the fixed ROM region that is never banked
    out. Measured on a single shard it took exact answers from 0/38 to
    36/38, so the MoE must carry it."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, N_EMBED)
        nn.init.normal_(self.emb.weight, std=0.3)
        self.pos = nn.Embedding(CTX, N_EMBED)
        nn.init.normal_(self.pos.weight, std=0.05)
        self.experts = nn.ModuleList(Expert() for _ in range(N_EXPERTS))
        self.router = TernaryLinear(N_EMBED, N_EXPERTS)

    def forward(self, idx, e):
        ew = fq_emb(self.emb.weight)
        x = ew[idx] + fq_emb(self.pos.weight)[:idx.shape[1]].unsqueeze(0)
        x = fq_act(x)
        ex = self.experts[e]
        for b in ex.blocks:
            x = b(x)
        return ex.ln_f(x) @ ew.T

    def route(self, idx, mask=None):
        """Mean-pool prompt embeddings -> expert logits."""
        ew = fq_emb(self.emb.weight)
        x = ew[idx]
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(1)
        return self.router(fq_act(x))

model = ElyaMoE().to(device)
tot = sum(p.numel() for p in model.parameters())
per_expert = sum(p.numel() for p in model.experts[0].parameters())
print(f"\ntotal {tot:,} params | shared emb {VOCAB*N_EMBED:,} | "
      f"per-expert {per_expert:,} | ACTIVE per token "
      f"{VOCAB*N_EMBED + per_expert:,}")

# ── router training data ──────────────────────────────────────────────
rp = torch.zeros(len(router_examples), CTX, dtype=torch.long)
rm = torch.zeros(len(router_examples), CTX)
ry = torch.zeros(len(router_examples), dtype=torch.long)
for i, (prompt, e) in enumerate(router_examples):
    b = prompt.encode("ascii", "replace")[:CTX]
    rp[i, :len(b)] = torch.tensor(list(b))
    rm[i, :len(b)] = 1.0
    ry[i] = e
rp, rm, ry = rp.to(device), rm.to(device), ry.to(device)

def batch(e, bs=256):
    d = data[e]
    ix = torch.randint(len(d) - CTX - 1, (bs,))
    x = torch.stack([d[i:i+CTX] for i in ix])
    y = torch.stack([d[i+1:i+CTX+1] for i in ix])
    return x.to(device), y.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0,
                        betas=(0.9, 0.95))
total_steps = STEPS * N_EXPERTS
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps,
                                                   eta_min=5e-5)
torch.manual_seed(1234)
evx = [batch(e, 512) for e in range(N_EXPERTS)]

@torch.no_grad()
def eval_all():
    model.eval()
    ls = []
    for e in range(N_EXPERTS):
        x, y = evx[e]
        ls.append(F.cross_entropy(model(x, e).view(-1, VOCAB), y.view(-1)).item())
    racc = (model.route(rp, rm).argmax(-1) == ry).float().mean().item()
    model.train()
    return ls, racc

print(f"\ntraining {total_steps:,} steps ({STEPS:,} per expert)")
t0, best, best_state = time.time(), 1e9, None
for step in range(total_steps):
    e = step % N_EXPERTS                       # round-robin over experts
    x, y = batch(e)
    loss = F.cross_entropy(model(x, e).view(-1, VOCAB), y.view(-1))
    loss = loss + PIN_W * F.relu(model.emb.weight.abs() - 1.8).pow(2).mean()
    # router trained on every step: it is tiny and needs the signal
    loss = loss + 0.5 * F.cross_entropy(model.route(rp, rm), ry)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 1000 == 0:
        ls, racc = eval_all()
        mean = sum(ls) / len(ls)
        if mean < best:
            best = mean
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if step % 8000 == 0:
            print(f"  {step:6d}/{total_steps}  mean={mean:.4f} best={best:.4f} "
                  f"router={racc*100:.0f}%  [" +
                  " ".join(f"{l:.2f}" for l in ls) + f"]  {time.time()-t0:.0f}s",
                  flush=True)

ls, racc = eval_all()
print(f"\nDone. best={best:.4f}  router={racc*100:.1f}%  {time.time()-t0:.0f}s")
model.load_state_dict(best_state); model.eval()
torch.save(best_state, OUT_PT)

# ── generations ───────────────────────────────────────────────────────
@torch.no_grad()
def gen(prompt, n=60):
    b = list(prompt.encode("ascii", "replace"))[:CTX]
    pi = torch.zeros(1, CTX, dtype=torch.long, device=device)
    pm = torch.zeros(1, CTX, device=device)
    pi[0, :len(b)] = torch.tensor(b, device=device)
    pm[0, :len(b)] = 1.0
    e = int(model.route(pi, pm).argmax(-1))
    x = torch.tensor([b], dtype=torch.long, device=device)
    out = []
    for _ in range(n):
        lg = model(x[:, -CTX:], e)[0, -1]
        nl = lg[10].item()
        lg[:32] = float("-inf"); lg[127:] = float("-inf"); lg[10] = nl
        t = int(lg.argmax())
        out.append(t)
        x = torch.cat([x, torch.tensor([[t]], device=device)], 1)
        if t == 10: break
    return e, bytes(out).decode("ascii", "replace")

PROMPTS = ["Who are you?: ", "What is your name?: ", "What is RustChain?: ",
           "What is the G4?: ", "Tell me about this dungeon.: ",
           "How do I earn RTC?: "]
vectors = []
print("\n-- generations (routed) --")
for p in PROMPTS:
    e, g = gen(p)
    vectors.append({"prompt": p, "expert": e, "expect": g})
    print(f"  [{NAMES[e]:9s}] {p!r} -> {g!r}")
OUT_VEC.write_text(json.dumps(vectors, indent=2))

# ── SGTM export ───────────────────────────────────────────────────────
def requant(s):
    S = 0
    while s * (1 << (S + 1)) <= 127 and S < 24:
        S += 1
    return max(1, min(127, round(s * (1 << S)))), S

def index_streams(wq_np):
    out = bytearray()
    for row in wq_np:
        adds = np.nonzero(row > 0)[0].astype(np.uint8)
        subs = np.nonzero(row < 0)[0].astype(np.uint8)
        out += struct.pack(">HH", len(adds), len(subs))
        out += adds.tobytes() + subs.tobytes()
    return bytes(out)

def tensor_blob(lin):
    wq, s = lin.quantized()
    M, S = requant(s.detach().item())
    return struct.pack(">HBB", M, S, 0) + index_streams(wq.detach().cpu().numpy())

# header
buf = bytearray()
buf += struct.pack(">4sBBBBHHHH", b"SGTM", N_EXPERTS, N_LAYERS, N_HEADS, 0,
                   N_EMBED, VOCAB, CTX, 7)   # bit2 = positional encoding
# shared: embedding
ew = fq_emb(model.emb.weight).detach().cpu().numpy()
buf += np.clip(np.round(ew * 64), -128, 127).astype(np.int8).tobytes()
pw = fq_emb(model.pos.weight).detach().cpu().numpy()
buf += np.clip(np.round(pw * 64), -128, 127).astype(np.int8).tobytes()
# shared: exp LUT
buf += np.array([round(16384 * math.exp(-i / 16.0)) for i in range(256)],
                dtype=">u2").tobytes()
# shared: router
buf += tensor_blob(model.router)
# expert offset table (filled after we know the sizes)
off_pos = len(buf)
buf += b"\0" * (4 * N_EXPERTS)

offsets = []
for e in range(N_EXPERTS):
    offsets.append(len(buf))
    ex = model.experts[e]
    for blk in ex.blocks:
        for lin in (blk.attn.wq, blk.attn.wk, blk.attn.wv, blk.attn.wo,
                    blk.wff1, blk.wff2):
            buf += tensor_blob(lin)
for i, o in enumerate(offsets):
    buf[off_pos + i*4: off_pos + i*4 + 4] = struct.pack(">I", o)

OUT_BIN.write_bytes(bytes(buf))
sz = len(buf)
print(f"\nSGTM blob: {sz:,} bytes ({sz/1024:.1f} KB) -> {OUT_BIN}")
print(f"  shared {offsets[0]:,} B | per-expert "
      f"{(sz - offsets[0])//N_EXPERTS:,} B")
print(f"  a 4MB cart holds ~{(4*1024*1024 - offsets[0]) // ((sz-offsets[0])//N_EXPERTS)} experts "
      f"with no mapper at all")
