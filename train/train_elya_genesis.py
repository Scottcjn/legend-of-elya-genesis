#!/usr/bin/env python3
"""
Legend of Elya — Genesis: ternary QAT training + SGT1 export.

Model (Genesis-sized):
  2 layers, 64 embed, 4 heads (hd=16), ctx 64, vocab 256 (byte-level)
  ~112K params. RMSNorm (no learned params), tied embedding, ReLU FFN 4x.

Quantization-aware training mirrors the C engine (src/elya_gpt.c):
  - matmul weights : ternary {-1,0,+1}, per-tensor scale (BitNet b1.58 STE)
  - activations    : int16 Q3.12  (step 1/4096, clamp ±8)
  - KV cache       : int8  Q4.4   (step 1/16,   clamp ±8)
  - embedding      : int8  Q2.6   (step 1/64,   clamp ±2), tied logits

Export: SGT1 blob (big-endian, 68000-native) + test vectors JSON.
Corpus: reused verbatim from the N64 train_sophia_v5.py (ast-extracted).
"""
import ast, json, math, struct, sys, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_LAYERS, N_EMBED, N_HEADS, VOCAB, CTX = 2, 64, 4, 256, 64
HEAD_DIM = N_EMBED // N_HEADS
FFN_DIM = N_EMBED * 4
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
# autotrain.py sweeps these; defaults match the v4 run
import os as _os
LR         = float(_os.environ.get("EG_LR", "5e-4"))
PIN_W      = float(_os.environ.get("EG_PIN", "1e-2"))
QA_REPEAT  = int(_os.environ.get("EG_QA_REPEAT", "1200"))

HERE = Path(__file__).resolve().parent
OUT_BIN = HERE.parent / "res" / "elya_genesis.bin"
OUT_VEC = HERE / "test_vectors.json"
OUT_PT  = HERE / "elya_genesis.pt"
V5 = Path("/home/scott/legend-of-elya-n64/train_sophia_v5.py")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); random.seed(42); np.random.seed(42)

# ── Corpus: vendored in-repo so a fresh clone builds ──────────────────
import json as _json
_c = _json.loads((HERE / "corpus.json").read_text())
QA_PAIRS, CORPUS_LINES = _c["QA_PAIRS"], _c["CORPUS_LINES"]

# EG_SHARD=<name> trains on ONE expert shard only. This is the capacity
# control: if a 114K model can speak cleanly when it only has to memorize
# ~30 QA pairs, capacity was the bottleneck and the MoE premise holds. If
# it still babbles, the bottleneck is the QAT noise floor and sharding
# cannot fix it.
_SHARD = _os.environ.get("EG_SHARD", "")
if _SHARD:
    import experts as _X
    _idx = _X.NAMES.index(_SHARD)
    _shards, _r, _g = _X.shard()
    QA_PAIRS = [l for l in _shards[_idx] if ":" in l]
    CORPUS_LINES = []
    print(f"SHARD MODE: {_SHARD} ({len(QA_PAIRS)} lines only)")

# v4: PURE QA corpus - the game only ever asks these; let the tiny
# model memorize its script instead of diluting over lore lines
qa_expanded = []
for _ in range(QA_REPEAT):
    l = QA_PAIRS[:]; random.shuffle(l); qa_expanded.extend(l)
all_lines = qa_expanded
random.shuffle(all_lines)
data_bytes = ("\n".join(all_lines) + "\n").encode("ascii", errors="replace")
print(f"Corpus: {len(data_bytes):,} bytes  QA={len(QA_PAIRS)}  BG={len(CORPUS_LINES)}")

# ── Fake-quant helpers (straight-through estimator) ───────────────────────
def ste(x, q):                      # forward=q, backward=identity
    return x + (q - x).detach()

def fq_act(x):                      # int16 Q3.12
    q = (x * 4096).round().clamp(-32768, 32767) / 4096
    return ste(x, q)

def fq_kv(x):                       # int8 Q4.4
    q = (x * 16).round().clamp(-128, 127) / 16
    return ste(x, q)

def fq_emb(w):                      # int8 Q2.6
    q = (w * 64).round().clamp(-128, 127) / 64
    return ste(w, q)

class TernaryLinear(nn.Module):
    """y = x @ (ternary(W) * s).T   — BitNet b1.58 style, per-tensor scale."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, std=0.5 / math.sqrt(in_f))

    def quantized(self):
        w = self.weight
        s = w.abs().mean().clamp(min=1e-5)
        wq = (w / s).round().clamp(-1, 1)
        return wq, s

    def forward(self, x):
        wq, s = self.quantized()
        w_eff = ste(self.weight, wq * s)
        return fq_act(F.linear(x, w_eff))

class RMSNorm(nn.Module):
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-8).sqrt()
        return fq_act(x / rms)

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = TernaryLinear(N_EMBED, N_EMBED)
        self.wk = TernaryLinear(N_EMBED, N_EMBED)
        self.wv = TernaryLinear(N_EMBED, N_EMBED)
        self.wo = TernaryLinear(N_EMBED, N_EMBED)
        self.register_buffer("mask",
            torch.tril(torch.ones(CTX, CTX)).view(1, 1, CTX, CTX))

    def forward(self, x):
        B, T, C = x.shape
        def proj(l, z): return l(z).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        q = proj(self.wq, x)
        k = fq_kv(proj(self.wk, x))     # KV cache is int8 Q4.4 on Genesis
        v = fq_kv(proj(self.wv, x))
        a = (q @ k.transpose(-2, -1)) * (HEAD_DIM ** -0.5)
        a = a.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        a = F.softmax(a, dim=-1)
        o = fq_act((a @ v).transpose(1, 2).contiguous().view(B, T, C))
        return self.wo(o)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(), RMSNorm()
        self.attn = Attn()
        self.wff1 = TernaryLinear(N_EMBED, FFN_DIM)
        self.wff2 = TernaryLinear(FFN_DIM, N_EMBED)

    def forward(self, x):
        x = fq_act(x + self.attn(self.ln1(x)))
        return fq_act(x + self.wff2(F.relu(self.wff1(self.ln2(x)))))

class ElyaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, N_EMBED)
        nn.init.normal_(self.emb.weight, std=0.3)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.ln_f = RMSNorm()

    def forward(self, idx):
        ew = fq_emb(self.emb.weight)
        x = fq_act(ew[idx])
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x) @ ew.T          # tied embedding

model = ElyaGPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}  device={device}")

# ── Train ──────────────────────────────────────────────────────────────────
data_arr = torch.tensor(list(data_bytes), dtype=torch.long)

def batch(bs=512):
    ix = torch.randint(len(data_arr) - CTX - 1, (bs,))
    x = torch.stack([data_arr[i:i+CTX] for i in ix])
    y = torch.stack([data_arr[i+1:i+CTX+1] for i in ix])
    return x.to(device), y.to(device)

# no weight decay on ternary weights: decay shrinks |W| under the
# quantization threshold and destabilizes the per-tensor scale
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0,
                        betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=5e-5)

# fixed held-out eval set: checkpoint on THIS, never on single-batch
# training loss (which rewards one lucky batch, not a good model)
torch.manual_seed(1234)
ex, ey = batch(1024)

@torch.no_grad()
def eval_loss():
    model.eval()
    l = F.cross_entropy(model(ex).view(-1, VOCAB), ey.view(-1)).item()
    model.train()
    return l

t0, best_loss, best_state = time.time(), 1e9, None
for step in range(N_STEPS):
    x, y = batch()
    loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
    emb_pin = F.relu(model.emb.weight.abs() - 1.8).pow(2).mean()
    loss = loss + PIN_W * emb_pin
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 250 == 0:
        ev = eval_loss()
        if ev < best_loss:
            best_loss = ev
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if step % 2500 == 0:
            print(f"  {step:6d}/{N_STEPS}  train={loss.item():.4f}  "
                  f"eval={ev:.4f}  best={best_loss:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

print(f"Done. best={best_loss:.4f}  {time.time()-t0:.0f}s")
model.load_state_dict(best_state); model.eval()
torch.save(best_state, OUT_PT)

# ── Greedy generation (printable ASCII), used for test vectors ────────────
@torch.no_grad()
def gen_greedy(prompt, n=60):
    toks = list(prompt.encode("ascii", "replace"))[-CTX:]
    x = torch.tensor([toks], dtype=torch.long, device=device)
    out = []
    for _ in range(n):
        lg = model(x[:, -CTX:])[0, -1]
        nl = lg[10].item()              # newline competes as stop token,
        lg[:32] = float("-inf"); lg[127:] = float("-inf")
        lg[10] = nl                     # matching the C engine exactly
        t = int(lg.argmax())
        out.append(t)
        x = torch.cat([x, torch.tensor([[t]], device=device)], dim=1)
        if t == ord("\n"):
            break
    return bytes(out).decode("ascii", "replace")

PROMPTS = [
    "Who are you?: ",
    "What is your name?: ",
    "Where are you from?: ",
    "What is your purpose?: ",
    "What is RustChain?: ",
    "Tell me of the realm.: ",
]
vectors = []
print("\n── Greedy generations (QAT model) ──")
for p in PROMPTS:
    g = gen_greedy(p)
    vectors.append({"prompt": p, "expect": g})
    print(f"  [{p}] -> {g!r}")
OUT_VEC.write_text(json.dumps(vectors, indent=2))

# ── SGT1 export (big-endian, 68000-native) ────────────────────────────────
# Header: 'SGT1', u8 layers, u8 heads, u16 embed, u16 vocab, u16 ctx,
#         u8 flags (bit0 = ternary), u8 pad
# Then: emb int8[vocab*embed] (Q2.6)
# Then per layer, tensors wq,wk,wv,wo,wff1,wff2:
#         u16 M(BE) u8 S u8 pad, packed 2-bit weights [out][in] MSB-first
#         encoding: 00=0, 01=+1, 10=-1, 11=reserved (tetranary ±2, P-panel)
# Then: exp LUT u16[256] (BE), lut[i] = round(16384 * exp(-i/16))

def pack_ternary(wq_np):
    flat = wq_np.astype(np.int8).flatten()
    pad = (-len(flat)) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.int8)])
    codes = np.where(flat == 0, 0, np.where(flat > 0, 1, 2)).astype(np.uint8)
    b = (codes[0::4] << 6) | (codes[1::4] << 4) | (codes[2::4] << 2) | codes[3::4]
    return b.astype(np.uint8).tobytes()

def pack_index_streams(wq_np):
    """SGT2 'T2' layout: per output row, an explicit list of activation
    indices to ADD and to SUBTRACT. The 68000 inner loop becomes
    `MOVE.B (A1)+,D0 ; ADD.W (A2,D0.W),D2` — no bit unpacking, no branch
    per weight, and zero weights cost literally nothing (they are simply
    absent from both lists). Trades ROM (cheap on a cartridge) for cycles
    (precious at 7.67 MHz). See docs/SPEED_PLAN.md.

    Row layout: u16 n_add (BE), u16 n_sub (BE), n_add bytes, n_sub bytes.
    in_dim <= 256 so every index fits a byte."""
    out = bytearray()
    for row in wq_np:
        adds = np.nonzero(row > 0)[0].astype(np.uint8)
        subs = np.nonzero(row < 0)[0].astype(np.uint8)
        out += struct.pack(">HH", len(adds), len(subs))
        out += adds.tobytes()
        out += subs.tobytes()
    return bytes(out)

def requant(s):
    """out_q12 = (acc_q12 * M) >> S  ≈ acc_q12 * s, with M ≤ 127 (int32-safe)."""
    S = 0
    while s * (1 << (S + 1)) <= 127 and S < 24:
        S += 1
    M = max(1, min(127, round(s * (1 << S))))
    return M, S

buf = bytearray()
buf += struct.pack(">4sBBHHHBB", b"SGT2", N_LAYERS, N_HEADS,
                   N_EMBED, VOCAB, CTX, 3, 0)   # flags: bit0 ternary, bit1 index-streams

ew = fq_emb(model.emb.weight).detach().cpu().numpy()
e8 = np.clip(np.round(ew * 64), -128, 127).astype(np.int8)
buf += e8.tobytes()
print(f"\nEmbedding: {e8.size} int8, |max|={np.abs(ew).max():.3f}")

for li, blk in enumerate(model.blocks):
    for name, lin in [("wq", blk.attn.wq), ("wk", blk.attn.wk),
                      ("wv", blk.attn.wv), ("wo", blk.attn.wo),
                      ("wff1", blk.wff1), ("wff2", blk.wff2)]:
        wq, s = lin.quantized()
        wq_np = wq.detach().cpu().numpy()
        M, S = requant(float(s))
        buf += struct.pack(">HBB", M, S, 0)
        buf += pack_index_streams(wq_np)
        nz = int((wq_np != 0).sum())
        print(f"  L{li}.{name}: s={float(s):.5f} M={M} S={S} "
              f"nonzero={100*nz/wq_np.size:.0f}%")

lut = np.array([round(16384 * math.exp(-i / 16.0)) for i in range(256)],
               dtype=">u2")
buf += lut.tobytes()

OUT_BIN.parent.mkdir(exist_ok=True)
OUT_BIN.write_bytes(buf)
print(f"\nSGT2 blob: {len(buf):,} bytes -> {OUT_BIN}")
print("=== DONE ===")
