"""
实验 10：D3PM 与 CTMC 对比
===========================
控制变量：词表、模板、Transformer 规模相同
变量：时间建模方式（离散步数 vs 连续时间）
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

VOCAB = ["A", "B", "C", "D", "[MASK]"]
T2I = {t: i for i, t in enumerate(VOCAB)}
I2T = {i: t for t, i in T2I.items()}
MASK_ID = T2I["[MASK]"]
V = len(VOCAB)


def generate_data(n, L):
    templates = [
        ["A", "B", "C", "D"] * (L // 4 + 1),
        ["A", "A", "B", "B", "C", "C", "D", "D"] * (L // 8 + 1),
        ["A", "B", "A", "B", "C", "D", "C", "D"] * (L // 8 + 1),
    ]
    templates = [t[:L] for t in templates]
    data = []
    for _ in range(n):
        t = templates[np.random.randint(len(templates))]
        seq = list(t)
        for i in range(L):
            if np.random.rand() < 0.05:
                seq[i] = VOCAB[np.random.randint(V - 1)]
        data.append([T2I[c] for c in seq])
    return torch.tensor(data, dtype=torch.long)


class Denoiser(nn.Module):
    def __init__(self, vocab_size, d=64, heads=4, layers=3, L=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.randn(1, L, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, vocab_size)

    def forward(self, x):
        h = self.emb(x) + self.pos[:, :x.size(1), :]
        return self.out(self.tf(h))


def template_match_rate(recovered, L, device):
    templates = [
        torch.tensor([T2I[c] for c in (["A", "B", "C", "D"] * (L // 4 + 1))[:L]], device=device),
        torch.tensor([T2I[c] for c in (["A", "A", "B", "B", "C", "C", "D", "D"] * (L // 8 + 1))[:L]], device=device),
        torch.tensor([T2I[c] for c in (["A", "B", "A", "B", "C", "D", "C", "D"] * (L // 8 + 1))[:L]], device=device),
    ]
    n = recovered.size(0)
    exact = 0
    near = 0
    for b in range(n):
        for tmpl in templates:
            if (recovered[b] == tmpl).all():
                exact += 1
                break
        for tmpl in templates:
            if (recovered[b] != tmpl).sum().item() <= 2:
                near += 1
                break
    return exact / n, near / n


def Q_absorbing(vocab_size):
    Q = torch.zeros(vocab_size, vocab_size)
    Q[:, MASK_ID] = 1.0
    Q[MASK_ID, MASK_ID] = 1.0
    return Q


def Q_bar(Q, t):
    return torch.matrix_power(Q, t)


def forward_d3pm(x0, Q_bt):
    B, L = x0.shape
    xt = torch.zeros_like(x0)
    for b in range(B):
        for s in range(L):
            xt[b, s] = torch.multinomial(Q_bt[x0[b, s]], 1).item()
    return xt


def train_d3pm(L, steps=10000, bs=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = Q_absorbing(V)
    model = Denoiser(V, L=L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = generate_data(10000, L).to(device)
    T_diff = 80
    losses = []

    for step in tqdm(range(steps), desc=f"D3PM L={L}"):
        idx = torch.randint(0, len(data), (bs,))
        x0 = data[idx]
        t = torch.randint(1, T_diff + 1, (1,)).item()
        Qbt = Q_bar(Q, t).to(device)
        xt = forward_d3pm(x0, Qbt)
        logits = model(xt)
        loss = F.cross_entropy(logits.view(-1, V), x0.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return model, Q, losses


def sample_d3pm(model, Q, n, L, T_diff=80):
    device = next(model.parameters()).device
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for t in range(T_diff, 0, -1):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            for b in range(n):
                for s in range(L):
                    if x[b, s] == MASK_ID or np.random.rand() < 1.0 / t:
                        x[b, s] = torch.multinomial(probs[b, s], 1).item()
    return x


class CTMCDiffusion:
    def __init__(self, vocab_size, lam=3.0, horizon=1.0):
        self.vocab_size = vocab_size
        self.lam = lam
        self.horizon = horizon
        self.mask_id = vocab_size - 1

    def forward(self, x0, t):
        r = 1.0 - np.exp(-self.lam * t)
        mask = torch.rand_like(x0.float()) < r
        return torch.where(mask, torch.tensor(self.mask_id), x0)


def train_ctmc(L, steps=10000, bs=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctmc = CTMCDiffusion(V, lam=3.0)
    model = Denoiser(V, L=L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = generate_data(10000, L).to(device)
    losses = []

    for step in tqdm(range(steps), desc=f"CTMC L={L}"):
        idx = torch.randint(0, len(data), (bs,))
        x0 = data[idx]
        t = np.random.uniform(0, ctmc.horizon)
        xt = ctmc.forward(x0, t).to(device)
        logits = model(xt)
        loss = F.cross_entropy(logits.view(-1, V), x0.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return model, ctmc, losses


def sample_ctmc(model, ctmc, n, L, tau=0.00625):
    device = next(model.parameters()).device
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=device)
    steps = int(ctmc.horizon / tau)
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            for b in range(n):
                for s in range(L):
                    if x[b, s] == MASK_ID:
                        if torch.rand(1).item() < ctmc.lam * tau:
                            x[b, s] = torch.multinomial(probs[b, s], 1).item()
    return x


def evaluate_method(model, test, method_name, Q=None, ctmc=None, L=16):
    device = test.device
    n = test.size(0)

    if method_name == "D3PM":
        Q_full = Q_bar(Q, 80).to(device)
        xT = forward_d3pm(test, Q_full)
        t0 = time.time()
        recovered = sample_d3pm(model, Q, n, L, T_diff=80)
        runtime = time.time() - t0
        nfe = 80
    else:
        xT = ctmc.forward(test, t=1.0).to(device)
        t0 = time.time()
        recovered = sample_ctmc(model, ctmc, n, L, tau=0.00625)
        runtime = time.time() - t0
        nfe = int(ctmc.horizon / 0.00625)

    recovered = recovered.to(device)
    acc = (recovered == test).float().mean().item()
    mask_pos = (xT == MASK_ID)
    masked_acc = (recovered[mask_pos] == test[mask_pos]).float().mean().item() if mask_pos.any() else 1.0

    model.eval()
    with torch.no_grad():
        logits = model(xT)
        top3 = logits.topk(3, dim=-1).indices
        top3_acc = (top3 == test.unsqueeze(-1)).any(dim=-1).float().mean().item()
        val_ce = F.cross_entropy(logits.view(-1, V), test.view(-1)).item()

    exact_rate, near_rate = template_match_rate(recovered, L, device)

    return {
        "method": method_name,
        "seq_len": L,
        "val_ce": val_ce,
        "token_acc": acc,
        "masked_acc": masked_acc,
        "top3_acc": top3_acc,
        "exact_rate": exact_rate,
        "near_rate": near_rate,
        "runtime": runtime,
        "nfe": nfe,
    }


def plot_comparison(all_results):
    lens = sorted(set(r["seq_len"] for r in all_results))
    methods = ["D3PM", "CTMC"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    metrics = [
        ("token_acc", "Token Accuracy"),
        ("masked_acc", "Masked Acc"),
        ("top3_acc", "Top-3 Acc"),
        ("near_rate", "Near Rate"),
        ("runtime", "Runtime (s)"),
        ("nfe", "NFE"),
    ]

    for ax, (key, title) in zip(axes.flat, metrics):
        for method in methods:
            xs = [r["seq_len"] for r in all_results if r["method"] == method]
            ys = [r[key] for r in all_results if r["method"] == method]
            ax.plot(xs, ys, "o-", label=method)
        ax.set_xlabel("Seq Length")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.set_xticks(lens)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=150)
    plt.close()


def main():
    print(f"{'='*60}")
    print("Experiment 10: D3PM vs CTMC")
    print(f"{'='*60}")

    all_results = []

    for L in [8, 16, 32]:
        print(f"\n--- Sequence Length = {L} ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test = generate_data(100, L).to(device)

        d3pm_model, Q, _ = train_d3pm(L, steps=8000)
        d3pm_res = evaluate_method(d3pm_model, test, "D3PM", Q=Q, L=L)
        all_results.append(d3pm_res)
        print(f"  D3PM: Acc={d3pm_res['token_acc']:.4%}, "
              f"Near={d3pm_res['near_rate']:.4%}, "
              f"Time={d3pm_res['runtime']:.3f}s, NFE={d3pm_res['nfe']}")

        ctmc_model, ctmc, _ = train_ctmc(L, steps=8000)
        ctmc_res = evaluate_method(ctmc_model, test, "CTMC", ctmc=ctmc, L=L)
        all_results.append(ctmc_res)
        print(f"  CTMC: Acc={ctmc_res['token_acc']:.4%}, "
              f"Near={ctmc_res['near_rate']:.4%}, "
              f"Time={ctmc_res['runtime']:.3f}s, NFE={ctmc_res['nfe']}")

    plot_comparison(all_results)

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment 10 complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
