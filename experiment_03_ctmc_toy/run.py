"""
实验 3：CTMC Discrete Diffusion 与 Tau-Leaping
===============================================
验证闭式 mask ratio 与 tau-leaping 的误差-效率权衡。
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


def generate_data(n=10000, L=16):
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


class CTMCDiffusion:
    def __init__(self, vocab_size, lam=3.0, horizon=1.0):
        self.vocab_size = vocab_size
        self.lam = lam
        self.horizon = horizon
        self.mask_id = vocab_size - 1

    def mask_ratio(self, t):
        return 1.0 - np.exp(-self.lam * t)

    def forward(self, x0, t):
        r = self.mask_ratio(t)
        mask = torch.rand_like(x0.float()) < r
        return torch.where(mask, torch.tensor(self.mask_id), x0)


def train(steps=10000, bs=256, lam=3.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctmc = CTMCDiffusion(V, lam=lam)
    model = Denoiser(V).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = generate_data(n=10000).to(device)
    losses = []

    model.train()
    for step in tqdm(range(steps), desc="Training CTMC"):
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


def tau_leaping(model, ctmc, n=100, L=16, tau=0.01):
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
    return x.cpu()


def evaluate(model, ctmc, n_eval=100):
    device = next(model.parameters()).device
    test = generate_data(n=n_eval).to(device)
    L = test.shape[1]

    xT = ctmc.forward(test, t=1.0).to(device)
    recovered = tau_leaping(model, ctmc, n=n_eval, L=L, tau=0.01).to(device)

    acc = (recovered == test).float().mean().item()
    mask_pos = (xT == MASK_ID)
    masked_acc = (recovered[mask_pos] == test[mask_pos]).float().mean().item() if mask_pos.any() else 1.0

    model.eval()
    with torch.no_grad():
        logits = model(xT)
        top3 = logits.topk(3, dim=-1).indices
        top3_acc = (top3 == test.unsqueeze(-1)).any(dim=-1).float().mean().item()
        val_ce = F.cross_entropy(logits.view(-1, V), test.view(-1)).item()

    templates = [
        torch.tensor([T2I[c] for c in (["A", "B", "C", "D"] * 4)[:L]]),
        torch.tensor([T2I[c] for c in (["A", "A", "B", "B", "C", "C", "D", "D"] * 2)[:L]]),
        torch.tensor([T2I[c] for c in (["A", "B", "A", "B", "C", "D", "C", "D"] * 2)[:L]]),
    ]
    exact = 0
    near = 0
    for b in range(n_eval):
        for tmpl in templates:
            if (recovered[b] == tmpl.to(device)).all():
                exact += 1
                break
        for tmpl in templates:
            if (recovered[b] != tmpl.to(device)).sum().item() <= 2:
                near += 1
                break

    return {
        "val_ce": val_ce,
        "token_acc": acc,
        "masked_acc": masked_acc,
        "top3_acc": top3_acc,
        "exact_rate": exact / n_eval,
        "near_rate": near / n_eval,
    }


def tau_ablation(model, ctmc, test_data):
    L = test_data.shape[1]
    results = []
    for tau in [0.1, 0.05, 0.02, 0.01, 0.005]:
        t0 = time.time()
        recovered = tau_leaping(model, ctmc, n=100, L=L, tau=tau).to(test_data.device)
        runtime = time.time() - t0
        nfe = int(ctmc.horizon / tau)

        acc = (recovered == test_data).float().mean().item()

        from collections import Counter
        real_counts = Counter(test_data.cpu().flatten().tolist())
        gen_counts = Counter(recovered.cpu().flatten().tolist())
        tv = 0
        for k in set(list(real_counts.keys()) + list(gen_counts.keys())):
            p = real_counts.get(k, 0) / test_data.numel()
            q = gen_counts.get(k, 0) / recovered.numel()
            tv += abs(p - q)
        tv /= 2

        results.append({
            "tau": tau,
            "tv_distance": tv,
            "accuracy": acc,
            "runtime": runtime,
            "nfe": nfe,
        })
    return results


def plot_mask_ratio_check(ctmc):
    ts = np.linspace(0, 1, 100)
    closed = [ctmc.mask_ratio(t) for t in ts]

    x0 = generate_data(n=5000, L=16)
    empirical = []
    for t in ts:
        xt = ctmc.forward(x0, t)
        ratio = (xt == MASK_ID).float().mean().item()
        empirical.append(ratio)

    plt.figure(figsize=(6, 4))
    plt.plot(ts, closed, "b-", label=r"Closed: $1 - e^{-\lambda t}$")
    plt.plot(ts, empirical, "r--", label="Empirical")
    plt.xlabel("t")
    plt.ylabel("Mask Ratio")
    plt.legend()
    plt.title("CTMC Mask Ratio Validation")
    plt.savefig(os.path.join(RESULTS_DIR, "mask_ratio_check.png"), dpi=150)
    plt.close()


def plot_loss(losses):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("CE Loss")
    plt.title("CTMC Training Loss")
    plt.savefig(os.path.join(RESULTS_DIR, "loss.png"), dpi=150)
    plt.close()


def plot_tau_ablation(results):
    taus = [r["tau"] for r in results]
    tvs = [r["tv_distance"] for r in results]
    runtimes = [r["runtime"] for r in results]
    nfes = [r["nfe"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(taus, tvs, "o-")
    axes[0].set_xlabel("tau")
    axes[0].set_ylabel("TV Distance")
    axes[0].set_title("Error vs tau")
    axes[0].invert_xaxis()

    axes[1].plot(nfes, tvs, "o-")
    axes[1].set_xlabel("NFE")
    axes[1].set_ylabel("TV Distance")
    axes[1].set_title("Error vs NFE")

    axes[2].plot(nfes, runtimes, "o-")
    axes[2].set_xlabel("NFE")
    axes[2].set_ylabel("Runtime (s)")
    axes[2].set_title("Runtime vs NFE")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tau_ablation.png"), dpi=150)
    plt.close()


def plot_recovery_by_time(model, ctmc, test_data):
    device = test_data.device
    rs = np.linspace(0.1, 0.9, 9)
    accs = []
    model.eval()
    for r in rs:
        t = -np.log(1 - r) / ctmc.lam
        xT = ctmc.forward(test_data, t)
        with torch.no_grad():
            logits = model(xT)
            pred = logits.argmax(dim=-1)
        mask = (xT != test_data)
        if mask.any():
            acc = (pred[mask] == test_data[mask]).float().mean().item()
        else:
            acc = 1.0
        accs.append(acc)

    plt.figure(figsize=(6, 4))
    plt.plot(rs, accs, "o-")
    plt.xlabel("Mask Ratio")
    plt.ylabel("Recovery Accuracy")
    plt.title("Recovery Difficulty vs Mask Ratio")
    plt.savefig(os.path.join(RESULTS_DIR, "recovery_by_time.png"), dpi=150)
    plt.close()


def main():
    print(f"{'='*60}")
    print("Experiment 3: CTMC Discrete Diffusion")
    print(f"{'='*60}")

    model, ctmc, losses = train(steps=10000)
    plot_mask_ratio_check(ctmc)
    plot_loss(losses)

    device = next(model.parameters()).device
    test = generate_data(n=100).to(device)
    metrics = evaluate(model, ctmc)

    print(f"\n  Val CE:       {metrics['val_ce']:.4f}")
    print(f"  Token Acc:    {metrics['token_acc']:.4%}")
    print(f"  Masked Acc:   {metrics['masked_acc']:.4%}")
    print(f"  Top-3 Acc:    {metrics['top3_acc']:.4%}")
    print(f"  Near Rate:    {metrics['near_rate']:.4%}")

    ablation = tau_ablation(model, ctmc, test)
    plot_tau_ablation(ablation)
    plot_recovery_by_time(model, ctmc, test)

    print("\n  Tau-Leaping Ablation:")
    for r in ablation:
        print(f"    tau={r['tau']:.3f}: TV={r['tv_distance']:.4f}, "
              f"NFE={r['nfe']}, Time={r['runtime']:.3f}s")

    output = {
        "metrics": metrics,
        "tau_ablation": ablation,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment 3 complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
