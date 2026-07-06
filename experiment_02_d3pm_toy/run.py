"""
实验 2：D3PM 离散状态 Toy
========================
比较 uniform/absorbing/structured 三类 corruption 的恢复难度。
"""

import json
import os
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


def Q_uniform(vocab_size):
    return torch.ones(vocab_size, vocab_size) / vocab_size


def Q_absorbing(vocab_size, mask_id=MASK_ID):
    Q = torch.zeros(vocab_size, vocab_size)
    Q[:, mask_id] = 1.0
    Q[mask_id, mask_id] = 1.0
    return Q


def Q_structured(vocab_size):
    Q = torch.eye(vocab_size) * 0.7
    Q[0, 1] = Q[1, 0] = 0.15
    Q[2, 3] = Q[3, 2] = 0.15
    for i in range(vocab_size - 1):
        Q[i, MASK_ID] = 0.1
        Q[i, i] -= 0.1
    Q[MASK_ID, MASK_ID] = 1.0
    Q = Q / Q.sum(dim=1, keepdim=True)
    return Q


def Q_bar(Q, t):
    return torch.matrix_power(Q, t)


def generate_data(n=10000, L=16):
    templates = [
        ["A", "B", "C", "D"] * (L // 4),
        ["A", "A", "B", "B", "C", "C", "D", "D"] * (L // 8),
        ["A", "B", "A", "B", "C", "D", "C", "D"] * (L // 8),
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


def forward_diffuse(x0, Q_bt):
    B, L = x0.shape
    xt = torch.zeros_like(x0)
    for b in range(B):
        for s in range(L):
            xt[b, s] = torch.multinomial(Q_bt[x0[b, s]], 1).item()
    return xt


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


def train(corruption_type, steps=10000, bs=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if corruption_type == "uniform":
        Q = Q_uniform(V)
    elif corruption_type == "absorbing":
        Q = Q_absorbing(V)
    elif corruption_type == "structured":
        Q = Q_structured(V)
    else:
        raise ValueError(corruption_type)

    model = Denoiser(V).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = generate_data(n=10000).to(device)
    T_diff = 100
    losses = []

    model.train()
    for step in tqdm(range(steps), desc=f"Training {corruption_type}"):
        idx = torch.randint(0, len(data), (bs,))
        x0 = data[idx]
        t = torch.randint(1, T_diff + 1, (1,)).item()
        Qbt = Q_bar(Q, t).to(device)
        xt = forward_diffuse(x0, Qbt)
        logits = model(xt)
        loss = F.cross_entropy(logits.view(-1, V), x0.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return model, Q, losses


def sample_reverse(model, Q, n=100, L=16, T_diff=100):
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
    return x.cpu()


def evaluate(model, Q, corruption_type, n_eval=100):
    device = next(model.parameters()).device
    test = generate_data(n=n_eval).to(device)
    L = test.shape[1]
    Q_full = Q_bar(Q, 100).to(device)
    xT = forward_diffuse(test, Q_full)

    recovered = sample_reverse(model, Q, n=n_eval, L=L)
    recovered = recovered.to(device)
    acc = (recovered == test).float().mean().item()

    model.eval()
    with torch.no_grad():
        logits = model(xT)
        top3 = logits.topk(3, dim=-1).indices
        top3_acc = (top3 == test.unsqueeze(-1)).any(dim=-1).float().mean().item()

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

    model.eval()
    with torch.no_grad():
        logits = model(xT)
        val_ce = F.cross_entropy(logits.view(-1, V), test.view(-1)).item()

    return {
        "val_ce": val_ce,
        "token_acc": acc,
        "top3_acc": top3_acc,
        "exact_rate": exact / n_eval,
        "near_rate": near / n_eval,
    }


def plot_Q_heatmap(Q, name):
    plt.figure(figsize=(5, 4))
    plt.imshow(Q.numpy(), cmap="YlOrRd", vmin=0, vmax=1)
    plt.xticks(range(V), VOCAB)
    plt.yticks(range(V), VOCAB)
    plt.colorbar()
    plt.title(f"Q matrix: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"Q_heatmap_{name}.png"), dpi=150)
    plt.close()


def plot_loss(losses, name):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Training Loss: {name}")
    plt.savefig(os.path.join(RESULTS_DIR, f"loss_{name}.png"), dpi=150)
    plt.close()


def main():
    all_results = {}

    for ctype in ["uniform", "absorbing", "structured"]:
        print(f"\n{'='*60}")
        print(f"Experiment 2: D3PM with {ctype} corruption")
        print(f"{'='*60}")

        if ctype == "uniform":
            Q = Q_uniform(V)
        elif ctype == "absorbing":
            Q = Q_absorbing(V)
        else:
            Q = Q_structured(V)
        plot_Q_heatmap(Q, ctype)

        model, Q_trained, losses = train(ctype)
        plot_loss(losses, ctype)

        metrics = evaluate(model, Q_trained, ctype)
        all_results[ctype] = metrics

        print(f"\n  Val CE:     {metrics['val_ce']:.4f}")
        print(f"  Token Acc:  {metrics['token_acc']:.4%}")
        print(f"  Top-3 Acc:  {metrics['top3_acc']:.4%}")
        print(f"  Exact Rate: {metrics['exact_rate']:.4%}")
        print(f"  Near Rate:  {metrics['near_rate']:.4%}")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment 2 complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
