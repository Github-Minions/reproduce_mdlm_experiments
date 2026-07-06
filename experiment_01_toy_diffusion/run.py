"""
实验 1：连续二维 Toy Diffusion
=============================
验证 DDPM 在三个 2D 数据集上的前向加噪与反向采样。
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_s_curve
from scipy.spatial.distance import cdist
from tqdm import tqdm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_two_moons(n=2048, noise=0.05):
    X, _ = make_moons(n_samples=n, noise=noise)
    return torch.tensor(X, dtype=torch.float32)


def make_swiss_roll(n=2048, noise=0.5):
    X, _ = make_s_curve(n_samples=n, noise=noise)
    X = X[:, [0, 2]]
    return torch.tensor(X, dtype=torch.float32)


def make_gaussian_mixture(n=2048, k=8):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 3.0
    per = n // k
    X = []
    for c in centers:
        X.append(np.random.randn(per, 2) * 0.5 + c)
    X = np.concatenate(X, axis=0)
    np.random.shuffle(X)
    return torch.tensor(X[:n], dtype=torch.float32)


DATASETS = {
    "two_moons": make_two_moons,
    "swiss_roll": make_swiss_roll,
    "gaussian_mixture": make_gaussian_mixture,
}


class ContinuousDiffusion:
    def __init__(self, T=1000, beta1=1e-4, betaT=0.02):
        self.T = T
        self.beta = torch.linspace(beta1, betaT, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x0, t):
        ab = self.alpha_bar[t].view(-1, 1)
        eps = torch.randn_like(x0)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps, eps

    def prior(self, n):
        return torch.randn(n, 2)


class ScoreNet(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, 2),
        )

    def forward(self, x, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        texp = t.float().view(-1, 1).expand(x.size(0), 1)
        return self.net(torch.cat([x, texp], dim=-1))


def train(dataset_name, steps=20000, bs=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_fn = DATASETS[dataset_name]
    diff = ContinuousDiffusion(T=1000)
    model = ScoreNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in tqdm(range(steps), desc=f"Training {dataset_name}"):
        x0 = data_fn(n=bs).to(device)
        t = torch.randint(0, diff.T, (bs,))
        xt, eps = diff.forward(x0, t)
        xt, eps = xt.to(device), eps.to(device)
        tnorm = t.to(device).float() / diff.T

        pred_eps = model(xt, tnorm)
        loss = F.mse_loss(pred_eps, eps)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return model, losses


def sample(model, diff, n=2048):
    device = next(model.parameters()).device
    x = diff.prior(n).to(device)
    model.eval()
    with torch.no_grad():
        for t in range(diff.T - 1, -1, -1):
            t_batch = torch.full((n,), t / diff.T, device=device)
            pred_eps = model(x, t_batch)
            a, ab, b = diff.alpha[t], diff.alpha_bar[t], diff.beta[t]
            x = (x - b / torch.sqrt(1 - ab) * pred_eps) / torch.sqrt(a)
            if t > 0:
                x = x + torch.sqrt(b) * torch.randn_like(x)
    return x.cpu()


def mmd_rbf(X, Y, gamma=1.0):
    X, Y = X.numpy(), Y.numpy()
    XX = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    YY = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    XY = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def sliced_wasserstein(X, Y, n_proj=100):
    X, Y = X.numpy(), Y.numpy()
    d = X.shape[1]
    dists = []
    for _ in range(n_proj):
        theta = np.random.randn(d)
        theta /= np.linalg.norm(theta)
        px, py = X @ theta, Y @ theta
        px.sort(); py.sort()
        dists.append(np.mean(np.abs(px - py)))
    return float(np.mean(dists))


def histogram_kl(X, Y, bins=50):
    X, Y = X.numpy(), Y.numpy()
    x0_min = min(X[:, 0].min(), Y[:, 0].min())
    x0_max = max(X[:, 0].max(), Y[:, 0].max())
    x1_min = min(X[:, 1].min(), Y[:, 1].min())
    x1_max = max(X[:, 1].max(), Y[:, 1].max())
    HX, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=bins,
                               range=[[x0_min, x0_max], [x1_min, x1_max]])
    HY, _, _ = np.histogram2d(Y[:, 0], Y[:, 1], bins=bins,
                               range=[[x0_min, x0_max], [x1_min, x1_max]])
    HX = HX / HX.sum() + 1e-10
    HY = HY / HY.sum() + 1e-10
    return float(np.sum(HX * np.log(HX / HY)))


def mean_cov_error(X, Y):
    mean_err = torch.norm(X.mean(0) - Y.mean(0)).item()
    cov_err = torch.norm(torch.cov(X.T) - torch.cov(Y.T), p="fro").item()
    return mean_err, cov_err


def plot_forward_noise(data_fn, diff, name):
    x0 = data_fn(n=1000)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    ts = [0, diff.T // 3, 2 * diff.T // 3, diff.T - 1]
    for ax, t in zip(axes, ts):
        xt, _ = diff.forward(x0, torch.tensor([t]))
        ax.scatter(xt[:, 0], xt[:, 1], s=2, alpha=0.5)
        ax.set_title(f"t={t}")
        ax.set_aspect("equal")
    plt.suptitle(f"Forward Noising: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"forward_{name}.png"), dpi=150)
    plt.close()


def plot_loss(losses, name):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss: {name}")
    plt.savefig(os.path.join(RESULTS_DIR, f"loss_{name}.png"), dpi=150)
    plt.close()


def plot_generated_vs_real(real, gen, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(real[:, 0], real[:, 1], s=2, alpha=0.5)
    axes[0].set_title("Real")
    axes[0].set_aspect("equal")
    axes[1].scatter(gen[:, 0], gen[:, 1], s=2, alpha=0.5)
    axes[1].set_title("Generated")
    axes[1].set_aspect("equal")
    plt.suptitle(f"{name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"gen_{name}.png"), dpi=150)
    plt.close()


def sampling_ablation(model, data_fn, diff, steps_list=[10, 50, 100, 500]):
    real = data_fn(n=2048)
    device = next(model.parameters()).device
    results = {}

    for n_steps in steps_list:
        indices = torch.linspace(diff.T - 1, 0, n_steps).long()
        x = diff.prior(2048).to(device)
        model.eval()
        with torch.no_grad():
            for i, t in enumerate(indices):
                t_batch = torch.full((2048,), t.item() / diff.T, device=device)
                pred_eps = model(x, t_batch)
                a, ab, b = diff.alpha[t], diff.alpha_bar[t], diff.beta[t]
                x = (x - b / torch.sqrt(1 - ab) * pred_eps) / torch.sqrt(a)
                if i < len(indices) - 1:
                    t_next = indices[i + 1]
                    b_eff = 1 - diff.alpha_bar[t_next] / diff.alpha_bar[t]
                    x = x + torch.sqrt(torch.abs(b_eff)) * torch.randn_like(x)
        gen = x.cpu()
        results[n_steps] = {
            "mmd": mmd_rbf(real, gen),
            "sw": sliced_wasserstein(real, gen),
        }
    return results


def main():
    all_results = {}

    for name in ["two_moons", "swiss_roll", "gaussian_mixture"]:
        print(f"\n{'='*60}")
        print(f"Experiment 1: {name}")
        print(f"{'='*60}")

        model, losses = train(name, steps=20000)
        diff = ContinuousDiffusion(T=1000)
        plot_forward_noise(DATASETS[name], diff, name)
        plot_loss(losses, name)

        gen = sample(model, diff, n=2048)
        real = DATASETS[name](n=2048)
        plot_generated_vs_real(real, gen, name)

        metrics = {
            "mmd_rbf": mmd_rbf(real, gen),
            "sliced_wasserstein": sliced_wasserstein(real, gen),
            "histogram_kl": histogram_kl(real, gen),
            "mean_error": mean_cov_error(real, gen)[0],
            "cov_error": mean_cov_error(real, gen)[1],
        }
        ablation = sampling_ablation(model, DATASETS[name], diff,
                                      steps_list=[10, 50, 100, 500])
        metrics["sampling_ablation"] = ablation
        all_results[name] = metrics

        print(f"\nResults for {name}:")
        for k, v in metrics.items():
            if k != "sampling_ablation":
                print(f"  {k:20s}: {v:.6f}")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Experiment 1 complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
