"""NPZ-type growth kinetics with three constraint modes and log-space ODE residuals."""
from __future__ import annotations
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DEFAULT_RANGES = {"mu_max": (0.1, 1.5), "K_N": (0.01, 0.5), "K_P": (0.001, 0.05), "r": (0.01, 0.2), "m": (0.005, 0.1), "K_I": (50.0, 1000.0)}


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x)) if x > 1e-6 else -10.0


class PhysicsParams(nn.Module):
    """mode: 'sigmoid' (hard bounds), 'none' (positivity only), 'soft' (positivity + out-of-range penalty).
    param (none/soft only): 'softplus' (v1-v3: softplus of a raw value in natural units) or 'exp' (v4: range centre x exp(raw)).
    With 'exp' every mode starts at the range centre with raw = 0 and moves on a log scale, so the modes differ only in how
    they bound the value; with 'softplus' K_I (~500) could hardly move under Adam steps of size lr."""
    def __init__(self, mode: str, n_sites: int, kn_max: float = 0.5, kp_max: float = 0.05, use_light: bool = True, theta: float = 1.047,
                 param: str = "softplus", null: bool = False):
        super().__init__()
        self.null = null                                         # placebo: growth term zero, net = -(r + m)
        assert mode in ("sigmoid", "none", "soft") and param in ("softplus", "exp")
        self.mode, self.theta, self.use_light, self.param = mode, theta, use_light, param
        self.ranges = dict(DEFAULT_RANGES); self.ranges["K_N"] = (0.01, kn_max); self.ranges["K_P"] = (0.001, kp_max)
        self.mu_site = nn.Embedding(n_sites, 1)
        self.raw = nn.ParameterDict()
        zero_init = mode == "sigmoid" or param == "exp"
        for k in ("K_N", "K_P", "r", "m", "K_I"):
            lo, hi = self.ranges[k]; c = (lo + hi) / 2
            self.raw[k] = nn.Parameter(torch.tensor(0.0 if zero_init else _inv_softplus(c), dtype=torch.float32))
        lo, hi = self.ranges["mu_max"]; c = (lo + hi) / 2
        nn.init.constant_(self.mu_site.weight, 0.0 if zero_init else _inv_softplus(c))

    def _map(self, raw: torch.Tensor, key: str) -> torch.Tensor:
        lo, hi = self.ranges[key]
        if self.mode == "sigmoid":
            return lo + torch.sigmoid(raw) * (hi - lo)
        if self.param == "exp":
            return (lo + hi) / 2 * torch.exp(raw)
        return F.softplus(raw) + 1e-6

    def value(self, key: str, sid: torch.Tensor | None = None) -> torch.Tensor:
        if key == "mu_max":
            return self._map(self.mu_site(sid).squeeze(-1) if sid is not None else self.mu_site.weight.squeeze(-1), key)
        return self._map(self.raw[key], key)

    def penalty(self) -> torch.Tensor:
        """Soft out-of-range penalty (dimensionless, scaled by range^2). Zero in other modes."""
        if self.mode != "soft":
            return torch.zeros((), device=self.raw["K_N"].device)
        pen = torch.zeros((), device=self.raw["K_N"].device)
        for k in ("K_N", "K_P", "r", "m", "K_I"):
            lo, hi = self.ranges[k]; v = self.value(k)
            pen = pen + (F.relu(v - hi) ** 2 + F.relu(lo - v) ** 2) / (hi - lo) ** 2
        lo, hi = self.ranges["mu_max"]; v = self.value("mu_max")
        pen = pen + ((F.relu(v - hi) ** 2 + F.relu(lo - v) ** 2) / (hi - lo) ** 2).mean()
        return pen

    def factors(self, T, N, P, I, sid):
        """All raw units. Returns dict of f_T, f_N, f_P, f_I, mu, net (broadcast over T's shape)."""
        mu_max = self.value("mu_max", sid)
        while mu_max.dim() < T.dim():
            mu_max = mu_max.unsqueeze(-1)
        f_T = self.theta ** (torch.clamp(T, 0.0, 40.0) - 20.0)
        K_N, K_P = self.value("K_N"), self.value("K_P")
        N, P = F.relu(N), F.relu(P)
        f_N, f_P = N / (K_N + N + 1e-8), P / (K_P + P + 1e-8)
        if self.use_light:
            K_I = self.value("K_I"); I = F.relu(I); f_I = I / (K_I + I + 1e-8)
        else:
            f_I = torch.ones_like(T)
        mu = mu_max * f_T * f_N * f_P * f_I
        if self.null:
            mu = torch.zeros_like(mu)
        net = mu - self.value("r") - self.value("m")
        return {"f_T": f_T, "f_N": f_N, "f_P": f_P, "f_I": f_I, "mu": mu, "net": net}

    def snapshot(self) -> dict:
        with torch.no_grad():
            d = {k: float(self.value(k)) for k in ("K_N", "K_P", "r", "m", "K_I")}
            d["mu_max_sites"] = [float(x) for x in self.value("mu_max")]
            return d


def physics_losses(phys: PhysicsParams, pr: torch.Tensor, pv: torch.Tensor, sid: torch.Tensor, pred_log1p_h1: torch.Tensor | None, eps: float = 1e-3):
    """pr, pv: (B, L, 5) raw values / validity for [T, N, P, C, PAR]. Returns dict of scalar losses."""
    T, N, P, C, I = pr[..., 0], pr[..., 1], pr[..., 2], pr[..., 3], pr[..., 4]
    vT, vN, vP, vC, vI = pv[..., 0], pv[..., 1], pv[..., 2], pv[..., 3], pv[..., 4]
    fac = phys.factors(T, N, P, I, sid)
    net = fac["net"]                                           # (B, L) day^-1
    drivers_ok = vT & vN & vP & vI
    # sequence residual on consecutive observed chlorophyll pairs: ln C_{t+1} - ln C_t - net_t * 1 day
    pair_ok = vC[:, :-1] & vC[:, 1:] & drivers_ok[:, :-1]
    res = torch.log(C[:, 1:] + eps) - torch.log(C[:, :-1] + eps) - net[:, :-1]
    seq_loss = F.smooth_l1_loss(res[pair_ok], torch.zeros_like(res[pair_ok])) if pair_ok.any() else torch.zeros((), device=pr.device)
    # prediction residual for horizon 1 at the anchor (last window day)
    if pred_log1p_h1 is not None:
        ok = vC[:, -1] & drivers_ok[:, -1]
        C_pred = torch.expm1(pred_log1p_h1).clamp_min(0.0)
        res_p = torch.log(C_pred + eps) - torch.log(C[:, -1] + eps) - net[:, -1]
        pred_loss = F.smooth_l1_loss(res_p[ok], torch.zeros_like(res_p[ok])) if ok.any() else torch.zeros((), device=pr.device)
        nonneg = F.relu(-pred_log1p_h1).mean()
    else:
        pred_loss = nonneg = torch.zeros((), device=pr.device)
    return {"seq": seq_loss, "pred": pred_loss, "nonneg": nonneg, "n_pairs": int(pair_ok.sum())}


def converge_physics(phys: PhysicsParams, pr, pv, sid, pred_log1p_h1=None, w_phys: float = 1.0, pen_weight: float = 1.0,
                     steps: int = 2500, lr: float = 0.05, lbfgs_iter: int = 300, fixed: tuple = ()) -> dict:
    """Fit the kinetic parameters alone to convergence with the network outputs held fixed (full batch, no weight decay).
    Objective = the physics part of the training loss, w_phys * (seq + pred) + pen_weight * penalty. Modifies `phys` in place.
    `fixed`: parameter keys (e.g. "K_N") held at their current value, for profiles."""
    params = [p for n_, p in phys.named_parameters() if n_.replace("raw.", "") not in fixed]
    for p in params:
        p.requires_grad_(True)

    def objective():
        L = physics_losses(phys, pr, pv, sid, pred_log1p_h1)
        return w_phys * (L["seq"] + L["pred"]) + pen_weight * phys.penalty()
    with torch.no_grad():
        start = float(objective())
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad(); loss = objective(); loss.backward(); opt.step()
    lb = torch.optim.LBFGS(params, lr=0.5, max_iter=lbfgs_iter, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad(); l = objective(); l.backward(); return l
    lb.step(closure)
    lb.zero_grad(); final = objective(); final.backward()
    gnorm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None)))
    lb.zero_grad()
    return {"loss_start": start, "loss_final": float(final.detach()), "grad_norm_final": gnorm, "adam_steps": steps, "adam_lr": lr, "lbfgs_max_iter": lbfgs_iter}


def kinetic_r2(phys: PhysicsParams, pr, pv, sid, eps: float = 1e-3) -> dict:
    """Share of the observed day-to-day change in ln Chl-a explained by the kinetic net growth rate (consecutive observed pairs)."""
    with torch.no_grad():
        T, N, P, C, I = pr[..., 0], pr[..., 1], pr[..., 2], pr[..., 3], pr[..., 4]
        vT, vN, vP, vC, vI = pv[..., 0], pv[..., 1], pv[..., 2], pv[..., 3], pv[..., 4]
        net = phys.factors(T, N, P, I, sid)["net"]
        ok = vC[:, :-1] & vC[:, 1:] & (vT & vN & vP & vI)[:, :-1]
        y = (torch.log(C[:, 1:] + eps) - torch.log(C[:, :-1] + eps))[ok].double().cpu().numpy()
        n = net[:, :-1][ok].double().cpu().numpy()
    if len(y) < 3:
        return {"n_pairs": int(len(y))}
    r = y - n
    return {"n_pairs": int(len(y)), "r2": float(1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "corr": float(np.corrcoef(n, y)[0, 1]) if n.std() > 0 else float("nan"), "mean_obs": float(y.mean()), "mean_net": float(n.mean())}


def physics_weight(epoch: int, w_min=0.01, w_max=0.3, warmup=10, curriculum=70) -> float:
    if epoch < warmup:
        return w_min
    if epoch >= curriculum:
        return w_max
    return w_min + (w_max - w_min) * (epoch - warmup) / max(1, curriculum - warmup)
