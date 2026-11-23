"""Pilot of process models with a transport term along the main stem of the Nakdong River (12 stations).
Variants, trained jointly on the 12 Nakdong stations of the national benchmark:
  B1  an xLSTM predicts, for each day and station, the parameters of a growth-and-transport mass balance (sigmoid bounds);
      the mass balance, iterated along the river network, makes the 1-14-day forecasts;
  B0  the same without transport (growth only);
  C1  xLSTM without process terms (black box) with the paper's residual head;
  C2  C1 with the upstream stations' Chl-a channels as extra inputs.
Usage: chain_nakdong.py --kind B1 --record 36 --seed 42"""
from __future__ import annotations
import argparse, copy, json, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "clean_revision")); sys.path.insert(0, str(ROOT / "scripts" / "national"))
import nat as N                                                                  # noqa: E402
C = N.patch()
from models import build_backbone                                                # noqa: E402

OUT = ROOT / "results" / "chain_nakdong_20261027"
CHAIN = ["S02023", "S02014", "S02022", "S02020", "S02021", "S02015", "S02009", "S02002", "S02005", "S02024", "S02006", "S02017"]
UP = {"S02014": ["S02023"], "S02022": ["S02014"], "S02020": ["S02022"], "S02021": ["S02020"], "S02015": ["S02021"], "S02009": ["S02015"],
      "S02002": ["S02009"], "S02005": ["S02002"], "S02006": ["S02005", "S02024"], "S02017": ["S02006"]}
H, HMAX, L = [1, 3, 7, 14], 14, 5                      # horizons; rollout length; days of upstream history (= max travel time)
THETA, KN, KP, KI = 1.047, 0.255, 0.0255, 525.0          # fixed at the centres of the paper's literature ranges
R_MU, R_LOSS, R_W, R_TAU = (0.1, 1.5), (0.015, 0.3), (0.0, 0.9), (0.0, 5.0)
S = len(CHAIN); IDX = {s: i for i, s in enumerate(CHAIN)}
N_OUT = {"B1": 6, "B0": 2, "C1": 4, "C2": 4}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def prepare(rec):
    df = N.load_daily()
    end = pd.Timestamp(N.RECORD_END); start = end - pd.DateOffset(months=rec) + pd.Timedelta(days=1)
    days = pd.date_range(start, end); vstart = days[-max(1, int(round(len(days) * N.VAL_FRAC)))]; tr_end = vstart - pd.Timedelta(days=1)
    sc = C.Scaler().fit(df, ((df.date >= start) & (df.date <= end)).to_numpy(), CHAIN)     # inside the record only
    arrs = {s: C.build_arrays(df, sc, [s], C.SEQ_LEN, H) for s in CHAIN}
    anc = arrs[CHAIN[0]]["anchor"]
    for s in CHAIN[1:]:
        anc = np.intersect1d(anc, arrs[s]["anchor"])
    assert (np.diff(anc) == np.timedelta64(1, "D")).all(), "anchors must be consecutive days"
    for s in CHAIN:                                                                       # common anchor days
        keep = np.isin(arrs[s]["anchor"], anc)
        arrs[s] = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in arrs[s].items()}
    fn = arrs[CHAIN[0]]["feat_names"]
    X = np.stack([arrs[s]["X"] for s in CHAIN], 1).astype(np.float32)                    # (T, S, 30, F)
    ch = [fn.index(c) for c in ("chlorophyll_a", "chlorophyll_a__mask", "chlorophyll_a__age", "chlorophyll_a__last_obs")]
    up = np.zeros(X.shape[:3] + (8,), np.float32)                                        # upstream Chl-a channels (C2)
    for s, us in UP.items():
        for k, u in enumerate(us):
            up[:, IDX[s], :, 4 * k:4 * k + 4] = X[:, IDX[u]][:, :, ch]
    yraw = np.stack([arrs[s]["y_raw"] for s in CHAIN], 1); ym = np.stack([arrs[s]["y_mask"] for s in CHAIN], 1)
    base = np.stack([sc.y_unscale(arrs[s]["base"].astype(float)) for s in CHAIN], 1)        # last measured Chl-a (raw)
    pad = np.concatenate([np.repeat(base[:1], L, 0), base], 0)
    hist = np.stack([pad[j:j + len(anc)] for j in range(L + 1)], -1)                         # (T, S, L+1): t-L .. t
    pr = np.stack([arrs[s]["phys_raw"][:, -7:, :] for s in CHAIN], 1); pv = np.stack([arrs[s]["phys_valid"][:, -7:, :] for s in CHAIN], 1)
    den = pv.sum(2); drv = np.where(den > 0, (pr * pv).sum(2) / np.maximum(den, 1), np.nan)   # 7-day means of T, N, P, C, PAR
    inrec = (anc >= np.datetime64(start.date())) & (anc <= np.datetime64(end.date()))
    for k in range(5):
        med = np.nanmedian(drv[inrec][:, :, k], 0); med = np.where(np.isnan(med), np.nanmedian(drv[inrec][:, :, k]), med)
        drv[:, :, k] = np.where(np.isnan(drv[:, :, k]), med[None, :], drv[:, :, k])
    tgt = arrs[CHAIN[0]]["target"]
    sel = lambda lo, hi: np.where((anc >= np.datetime64(lo)) & (anc <= np.datetime64(hi)))[0]
    D = {"X": X, "up": up, "base": base.astype(np.float32), "hist": hist.astype(np.float32), "drv": drv[:, :, [0, 1, 2, 4]].astype(np.float32),
         "ylog": np.log1p(np.nan_to_num(yraw, nan=0.0)).astype(np.float32), "yraw": yraw, "anc": anc, "tgt": tgt,
         "m_tr": (ym * (tgt <= np.datetime64(tr_end.date()))[:, None, :]).astype(np.float32),
         "m_va": (ym * (tgt <= np.datetime64(end.date()))[:, None, :]).astype(np.float32), "m_te": ym.astype(np.float32),
         "tr": sel(start.date(), tr_end.date()), "va": sel(vstart.date(), end.date()), "te": sel(N.SPLITS["test"][0], N.SPLITS["test"][1]),
         "record": {"start": str(start.date()), "val_start": str(vstart.date()), "end": str(end.date())}}
    d = (np.log1p(np.nan_to_num(yraw, nan=0.0)) - np.log1p(base)[:, :, None])[D["tr"]]; m = D["m_tr"][D["tr"]] > 0
    var = np.array([d[..., k][m[..., k]].var() if m[..., k].sum() > 2 else 1.0 for k in range(len(H))])
    hw = 1 / np.maximum(var, 1e-6); D["hw"] = (hw / hw.mean()).astype(np.float32)
    return D


def bound(z, r):
    return r[0] + torch.sigmoid(z) * (r[1] - r[0])


class Net(nn.Module):
    def __init__(self, kind, n_feat):
        super().__init__()
        self.kind = kind
        self.bb = build_backbone("xLSTM", n_feat + (8 if kind == "C2" else 0), len(C.STATION_IDS), C.SEQ_LEN, N_OUT[kind])
        self.register_buffer("sid", torch.tensor([C.STATION_IDS[s] for s in CHAIN]))

    def raw(self, x):
        B = x.shape[0]
        o = self.bb(x.reshape(B * S, *x.shape[2:]), self.sid.repeat(B))
        return (o[0] if isinstance(o, tuple) else o).reshape(B, S, -1)

    def forward(self, x, base, hist, drv, transport=True):
        """Returns log1p forecasts (B, S, 4) and, for the process models, the daily parameters."""
        z = self.raw(x)
        if self.kind in ("C1", "C2"):
            return torch.log1p(base)[..., None] + z, None
        mu, loss = bound(z[..., 0], R_MU), bound(z[..., 1], R_LOSS)
        T, Nn, P, I = drv[..., 0].clamp(0, 40), drv[..., 1].clamp_min(0), drv[..., 2].clamp_min(0), drv[..., 3].clamp_min(0)
        phi = THETA ** (T - 20) * Nn / (KN + Nn) * P / (KP + P) * I / (KI + I)
        g = (mu * phi - loss).clamp(-1.0, 1.0); eg = torch.exp(g)
        par = {"mu": mu, "loss": loss, "g": g}
        use_t = self.kind == "B1" and transport
        if self.kind == "B1":
            par.update(w=bound(z[..., 2], R_W), share=torch.sigmoid(z[..., 3]), tau1=bound(z[..., 4], R_TAU), tau2=bound(z[..., 5], R_TAU))
        state = [hist[:, i, :] for i in range(S)]; out = []
        for k in range(1, HMAX + 1):
            for i, s in enumerate(CHAIN):
                own = state[i][:, -1]
                if use_t and s in UP:
                    inflow = torch.zeros_like(own)
                    for slot, u in enumerate(UP[s]):
                        sj = state[IDX[u]]                                       # upstream already advanced to t + k
                        pos = (L + k) - (par["tau1"][:, i] if slot == 0 else par["tau2"][:, i])
                        lo = pos.floor().long().clamp(0, sj.shape[1] - 1); hi = (lo + 1).clamp(max=sj.shape[1] - 1); fr = pos - lo.float()
                        val = (1 - fr) * sj.gather(1, lo[:, None]).squeeze(1) + fr * sj.gather(1, hi[:, None]).squeeze(1)
                        wgt = torch.ones_like(own) if len(UP[s]) == 1 else (par["share"][:, i] if slot == 0 else 1 - par["share"][:, i])
                        inflow = inflow + wgt * val
                    mixed = (1 - par["w"][:, i]) * own + par["w"][:, i] * inflow
                else:
                    mixed = own
                state[i] = torch.cat([state[i], (mixed * eg[:, i]).clamp(1e-3, 1e4)[:, None]], 1)
            if k in H:
                out.append(torch.stack([state[i][:, -1] for i in range(S)], 1))
        return torch.log1p(torch.stack(out, -1)), par


def tensors(D, idx, kind, dev):
    x = D["X"][idx]
    if kind == "C2":
        x = np.concatenate([x, D["up"][idx]], -1)
    return [torch.tensor(a, device=dev) for a in (x, D["base"][idx], D["hist"][idx], D["drv"][idx])]


def mse(pred, y, m, hw):
    """Weighted mean squared error over measured targets, as in the paper's training script."""
    return ((pred - y) ** 2 * m * hw).sum() / (m * hw).sum().clamp_min(1.0)


def evaluate(net, D, idx, mask, kind, dev, hw, bs=128):
    net.eval(); num, den = 0.0, 0.0
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            j = idx[b:b + bs]; x, base, hist, drv = tensors(D, j, kind, dev)
            p, _ = net(x, base, hist, drv)
            y, m = torch.tensor(D["ylog"][j], device=dev), torch.tensor(mask[j], device=dev)
            num += float(((p - y) ** 2 * m * hw).sum()); den += float((m * hw).sum())
    return num / max(den, 1.0)


def train(kind, D, lr, seed, dev, epochs=100, patience=15, bs=32):
    seed_all(seed); net = Net(kind, D["X"].shape[-1]).to(dev); hw = torch.tensor(D["hw"], device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    tr = D["tr"][(D["m_tr"][D["tr"]].sum((1, 2)) > 0)]; rng = np.random.RandomState(seed)
    best, best_state, bad, ep_best = np.inf, None, 0, -1
    for ep in range(epochs):
        net.train(); perm = rng.permutation(tr)
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]; x, base, hist, drv = tensors(D, j, kind, dev)
            p, _ = net(x, base, hist, drv)
            loss = mse(p, torch.tensor(D["ylog"][j], device=dev), torch.tensor(D["m_tr"][j], device=dev), hw)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        v = evaluate(net, D, D["va"], D["m_va"], kind, dev, hw)
        if ep % 10 == 0:
            print(f"[{kind} lr {lr:g}] ep {ep} val {v:.4f} best {best:.4f}@{ep_best}", flush=True)
        if v < best - 1e-6:
            best, best_state, bad, ep_best = v, copy.deepcopy(net.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, best, ep_best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True, choices=list(N_OUT)); ap.add_argument("--record", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True); a = ap.parse_args()
    tag = f"{a.kind}_r{a.record}_s{a.seed}"; rd = OUT / "runs" / tag; rd.mkdir(parents=True, exist_ok=True)
    if (rd / "DONE").exists():
        print("skip", tag); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); t0 = time.time()
    D = prepare(a.record)
    runs = [(lr, *train(a.kind, D, lr, a.seed, dev)) for lr in (1e-3, 3e-4, 1e-4)]
    lr, net, best, ep = min(runs, key=lambda r: r[2])
    te = D["te"]; rows, prow = [], []
    with torch.no_grad():
        for b in range(0, len(te), 128):
            j = te[b:b + 128]; x, base, hist, drv = tensors(D, j, a.kind, dev)
            p, par = net.eval()(x, base, hist, drv); p = p.cpu().numpy()
            if a.kind == "B1":
                p0, _ = net(x, base, hist, drv, transport=False); p0 = p0.cpu().numpy()
            for bi, t in enumerate(j):
                for i, s in enumerate(CHAIN):
                    for k, h in enumerate(H):
                        rows.append({"station": s, "anchor_date": str(D["anc"][t]), "horizon": h, "target_date": str(D["tgt"][t, k]),
                                     "observed": bool(D["m_te"][t, i, k] > 0), "y_true_raw": float(D["yraw"][t, i, k]) if D["m_te"][t, i, k] > 0 else np.nan,
                                     "y_pred_raw": float(np.expm1(p[bi, i, k]))})
                    if par is not None:
                        r = {"station": s, "anchor_date": str(D["anc"][t]), **{k2: float(v[bi, i]) for k2, v in par.items()},
                             "base": float(D["base"][t, i]), "dlog7_full": float(p[bi, i, 2] - np.log1p(D["base"][t, i]))}
                        if a.kind == "B1":
                            r["dlog7_no_transport"] = float(p0[bi, i, 2] - np.log1p(D["base"][t, i]))
                        prow.append(r)
    pd.DataFrame(rows).to_csv(rd / "predictions_test.csv", index=False)
    if prow:
        pd.DataFrame(prow).to_csv(rd / "params_test.csv", index=False)
    json.dump({"kind": a.kind, "record_months": a.record, "seed": a.seed, "lr_selected": lr, "best_val": best, "best_epoch": ep, "record": D["record"],
               "lr_table": [{"lr": r[0], "best_val": r[2], "best_epoch": r[3]} for r in runs], "time_s": time.time() - t0, "stations": CHAIN, "upstream": UP},
              open(rd / "config.json", "w"), indent=1)
    (rd / "DONE").write_text("ok"); print(f"[{tag}] done {(time.time() - t0) / 60:.1f} min, lr {lr}, val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
