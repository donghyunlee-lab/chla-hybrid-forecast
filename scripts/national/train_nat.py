"""Training driver of the national benchmark. Residual (log-change from last observation) or level head; configurable horizons with per-horizon
loss normalisation; source-only scaler; record-length scarcity for targets; learning-rate grid selected on the
validation year for target modes; zero-shot evaluation. Saves every artifact needed for later figures and statistics.

Options of the physics runs (the defaults give the settings of the pure-network runs): --phys_param exp (none/soft on a log scale around the range centre), --phys_wd 0 and
--phys_lr (kinetic parameters in their own optimiser group), --warmup 0 --curriculum 0 (physics weight fixed at --w_max
from the first epoch), --converge_physics 1 (after training, fit the kinetic parameters alone to convergence with the
network fixed and report both estimates), --record_end/--val_frac_in_record (short records that end at a given date,
validated inside the record).

--record_in_pretrain 1 applies --record_months, --record_end and --val_frac_in_record to the pooled source stations in
pretrain mode (default 0: the record options apply to the target stations only).

The training and evaluation code are those of scripts/clean_revision/train_dl.py. For the 37 stations of the national
benchmark, the data module is patched by scripts/national/nat.py (stations = WEIS codes, periods, loader), --fold sets the
leave-one-basin-out source and target stations, and arrays are built only for the stations a run uses."""
from __future__ import annotations
import argparse, copy, json, random, sys, time, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clean_revision"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import nat as N
N.patch()
from physics import PhysicsParams, physics_losses, physics_weight, converge_physics, kinetic_r2
from models import build_backbone, Net

INV = {v: k for k, v in C.STATION_IDS.items()}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def to_t(a, dev):
    return torch.as_tensor(a, device=dev)


def wmse(pred, y, m, w):
    return ((pred - y) ** 2 * m * w).sum() / (m * w).sum().clamp_min(1.0)


def forward(net, X, sid, base, residual):
    o = net(X, sid)
    return o + base[:, None] if residual else o


def predict(net, arr, idx, dev, residual, bs=512):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            out.append(forward(net, to_t(arr["X"][j], dev), to_t(arr["sid"][j], dev), to_t(arr["base"][j], dev), residual).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, arr["y"].shape[1]), np.float32)


def pred_frame(arr, idx, ym, ps, scaler, horizons):
    rows = []
    for n, i in enumerate(idx):
        for hi, h in enumerate(horizons):
            rows.append({"station": INV[int(arr["sid"][i])], "anchor_date": str(arr["anchor"][i]), "horizon": h, "target_date": str(arr["target"][i][hi]),
                         "observed": bool(ym[n, hi] > 0), "last_obs_raw": float(scaler.y_unscale(arr["base"][i])) if arr["base_valid"][i] else np.nan,
                         "y_true_raw": float(arr["y_raw"][i, hi]) if ym[n, hi] > 0 else np.nan, "y_true_scaled": float(arr["y"][i, hi]) if ym[n, hi] > 0 else np.nan,
                         "y_pred_scaled": float(ps[n, hi]), "y_pred_raw": float(scaler.y_unscale(ps[n, hi]))})
    df = pd.DataFrame(rows); df["target_date"] = pd.to_datetime(df["target_date"]); return df


def evaluate_split(frame, name):
    obs = frame[frame.observed]
    out = {name: C.evaluate_predictions(obs)}
    if name == "test":
        for yr in sorted(obs.target_date.dt.year.unique()):
            out[f"test_{yr}"] = C.evaluate_predictions(obs[obs.target_date.dt.year == yr])
    return out


def latent_frame(net, arr, idx, dev):
    pr, pv, sid = to_t(arr["phys_raw"][idx], dev), to_t(arr["phys_valid"][idx], dev), to_t(arr["sid"][idx], dev)
    with torch.no_grad():
        fac = net.physics.factors(pr[..., 0], pr[..., 1], pr[..., 2], pr[..., 4], sid)
    d = {"station": [INV[int(s)] for s in arr["sid"][idx]], "anchor_date": [str(x) for x in arr["anchor"][idx]]}
    for k, name in enumerate(C.PHYS_VARS):
        d[f"{name}_raw"] = arr["phys_raw"][idx][:, -1, k]; d[f"{name}_valid"] = arr["phys_valid"][idx][:, -1, k]
    for k, v in fac.items():
        d[k] = v[:, -1].cpu().numpy()
    return pd.DataFrame(d)


def train_once(net, init_state, data, a, physics, hw, lr, dev, scaler, h1_idx, tag):
    net.load_state_dict(init_state)
    phys_p = [p for n_, p in net.named_parameters() if n_.startswith("physics.") and p.requires_grad]
    other_p = [p for n_, p in net.named_parameters() if not n_.startswith("physics.") and p.requires_grad]
    groups = [{"params": other_p, "lr": lr, "weight_decay": a.wd}]
    if phys_p:
        groups.append({"params": phys_p, "lr": a.phys_lr if a.phys_lr > 0 else lr, "weight_decay": a.phys_wd if a.phys_wd >= 0 else a.wd})
    params = other_p + phys_p
    opt = torch.optim.Adam(groups)
    Xtr, Str, Ytr, Mtr, Btr, PRtr, PVtr = data["train"]; Xva, Sva, Yva, Mva, Bva = data["val"]
    hist, traj = [], []; best, best_ep, best_state, wait = np.inf, -1, None, 0; n = len(Xtr)
    for ep in range(a.epochs):
        net.train(); te0 = time.time(); perm = torch.randperm(n, device=dev)
        acc = {"total": 0.0, "data": 0.0, "phys_seq": 0.0, "phys_pred": 0.0, "nonneg": 0.0, "pen": 0.0, "pairs": 0}; nb = 0
        w = physics_weight(ep, a.w_min, a.w_max, a.warmup, a.curriculum) if physics is not None else 0.0
        for i in range(0, n, a.batch):
            b = perm[i:i + a.batch]
            pred = forward(net, Xtr[b], Str[b], Btr[b], a.residual)
            data_loss = wmse(pred, Ytr[b], Mtr[b], hw); loss = data_loss
            if physics is not None:
                p1 = pred[:, h1_idx] * (scaler.y_hi - scaler.y_lo) + scaler.y_lo if h1_idx is not None else None
                pl = physics_losses(physics, PRtr[b], PVtr[b], Str[b], p1); pen = physics.penalty()
                loss = loss + w * (pl["seq"] + pl["pred"] + pl["nonneg"]) + a.pen_weight * pen
                acc["phys_seq"] += pl["seq"].item(); acc["phys_pred"] += pl["pred"].item(); acc["nonneg"] += pl["nonneg"].item(); acc["pen"] += pen.item(); acc["pairs"] += pl["n_pairs"]
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            acc["total"] += loss.item(); acc["data"] += data_loss.item(); nb += 1
        net.eval()
        with torch.no_grad():
            pv = forward(net, Xva, Sva, Bva, a.residual); val_loss = float(wmse(pv, Yva, Mva, hw))
            val_h = [float(wmse(pv[:, k:k + 1], Yva[:, k:k + 1], Mva[:, k:k + 1], hw[k:k + 1] * 0 + 1)) for k in range(Yva.shape[1])]
        row = {"epoch": ep, "lr": lr, "w_phys": w, "val_loss": val_loss, "epoch_time_s": time.time() - te0}
        row.update({k: (v / max(nb, 1) if k != "pairs" else v) for k, v in acc.items()})
        row.update({f"val_mse_h{h}": val_h[k] for k, h in enumerate(a.hz)})
        if physics is not None:
            snap = physics.snapshot(); traj.append({"epoch": ep, "lr": lr, **{k: v for k, v in snap.items() if k != "mu_max_sites"}, **{f"mu_max_{s}": snap["mu_max_sites"][i] for s, i in C.STATION_IDS.items()}})
        hist.append(row)
        if val_loss < best - 1e-6:
            best, best_ep, wait = val_loss, ep, 0; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
        if ep % 10 == 0:
            print(f"[{tag}] lr {lr:g} ep {ep} train {row['total']:.4f} val {val_loss:.4f} best {best:.4f}@{best_ep}", flush=True)
        if wait >= a.patience:
            break
    w_best = hist[best_ep]["w_phys"] if 0 <= best_ep < len(hist) else float("nan")
    return {"best_state": best_state, "best_val": best, "best_ep": best_ep, "hist": hist, "traj": traj, "lr": lr, "epochs_run": len(hist), "w_phys_best": w_best}


def physics_report(net, arr, tr, te, dev, a, scaler, h1_idx, w_fit):
    """Kinetic explanatory power at the jointly trained values, then the kinetic parameters fitted alone to convergence with
    the network held fixed (objective = physics part of the training loss). The network weights are not changed."""
    ph = net.physics
    def tens(idx):
        return to_t(arr["phys_raw"][idx], dev), to_t(arr["phys_valid"][idx], dev), to_t(arr["sid"][idx], dev)
    prt, pvt, st = tens(tr); pre, pve, se = tens(te)
    p1 = None
    if h1_idx is not None:
        p1 = to_t((predict(net, arr, tr, dev, a.residual)[:, h1_idx] * (scaler.y_hi - scaler.y_lo) + scaler.y_lo).astype(np.float32), dev)
    rep = {"w_fit": w_fit, "joint": {"params": ph.snapshot(), "kinetic_r2_train": kinetic_r2(ph, prt, pvt, st), "kinetic_r2_test": kinetic_r2(ph, pre, pve, se)}}
    fit_ph = copy.deepcopy(ph)
    fit = converge_physics(fit_ph, prt, pvt, st, p1, w_phys=w_fit, pen_weight=a.pen_weight)
    conv = fit_ph.snapshot()
    rep["converged"] = {"params": conv, "fit": fit, "kinetic_r2_train": kinetic_r2(fit_ph, prt, pvt, st), "kinetic_r2_test": kinetic_r2(fit_ph, pre, pve, se),
                        "position_in_range": {k: (conv[k] - ph.ranges[k][0]) / (ph.ranges[k][1] - ph.ranges[k][0]) for k in ("K_N", "K_P", "r", "m", "K_I")}}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--backbone", default="xLSTM")
    ap.add_argument("--physics", default="off", choices=["off", "none", "soft", "sigmoid"])
    ap.add_argument("--kn_max", type=float, default=0.5); ap.add_argument("--kp_max", type=float, default=0.05)
    ap.add_argument("--light", type=int, default=1); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="pretrain", choices=["pretrain", "zero_shot", "scratch", "linear_probe", "full_tune"])
    ap.add_argument("--target_station", default=None); ap.add_argument("--init_ckpt", default=None)
    ap.add_argument("--record_months", type=int, default=24, help="target record length used for training (months from 2021-01)")
    ap.add_argument("--horizons", default=",".join(map(str, C.HORIZONS)))
    ap.add_argument("--residual", type=int, default=1, help="1: predict log-change from the last observed chlorophyll-a; 0: level head")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--patience", type=int, default=15); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="pretrain learning rate"); ap.add_argument("--lr_grid", default="1e-3,3e-4,1e-4", help="target modes: selected on validation")
    ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--w_min", type=float, default=0.01); ap.add_argument("--w_max", type=float, default=0.3); ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--curriculum", type=int, default=70)
    ap.add_argument("--pen_weight", type=float, default=1.0); ap.add_argument("--out", default=str(C.RESULTS))
    ap.add_argument("--phys_param", default="softplus", choices=["softplus", "exp"]); ap.add_argument("--phys_lr", type=float, default=-1.0, help="<=0: same as the learning rate")
    ap.add_argument("--phys_wd", type=float, default=-1.0, help="<0: same as --wd"); ap.add_argument("--converge_physics", type=int, default=0)
    ap.add_argument("--phys_null", type=int, default=0, help="placebo: kinetic growth term set to zero, so the physics loss pulls the one-day change towards a constant rate")
    ap.add_argument("--record_end", default="", help="v4: the record is the --record_months months ending on this date (empty: v3, first months from 2021-01)")
    ap.add_argument("--val_frac_in_record", type=float, default=0.0, help="v4 with --record_end: share of the record's last anchor days held out for validation (0: validation year)")
    ap.add_argument("--record_in_pretrain", type=int, default=0, help="v7: apply the record options to the source stations in pretrain mode")
    ap.add_argument("--fold", default="", help="national: held-out basin (Han, Nakdong, Geum, Yeongsan)")
    a = ap.parse_args()
    N.set_fold(a.fold)
    a.hz = [int(x) for x in a.horizons.split(",")]; h1_idx = a.hz.index(1) if 1 in a.hz else None
    seed_all(a.seed); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(a.out) / "runs" / a.tag; run_dir.mkdir(parents=True, exist_ok=True); t0 = time.time()

    # ---- data (scaler from source stations' training period only) ----
    df = C.load_daily()
    train_mask = ((df["date"] >= C.SPLITS["train"][0]) & (df["date"] <= C.SPLITS["train"][1])).to_numpy()
    scaler = C.Scaler().fit(df, train_mask, C.SOURCE)
    arr = C.build_arrays(df, scaler, list(C.SOURCE) if a.mode == "pretrain" else [a.target_station], C.SEQ_LEN, a.hz)
    if a.mode == "pretrain":
        tr_st = ev_st = C.SOURCE
    else:
        assert a.target_station in C.TARGETS; tr_st = ev_st = [a.target_station]
    tr, va, te = C.split_indices(arr, "train", tr_st), C.split_indices(arr, "val", ev_st), C.split_indices(arr, "test", ev_st)
    ym_tr, ym_va, ym_te = None, C.mask_targets_to_split(arr, va, "val"), C.mask_targets_to_split(arr, te, "test")
    record = None
    if a.mode != "pretrain" or a.record_in_pretrain:
        if a.record_end:
            tr, va_rec, tr_end, rec_end = C.record_window(arr, tr, a.record_months, a.record_end, a.val_frac_in_record)
            ym_tr = C.mask_targets_after(arr, tr, tr_end)
            if va_rec is not None:
                va, ym_va = va_rec, C.mask_targets_after(arr, va_rec, rec_end)
            record = {"first_anchor": str(arr["anchor"][tr].min()) if len(tr) else None, "train_targets_end": str(tr_end.date()), "record_end": str(rec_end.date()),
                      "validation": "in_record" if va_rec is not None else "validation_year", "val_first_anchor": str(arr["anchor"][va].min()) if len(va) else None}
        else:
            tr = C.record_subset(arr, tr, a.record_months)
    if ym_tr is None:
        ym_tr = C.mask_targets_to_split(arr, tr, "train")
    keep = ym_tr.sum(1) > 0; tr, ym_tr = tr[keep], ym_tr[keep]
    # per-horizon loss normalisation from the training targets (residual or level, as trained)
    res = arr["y"][tr] - (arr["base"][tr][:, None] if a.residual else 0.0)
    var = np.array([np.var(res[ym_tr[:, k] > 0, k]) if (ym_tr[:, k] > 0).sum() > 2 else 1.0 for k in range(len(a.hz))])
    hw_np = 1.0 / np.maximum(var, 1e-6); hw_np = hw_np / hw_np.mean(); hw = torch.tensor(hw_np, dtype=torch.float32, device=dev)

    # ---- model ----
    n_feat = arr["X"].shape[2]
    backbone = build_backbone(a.backbone, n_feat, len(C.STATION_IDS), C.SEQ_LEN, len(a.hz), a.hidden, a.layers, a.dropout)
    physics = PhysicsParams(a.physics, len(C.STATION_IDS), a.kn_max, a.kp_max, bool(a.light), param=a.phys_param, null=bool(a.phys_null)) if a.physics != "off" else None
    net = Net(backbone, physics).to(dev)
    if a.init_ckpt:
        net.load_state_dict(torch.load(a.init_ckpt, map_location=dev), strict=True)
        if a.mode != "pretrain":
            with torch.no_grad():
                tid = C.STATION_IDS[a.target_station]; sids = [C.STATION_IDS[x] for x in C.SOURCE]
                emb = getattr(net.backbone, "station_embedding", None)
                if emb is not None:
                    emb.weight[tid] = emb.weight[sids].mean(0)
                if physics is not None:
                    physics.mu_site.weight[tid] = physics.mu_site.weight[sids].mean(0)
    elif a.residual:
        with torch.no_grad():
            net.backbone.mean_predictor.weight.zero_(); net.backbone.mean_predictor.bias.zero_()
    if a.mode == "linear_probe":
        for n_, p in net.backbone.named_parameters():
            p.requires_grad = n_.startswith("mean_predictor") or n_.startswith("station_embedding")
    init_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    data = {"train": (to_t(arr["X"][tr], dev), to_t(arr["sid"][tr], dev), to_t(arr["y"][tr], dev), to_t(ym_tr, dev), to_t(arr["base"][tr], dev), to_t(arr["phys_raw"][tr], dev), to_t(arr["phys_valid"][tr], dev)),
            "val": (to_t(arr["X"][va], dev), to_t(arr["sid"][va], dev), to_t(arr["y"][va], dev), to_t(ym_va, dev), to_t(arr["base"][va], dev))}

    # ---- training (lr grid on validation for target modes) ----
    if a.mode == "zero_shot":
        best = {"best_state": init_state, "best_val": float("nan"), "best_ep": -1, "hist": [], "traj": [], "lr": float("nan"), "epochs_run": 0, "w_phys_best": float("nan")}; lr_table = []
    else:
        lrs = [a.lr] if a.mode == "pretrain" else [float(x) for x in a.lr_grid.split(",")]
        runs = [train_once(net, init_state, data, a, physics, hw, lr, dev, scaler, h1_idx, a.tag) for lr in lrs]
        best = min(runs, key=lambda r: r["best_val"]); lr_table = [{"lr": r["lr"], "best_val": r["best_val"], "best_epoch": r["best_ep"], "epochs_run": r["epochs_run"]} for r in runs]
    net.load_state_dict(best["best_state"]); torch.save(best["best_state"], run_dir / "best_model.pt"); train_time = time.time() - t0

    # ---- artifacts ----
    pd.DataFrame(best["hist"]).to_csv(run_dir / "history.csv", index=False)
    if best["traj"]:
        pd.DataFrame(best["traj"]).to_csv(run_dir / "params_trajectory.csv", index=False)
    metrics = {}
    fva = pred_frame(arr, va, ym_va, predict(net, arr, va, dev, a.residual), scaler, a.hz); fva.to_csv(run_dir / "predictions_val.csv", index=False); metrics.update(evaluate_split(fva, "val"))
    fte = pred_frame(arr, te, ym_te, predict(net, arr, te, dev, a.residual), scaler, a.hz); fte.to_csv(run_dir / "predictions_test.csv", index=False); metrics.update(evaluate_split(fte, "test"))
    prep = None
    if physics is not None:
        latent_frame(net, arr, te, dev).to_csv(run_dir / "latent_test.csv", index=False)
        metrics["physics_params_final"] = physics.snapshot(); metrics["physics_ranges"] = physics.ranges
        if a.converge_physics and a.mode != "zero_shot":
            w_fit = best["w_phys_best"] if np.isfinite(best["w_phys_best"]) else a.w_max
            try:                                                         # national: a failed post-hoc kinetic fit does not discard the forecasts
                prep = physics_report(net, arr, tr, te, dev, a, scaler, h1_idx, w_fit)
                C.write_json(prep, run_dir / "physics_report.json"); metrics["physics_params_converged"] = prep["converged"]["params"]
            except RuntimeError as e:
                prep = None; metrics["physics_converged_fit_failed"] = repr(e); print("converged kinetic fit failed:", repr(e), flush=True)
    C.write_json(metrics, run_dir / "metrics.json")
    cfg = {k: v for k, v in vars(a).items() if k != "hz"} | {"horizons": a.hz, "lr_selected": best["lr"], "lr_table": lr_table, "horizon_loss_weights": hw_np.tolist(),
                                                              "record": record, "w_phys_best": best["w_phys_best"],
                                                              "device": str(dev), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)), "n_features": int(n_feat),
                                                              "feature_names": arr["feat_names"], "seq_len": C.SEQ_LEN, "splits": C.SPLITS, "train_stations": tr_st, "eval_stations": ev_st,
                                                              "scaler_stations": C.SOURCE, "torch": torch.__version__, "started": dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds")}
    C.write_json(cfg, run_dir / "config.json"); C.write_json(scaler.to_dict(), run_dir / "scaler.json")
    info = {"params_total": net.n_params(), "params_backbone": net.n_params("backbone"), "best_epoch": best["best_ep"], "best_val_loss": best["best_val"], "epochs_run": best["epochs_run"], "train_time_s": train_time,
            "w_phys_best": best["w_phys_best"]}
    C.write_json(info, run_dir / "model_info.json")
    pool = lambda s, h: metrics.get(s, {}).get(f"h{h}", {}).get("pooled", {})
    C.append_index({"tag": a.tag, "backbone": a.backbone, "physics": a.physics + ("-null" if a.phys_null else ""), "mode": a.mode, "target": a.target_station or "", "record_months": a.record_months, "residual": a.residual,
                    "horizons": a.horizons, "seed": a.seed, "kn_max": a.kn_max, "light": a.light, "lr_selected": best["lr"], "n_train": len(tr), "best_epoch": best["best_ep"],
                    "train_time_s": round(train_time, 1), "params": info["params_total"], **{f"test_h{h}_r2": pool("test", h).get("r2") for h in a.hz},
                    **{f"test2025_h{h}_r2": pool("test_2025", h).get("r2") for h in a.hz}, "K_N": metrics.get("physics_params_final", {}).get("K_N"), "K_P": metrics.get("physics_params_final", {}).get("K_P"),
                    "phys_param": a.phys_param, "phys_lr": a.phys_lr, "phys_wd": a.phys_wd, "w_max": a.w_max, "w_phys_best": best["w_phys_best"],
                    "record_end": a.record_end, "val_frac_in_record": a.val_frac_in_record,
                    **({"K_N_conv": prep["converged"]["params"]["K_N"], "K_P_conv": prep["converged"]["params"]["K_P"], "K_I_conv": prep["converged"]["params"]["K_I"],
                        "r_plus_m_conv": prep["converged"]["params"]["r"] + prep["converged"]["params"]["m"],
                        "kin_r2_test_joint": prep["joint"]["kinetic_r2_test"].get("r2"), "kin_r2_test_conv": prep["converged"]["kinetic_r2_test"].get("r2")} if prep else {}),
                    "finished": dt.datetime.now().isoformat(timespec="seconds")})
    (run_dir / "DONE").write_text("ok")
    extra = f"; w_phys at best {best['w_phys_best']}" if physics is not None else ""
    if prep:
        cp = prep["converged"]["params"]
        extra += f"; converged K_N {cp['K_N']:.3g} K_P {cp['K_P']:.3g} K_I {cp['K_I']:.4g} r+m {cp['r'] + cp['m']:.3g}; kinetic R2 test {prep['converged']['kinetic_r2_test'].get('r2', float('nan')):.3f}"
    print(f"[{a.tag}] done {train_time/60:.1f} min; lr {best['lr']}; test R2 " + " ".join(f"h{h}={pool('test', h).get('r2', float('nan')):.3f}" for h in a.hz) + extra, flush=True)


if __name__ == "__main__":
    main()
