"""Persistence baseline for the tables of the physics runs (no model is trained or changed).

Persistence repeats the starting point of the residual head: the last observed chlorophyll-a inside the 30-day input window
(common.build_arrays `base`; training median when the window has no observation), converted back to mg m-3.
Scored on exactly the keys (station, anchor date, horizon) of a reference ensemble, so the masks match Table 2 (source,
reference xLSTM) and Table 4 (each reservoir, reference fine-tuned PINN). Paired comparisons reuse stats_v4.compare
(30-day date-block bootstrap, 14/60 as sensitivity). The frozen stats are not touched.
Writes models.csv, comparisons.csv, station_comparisons.csv and summary.md to the folder OUT below (not part of this package)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import stats_v4 as S

OUT = S.V4 / "persistence_20261020"
NAME = {i: s for s, i in C.STATION_IDS.items()}


def persistence_frame():
    df = C.load_daily()
    train_mask = ((df["date"] >= C.SPLITS["train"][0]) & (df["date"] <= C.SPLITS["train"][1])).to_numpy()
    scaler = C.Scaler().fit(df, train_mask, C.SOURCE)
    arr = C.build_arrays(df, scaler, list(C.STATION_IDS), C.SEQ_LEN, C.HORIZONS)
    p_raw = scaler.y_unscale(arr["base"].astype(float))
    rows = []
    for k, h in enumerate(C.HORIZONS):
        ok = arr["y_mask"][:, k] > 0
        rows.append(pd.DataFrame({"station": [NAME[int(s)] for s in arr["sid"][ok]], "anchor_date": arr["anchor"][ok].astype(str),
                                  "horizon": h, "target_date": pd.to_datetime(arr["target"][ok, k]), "y_true_raw": arr["y_raw"][ok, k],
                                  "y_pred_raw": p_raw[ok], "base_valid": arr["base_valid"][ok]}))
    return pd.concat(rows, ignore_index=True)


def restrict(per, ref):
    """Persistence on the reference keys; y_true must agree with the reference file."""
    j = ref[S.KEY + ["y_true_raw"]].merge(per, on=S.KEY, how="left", suffixes=("_ref", ""))
    assert j.y_pred_raw.notna().all(), "persistence missing for some reference keys"
    assert np.allclose(j.y_true_raw, j.y_true_raw_ref), "observed values differ from the reference file"
    return j.drop(columns="y_true_raw_ref")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = persistence_frame(); mrows, crows, srows, notes = [], [], [], []
    groups = [("source", "", S.source_models(), "xLSTM", ["xLSTM", "Harmonic ARIMA", "XGBoost residual", "PINN sigmoid w0.3", "ARIMA"])]
    groups += [("transfer", tg, S.transfer_models(tg), "fine-tune PINN", ["fine-tune PINN", "fine-tune pure", "fine-tune placebo", "Harmonic ARIMA", "XGBoost target-only"])
               for tg in C.TARGETS]
    for group, target, models, ref_name, others in groups:
        ens = {}
        for name in set(others) | {ref_name}:
            e, fr = S.ensemble(models[name])
            if e is None:
                notes.append(f"missing {group} {target}: {name}"); continue
            ens[name] = e
        p = restrict(per, ens[ref_name])
        notes.append(f"{group} {target}: n per horizon " + ", ".join(f"{h}: {int((p.horizon == h).sum())}" for h in C.HORIZONS)
                     + f"; windows without an observation (median used): {int((~p.base_valid).sum())}")
        mrows += S.model_rows(group, target, "24m", "Persistence", p.drop(columns="base_valid"), [])
        for na in others:
            if na in ens:
                c, s = S.compare(ens[na], p.drop(columns="base_valid"), group, target, "24m", na, "Persistence"); crows += c; srows += s
        print("done", group, target, flush=True)
    mt, ct, st = pd.DataFrame(mrows), pd.DataFrame(crows), pd.DataFrame(srows)
    mt.to_csv(OUT / "models.csv", index=False); ct.to_csv(OUT / "comparisons.csv", index=False); st.to_csv(OUT / "station_comparisons.csv", index=False)
    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    ci = lambda r, nm: f"{r[f'diff_{nm}']:+.3f} [{r[f'ci30_{nm}_lo']:+.3f}, {r[f'ci30_{nm}_hi']:+.3f}]" + (" *" if r[f"excl0_{nm}"] else "")
    lines = ["# Persistence baseline (v4 masks; frozen numbers unchanged)", ""] + [f"- {n}" for n in notes] + [""]
    for (group, target), g in mt.groupby(["group", "target"], sort=False):
        v = g.set_index("horizon")
        lines += [f"## {group} {target}".strip(), "", "| | h1 | h3 | h7 | h14 | h7 station mean | h14 station mean |", "|---|---|---|---|---|---|---|",
                  "| Persistence | " + " | ".join(f3(v.r2_pooled.get(h)) for h in C.HORIZONS) + f" | {f3(v.r2_station_mean.get(7))} | {f3(v.r2_station_mean.get(14))} |", ""]
        cg = ct[(ct.group == group) & (ct.target == target)]
        lines += ["| comparison | h | pooled R2 difference | station-mean difference | 14-day blocks (pooled) | 60-day blocks (pooled) |", "|---|---|---|---|---|---|"]
        for _, r in cg.iterrows():
            lines.append(f"| {r.comparison} | {r.horizon} | {ci(r, 'pooled')} | {ci(r, 'station_mean')} | [{r.ci14_pooled_lo:+.3f}, {r.ci14_pooled_hi:+.3f}] | [{r.ci60_pooled_lo:+.3f}, {r.ci60_pooled_hi:+.3f}] |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__":
    main()
