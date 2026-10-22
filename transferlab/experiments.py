"""Run post-hoc diagnostics from a fresh, independent implementation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import audit, load_panel, with_label_maturity, training_before, evaluate_window, file_sha256
from .models import fit_models, predict, MODELS, calibration_line
from .metrics import date_metrics, summarize
from .inference import (block_bootstrap, paired_date_difference,
                        factorial_paired_contrasts, factorial_interaction_bootstrap)


def _save(df: pd.DataFrame, out: Path, name: str):
    df.to_csv(out/name,index=False,float_format="%.10g")


def split_identities(ids: list[str], frac: float, seed: int, split: int) -> tuple[set[str],set[str]]:
    if not 0 < frac < 0.5:
        raise ValueError("test_identity_fraction should be in (0,0.5)")
    rng = np.random.default_rng(np.random.SeedSequence([seed, split]))
    ids = np.array(sorted(ids))
    selected = set(rng.choice(ids, size=max(5,int(round(frac*len(ids)))),replace=False).tolist())
    return set(ids.tolist()) - selected, selected


def _run_one(df: pd.DataFrame, train_end: str, test_start: str, test_end: str,
             seen: set[str], unseen: set[str], alpha: float, min_stocks: int,
             split_id: int):
    train = training_before(df,train_end,seen)
    bundle = fit_models(train, alpha)
    eva = evaluate_window(df,test_start,test_end,unseen)
    if eva.empty:
        raise ValueError(f"No test observations for split={split_id}")
    preds = predict(bundle,eva)
    preds["split"] = split_id
    metrics = date_metrics(preds,min_stocks)
    metrics["split"] = split_id
    counts = {"split":split_id, "train_rows":len(train),"test_rows":len(eva),
              "train_identities":train.ticker.nunique(),"test_identities":eva.ticker.nunique(),
              "eligible_dates":metrics.date.nunique()}
    calibration = calibration_line(bundle)
    calibration.update({
        "fit_kind": "heldout_split",
        "split": int(split_id),
        "train_end_cutoff": str(pd.Timestamp(train_end).date()),
        "test_start_bound": str(pd.Timestamp(test_start).date()),
        "test_end_bound": str(pd.Timestamp(test_end).date()),
        "first_train_signal": str(train.date.min().date()),
        "last_train_signal": str(train.date.max().date()),
        "first_train_maturity": str(train.maturity_date.min().date()),
        "last_train_maturity": str(train.maturity_date.max().date()),
        "train_rows": len(train),
        "train_dates": train.date.nunique(),
        "train_identities": train.ticker.nunique(),
        "test_rows": len(eva),
        "test_dates": eva.date.nunique(),
        "test_identities": eva.ticker.nunique(),
        "training_ret63_target": "future_clipped_QQQ_relative_log_return_y",
        "calibration_fitted_on": "matured_training_rows_only",
        "calibration_formula": "intercept + slope * unscaled_ret63",
        "test_outcomes_used_for_calibration": False,
    })
    if train.maturity_date.max() > pd.Timestamp(train_end):
        raise AssertionError("A training label has not matured by fit cutoff")
    if not set(train.ticker).isdisjoint(set(eva.ticker)):
        raise AssertionError("Held-out identity leaked into training")
    return preds, metrics, counts, bundle, calibration


def frozen_diagnostic_splits(df: pd.DataFrame, cfg: dict, output: Path):
    # The identity pool is defined from history available by the training cutoff,
    # not from future test outcomes. Frozen roster reuse != new independent confirmation.
    eligible = sorted(df.loc[df.date <= pd.Timestamp(cfg["train_end"]),"ticker"].unique())
    preds, metrics, split_info, roster, cal_rows = [], [], [], [], []
    for i in range(cfg["splits"]):
        seen, unseen = split_identities(eligible,cfg["test_identity_fraction"],cfg["seed"],i)
        p,m,c,_,cal = _run_one(df,cfg["train_end"],cfg["test_start"],cfg["test_end"],
                           seen,unseen,cfg["ridge_alpha"],cfg["minimum_stocks_per_date"],i)
        preds.append(p);metrics.append(m);split_info.append(c);cal_rows.append(cal)
        roster += [{"split":i,"ticker":s,"role":"train"} for s in sorted(seen)]
        roster += [{"split":i,"ticker":s,"role":"held_out"} for s in sorted(unseen)]
    p,m = pd.concat(preds,ignore_index=True),pd.concat(metrics,ignore_index=True)
    _save(p,output,"heldout_predictions.csv")
    _save(m,output,"heldout_date_metrics.csv")
    _save(pd.DataFrame(split_info),output,"split_sizes.csv")
    _save(pd.DataFrame(cal_rows),output,"ret63_calibration_by_split.csv")
    _save(pd.DataFrame(roster),output,"frozen_identity_roster.csv")
    _save(m.groupby("model",as_index=False)[["mse","mae","spearman","top_decile_lift","top1_regret","between_mse","within_mse","cs_mae"]].mean(),output,"heldout_summary.csv")
    inferred=[]
    for metric in ["spearman","top_decile_lift","mae","within_mse"]:
        d = paired_date_difference(m,"cs_resid_ridge","state_ridge",metric)
        _save(d,output,f"paired_by_date_{metric}.csv")
        bt = block_bootstrap(d,cfg["bootstrap_block_months"],cfg["bootstrap_replicates"],cfg["seed"]+len(metric))
        bt["metric"]=metric;bt["contrast"]="cs_resid_ridge - state_ridge"
        inferred.append(bt)
    # Explicit forecast comparability test, calibrated Ret63 fitted on training dates only.
    for metric in ["mae","spearman"]:
        d=paired_date_difference(m,"state_ridge","ret63_calibrated",metric)
        bt=block_bootstrap(d,cfg["bootstrap_block_months"],cfg["bootstrap_replicates"],cfg["seed"]+123+len(metric))
        bt["metric"]=metric;bt["contrast"]="state_ridge - ret63_calibrated"
        inferred.append(bt)
    inf=pd.concat(inferred,ignore_index=True)
    _save(inf,output,"dependence_robust_intervals.csv")
    return m, inf


def factorial_2x2(df: pd.DataFrame, cfg: dict, output: Path):
    end=cfg["factorial_train_end"]
    eligible=sorted(df.loc[df.date <= pd.Timestamp(end),"ticker"].unique())
    seen,unseen=split_identities(eligible,cfg["test_identity_fraction"],cfg["seed"],1000)
    train=training_before(df,end,seen)
    bundle=fit_models(train,cfg["ridge_alpha"])
    cal = calibration_line(bundle)
    cal.update({"fit_kind":"factorial_fixed_roster","split":1000,
        "train_end_cutoff":str(pd.Timestamp(end).date()),
        "first_train_signal":str(train.date.min().date()),
        "last_train_signal":str(train.date.max().date()),
        "first_train_maturity":str(train.maturity_date.min().date()),
        "last_train_maturity":str(train.maturity_date.max().date()),
        "train_rows":len(train),"train_dates":train.date.nunique(),
        "train_identities":train.ticker.nunique(),
        "training_ret63_target":"future_clipped_QQQ_relative_log_return_y",
        "calibration_fitted_on":"matured_training_rows_only",
        "calibration_formula":"intercept + slope * unscaled_ret63",
        "test_outcomes_used_for_calibration":False})
    if train.maturity_date.max() > pd.Timestamp(end):
        raise AssertionError("Unmatured label reached factorial training")
    _save(pd.DataFrame([cal]),output,"ret63_calibration_factorial.csv")
    all_m=[]; all_pred=[];counts=[]
    for time_name,start,stop in [("near",cfg["factorial_near_start"],cfg["factorial_near_end"]),
                                 ("future",cfg["factorial_far_start"],cfg["factorial_far_end"])]:
        for identity_name,subset in [("seen",seen),("unseen",unseen)]:
            eva=evaluate_window(df,start,stop,subset)
            p=predict(bundle,eva)
            m=date_metrics(p,cfg["minimum_stocks_per_date"])
            p["time_regime"]=time_name;p["identity_regime"]=identity_name
            m["time_regime"]=time_name;m["identity_regime"]=identity_name
            all_pred.append(p);all_m.append(m)
            counts.append({"time_regime":time_name,"identity_regime":identity_name,
                           "rows":len(p),"dates":m.date.nunique(),"identities":p.ticker.nunique()})
    mm=pd.concat(all_m,ignore_index=True);pp=pd.concat(all_pred,ignore_index=True)
    _save(mm,output,"factorial_date_metrics.csv")
    _save(pp,output,"factorial_predictions.csv")
    _save(pd.DataFrame(counts),output,"factorial_cell_sizes.csv")
    summ=mm.groupby(["time_regime","identity_regime","model"],as_index=False)[["mae","spearman","top_decile_lift","between_mse","within_mse"]].mean()
    _save(summ,output,"factorial_summary.csv")
    # Pair seen/unseen *within calendar month*. Then regime contrast, not
    # causal DiD; fixed roster and distinct asset populations limit meaning.
    for metric in ["spearman", "top_decile_lift", "mae"]:
        paired = factorial_paired_contrasts(mm, metric=metric)
        _save(paired, output, f"factorial_paired_date_{metric}.csv")
        inter = factorial_interaction_bootstrap(paired,cfg["bootstrap_block_months"],
                           cfg["bootstrap_replicates"],cfg["seed"]+791+len(metric))
        inter["metric"]=metric
        _save(inter, output, f"factorial_interaction_{metric}.csv")
    # Descriptive identity effects cannot be causal, particularly since
    # constituent counts and relative rankings change across subuniverses.
    result={"factorial_training_end":end,"seen_ids":len(seen),"never_trained_ids":len(unseen),
            "interpretation":"Fixed-roster associational 2x2; not causal or newly-listed-identity effect.",
            "interaction_estimand":"(future unseen - future seen) - (near unseen - near seen) for paired CS-State date-level metric",
            "bootstrap":"within-regime independent calendar blocks (cross-boundary dependence ignored)",
            "not_independent_confirmation":True}
    (output/"factorial_protocol.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return summ


def run(config_path: str | Path, data_path: str | Path, calendar_path: str | Path, out_path: str | Path):
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    out=Path(out_path);out.mkdir(parents=True,exist_ok=True)
    df=with_label_maturity(load_panel(data_path),calendar_path,int(cfg["horizon"]))
    row_audit=audit(df)
    (out/"data_audit.json").write_text(json.dumps(row_audit,indent=2),encoding="utf-8")
    m, intervals=frozen_diagnostic_splits(df,cfg,out)
    factorial=factorial_2x2(df,cfg,out)
    manifest={"analysis_status":cfg["analysis_status"],
              "message":"New analyses on historical reused dataset; do NOT advertise as independent confirmation.",
              "config":cfg,"data_panel_sha256":file_sha256(data_path),
              "trading_calendar_sha256":file_sha256(calendar_path),
              "config_sha256":file_sha256(config_path),"audit":row_audit,
              "software":"fresh identity-transfer-audit implementation, no historical modules imported",
              "output_definition":"Date-averaged paired model contrasts; 5000-rep circular block percentile exploratory CI; null-centered Monte Carlo p only with >=4 nominal blocks",
              "inference_warning":"24 future months; near 12 months; long blocks only descriptive; within-regime resampling does not capture cross-regime boundary dependence"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    make_plots(m, factorial, out)
    (out/"INTERPRETATION.txt").write_text(
        "POST-HOC ROBUSTNESS, NOT NEW INDEPENDENT CONFIRMATION.\n"
        "Report all results including negative outcomes.\n"
        "Block confidence intervals depend on date-block choice and do not fix survivorship or vendor-revision bias.\n"
        "MAE comparison against raw Ret63 is not valid; use ret63_calibrated.\n"
        "Bootstrap p is exploratory null-centered resampling with plus-one correction, not formal proof.\n"
        "Blocks yielding <4 nominal nonoverlapping blocks are DESCRIPTIVE ONLY: no p-value.\n"
        "2x2 interaction is an associational fixed-roster contrast, not causal DiD; near=12 months.\n"
        "No execution-level alpha or investment performance is measured here.\n",encoding="utf-8")
    return {"out":str(out),"dates":row_audit["dates"],"identities":row_audit["identities"],
            "intervals":len(intervals),"factorial_cells":len(factorial)}


def make_plots(m:pd.DataFrame,factorial:pd.DataFrame,out:Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.2))
    summ=m.groupby("model")[["mae","spearman"]].mean()
    for name in ["state_ridge","cs_resid_ridge","ret63_calibrated"]:
        ax.scatter(summ.loc[name,"mae"],summ.loc[name,"spearman"],s=65)
        ax.annotate(name,(summ.loc[name,"mae"],summ.loc[name,"spearman"]),xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.set_xlabel("Mean date-level MAE (calibrated forecast scale)")
    ax.set_ylabel("Mean date-level Spearman")
    ax.set_title("Prediction accuracy versus same-date ordering")
    ax.axhline(0,linewidth=.7);fig.tight_layout();fig.savefig(out/"prediction_ranking_scatter.png",dpi=160);plt.close(fig)
    sub=factorial[factorial.model=="cs_resid_ridge"]
    fig,ax=plt.subplots(figsize=(7,4))
    for j,t in enumerate(["near","future"]):
        for k,identity in enumerate(["seen","unseen"]):
            row=sub[(sub.time_regime==t)&(sub.identity_regime==identity)]
            if not row.empty:
                ax.bar(j*3+k,float(row.spearman.iloc[0]),width=.8)
    ax.set_xticks([0,1,3,4], ["near/seen","near/unseen","future/seen","future/unseen"],rotation=15)
    ax.set_ylabel("Mean date-level Spearman")
    ax.set_title("2x2 shift diagnostic (descriptive)")
    ax.axhline(0,linewidth=.7);fig.tight_layout();fig.savefig(out/"factorial_spearman.png",dpi=160);plt.close(fig)
