"""Command line entrypoint. All writes go to user-specified output directory."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .data import audit,load_panel,with_label_maturity,file_sha256
from .experiments import run
from .synthetic import sanity

def main(argv=None):
    p=argparse.ArgumentParser(prog="identity-audit",description="Post-hoc ranking transfer diagnostics")
    s=p.add_subparsers(dest="command",required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument("--data",default="data/asset_panel_h21.csv.gz")
    common.add_argument("--calendar",default="data/market_sessions.csv")
    common.add_argument("--config",default="configs/primary.json")
    q=s.add_parser("audit",parents=[common],help="Check data schema and maturity dates")
    r=s.add_parser("run",parents=[common],help="Run paired diagnostics + 2x2 + inference")
    r.add_argument("--out",default="runs/full_run")
    t=s.add_parser("synthetic",help="Run falsification demonstration with synthetic data")
    t.add_argument("--seed",type=int,default=20261020)
    args=p.parse_args(argv)
    if args.command=="synthetic":
        print(json.dumps(sanity(args.seed),indent=2));return 0
    cfg=json.loads(Path(args.config).read_text())
    if args.command=="audit":
        a=with_label_maturity(load_panel(args.data),args.calendar,int(cfg["horizon"]))
        print(json.dumps({**audit(a),"data_sha256":file_sha256(args.data)},indent=2));return 0
    if args.command=="run":
        print(json.dumps(run(args.config,args.data,args.calendar,args.out),indent=2));return 0
    return 2

if __name__=="__main__":
    raise SystemExit(main())
