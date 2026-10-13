#!/usr/bin/env python3
"""Run every experiment of the paper with its settings (configs/paper_settings.json) and write the report.

  python reproduce.py --raw-dir data/tubes --site-map site_map.csv --out results

Steps (each one runs as its own process with its own log in <out>/logs; a failing step is reported and the next
one still runs; finished fold runs are reused, so after an interruption simply start the command again):
  1 data-quality audit and exclusion list                         scripts/data_audit.py         -> <out>/audit
  2 input B2, leave one tube out: proposed models, references,
    federated baselines, privacy, ablation and controls          scripts/failrisk.py           -> <out>/B2/loo
  3 input B2, leave one hospital out                              scripts/failrisk.py           -> <out>/B2/loho
  4 input A2 (without failure precursors), both protocols         scripts/failrisk.py           -> <out>/A2/...
  5 DTG weights from truncated logs and under DP (no training)    scripts/failrisk.py --analysis-only
  6 computational and communication cost                          scripts/failrisk_complexity.py
  7 comparison report with every table                            scripts/make_report.py        -> <out>/comparison
  8 convergence figure and the other figures                      scripts/plot_convergence.py, scripts/failrisk_figures.py
If the audit excludes tubes, steps 2-4 are also run without them (<out>/B2_incl, <out>/A2_incl), and the report uses
those runs as the primary analysis. --device auto (default) uses the first CUDA GPU if PyTorch sees one, otherwise
the CPU; --workers auto runs 3 fold processes on a GPU and 1 on a CPU (same results).
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True, help="one CSV per failed tube (device_*.csv, from scripts/convert_tua.py)")
    ap.add_argument("--site-map", required=True, help="CSV with columns tube, hospital (see site_map_template.csv)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "paper_settings.json"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", default="auto", help="parallel fold processes ('auto': 3 on a GPU, 1 on a CPU)")
    ap.add_argument("--steps", default="audit,B2_loo,B2_loho,A2,stability,complexity,report,figures")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    tr = cfg["training"]
    out = os.path.abspath(a.out)
    logs = os.path.join(out, "logs")
    os.makedirs(logs, exist_ok=True)
    if a.workers == "auto":
        try:
            import torch
            workers = 3 if (a.device != "cpu" and torch.cuda.is_available()) else 1
        except Exception:
            workers = 1
    else:
        workers = max(1, int(a.workers))
    steps = {s.strip() for s in a.steps.split(",") if s.strip()}
    py = sys.executable
    status = []

    def step(name, args):
        t0 = dt.datetime.now()
        print(f"=== {name}  (started {t0:%Y-%m-%d %H:%M}) ===", flush=True)
        code = -1
        try:
            with open(os.path.join(logs, f"{name}.log"), "w", encoding="utf-8", errors="replace") as fh:
                proc = subprocess.Popen([py, "-u"] + args, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace",
                                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                for line in proc.stdout:
                    fh.write(line)
                    fh.flush()
                    print(line, end="", flush=True)
                code = proc.wait()
        except Exception as e:
            print(f"step {name} could not run: {type(e).__name__}: {e}", flush=True)
        mins = (dt.datetime.now() - t0).total_seconds() / 60
        state = "ok" if code == 0 else f"FAILED (exit code {code})"
        print(f"=== {name} finished in {mins:.0f} min: {state} ===", flush=True)
        status.append(f"{name}: {state} ({mins:.0f} min)")
        return code == 0

    cache = os.path.join(out, "cache")
    base = ["scripts/failrisk.py", "--raw-dir", a.raw_dir, "--cache", cache, "--skip-existing",
            "--seeds", ",".join(str(s) for s in tr["seeds"]), "--seq-days", str(tr["seq_days"]),
            "--batch", str(tr["batch"]), "--lr", str(tr["lr"]), "--local-epochs", str(tr["local_epochs"]),
            "--rounds", str(tr["max_rounds"]), "--min-rounds", str(tr["min_rounds"]), "--patience", str(tr["patience"]),
            "--val-tubes", str(tr["validation_tubes"]), "--lam-mono", str(tr["lambda_mono"]),
            "--lam-reg", str(tr["lambda_reg_default"]), "--tune-reg", ",".join(str(x) for x in tr["lambda_reg_grid"]),
            "--tune-tau", ",".join(str(x) for x in tr["dtg2_tau_grid"]), "--min-life", str(cfg["data"]["min_life_scans"]),
            "--min-dtf", str(cfg["data"]["min_days_before_failure"]), "--mu", str(cfg["baselines"]["fedprox_mu"]),
            "--scaffold-lr", str(cfg["baselines"]["scaffold_lr"]), "--cfl-warmup", str(cfg["baselines"]["cfl_warmup_rounds"]),
            "--cdp-clip", str(cfg["privacy"]["dp_fedavg_clip"]), "--dp-rclip", str(cfg["privacy"]["trend_rate_clip"]),
            "--dp-lag", str(cfg["privacy"]["trend_lag_records"]), "--bootstrap", str(cfg["evaluation"]["bootstrap"]),
            "--device", a.device, "--workers", str(workers)]
    main_v = ",".join(cfg["variants"]["main"])
    all_v = ",".join(cfg["variants"]["main"] + cfg["variants"]["supplementary"])

    audit = os.path.join(out, "audit")
    if "audit" in steps:
        step("1_data_audit", ["scripts/data_audit.py", "--raw-dir", a.raw_dir, "--site-map", a.site_map,
                              "--min-life", str(cfg["data"]["min_life_scans"]), "--out", audit])
    excl = os.path.join(audit, "exclusion.csv")
    n_ex = 0
    if os.path.exists(excl):
        import pandas as pd
        E = pd.read_csv(excl)
        n_ex = int(E.excluded.astype(str).str.lower().isin(["true", "1"]).sum())
    print(f"tubes excluded by the data-quality criteria: {n_ex}", flush=True)
    sets = [("", [])] + ([("_incl", ["--exclude-file", excl])] if n_ex else [])
    for suffix, ex in sets:
        b2 = os.path.join(out, f"B2{suffix}")
        a2 = os.path.join(out, f"A2{suffix}")
        if "B2_loo" in steps:
            step(f"2_B2{suffix}_loo", base + ex + ["--input", "daily-B2", "--variants", all_v,
                                                   "--out", os.path.join(b2, "loo")])
        if "B2_loho" in steps:
            step(f"3_B2{suffix}_loho", base + ex + ["--input", "daily-B2", "--variants", main_v, "--protocol", "loho",
                                                    "--site-map", a.site_map, "--out", os.path.join(b2, "loho")])
        if "A2" in steps:
            step(f"4_A2{suffix}_loo", base + ex + ["--input", "daily-A2", "--variants", main_v,
                                                   "--out", os.path.join(a2, "loo")])
            step(f"4_A2{suffix}_loho", base + ex + ["--input", "daily-A2", "--variants", main_v, "--protocol", "loho",
                                                    "--site-map", a.site_map, "--out", os.path.join(a2, "loho")])
    key = "B2_incl" if n_ex else "B2"
    loo = os.path.join(out, key, "loo")
    if "stability" in steps:
        step("5_dtg_stability", base + (["--exclude-file", excl] if n_ex else []) +
             ["--input", "daily-B2", "--variants", main_v, "--out", loo, "--analysis-only",
              "--dp-draws", str(cfg["privacy"]["trend_dp_draws"])])
    comp = os.path.join(out, "comparison")
    if "complexity" in steps:
        step("6_complexity", ["scripts/failrisk_complexity.py", "--run", loo, "--cache", cache, "--out", comp,
                              "--device", a.device])
    if "report" in steps:
        step("7_report", ["scripts/make_report.py", "--root", out, "--bootstrap", str(cfg["evaluation"]["bootstrap"])])
    if "figures" in steps:
        figs = os.path.join(out, key, "figures")
        step("8_convergence", ["scripts/plot_convergence.py", "--loo", loo, "--out", figs])
        fa = ["scripts/failrisk_figures.py", "--loo", loo, "--out", figs, "--site-map", a.site_map]
        loho = os.path.join(out, key, "loho")
        if os.path.exists(os.path.join(loho, "summary.csv")):
            fa += ["--loho", loho]
        step("8_figures", fa)
    with open(os.path.join(logs, "ALL_DONE.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"finished {dt.datetime.now():%Y-%m-%d %H:%M}\n" + "\n".join(status) + "\n")
    print("\n".join(status))
    print(f"report: {os.path.join(comp, 'comparison_report.txt')}")


if __name__ == "__main__":
    main()
