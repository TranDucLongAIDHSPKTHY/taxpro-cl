"""Sequential runner for the confirmatory evaluation (docs/confirmatory_protocol.md,
Sections 5-6, with Deviations 1-2) on the selected held-out dataset.

Every run sets TAXPRO_DEFER_TEST=1, so no test metric is computed here. Selections
use validation Recall@20 only: three-seed means at each run's best-validation epoch
(the first epoch with the highest validation Recall@20, as in batch_test.py).

Order (Deviation 1): Step 1 (taxonomy policy), Step 2 (TaxPro-CL grid, rule R),
the tuned-SimGCL grid of Step 3 (rule R), Step 4 (factorial V0-V2); then
confirmatory_selection.json and its SHA-256 are written; then the four remaining
default baselines. With --then-a7 the A7 grid runner is resumed afterwards, so the
GPU is used sequentially without idle time.

Usage:
    python -m tools.experiments.run_confirmatory --dataset office-products [--then-a7] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 0, 1)
GROUPS = ("near_cold", "long_tail", "warm", "overall")
POLICIES = ("no_merge", "merge_t5", "merge_t10", "merge_t15")
TAX_TEMPS = (0.10, 0.125, 0.15)
TAX_GAMMA = (1.5, 5.0)
SIM_TEMPS = (0.05, 0.10, 0.15, 0.20)
SIM_EPS = (0.05, 0.10, 0.20)

COMMON = {"dataset_path": "./dataset_verify/",
          "evaluation_protocol_path": "./preprocessed/evaluation_protocol/split_seed_42/",
          "top_K": "[10, 20]", "selection_K": "20", "embedding_size": "64", "num_worker": "4",
          "learn_rate": "0.001", "reg_lambda": "0.0001", "GCN_layer": "3", "sparsity_test": "0"}
# Section 5: the final ACS configuration (manuscript Table 6); checked against the ACS
# main-run manifest at start-up.
TAXPRO_BASE = {**COMMON, "training_epochs": "200", "early_stopping": "20", "interval": "1", "batch_size": "2048",
               "test_batch_size": "2048", "warm_start_epochs": "20", "mu": "0.9", "epsilon_max": "0.2",
               "delta": "1e-8", "temperature": "0.125", "ssl_lambda": "0.5", "taxonomy_policy": "no_merge",
               "prototype_mode": "leaf", "prototype_update": "ema", "augmentation_direction": "taxonomy",
               "unknown_taxonomy_policy": "exclude_cl", "symmetric_info_nce": "false",
               "use_adaptive_epsilon": "true", "use_leaf_aware_infonce": "true", "direction_source": "prototype",
               "gamma_cold": "1.5", "gamma_warm": "1.0", "same_leaf_weight": "0.0", "degree_near_cold_max": "5",
               "degree_long_tail_max": "10", "use_user_ssl": "true", "epsilon_user": "0.05",
               "ssl_lambda_user": "0.5", "temperature_user": "0.2", "isotropic_blend": "0.0"}
# Framework defaults with the training budgets of manuscript Table 5 (as run on ACS).
BASELINES = {
    "SimGCL": {**COMMON, "training_epochs": "200", "early_stopping": "20", "interval": "1", "batch_size": "2048",
               "test_batch_size": "2048", "ssl_lambda": "0.5", "temperature": "0.2", "epsilon": "0.05"},
    "LightGCN": {**COMMON, "training_epochs": "1000", "early_stopping": "10", "interval": "10",
                 "batch_size": "1024", "test_batch_size": "1024"},
    "SGL": {**COMMON, "training_epochs": "200", "early_stopping": "20", "interval": "1", "batch_size": "2048",
            "test_batch_size": "200", "ssl_lambda": "0.1", "ssl_ratio": "0.1", "aug_type": "ed",
            "temperature": "0.2"},
    "XSimGCL": {**COMMON, "training_epochs": "200", "early_stopping": "30", "interval": "1", "batch_size": "2048",
                "test_batch_size": "2048", "ssl_lambda": "0.2", "temperature": "0.15", "epsilon": "0.2",
                "cl_layer": "1"},
    "NCL": {**COMMON, "training_epochs": "500", "early_stopping": "20", "interval": "1", "batch_size": "2048",
            "test_batch_size": "2048", "ssl_lambda": "1e-6", "proto_lambda": "1e-7", "temperature": "0.05",
            "cl_layer": "1", "alpha": "1.5", "k": "2000"},
}
ACS_MANIFEST_DIRS = {
    "TaxPro-CL": "log/p0/taxprocl/arts-crafts-and-sewing/taxpro-cl-FINAL-no_merge-temp0.125-gammacold1.5-sameleaf0",
    "SimGCL": "log/p0/baseline/SimGCL/arts-crafts-and-sewing/simgcl-c8b32441f415",
    "LightGCN": "log/p0/baseline/LightGCN/arts-crafts-and-sewing/lightgcn-43de3b4b3ff5",
    "SGL": "log/p0/baseline/SGL/arts-crafts-and-sewing/sgl-9cd8e529a74c",
    "XSimGCL": "log/p0/baseline/XSimGCL/arts-crafts-and-sewing/xsimgcl-fd0112231ebb",
    "NCL": "log/p0/baseline/NCL/arts-crafts-and-sewing/ncl-0f7e1a630c06",
}


class Runner:
    def __init__(self, dataset, device, dry_run):
        self.dataset, self.device, self.dry_run = dataset, device, dry_run
        self.base = ROOT / "log" / "p0" / "confirmatory" / dataset
        self.progress = self.base / "progress.log"
        self.failed = False

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        if not self.dry_run:
            self.base.mkdir(parents=True, exist_ok=True)
            with self.progress.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    @staticmethod
    def status(run_dir):
        try:
            return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError):
            return None

    def run(self, model, cell, config, seed):
        run_dir = self.base / model / cell / f"seed{seed}"
        if self.status(run_dir) == "completed":
            return run_dir
        if self.dry_run:
            print(f"  would run {model} {cell} seed{seed}")
            return run_dir
        cmd = [sys.executable, "main.py", "--model", model, "--seed", str(seed), "--device", self.device]
        for key, value in {**config, "dataset": self.dataset}.items():
            cmd += ["--" + key, str(value)]
        if self.status(run_dir) == "running" and (run_dir / "last_model.pt").exists():
            cmd += ["--resume_checkpoint", str((run_dir / "last_model.pt").relative_to(ROOT))]
            self.log(f"RESUME {model} {cell} seed{seed}")
        else:
            self.log(f"START {model} {cell} seed{seed}")
        run_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, TAXPRO_DEFER_TEST="1", TAXPROCL_RUN_OUTPUT_DIR=str(run_dir.relative_to(ROOT)))
        t0 = time.time()
        for attempt in (1, 2):
            rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
            if rc == 0 and self.status(run_dir) == "completed":
                break
            self.log(f"FAILED attempt {attempt} (exit {rc}) {model} {cell} seed{seed}")
            if attempt == 1 and (run_dir / "last_model.pt").exists() and "--resume_checkpoint" not in cmd:
                cmd += ["--resume_checkpoint", str((run_dir / "last_model.pt").relative_to(ROOT))]
        else:
            raise RuntimeError(f"run failed twice: {model} {cell} seed{seed}")
        for name in ("final_test_metrics.json", "final_test_group_metrics.json"):
            if (run_dir / name).exists():
                raise RuntimeError(f"test metrics were written despite TAXPRO_DEFER_TEST: {run_dir}")
        self.log(f"DONE ({time.time() - t0:.0f}s) {model} {cell} seed{seed}")
        return run_dir

    def cell(self, model, cell, config):
        dirs = [self.run(model, cell, config, s) for s in SEEDS]
        return None if self.dry_run else validation_means(dirs)


def best_validation(run_dir):
    history = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    best = None
    for entry in history:  # first epoch with the highest primary value, as batch_test.py
        if best is None or entry["primary_value"] > best["primary_value"]:
            best = entry
    g = best["metrics"]["groupwise"]
    return {k: float(g[k]["recall"]["20"]) for k in GROUPS}


def validation_means(dirs):
    per_seed = [best_validation(d) for d in dirs]
    return {"mean": {g: statistics.fmean(p[g] for p in per_seed) for g in GROUPS},
            "sd": {g: statistics.stdev(p[g] for p in per_seed) for g in GROUPS},
            "per_seed": per_seed, "run_dirs": [str(d.relative_to(ROOT)) for d in dirs]}


def rule_r(cells):
    """cells: ordered list of (name, means). Overall within 2% of the best, then highest LT; ties to the first."""
    top = max(m["mean"]["overall"] for _, m in cells)
    eligible = [(n, m) for n, m in cells if m["mean"]["overall"] >= 0.98 * top]
    best = eligible[0]
    for n, m in eligible[1:]:
        if m["mean"]["long_tail"] > best[1]["mean"]["long_tail"]:
            best = (n, m)
    return best[0]


def rule_ov(cells):
    """Deviation 2 sensitivity rule: highest validation Overall; ties to the first."""
    best = cells[0]
    for n, m in cells[1:]:
        if m["mean"]["overall"] > best[1]["mean"]["overall"]:
            best = (n, m)
    return best[0]


def policy_rule(cells):
    """Manuscript Section 4.1: nominal top scorer on validation Overall; no_merge if within the top's seed SD."""
    top_name, top = max(cells, key=lambda c: c[1]["mean"]["overall"])
    nm = dict(cells)["no_merge"]
    if top_name != "no_merge" and top["mean"]["overall"] - nm["mean"]["overall"] < top["sd"]["overall"]:
        return "no_merge"
    return top_name


def check_configs():
    for model, rel in ACS_MANIFEST_DIRS.items():
        cfg = json.loads((ROOT / rel / "seed42" / "run_manifest.json").read_text(encoding="utf-8"))["configuration"]
        cfg = {k: v for k, v in cfg.items() if k not in ("dataset", "resume_checkpoint")}
        ours = TAXPRO_BASE if model == "TaxPro-CL" else BASELINES[model]
        assert cfg == ours, (model, set(cfg.items()) ^ set(ours.items()))


def taxpro_cfg(policy, temp, gamma, direction="taxonomy", adaptive="true"):
    return {**TAXPRO_BASE, "taxonomy_policy": policy, "temperature": str(temp), "gamma_cold": str(gamma),
            "augmentation_direction": direction, "use_adaptive_epsilon": adaptive}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--then-a7", action="store_true")
    args = ap.parse_args(argv)
    check_configs()
    r = Runner(args.dataset, args.device, args.dry_run)
    sel_path = r.base / "confirmatory_selection.json"
    try:
        # Step 1: taxonomy policy
        step1 = [(p, r.cell("TaxPro-CL", f"step1-{p}", taxpro_cfg(p, 0.125, 1.5))) for p in POLICIES]
        policy = "no_merge" if args.dry_run else policy_rule(step1)
        r.log(f"Step 1 selected policy: {policy}")
        # Step 2: TaxPro-CL grid under the selected policy (base cell reused from Step 1)
        step2 = []
        for t in TAX_TEMPS:
            for g in TAX_GAMMA:
                if (t, g) == (0.125, 1.5):
                    step2.append((f"temp{t}-gamma{g}", dict(step1)[policy]))
                else:
                    step2.append((f"temp{t}-gamma{g}", r.cell("TaxPro-CL", f"step2-{policy}-temp{t}-gamma{g}",
                                                               taxpro_cfg(policy, t, g))))
        tax_cell = step2[0][0] if args.dry_run else rule_r(step2)
        r.log(f"Step 2 selected TaxPro-CL cell: {tax_cell}")
        t_sel, g_sel = (float(x) for x in tax_cell.replace("temp", "").split("-gamma"))
        # Step 3a: tuned SimGCL grid
        step3 = []
        for t in SIM_TEMPS:
            for e in SIM_EPS:
                cfg = {**BASELINES["SimGCL"], "temperature": str(t), "epsilon": str(e)}
                step3.append((f"temp{t}-eps{e}", r.cell("SimGCL", f"grid-temp{t}-eps{e}", cfg)))
        sim_cell = step3[0][0] if args.dry_run else rule_r(step3)
        r.log(f"Step 3 selected tuned-SimGCL cell: {sim_cell}")
        # Step 4: factorial at the selected configuration and policy (V3 = selected cell)
        fact = {}
        for name, direction, adaptive in (("V0", "random", "false"), ("V1", "taxonomy", "false"),
                                          ("V2", "random", "true")):
            fact[name] = r.cell("TaxPro-CL", f"step4-{name}-{policy}-temp{t_sel}-gamma{g_sel}",
                                taxpro_cfg(policy, t_sel, g_sel, direction, adaptive))
        if not args.dry_run:
            selection = {
                "protocol": "docs/confirmatory_protocol.md (with Deviations 1-2)", "dataset": args.dataset,
                "rule_R": "Overall within 2% of the best validation Overall, then highest validation Long-Tail",
                "step1_policy": {"selected": policy, "cells": dict(step1)},
                "step2_taxpro": {"selected": tax_cell, "selected_rule_ov_sensitivity": rule_ov(step2),
                                 "cells": dict(step2)},
                "step3_tuned_simgcl": {"selected": sim_cell, "selected_rule_ov_sensitivity": rule_ov(step3),
                                       "default_cell": "temp0.2-eps0.05", "cells": dict(step3)},
                "step4_factorial": {**fact, "V3": f"= step2 selected cell {tax_cell}"},
                "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if not sel_path.exists():
                sel_path.write_bytes((json.dumps(selection, indent=2) + "\n").encode("utf-8"))
                digest = hashlib.sha256(sel_path.read_bytes()).hexdigest()
                sel_path.with_suffix(".sha256").write_bytes(f"{digest}  confirmatory_selection.json\n".encode())
                r.log(f"Selection frozen: {sel_path} sha256={digest}")
        # Remaining default baselines (SimGCL's default is a grid cell)
        for model in ("LightGCN", "SGL", "XSimGCL", "NCL"):
            r.cell(model, "default", BASELINES[model])
        r.log("Confirmatory runs complete; test split still sealed")
    except Exception as exc:  # keep the GPU busy with A7 even if the protocol stops
        r.log(f"STOPPED: {exc!r}")
        r.failed = True
    if args.then_a7 and not args.dry_run:
        r.log("Resuming the A7 grid runner")
        subprocess.run([sys.executable, "-m", "tools.experiments.run_a7_full_grid", "--device", args.device], cwd=ROOT)
    return 1 if r.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
