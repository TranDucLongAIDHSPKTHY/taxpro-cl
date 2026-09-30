"""Add the held-out (Office-Products) runs whose test split was opened to results_manifest.csv
and their per-seed test metrics to metrics_seed.csv (main paper Table 12; Online Resource 1,
Section S36). Idempotent: rows already present (same run_id) are replaced, never duplicated.

Runs: the selected TaxPro-CL cell, the factorial controls V0-V2, the SimGCL cells selected by
rule R and rule Ov, the default SimGCL cell, and the four other default baselines; seeds 42, 0, 1.

Usage: python -m tools.analysis.update_manifest_confirmatory
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = "office-products"
BASE = ROOT / "log" / "p0" / "confirmatory" / DATASET
MANIFEST = ROOT / "results" / "results_manifest.csv"
METRICS = ROOT / "results" / "metrics_seed.csv"
SEEDS = (42, 0, 1)
GROUPS = ("near_cold", "long_tail", "warm", "overall")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runs():
    sel = json.loads((BASE / "confirmatory_selection.json").read_text(encoding="utf-8"))
    pol, tax = sel["step1_policy"]["selected"], sel["step2_taxpro"]["selected"]
    sim = sel["step3_tuned_simgcl"]
    tables = "Table 12; Online Resource 1 Tables S36-S36c (held-out evaluation)"
    out = [("TaxPro-CL", "HO-TaxPro-CL-selected", BASE / "TaxPro-CL" / f"step2-{pol}-{tax}", tables)]
    for v in ("V0", "V1", "V2"):
        out.append(("TaxPro-CL", f"HO-{v}", BASE / "TaxPro-CL" / f"step4-{v}-{pol}-{tax}", "Online Resource 1 Tables S36b-S36c"))
    out += [("SimGCL", "HO-SimGCL-tuned-R", BASE / "SimGCL" / f"grid-{sim['selected']}", tables),
            ("SimGCL", "HO-SimGCL-tuned-Ov", BASE / "SimGCL" / f"grid-{sim['selected_rule_ov_sensitivity']}", tables),
            ("SimGCL", "HO-SimGCL-default", BASE / "SimGCL" / f"grid-{sim['default_cell']}", tables)]
    for model, d in (("LightGCN", "LightGCN"), ("SGL-ED", "SGL"), ("XSimGCL", "XSimGCL"), ("NCL", "NCL")):
        out.append((model, f"HO-{model}-default", BASE / d / "default", "Online Resource 1 Table S36b (descriptive)"))
    return out


def main():
    split_dir = ROOT / "dataset_verify" / DATASET
    split_sha = {s: sha256_of(split_dir / f"{s}.txt") for s in ("train", "validation", "test")}
    fields = list(csv.DictReader(open(MANIFEST, encoding="utf-8")).fieldnames)
    mrows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))]
    xrows = [r for r in csv.DictReader(open(METRICS, encoding="utf-8"))]
    xfields = list(csv.DictReader(open(METRICS, encoding="utf-8")).fieldnames)
    new_m, new_x = [], []
    for model, variant, fam, tables in runs():
        for seed in SEEDS:
            run_dir = fam / f"seed{seed}"
            man = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            cfg = man["configuration"]
            assert (run_dir / "final_test_group_metrics.json").exists(), run_dir
            pol = cfg.get("taxonomy_policy", "")
            tax_manifest = ROOT / "metadata" / "taxonomy_variants" / DATASET / pol / "manifest.json"
            run_id = f"{model}|{DATASET}|{variant}|seed{seed}"
            row = {f: "" for f in fields}
            row.update({
                "run_id": run_id, "model": model, "dataset": DATASET, "variant": variant, "seed": str(seed),
                "status": man.get("status", ""), "run_dir": str(run_dir.relative_to(ROOT)).replace("/", "\\"),
                "git_commit": man.get("environment", {}).get("git_commit", "") if isinstance(man.get("environment"), dict) else "",
                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                "train_split_sha256": split_sha["train"], "validation_split_sha256": split_sha["validation"],
                "test_split_sha256": split_sha["test"], "split_hash_source": "computed from dataset_verify/office-products",
                "taxonomy_hash": sha256_of(tax_manifest) if pol and tax_manifest.exists() else "",
                "checkpoint_sha256_best_validation_model": sha256_of(run_dir / "best_validation_model.pt"),
                "v3_is_same_checkpoint_as_taxprocl_main": "N/A",
                "selection_metric": "validation Recall@20 Overall",
                "test_policy": "sealed during training (TAXPRO_DEFER_TEST=1); opened once by evaluate_sealed_test.py",
                "completed_epochs": str(man.get("completed_epochs", "")), "duration_seconds": str(man.get("duration_seconds", "")),
                "temperature": str(cfg.get("temperature", "")), "temperature_user": str(cfg.get("temperature_user", "")),
                "epsilon_max": str(cfg.get("epsilon_max", cfg.get("epsilon", ""))),
                "warm_start_epochs": str(cfg.get("warm_start_epochs", "")),
                "augmentation_direction": str(cfg.get("augmentation_direction", "")),
                "use_adaptive_epsilon": str(cfg.get("use_adaptive_epsilon", "")), "taxonomy_policy": pol,
                "used_in_tables": tables,
            })
            new_m.append(row)
            g = json.loads((run_dir / "final_test_group_metrics.json").read_text(encoding="utf-8"))
            for grp in GROUPS:
                new_x.append({"run_id": run_id, "model": model, "dataset": DATASET, "variant": variant, "seed": str(seed),
                              "group": grp, "eligible_users": str(g[grp].get("eligible_users", "")),
                              "relevant_items": str(g[grp].get("relevant_items", "")),
                              "recall_at_10": repr(float(g[grp]["recall"]["10"])), "recall_at_20": repr(float(g[grp]["recall"]["20"])),
                              "ndcg_at_10": repr(float(g[grp]["ndcg"]["10"])), "ndcg_at_20": repr(float(g[grp]["ndcg"]["20"]))})
    ids = {r["run_id"] for r in new_m}
    mrows = [r for r in mrows if r["run_id"] not in ids] + new_m
    xrows = [r for r in xrows if r["run_id"] not in ids] + new_x
    for path, flds, rows in ((MANIFEST, fields, mrows), (METRICS, xfields, xrows)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=flds, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print(f"manifest: {len(new_m)} held-out rows (total {len(mrows)}); metrics_seed: {len(new_x)} rows (total {len(xrows)})")


if __name__ == "__main__":
    main()
