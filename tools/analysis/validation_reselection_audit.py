"""Retrospective validation-only re-selection of the TaxPro-CL configuration
(Online Resource 1, Section S35; main paper Section 4.3 and Limitation 10).

TaxPro-CL's configuration was chosen during development with test-split metrics in
view. This script asks, over the runs that were retained, which configuration a rule
that looks only at the validation split would have selected, and what the selected
configuration scores on the test split relative to the five baselines.

Candidates (per dataset): every retained TaxPro-CL run family with all three training
seeds (0, 1, 42) whose resolved configuration equals the dataset's main configuration
except in the hyperparameters that were tuned (main paper Table 4: temperature,
temperature_user, epsilon_max, gamma_cold, gamma_warm, warm_start_epochs,
taxonomy_policy). This keeps the method as specified in main paper Section 3 and
excludes the factorial controls, component variants (parent prototype, isotropic blend,
prototype-construction variants, same-leaf soft positives) and early-version runs. A
family whose tuned values equal the main configuration's counts as the main
configuration. A second pool, reported for comparison, also admits the component
variants.

Rules (validation split, best-validation checkpoint, three-seed means):
  overall      highest Overall Recall@20
  lt           highest Long-Tail Recall@20
  nc           highest Near-Cold Recall@20
  lt_within_x  highest Long-Tail Recall@20 among candidates whose Overall Recall@20 is
               within x% (x = 1, 2, 3) of the best candidate's
Ties are broken by the next criterion (Overall for lt, Long-Tail for nc).

The rules were fixed after every retained run's validation and test results were
known, so the output is descriptive: it shows how much the reported comparison depends
on the selection criterion, not a validation-selected estimate.

Inference only (reads each run's final_test_metrics.json and config_resolved.json).
Output: results/validation_reselection_audit.json; --latex prints Table S35 rows.

Usage:
    python -m tools.analysis.validation_reselection_audit [--latex]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "log" / "p0" / "taxprocl"
DATASETS = ["amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"]
SEEDS = ("seed0", "seed1", "seed42")
GROUPS = ("near_cold", "long_tail", "overall", "warm")
TUNED = ("temperature", "temperature_user", "epsilon_max", "gamma_cold", "gamma_warm",
         "warm_start_epochs", "taxonomy_policy")
# keys that do not define the method (bookkeeping, paths, evaluation plumbing)
IGNORED = {"seed", "dataset", "num_worker", "resume_checkpoint", "dataset_path",
           "evaluation_protocol_path", "log_dir", "output_dir", "run_name", "test_batch_size",
           "interval", "top_K", "selection_K", "sparsity_test", "mixture_alpha", "delta", "device"}
# component switches, used to admit variants in the comparison pool
COMPONENT_KEYS = {"prototype_mode", "isotropic_blend", "prototype_weighting", "prototype_leave_one_out",
                  "asymmetric_view_direction", "same_leaf_weight"}
# values that older runs did not write out; these equal the code defaults
DEFAULTS = {"augmentation_direction": "taxonomy", "prototype_weighting": "interaction",
            "prototype_leave_one_out": "false", "prototype_init_scope": "all_valid",
            "isotropic_blend": "0.0", "asymmetric_view_direction": "false", "same_leaf_weight": "0.0",
            "taxonomy_granularity": "leaf", "prototype_mode": "leaf", "direction_source": "prototype",
            "use_leaf_aware_infonce": "true", "symmetric_info_nce": "false"}
TOLERANCES = (1, 2, 3)
BASELINES = ("LightGCN", "SGL-ED", "SimGCL", "XSimGCL", "NCL")


def _flatten(d, out=None):
    out = {} if out is None else out
    for key, value in d.items():
        if isinstance(value, dict):
            _flatten(value, out)
        else:
            out[key] = value
    return out


def normalize_config(cfg):
    """Lower-cased string values, code defaults filled in, numbers in one canonical form."""
    c = {k: str(v).lower() for k, v in _flatten(cfg).items() if v is not None}
    for key, value in DEFAULTS.items():
        c.setdefault(key, value)
    for key in list(c):
        try:
            c[key] = repr(float(c[key]))
        except ValueError:
            pass
    return c


def differing_keys(config, reference):
    """Method-defining keys (not tuned, not ignored) on which config differs from reference."""
    keys = (set(config) | set(reference)) - set(TUNED) - IGNORED
    return sorted(k for k in keys if config.get(k) != reference.get(k))


def _recall20(block, group):
    return float(block["groupwise"][group]["recall"]["20"])


def load_families(dataset):
    """Every run family of a dataset with all three seeds: normalized config and per-seed metrics."""
    families = {}
    for metrics_file in (LOG / dataset).rglob("final_test_metrics.json"):
        seed_dir = metrics_file.parent
        if seed_dir.name in SEEDS:
            families.setdefault(seed_dir.parent, []).append(seed_dir)
    out = []
    for family_dir, seed_dirs in families.items():
        if {s.name for s in seed_dirs} != set(SEEDS):
            continue
        seed_dirs = sorted(seed_dirs, key=lambda s: SEEDS.index(s.name))
        configs = [normalize_config(json.loads((s / "config_resolved.json").read_text(encoding="utf-8")))
                   for s in seed_dirs]
        per_seed = []
        for s in seed_dirs:
            m = json.loads((s / "final_test_metrics.json").read_text(encoding="utf-8"))
            per_seed.append({"val": {g: _recall20(m["best_validation_metrics"], g) for g in GROUPS},
                             "test": {g: _recall20(m["test_metrics"], g) for g in GROUPS}})
        out.append({
            "family": str(family_dir.relative_to(LOG / dataset)).replace("\\", "/"),
            "config": configs[0],
            "val": {g: statistics.mean(p["val"][g] for p in per_seed) for g in GROUPS},
            "val_sd": {g: statistics.stdev(p["val"][g] for p in per_seed) for g in GROUPS},
            "test": {g: statistics.mean(p["test"][g] for p in per_seed) for g in GROUPS},
        })
    return out


def main_family_name(dataset):
    """The TaxPro-CL-main family of a dataset, as recorded in results/results_manifest.csv."""
    with open(ROOT / "results" / "results_manifest.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == "TaxPro-CL" and row["dataset"] == dataset and row["variant"] == "TaxPro-CL-main":
                parts = row["run_dir"].replace("\\", "/").split("/")
                return "/".join(parts[parts.index(dataset) + 1:-1])
    raise KeyError(dataset)


def build_pools(families, main_name):
    """Candidate pools: the method's own hyperparameter variants, and the same plus component variants."""
    main = next(f for f in families if f["family"] == main_name)
    ref = main["config"]
    method, with_components = [], []
    for fam in families:
        diff = differing_keys(fam["config"], ref)
        tuned = {k: fam["config"].get(k) for k in TUNED if fam["config"].get(k) != ref.get(k)}
        fam = dict(fam, tuned_difference=tuned, component_difference=diff)
        if fam["family"] != main_name and not tuned and not diff:
            continue  # an independent re-run of the main configuration; the main run represents it
        if not diff:
            method.append(fam)
            with_components.append(fam)
        elif set(diff) <= COMPONENT_KEYS:
            with_components.append(fam)
    return main, method, with_components


def select(candidates, rule):
    """Candidate chosen by one validation-only rule."""
    if rule == "overall":
        return max(candidates, key=lambda c: c["val"]["overall"])
    if rule == "lt":
        return max(candidates, key=lambda c: (c["val"]["long_tail"], c["val"]["overall"]))
    if rule == "nc":
        return max(candidates, key=lambda c: (c["val"]["near_cold"], c["val"]["long_tail"]))
    if rule.startswith("lt_within_"):
        tolerance = float(rule.rsplit("_", 1)[1]) / 100.0
        best = max(c["val"]["overall"] for c in candidates)
        eligible = [c for c in candidates if c["val"]["overall"] >= (1.0 - tolerance) * best]
        return max(eligible, key=lambda c: (c["val"]["long_tail"], c["val"]["overall"]))
    raise ValueError(rule)


RULES = ("overall", "lt", "nc") + tuple("lt_within_%d" % x for x in TOLERANCES)


def compare_with_baselines(test, table):
    """Percentage change against SimGCL and rank among the six methods (1 = highest mean)."""
    out = {}
    for group in ("near_cold", "long_tail", "overall"):
        simgcl = table["SimGCL"][group]["recall20_mean"]
        others = [table[b][group]["recall20_mean"] for b in BASELINES]
        out[group] = {
            "test_recall20": test[group],
            "pct_vs_simgcl": 100.0 * (test[group] / simgcl - 1.0) if simgcl else None,
            "rank_of_six": 1 + sum(1 for v in others if v > test[group]),
        }
    return out


def describe(fam, main_name):
    if fam["family"] == main_name:
        return "main configuration"
    parts = []
    for key, value in fam["tuned_difference"].items():
        parts.append("%s=%s" % (key, value))
    for key in fam["component_difference"]:
        parts.append("%s=%s" % (key, fam["config"].get(key)))
    return ", ".join(parts)


def run():
    table = json.loads((ROOT / "results" / "main_results_table.json").read_text(encoding="utf-8"))
    report = {}
    for dataset in DATASETS:
        main_name = main_family_name(dataset)
        families = load_families(dataset)
        main, method, with_components = build_pools(families, main_name)
        entry = {"main_family": main_name,
                 "candidates": [{"family": f["family"], "differs_from_main": describe(f, main_name),
                                 "val": f["val"], "val_sd": f["val_sd"], "test": f["test"]} for f in method],
                 "comparison_pool_extra": [f["family"] for f in with_components if f not in method],
                 "main": {"family": main_name,
                          "vs_baselines": compare_with_baselines(main["test"], table[dataset])},
                 "selections": {}, "selections_with_component_variants": {}}
        for pool_name, pool in (("selections", method), ("selections_with_component_variants", with_components)):
            for rule in RULES:
                chosen = select(pool, rule)
                entry[pool_name][rule] = {
                    "family": chosen["family"],
                    "is_main_configuration": chosen["family"] == main_name,
                    "configuration": describe(chosen, main_name),
                    "val": chosen["val"],
                    "vs_baselines": compare_with_baselines(chosen["test"], table[dataset]),
                }
        report[dataset] = entry
    return report


PRETTY = {"temperature": r"$\tau$", "temperature_user": r"$\tau_{\text{user}}$", "epsilon_max": r"$\epsilon_{\max}$",
          "gamma_cold": r"$\gamma_{\text{cold}}$", "gamma_warm": r"$\gamma_{\text{warm}}$",
          "warm_start_epochs": "warm-start", "taxonomy_policy": "policy"}


def pretty_configuration(text):
    """'temperature=0.2, warm_start_epochs=0.0' -> '$\\tau$=0.2, warm-start epochs=0' for the table."""
    if text == "main configuration":
        return text
    parts = []
    for item in text.split(", "):
        key, value = item.split("=", 1)
        try:
            number = float(value)
            value = str(int(number)) if number.is_integer() else "%g" % number
        except ValueError:
            value = r"\texttt{%s}" % value.replace("_", r"\_")
        parts.append("%s=%s" % (PRETTY.get(key, key.replace("_", r"\_")), value))
    return ", ".join(parts)


def latex_rows(report):
    names = {"amazon-book": "AB", "yelp2018": "Yelp", "musical-instruments": "MI", "arts-crafts-and-sewing": "ACS"}
    labels = {"overall": "Ov", "lt": "LT", "nc": "NC", "lt_within_1": "LT$|$1\\%", "lt_within_2": "LT$|$2\\%",
              "lt_within_3": "LT$|$3\\%"}
    lines = []
    for dataset, entry in report.items():
        groups = {}
        for rule, sel in entry["selections"].items():
            groups.setdefault(sel["family"], []).append(rule)
        rows = [("Reported", entry["main"]["vs_baselines"], "main configuration")]
        for family, rules in groups.items():
            sel = entry["selections"][rules[0]]
            rule_text = "all six" if len(rules) == len(RULES) else ", ".join(labels[r] for r in rules)
            rows.append((rule_text, sel["vs_baselines"], sel["configuration"]))
        for rule_text, vs, config in rows:
            cells = []
            for group in ("near_cold", "long_tail", "overall"):
                pct = vs[group]["pct_vs_simgcl"]
                cells.append(("%+.2f\\%%" % pct).replace("-", "$-$"))
            lines.append("%s & %s & %s & %s & %s & %s & %d & %d \\\\" % (
                names[dataset], rule_text, pretty_configuration(config), *cells,
                vs["near_cold"]["rank_of_six"], vs["long_tail"]["rank_of_six"]))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "validation_reselection_audit.json")
    parser.add_argument("--latex", action="store_true", help="print the rows of Online Resource 1, Table S35")
    args = parser.parse_args()
    report = run()
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", args.out)
    if args.latex:
        print("\n".join(latex_rows(report)))


if __name__ == "__main__":
    main()
