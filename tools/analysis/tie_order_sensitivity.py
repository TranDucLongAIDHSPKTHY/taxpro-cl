"""Sensitivity of the per-user NDCG@20 analyses to the order of tied scores (Online Resource 1, Section S13e).

Test scores are sigmoid outputs, which saturate at 1.0 in float32, so a user's highest-scored items can tie. The
stored evaluation takes the top 20 with torch.topk, and the per-user analyses reproduce that order (head_k in
tools/ranking/inference.py); PyTorch does not guarantee the order of tied values in torch.topk. This script

  1. counts, for every checkpoint it scores, the users whose top 20 contains tied scores and the users with a tie
     between positions 20 and 21; only the latter could change the top-20 set, and hence Recall@20;
  2. repeats the per-user analyses with a deterministic order: the pre-sigmoid scores (float64, no
     saturation), ties by item index (stable sort). The analyses are the RQ1 factorial and its Holm families
     (factorial_multiplicity), the parent prototype (a5_leaf_vs_parent_mergedt10), the prototype-construction
     variants (prototype_variants_ndcg), the Office held-out analysis (confirmatory_analysis), and the label and
     gradient-path controls of the extended evaluation (extended_analysis.analyse_controls; Section S41), whose
     opened checkpoints are re-scored, not re-opened for any selection;
  3. compares every Recall@20 and NDCG@20 decision of these analyses between the two orders: whether the interval
     excludes zero (and its sign), whether p < 0.05, the Holm decisions (8-cell families and the pooled 32-cell
     family of the factorial; the Office families; the control families), and the Office endpoint label.

Inference on saved checkpoints only. Writes results/tie_order/*.json and results/tie_order_sensitivity.json.
Usage: python -m tools.analysis.tie_order_sensitivity [--parts factorial a5 variants office extended]
(--parts re-runs the named analyses; the comparison always reads every part's output, and the tie counts of all
parts are kept in results/tie_order/ties_by_checkpoint.json.)
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.analysis.a5_leaf_vs_parent_mergedt10 as a5  # noqa: E402
import tools.analysis.confirmatory_analysis as ca  # noqa: E402
import tools.analysis.extended_analysis as ea  # noqa: E402
import tools.analysis.factorial_amazon_book as fab  # noqa: E402
import tools.analysis.factorial_multiplicity as fm  # noqa: E402
import tools.analysis.prototype_variants_ndcg as pvn  # noqa: E402
from tools.analysis.export_factorial_pvalues import holm_adjusted  # noqa: E402
import tools.experiments.evaluate_sealed_test as est  # noqa: E402
from tools.ranking import inference  # noqa: E402

K = 20
OUT_DIR = ROOT / "results" / "tie_order"
SUMMARY = ROOT / "results" / "tie_order_sensitivity.json"
TIES_FILE = OUT_DIR / "ties_by_checkpoint.json"
PARTS = ("factorial", "a5", "variants", "office", "extended")
BATCH = 256

# run directory -> {"users": set, "tie_within_top20": set, "tie_at_20_21": set, "logit_tie_within_top21": set}
TIES: dict[str, dict[str, set]] = {}


def logit_order_and_rank(model, dataset, device, batch_users, split="test", head_k=None):
    """Drop-in replacement of inference.compute_batch_order_and_rank: order on the pre-sigmoid scores, ties by item
    index. head_k is accepted and ignored (the order is deterministic). Also records the ties of the sigmoid scores
    that the stored evaluation ranks (the same float32 values the model's sigmoid produces)."""
    batch_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
    saved = model.activation
    model.activation = torch.nn.Identity()
    try:
        with torch.no_grad():
            logits = model.get_rating_for_test(batch_tensor)
    finally:
        model.activation = saved
    with torch.no_grad():
        sigmoid = saved(logits)
        logits = logits.double()
        excluded = dataset.get_user_pos_items_for_evaluation(batch_users, split)
        for row, items in enumerate(excluded):
            if len(items):
                sigmoid[row, list(items)] = float("-inf")
                logits[row, list(items)] = float("-inf")
        top = torch.topk(sigmoid, K + 1).values
        within = (top[:, 1:K] == top[:, :K - 1]).any(dim=1).cpu().numpy()
        boundary = (top[:, K - 1] == top[:, K]).cpu().numpy()
        values, order = torch.sort(logits, dim=1, descending=True, stable=True)
        logit_within = (values[:, 1:K + 1] == values[:, :K]).any(dim=1).cpu().numpy()
        num_items = logits.shape[1]
        rank = torch.empty_like(order)
        positions = torch.arange(1, num_items + 1, device=device).unsqueeze(0).expand(order.shape[0], -1)
        rank.scatter_(1, order, positions)
    run = str(Path(dataset.training_output_dir).resolve().relative_to(ROOT)).replace("\\", "/")
    rec = TIES.setdefault(run, {"users": set(), "tie_within_top20": set(), "tie_at_20_21": set(),
                                "logit_tie_within_top21": set()})
    for row, user in enumerate(batch_users):
        rec["users"].add(int(user))
        if within[row]:
            rec["tie_within_top20"].add(int(user))
        if boundary[row]:
            rec["tie_at_20_21"].add(int(user))
        if logit_within[row]:
            rec["logit_tie_within_top21"].add(int(user))
    return order, rank


def install():
    for module in (inference, fm, fab, pvn):
        module.compute_batch_order_and_rank = logit_order_and_rank


def office_confirmatory(device):
    """confirmatory_analysis on Office (P1, the secondary Holm family, the factorial cells, the rule-Ov sensitivity
    and the NDCG@20 contrasts), from the same runs, with the order installed by install()."""
    stored = json.loads((ROOT / "results" / "confirmatory" / "confirmatory_analysis_office-products.json")
                        .read_text(encoding="utf-8"))
    rel = stored["run_dirs"]
    scores = fm.score_checkpoint_set("office-products", {k: v for k, v in rel.items() if k != "V3"}, device)
    scores["V3"] = scores["TaxPro-CL"]
    means = {k: ca.group_means(v) for k, v in rel.items()}
    p1 = ca.contrast(scores, "TaxPro-CL", "SimGCL_R", "recall", "long_tail")
    p1["interpretation"] = ca.label_p1(p1, means["SimGCL_R"]["long_tail"]["recall"]["mean"])
    secondary = [dict(name="LT vs default SimGCL", **ca.contrast(scores, "TaxPro-CL", "SimGCL_default", "recall", "long_tail")),
                 dict(name="NC vs tuned SimGCL", **ca.contrast(scores, "TaxPro-CL", "SimGCL_R", "recall", "near_cold")),
                 dict(name="NC vs default SimGCL", **ca.contrast(scores, "TaxPro-CL", "SimGCL_default", "recall", "near_cold"))]
    fm.holm_bonferroni(secondary, "p_signflip", "holm_significant")
    factorial = {}
    for factor, pairs in (("direction", (("V1", "V0"), ("V3", "V2"))), ("epsilon_adaptivity", (("V2", "V0"), ("V3", "V1")))):
        cells = []
        for group in ("near_cold", "long_tail"):
            cs = {f"{a}-{b}": ca.contrast(scores, a, b, "recall", group) for a, b in pairs}
            cells.append({"group": group, "contrasts": cs, "cell_p": max(c["p_signflip"] for c in cs.values()),
                          "both_positive": all(c["mean_diff"] > 0 for c in cs.values())})
        fm.holm_bonferroni(cells, "cell_p", "holm_significant")
        for c in cells:
            c["supported"] = bool(c["holm_significant"] and c["both_positive"])
        factorial[factor] = cells
    sens = {"P1_rule_Ov": ca.contrast(scores, "TaxPro-CL", "SimGCL_Ov", "recall", "long_tail"),
            "NC_rule_Ov": ca.contrast(scores, "TaxPro-CL", "SimGCL_Ov", "recall", "near_cold")}
    sens["P1_rule_Ov"]["interpretation"] = ca.label_p1(sens["P1_rule_Ov"], means["SimGCL_Ov"]["long_tail"]["recall"]["mean"])
    ndcg = {"LT vs tuned SimGCL (NDCG@20)": ca.contrast(scores, "TaxPro-CL", "SimGCL_R", "ndcg", "long_tail"),
            "NC vs tuned SimGCL (NDCG@20)": ca.contrast(scores, "TaxPro-CL", "SimGCL_R", "ndcg", "near_cold")}
    out = {"dataset": "office-products", "run_dirs": rel, "P1": p1, "secondary_holm": secondary,
           "factorial": factorial, "sensitivity_rule_Ov": sens, "descriptive_ndcg": ndcg}
    return stored, out


def logit_ranked_lists(model, dataset, device, config, split="test"):
    """evaluate_sealed_test.ranked_lists with the deterministic order: the same users and list length."""
    target_dict = dataset.validation_dict if split == "validation" else dataset.test_dict
    users = [int(u) for u in target_dict.keys()]
    top_k = max(ast.literal_eval(str(config["top_K"])))
    rankings = {}
    model = model.eval()
    for start in range(0, len(users), BATCH):
        batch = users[start:start + BATCH]
        order, _ = logit_order_and_rank(model, dataset, device, batch, split)
        for user, items in zip(batch, order[:, :top_k].cpu().tolist()):
            rankings[user] = items
    return rankings


_PER_USER = {}


def logit_per_user_values(run_dir, device):
    """extended_analysis.per_user_values, re-scored from the checkpoint with the deterministic order."""
    key = str(run_dir)
    if key not in _PER_USER:
        model, dataset, config, _ = inference.load_model(run_dir, device)
        _PER_USER[key] = est.per_user_arrays(model, dataset, device, config)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return _PER_USER[key]


def extended_controls(device):
    """The four control contrasts of Section S41 (all datasets, groups, metrics; Holm over NC/LT per contrast and
    metric), with the generator stream of extended_analysis.main, whose first analysis they are."""
    est.ranked_lists = logit_ranked_lists
    ea.per_user_values = logit_per_user_values
    return ea.analyse_controls(ea.new_run_index(), np.random.default_rng(42), device)


def save_ties():
    """Merge this run's tie counts into TIES_FILE (a checkpoint scored again keeps the entry with more users)."""
    merged = json.loads(TIES_FILE.read_text(encoding="utf-8")) if TIES_FILE.exists() else {}
    for run, rec in TIES.items():
        counts = {k: len(v) for k, v in rec.items()}
        if run not in merged or counts["users"] >= merged[run]["users"]:
            merged[run] = counts
    TIES_FILE.write_bytes((json.dumps(dict(sorted(merged.items())), indent=2) + "\n").encode("utf-8"))
    return merged


def sign(x):
    return "+" if x > 0 else ("-" if x < 0 else "0")


def decision(stat):
    return {"excludes_zero": bool(stat["excludes_zero"]), "sign": sign(stat["mean_diff"]),
            "mean_diff": stat["mean_diff"], "ci95": [stat["ci95_lo"], stat["ci95_hi"]]}


def office_decision(stat):
    out = {"excludes_zero": bool(stat["ci95"][0] > 0 or stat["ci95"][1] < 0), "sign": sign(stat["mean_diff"]),
           "p_below_0.05": stat["p_signflip"] < 0.05, "mean_diff": stat["mean_diff"], "ci95": stat["ci95"],
           "p": stat["p_signflip"]}
    if "holm_significant" in stat:
        out["holm_significant"] = bool(stat["holm_significant"])
    if "interpretation" in stat:
        out["label"] = stat["interpretation"]["label"]
    return out


def holm_tables(mult):
    """Cell p (intersection-union) of the 32 current cells, Holm decisions in the four 8-cell families and in the
    pooled family, as export_factorial_pvalues computes them."""
    pairs = {"direction": ("direction_fixed_eps__V1-V0", "direction_adaptive_eps__V3-V2"),
             "epsilon": ("eps_random_dir__V2-V0", "eps_taxonomy_dir__V3-V1")}
    rows = []
    for key, groups in mult["combined_cells"].items():
        dataset, cset, factor, metric = key.split("|")
        if cset != "current":
            continue
        for group in groups:
            ps = [mult["cells"][f"{dataset}|{cset}"][f"{c}|{metric}|{group}"]["p_signflip_1e5_seed42"]
                  for c in pairs[factor]]
            rows.append({"dataset": dataset, "factor": factor, "metric": metric, "group": group, "p_cell": max(ps)})
    assert len(rows) == 32, len(rows)
    for factor in pairs:
        for metric in ("recall", "ndcg"):
            fam = [r for r in rows if r["factor"] == factor and r["metric"] == metric]
            adj, ranks = holm_adjusted([r["p_cell"] for r in fam])
            still = True
            for i in sorted(range(8), key=lambda k: ranks[k]):
                still = still and fam[i]["p_cell"] <= 0.05 / (8 - ranks[i] + 1)
                fam[i]["holm_significant_8"], fam[i]["holm_adjusted_p_8"] = still, adj[i]
    adj, ranks = holm_adjusted([r["p_cell"] for r in rows])
    still = True
    for i in sorted(range(32), key=lambda k: ranks[k]):
        still = still and rows[i]["p_cell"] <= 0.05 / (32 - ranks[i] + 1)
        rows[i]["holm_significant_32"], rows[i]["holm_adjusted_p_32"] = still, adj[i]
    return {f"{r['dataset']}|{r['factor']}|{r['metric']}|{r['group']}": r for r in rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    install()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if "factorial" in args.parts:
        fm.main(["--device", args.device, "--skip-precision-rerun", "--output",
                 str(OUT_DIR / "factorial_multiplicity_logit.json")])
        print(f"[tie-order] factorial done, elapsed={time.time() - t0:.0f}s", flush=True)
    if "a5" in args.parts:
        a5.main(["--device", args.device, "--output", str(OUT_DIR / "a5_leaf_vs_parent_mergedt10_logit.json")])
        print(f"[tie-order] A5 done, elapsed={time.time() - t0:.0f}s", flush=True)
    if "variants" in args.parts:
        pvn.main(["--device", args.device, "--output", str(OUT_DIR / "prototype_variants_ndcg_logit.json")])
        print(f"[tie-order] prototype variants done, elapsed={time.time() - t0:.0f}s", flush=True)
    if "office" in args.parts:
        _, office = office_confirmatory(device)
        (OUT_DIR / "confirmatory_analysis_office-products_logit.json").write_bytes(
            (json.dumps(office, indent=2) + "\n").encode("utf-8"))
        print(f"[tie-order] Office done, elapsed={time.time() - t0:.0f}s", flush=True)
    if "extended" in args.parts:
        controls = extended_controls(device)
        (OUT_DIR / "extended_controls_logit.json").write_bytes((json.dumps(controls, indent=2) + "\n").encode("utf-8"))
        print(f"[tie-order] extended controls done, elapsed={time.time() - t0:.0f}s", flush=True)
    tie_counts = save_ties()
    office_stored = json.loads((ROOT / "results" / "confirmatory" / "confirmatory_analysis_office-products.json")
                               .read_text(encoding="utf-8"))
    office_logit = json.loads((OUT_DIR / "confirmatory_analysis_office-products_logit.json").read_text(encoding="utf-8"))

    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    decisions = []

    changes = {"recall": [], "ndcg": []}   # absolute change of every compared mean difference

    def add(analysis, metric, key, evaluator, logit):
        # A decision: whether the interval excludes zero (and, if it does under either order, its sign), whether
        # p < 0.05, the Holm decisions and the Office label; estimates and p-values are reported, not compared.
        keys = ("excludes_zero", "p_below_0.05", "holm_significant", "holm_significant_8", "holm_significant_32",
                "supported", "label")
        same = all(evaluator[k] == logit[k] for k in keys if k in evaluator)
        if evaluator.get("excludes_zero") or logit.get("excludes_zero"):
            same = same and evaluator["sign"] == logit["sign"]
        if "mean_diff" in evaluator:
            changes[metric].append(abs(evaluator["mean_diff"] - logit["mean_diff"]))
        decisions.append({"analysis": analysis, "metric": metric, "key": key, "evaluator_order": evaluator,
                          "logit_order": logit, "same": same})

    # RQ1 factorial: every contrast (all datasets and groups, both checkpoint sets) and the Holm decisions.
    mult_e = load(ROOT / "results" / "factorial_multiplicity.json")
    mult_l = load(OUT_DIR / "factorial_multiplicity_logit.json")
    for prefix, cells in mult_e["cells"].items():
        for key, cell in cells.items():
            other = mult_l["cells"][prefix][key]
            e, l_ = decision(cell), decision(other)
            e["p_below_0.05"], l_["p_below_0.05"] = cell["p_signflip_1e5_seed42"] < 0.05, other["p_signflip_1e5_seed42"] < 0.05
            e["p"], l_["p"] = cell["p_signflip_1e5_seed42"], other["p_signflip_1e5_seed42"]
            add("factorial", cell["metric"], f"{prefix}|{key}", e, l_)
    holm_e, holm_l = holm_tables(mult_e), holm_tables(mult_l)
    pick = lambda r: {k: r[k] for k in ("holm_significant_8", "holm_significant_32", "holm_adjusted_p_8",  # noqa: E731
                                        "holm_adjusted_p_32")} | {"p": r["p_cell"]}
    for key, row in holm_e.items():
        add("factorial_holm", row["metric"], key, pick(row), pick(holm_l[key]))

    # Parent prototype (A5) and the prototype-construction variants (NDCG@20 companion, Table S14c).
    a5_e = load(ROOT / "results" / "a5_leaf_vs_parent_mergedt10.json")["comparison_parent_minus_leaf"]
    a5_l = load(OUT_DIR / "a5_leaf_vs_parent_mergedt10_logit.json")["comparison_parent_minus_leaf"]
    for metric in ("recall", "ndcg"):
        for group, stat in a5_e[metric].items():
            add("a5_parent_minus_leaf", metric, group, decision(stat), decision(a5_l[metric][group]))
    pv_e = {(r["variant"], r["group"]): r for r in load(ROOT / "results" / "prototype_variants_ndcg.json")}
    pv_l = {(r["variant"], r["group"]): r for r in load(OUT_DIR / "prototype_variants_ndcg_logit.json")}
    for key, stat in pv_e.items():
        add("prototype_variants", "ndcg", "|".join(key), decision(stat), decision(pv_l[key]))

    # Office held-out analysis: P1, secondary family, factorial cells, rule-Ov sensitivity, NDCG@20 contrasts.
    o_e, o_l = office_stored, office_logit
    add("office", "recall", "P1", office_decision(o_e["P1"]), office_decision(o_l["P1"]))
    for se, sl in zip(o_e["secondary_holm"], o_l["secondary_holm"]):
        add("office", "recall", "secondary|" + se["name"], office_decision(se), office_decision(sl))
    for factor, cells in o_e["factorial"].items():
        for ce, cl in zip(cells, o_l["factorial"][factor]):
            e = {"holm_significant": ce["holm_significant"], "supported": ce["supported"], "p": ce["cell_p"]}
            l_ = {"holm_significant": cl["holm_significant"], "supported": cl["supported"], "p": cl["cell_p"]}
            add("office_factorial", "recall", f"{factor}|{ce['group']}", e, l_)
            for name, stat in ce["contrasts"].items():
                add("office_factorial", "recall", f"{factor}|{ce['group']}|{name}", office_decision(stat),
                    office_decision(cl["contrasts"][name]))
    for name, stat in o_e["sensitivity_rule_Ov"].items():
        add("office", "recall", "rule_Ov|" + name, office_decision(stat), office_decision(o_l["sensitivity_rule_Ov"][name]))
    for name, stat in o_e["descriptive_ndcg"].items():
        add("office", "ndcg", name, office_decision(stat), office_decision(o_l["descriptive_ndcg"][name]))

    # Label and gradient-path controls (Section S41): every cell, and the Holm decisions over NC/LT.
    ext_e = load(ROOT / "results" / "extended" / "extended_analysis.json")["controls"]
    ext_l = load(OUT_DIR / "extended_controls_logit.json")
    for name, part in ext_e.items():
        assert part["not_run"] == ext_l[name]["not_run"], name
        other = {(c["dataset"], c["group"], c["metric"]): c for c in ext_l[name]["cells"]}
        for cell in part["cells"]:
            o = other[(cell["dataset"], cell["group"], cell["metric"])]
            e, l_ = decision(cell), decision(o)
            e["p_below_0.05"], l_["p_below_0.05"] = cell["p"] < 0.05, o["p"] < 0.05
            e["p"], l_["p"] = cell["p"], o["p"]
            if "holm_significant" in cell:
                e["holm_significant"], l_["holm_significant"] = bool(cell["holm_significant"]), bool(o["holm_significant"])
            add("extended_controls", cell["metric"], f"{name}|{cell['dataset']}|{cell['group']}", e, l_)

    ties = dict(sorted(tie_counts.items()))
    totals = {k: sum(t[k] for t in ties.values()) for k in ("users", "tie_within_top20", "tie_at_20_21",
                                                             "logit_tie_within_top21")}
    by_dataset = {}
    for run, t in ties.items():
        name = next(d for d in ("amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing",
                                "office-products") if f"/{d}/" in run)
        agg = by_dataset.setdefault(name, {"checkpoints": 0, "users": 0, "tie_within_top20": 0, "tie_at_20_21": 0})
        agg["checkpoints"] += 1
        for k in ("users", "tie_within_top20", "tie_at_20_21"):
            agg[k] += t[k]
    for agg in by_dataset.values():
        agg["share_tie_within_top20"] = agg["tie_within_top20"] / agg["users"]
    changed = [d for d in decisions if not d["same"]]
    summary = {"order_evaluator": "torch.topk on float32 sigmoid scores (stored evaluation; head_k=20 re-scoring)",
               "order_logit": "pre-sigmoid scores in float64, ties by item index (torch.sort, stable=True)",
               "checkpoints": len(ties), "tie_totals_user_checkpoint_pairs": totals, "ties_by_dataset": by_dataset,
               "ties_by_checkpoint": ties,
               "mean_diff_max_abs_change": {m: max(v) for m, v in changes.items()},
               "n_decisions_by_metric": {m: sum(d["metric"] == m for d in decisions) for m in changes},
               "n_decisions": len(decisions), "n_changed": len(changed),
               "changed": changed, "decisions": decisions}
    SUMMARY.write_bytes((json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"checkpoints": len(ties), "ties": totals, "by_dataset": by_dataset,
                      "mean_diff_change": {m: max(v) for m, v in changes.items()}, "decisions": len(decisions),
                      "changed": len(changed)},
                     indent=1))
    for d in changed:
        print("CHANGED", d["analysis"], d["metric"], d["key"], d["evaluator_order"], d["logit_order"])
    print(f"wrote {SUMMARY} elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
