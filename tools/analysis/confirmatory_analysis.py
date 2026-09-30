"""Analysis of the pre-registered confirmatory evaluation (docs/confirmatory_protocol.md,
Sections 7-8 and Deviation 2), run once after the sealed test split was opened.

Per-user test Recall@20 (and NDCG@20) differences, averaged over the three seed-matched
pairs before resampling (as main paper Table 11); two-sided paired sign-flip test (10^5
sign assignments, generator seed 42); percentile bootstrap (2,000 resamples, seed 42).

  P1          Long-Tail Recall@20, selected TaxPro-CL minus tuned SimGCL (rule R), alone
              at alpha 0.05, with the SESOI of 5% of the tuned SimGCL mean.
  Secondary   Holm over {LT vs default SimGCL, NC vs tuned SimGCL, NC vs default SimGCL}.
  Factorial   direction (V1-V0, V3-V2) and epsilon-adaptivity (V2-V0, V3-V1); cell p = the
              larger contrast p (intersection-union); Holm within each {NC, LT} family.
  Sensitivity P1 and NC with the rule-Ov SimGCL cell (Deviation 2).
  Descriptive three-seed test means of every group, Recall@20 and NDCG@20.

Usage: python -m tools.analysis.confirmatory_analysis --dataset office-products
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.factorial_multiplicity import (  # noqa: E402
    SEEDS, holm_bonferroni, pooled_diffs, score_checkpoint_set, sign_flip_pvalue)

SESOI = 0.05
GROUPS = ("near_cold", "long_tail", "warm", "overall")


def boot(values, n_boot=2000, seed=42):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.sort([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    q = lambda a: float(means[int(a * n_boot)])  # noqa: E731
    return {"n": len(v), "mean_diff": float(v.mean()), "ci95": [q(0.025), float(means[int(0.975 * n_boot) - 1])],
            "ci90": [q(0.05), float(means[int(0.95 * n_boot) - 1])]}


def contrast(scores, plus, minus, metric, group):
    diffs = pooled_diffs(scores, [(1, plus), (-1, minus)], metric, group)
    values = list(diffs.values())
    out = boot(values)
    out["p_signflip"] = float(sign_flip_pvalue(values, 100_000, np.random.default_rng(42)))
    return out


def group_means(rel):
    per = [json.loads((ROOT / rel / f"seed{s}" / "final_test_group_metrics.json").read_text(encoding="utf-8"))
           for s in SEEDS]
    return {g: {m: {"mean": statistics.fmean(float(p[g][m]["20"]) for p in per),
                    "std": statistics.stdev(float(p[g][m]["20"]) for p in per)} for m in ("recall", "ndcg")}
            for g in GROUPS}


def label_p1(stat, denom):
    rel = stat["mean_diff"] / denom
    rel90 = [x / denom for x in stat["ci90"]]
    sig = stat["p_signflip"] < 0.05
    if sig and rel > 0:
        base = "Long-Tail improvement over a comparably tuned SimGCL confirmed on a held-out dataset"
        qual = "practically meaningful" if rel >= SESOI else "below the smallest effect of interest"
    elif sig and rel < 0:
        base, qual = "reversed on the held-out dataset", None
    else:
        base = "not confirmed"
        qual = "equivalent within +/-5%" if (-SESOI < rel90[0] and rel90[1] < SESOI) else "inconclusive"
    return {"relative_difference": rel, "relative_ci90": rel90,
            "relative_ci95": [x / denom for x in stat["ci95"]], "label": base, "qualifier": qual}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="office-products")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    base = f"log/p0/confirmatory/{args.dataset}"
    sel = json.loads((ROOT / base / "confirmatory_selection.json").read_text(encoding="utf-8"))
    policy = sel["step1_policy"]["selected"]
    tax = sel["step2_taxpro"]["selected"]
    rel = {
        "TaxPro-CL": f"{base}/TaxPro-CL/step2-{policy}-{tax}",
        "SimGCL_R": f"{base}/SimGCL/grid-{sel['step3_tuned_simgcl']['selected']}",
        "SimGCL_Ov": f"{base}/SimGCL/grid-{sel['step3_tuned_simgcl']['selected_rule_ov_sensitivity']}",
        "SimGCL_default": f"{base}/SimGCL/grid-{sel['step3_tuned_simgcl']['default_cell']}",
    }
    if tax == "temp0.125-gamma1.5":
        rel["TaxPro-CL"] = f"{base}/TaxPro-CL/step1-{policy}"
    for v in ("V0", "V1", "V2"):
        rel[v] = f"{base}/TaxPro-CL/step4-{v}-{policy}-{tax.replace('temp', 'temp').replace('-gamma', '-gamma')}"
    rel["V3"] = rel["TaxPro-CL"]
    assert sel["step2_taxpro"]["selected_rule_ov_sensitivity"] == tax, "TaxPro-CL Ov cell differs: extend the script"

    scores = score_checkpoint_set(args.dataset, {k: v for k, v in rel.items() if k != "V3"}, torch.device(args.device))
    scores["V3"] = scores["TaxPro-CL"]
    means = {k: group_means(v) for k, v in rel.items()}
    denom = means["SimGCL_R"]["long_tail"]["recall"]["mean"]

    p1 = contrast(scores, "TaxPro-CL", "SimGCL_R", "recall", "long_tail")
    p1["interpretation"] = label_p1(p1, denom)

    secondary = [dict(name="LT vs default SimGCL", **contrast(scores, "TaxPro-CL", "SimGCL_default", "recall", "long_tail")),
                 dict(name="NC vs tuned SimGCL", **contrast(scores, "TaxPro-CL", "SimGCL_R", "recall", "near_cold")),
                 dict(name="NC vs default SimGCL", **contrast(scores, "TaxPro-CL", "SimGCL_default", "recall", "near_cold"))]
    holm_bonferroni(secondary, "p_signflip", "holm_significant")

    factorial = {}
    for factor, pairs in (("direction", (("V1", "V0"), ("V3", "V2"))), ("epsilon_adaptivity", (("V2", "V0"), ("V3", "V1")))):
        cells = []
        for group in ("near_cold", "long_tail"):
            cs = {f"{a}-{b}": contrast(scores, a, b, "recall", group) for a, b in pairs}
            cells.append({"group": group, "contrasts": cs, "cell_p": max(c["p_signflip"] for c in cs.values()),
                          "both_positive": all(c["mean_diff"] > 0 for c in cs.values())})
        holm_bonferroni(cells, "cell_p", "holm_significant")
        for c in cells:
            c["supported"] = bool(c["holm_significant"] and c["both_positive"])
        factorial[factor] = cells

    ov_denom = means["SimGCL_Ov"]["long_tail"]["recall"]["mean"]
    sens = {"P1_rule_Ov": contrast(scores, "TaxPro-CL", "SimGCL_Ov", "recall", "long_tail"),
            "NC_rule_Ov": contrast(scores, "TaxPro-CL", "SimGCL_Ov", "recall", "near_cold")}
    sens["P1_rule_Ov"]["interpretation"] = label_p1(sens["P1_rule_Ov"], ov_denom)
    ndcg = {"LT vs tuned SimGCL (NDCG@20)": contrast(scores, "TaxPro-CL", "SimGCL_R", "ndcg", "long_tail"),
            "NC vs tuned SimGCL (NDCG@20)": contrast(scores, "TaxPro-CL", "SimGCL_R", "ndcg", "near_cold")}

    out = {"dataset": args.dataset, "selection_sha256": (ROOT / base / "confirmatory_selection.sha256").read_text().split()[0],
           "run_dirs": rel, "P1": p1, "secondary_holm": secondary, "factorial": factorial,
           "sensitivity_rule_Ov": sens, "descriptive_ndcg": ndcg, "test_means": means}
    path = ROOT / "results" / "confirmatory" / f"confirmatory_analysis_{args.dataset}.json"
    path.write_bytes((json.dumps(out, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"P1": p1, "secondary": [(s["name"], s["mean_diff"], s["p_signflip"], s["holm_significant"]) for s in secondary],
                      "factorial": {f: [(c["group"], c["cell_p"], c["both_positive"], c["supported"]) for c in cs] for f, cs in factorial.items()},
                      "sens_P1_Ov": sens["P1_rule_Ov"]}, indent=1))
    for k, m in means.items():
        print(k, {g: round(1e3 * m[g]["recall"]["mean"], 3) for g in GROUPS})
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
