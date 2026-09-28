"""Replace the Amazon-Book blocks of the older per-script result JSONs with the
merge_t10 factorial results computed by a2_mergedt10_factorial_recompute.py,
so that every JSON cited by results/claim_evidence.csv agrees with the paper's
Online Resource 1 tables (Yelp2018/Musical-Instruments/Arts-Crafts-and-Sewing
blocks are left unchanged).

Also replaces the Yelp2018 block of factorial_direction_bootstrap_full_FIXED.json with
the temperature_user=0.15 result (factorial_direction_bootstrap_yelp_tempuser_fix.json).

Rewrites, in results/: factorial_direction_bootstrap_full_FIXED.json,
factorial_interaction_bootstrap.json, warmstart_removal_bootstrap.json,
b1_factorial_ndcg.json, b1_epsilon_ndcg.json.

Usage:
    python -m tools.analysis.merge_a2_ab_into_result_jsons
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
GROUPS = ("near_cold", "long_tail", "overall", "warm")


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def dump(name, obj):
    (RES / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> int:
    a2 = load("a2_mergedt10_factorial_bootstrap.json")["comparisons"]

    def rec(name):
        return {g: a2[name]["recall"][g] for g in GROUPS}

    # 1. direction/epsilon-adaptivity (Recall@20)
    fd = load("factorial_direction_bootstrap_full_FIXED.json")
    fd["amazon-book"] = {
        "V1_vs_V0": rec("direction_fixed_eps__V1-V0"),
        "V3_vs_V2": rec("direction_adaptive_eps__V3-V2"),
        "V2_vs_V0": rec("eps_random_dir__V2-V0"),
        "V3_vs_V1": rec("eps_taxonomy_dir__V3-V1"),
    }
    # Yelp2018 rows: the checkpoints trained with temperature_user=0.15 (the main configuration's value)
    fd["yelp2018"] = load("factorial_direction_bootstrap_yelp_tempuser_fix.json")["yelp2018"]
    dump("factorial_direction_bootstrap_full_FIXED.json", fd)

    # 2. interaction
    fi = load("factorial_interaction_bootstrap.json")
    fi["amazon-book"] = {
        g: {
            "direction_conditional_effect_fixed_epsilon_V1_minus_V0": a2["direction_fixed_eps__V1-V0"]["recall"][g],
            "direction_conditional_effect_adaptive_epsilon_V3_minus_V2": a2["direction_adaptive_eps__V3-V2"]["recall"][g],
            "interaction_V3V2_minus_V1V0": a2["interaction__(V3-V2)-(V1-V0)"]["recall"][g],
        }
        for g in GROUPS
    }
    dump("factorial_interaction_bootstrap.json", fi)

    # 3. warm-start removal
    dump("warmstart_removal_bootstrap.json", {
        "random_direction": rec("nowarm_random__V0nowarm-V0"),
        "taxonomy_direction": rec("nowarm_taxonomy__V3nowarm-V3"),
    })

    # 4/5. NDCG@20 rows (Near-Cold / Long-Tail)
    def ndcg_rows(existing, spec):
        rows = [r for r in existing if r["dataset"] != "amazon-book"]
        ab = []
        for comp, label, key in spec:
            for g in ("near_cold", "long_tail"):
                st = dict(a2[key]["ndcg"][g])
                st.update({"dataset": "amazon-book", "comparison": comp, "label": label, "group": g,
                           "n_seed_pairs": 3, "metric": "NDCG@20"})
                ab.append(st)
        return ab + rows

    dump("b1_factorial_ndcg.json", ndcg_rows(load("b1_factorial_ndcg.json"), [
        ("V1-V0", "direction_fixed_epsilon", "direction_fixed_eps__V1-V0"),
        ("V3-V2", "direction_adaptive_epsilon", "direction_adaptive_eps__V3-V2")]))
    dump("b1_epsilon_ndcg.json", ndcg_rows(load("b1_epsilon_ndcg.json"), [
        ("V2-V0", "epsilon_adaptivity_random_direction", "eps_random_dir__V2-V0"),
        ("V3-V1", "epsilon_adaptivity_taxonomy_direction", "eps_taxonomy_dir__V3-V1")]))
    print("Amazon-Book blocks replaced in 5 JSON files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
