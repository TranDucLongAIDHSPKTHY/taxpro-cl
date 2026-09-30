"""Per-user hit decomposition of the Yelp2018 factorial (main paper Table 8, footnote a; Online
Resource 1, Tables S13g-S13h).

For every variant V0-V3 (temperature_user=0.15 runs, the ones behind Table 8) and seed,
re-scores the saved best-validation checkpoint on the test split with the same full-catalog
ranking and train/validation filtering as the evaluator, and exports per user:
  t_NC, t_MT  number of the user's test positives in the Near-Cold / Mid-Tail item group
  h_NC, h_MT  how many of them are in the top 20, with the IDs of those items
Checks that the recomputed Near-Cold and Long-Tail Recall@20 equal final_test_group_metrics.json
and that Recall_LT = C_NC + C_MT, then reports per variant and seed:
  N = |U_NC|, H = #users with h_NC > 0, J = sum h_NC, S = sum h_NC / t_NC (so Recall_NC = S / N),
  C_NC = mean over U_LT of h_NC / (t_NC + t_MT), C_MT = mean over U_LT of h_MT / (t_NC + t_MT),
and, per pair of variants and seed, whether h_NC is identical per user and the differences in
C_NC and C_MT. The checkpoint and configuration hashes of every scored run and the hashes of the
split files are recorded, so the scored models can be matched to results_manifest.csv.

Inference only. Outputs results/yelp_factorial_hits.csv (per user),
results/yelp_lt_decomposition.csv (per variant and seed) and results/yelp_factorial_hits_summary.json.

Usage: python -m tools.analysis.yelp_factorial_hits
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch

from config_path.config_path import evaluation_protocol_dir
from tools.ranking import inference
from utility.utility_train.group_evaluator import load_targets

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {v: f"log/p0/taxprocl/yelp2018/A2-{v}-tempuser0.15" for v in ("V0", "V1", "V2", "V3")}
SEEDS = ("42", "0", "1")
K = 20
CONFIG_KEYS = ("temperature", "temperature_user", "augmentation_direction", "use_adaptive_epsilon", "taxonomy_policy", "gamma_cold")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = load_targets(evaluation_protocol_dir("yelp2018"), "test")
    nc = {u: set(i) for u, i in targets["near_cold"].items() if len(i)}
    lt = {u: set(i) for u, i in targets["long_tail"].items() if len(i)}
    users = sorted(lt)
    assert set(nc) <= set(lt)
    rows, summary, hnc = [], {}, {}
    for v, rel in VARIANTS.items():
        for s in SEEDS:
            run_dir = ROOT / rel / f"seed{s}"
            model, dataset, _, _ = inference.load_model(str(run_dir), device)
            h = {}
            for start in range(0, len(users), 512):
                batch = users[start:start + 512]
                _, rank = inference.compute_batch_order_and_rank(model, dataset, device, batch, split="test")
                rank = rank.cpu().numpy()
                for row, u in enumerate(batch):
                    mt = lt[u] - nc.get(u, set())
                    hit_nc = sorted(int(i) for i in nc.get(u, ()) if rank[row, i] <= K)
                    hit_mt = sorted(int(i) for i in mt if rank[row, i] <= K)
                    h[u] = (len(hit_nc), len(hit_mt), len(nc.get(u, ())), len(mt), hit_nc, hit_mt)
            del model
            hnc[(v, s)] = {u: h[u][0] for u in users}
            recall_nc = sum(h[u][0] / h[u][2] for u in nc) / len(nc)
            recall_lt = sum((h[u][0] + h[u][1]) / (h[u][2] + h[u][3]) for u in users) / len(users)
            ref = json.loads((run_dir / "final_test_group_metrics.json").read_text(encoding="utf-8"))
            assert abs(recall_nc - float(ref["near_cold"]["recall"]["20"])) < 1e-9, (v, s, recall_nc)
            assert abs(recall_lt - float(ref["long_tail"]["recall"]["20"])) < 1e-9, (v, s, recall_lt)
            c_nc = sum(h[u][0] / (h[u][2] + h[u][3]) for u in users) / len(users)
            c_mt = sum(h[u][1] / (h[u][2] + h[u][3]) for u in users) / len(users)
            assert abs(c_nc + c_mt - recall_lt) < 1e-12, (v, s)
            cfg = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["configuration"]
            summary[f"{v}_seed{s}"] = {
                "N": len(nc), "H": sum(1 for u in nc if h[u][0] > 0), "J": sum(h[u][0] for u in nc),
                "S": sum(h[u][0] / h[u][2] for u in nc), "recall_nc": recall_nc, "U_LT": len(users),
                "recall_lt": recall_lt, "C_NC": c_nc, "C_MT": c_mt,
                "nc_hit_users": sorted(int(u) for u in nc if h[u][0] > 0),
                "run_dir": rel + f"/seed{s}",
                "checkpoint_sha256": sha256_of(run_dir / "best_validation_model.pt"),
                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                "config": {k: cfg.get(k) for k in CONFIG_KEYS}}
            for u in users:
                rows.append({"variant": v, "seed": s, "user_id": u, "t_NC": h[u][2], "t_MT": h[u][3],
                             "h_NC": h[u][0], "h_MT": h[u][1], "hit_items_NC": " ".join(map(str, h[u][4])),
                             "hit_items_MT": " ".join(map(str, h[u][5]))})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    pairs = {}
    for a, b in (("V1", "V0"), ("V3", "V2"), ("V2", "V0"), ("V3", "V1")):
        for s in SEEDS:
            pairs[f"{a}-{b}_seed{s}"] = {
                "h_NC_identical_per_user": all(hnc[(a, s)][u] == hnc[(b, s)][u] for u in users),
                "delta_C_NC": summary[f"{a}_seed{s}"]["C_NC"] - summary[f"{b}_seed{s}"]["C_NC"],
                "delta_C_MT": summary[f"{a}_seed{s}"]["C_MT"] - summary[f"{b}_seed{s}"]["C_MT"]}
    out = ROOT / "results"
    with open(out / "yelp_factorial_hits.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(out / "yelp_lt_decomposition.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["variant", "seed", "N", "H", "J", "S", "recall_nc", "U_LT", "C_NC", "C_MT", "recall_lt", "checkpoint_sha256"])
        for key, val in summary.items():
            var, seed = key.split("_seed")
            w.writerow([var, seed, val["N"], val["H"], val["J"], repr(val["S"]), repr(val["recall_nc"]), val["U_LT"],
                        repr(val["C_NC"]), repr(val["C_MT"]), repr(val["recall_lt"]), val["checkpoint_sha256"]])
    split_dir = ROOT / "dataset_verify" / "yelp2018"
    split_sha = {n: sha256_of(split_dir / f"{n}.txt") for n in ("train", "validation", "test")}
    (out / "yelp_factorial_hits_summary.json").write_bytes(
        (json.dumps({"k": K, "split_sha256": split_sha, "per_variant_seed": summary, "pairs": pairs}, indent=2) + "\n")
        .encode("utf-8"))
    for key, val in summary.items():
        print(key, {k: val[k] for k in ("N", "H", "J", "S")}, "C_NC=%.8f C_MT=%.8f" % (val["C_NC"], val["C_MT"]))
    for key, val in pairs.items():
        print(key, val)


if __name__ == "__main__":
    main()
