"""Relative prototype-to-item distance by leaf size, in three embedding spaces
(Online Resource 1, Table S16; main paper Sections 4.1 and 5.5).

For each dataset's main TaxPro-CL checkpoint (seeds 42, 0, 1) and every
valid-taxonomy item i of leaf l, computes ||p_l - e_i|| / ||e_i||, with p_l the
trained EMA prototype, for three choices of e_i:
  - "pooled": the clean mean over propagated layers 1..L -- the space the
    prototype bank is initialized and updated from;
  - "layer1".."layerL": the clean propagated embedding at each layer -- the
    space in which Eq. (1) forms the direction;
  - "ego": the layer-0 lookup embedding (what
    tools/analysis/prototype_distance_by_leaf_size.py reports; it belongs to
    neither space above).
Items are grouped by the number of valid items in their leaf (1, 2, 3-10,
11-100, 101+) and the median is reported per group.

Inference only, no retraining.

Usage:
    python -m tools.analysis.prototype_distance_by_space [--device cuda]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ranking import checkpoint_selection  # noqa: E402
from tools.analysis.prototype_variants_overlap import load_model_dropping_stale_buffers  # noqa: E402
from tools.analysis.prototype_distance_by_leaf_size import BUCKETS, DATASETS  # noqa: E402


def analyse(dataset: str, seed: str, device: torch.device):
    choice = checkpoint_selection.select_checkpoint("TaxPro-CL", dataset)
    run_dir = choice.run_dir.parent / f"seed{seed}"
    model, data, _config, _name = load_model_dropping_stale_buffers(run_dir, device)
    model.eval()
    with torch.no_grad():
        num_users = data.num_users
        embedding = torch.cat([model.user_embedding.weight, model.item_embedding.weight])
        layers = []
        for _ in range(int(model.config["GCN_layer"])):
            embedding = torch.sparse.mm(model.Graph, embedding)
            layers.append(embedding[num_users:])
        spaces = {"ego": model.item_embedding.weight, "pooled": torch.stack(layers, 1).mean(1)}
        _users, items = model.aggregate(perturbed=False)
        if not torch.allclose(items, spaces["pooled"], atol=1e-5):
            raise RuntimeError("layer-averaged embedding does not match model.aggregate()")
        for index, layer in enumerate(layers):
            spaces[f"layer{index + 1}"] = layer
        prototypes = model.prototype_bank.prototypes
        leaf = model.item_to_leaf_id.long()
        valid = model.valid_train_mask.bool()
        sizes = collections.Counter(leaf[valid].tolist())
        size_of = torch.tensor([sizes.get(int(l), 0) if v else 0 for l, v in zip(leaf.tolist(), valid.tolist())],
                               device=device)
        rows = {"n_items": {name: int((valid & (size_of >= lo) & (size_of <= hi)).sum()) for name, lo, hi in BUCKETS}}
        for space, vectors in spaces.items():
            ratio = (prototypes[leaf.clamp_min(0)] - vectors).norm(dim=1) / vectors.norm(dim=1).clamp_min(1e-12)
            rows[space] = {}
            for name, lo, hi in BUCKETS:
                mask = valid & (size_of >= lo) & (size_of <= hi)
                rows[space][name] = float(ratio[mask].median()) if mask.any() else None
    return str(run_dir.relative_to(ROOT)).replace("\\", "/"), rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    results = {}
    for dataset in DATASETS:
        for seed in ("42", "0", "1"):
            run_dir, rows = analyse(dataset, seed, device)
            results[f"{dataset}|{seed}"] = {"run_dir": run_dir, **rows}
            print(dataset, seed, {k: v.get("1") for k, v in rows.items() if k != "n_items"}, flush=True)
    out = ROOT / "results" / "prototype_distance_by_space.json"
    out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print("Saved to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
