"""NDCG@20 for the RQ5 epsilon-adaptivity contrasts (V2-V0, V3-V1; Online
Resource 1, Table S17b), the NDCG@20 companion of Table S17. Same checkpoints,
per-user pooling across seeds, and bootstrap as factorial_ndcg_bootstrap.py,
which covers the direction contrasts (Table S13b). Yelp2018 uses the V0-V3
runs trained with temperature_user=0.15, the main configuration's value.

Inference only; no retraining."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.analysis.factorial_ndcg_bootstrap as base

base.COMPARISONS = [
    ("V2", "V0", "epsilon_adaptivity_random_direction"),
    ("V3", "V1", "epsilon_adaptivity_taxonomy_direction"),
]
base.DATASET_DIRS["yelp2018"] = {
    "V0": "log/p0/taxprocl/yelp2018/A2-V0-tempuser0.15",
    "V1": "log/p0/taxprocl/yelp2018/A2-V1-tempuser0.15",
    "V2": "log/p0/taxprocl/yelp2018/A2-V2-tempuser0.15",
    "V3": "log/p0/taxprocl/yelp2018/A2-V3-tempuser0.15",
}

if __name__ == "__main__":
    extra = ["--output", str(ROOT / "results" / "epsilon_ndcg_bootstrap.json")]
    raise SystemExit(base.main(sys.argv[1:] + extra))
