"""B1: NDCG@20 for the RQ5
epsilon-adaptivity factorial (V2-V0, V3-V1), extending b1_factorial_ndcg.py
(which covers only the direction comparisons, V1-V0/V3-V2, Table S13/S13b)
to the epsilon-adaptivity comparisons reported in Section S17 (Table S17,
Recall@20-only). Same checkpoints, same per-user pooling-across-seeds
procedure, same bootstrap machinery as b1_factorial_ndcg.py, computing
NDCG@20 instead of Recall@20. Yelp2018 uses the temperature_user=0.15
corrected V0-V3 checkpoints, matching Table S17's own Yelp2018 rows.

Inference only; no retraining; no new checkpoints created.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.analysis.b1_factorial_ndcg as base

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
    extra = ["--output", str(ROOT / "results" / "b1_epsilon_ndcg.json")]
    raise SystemExit(base.main(sys.argv[1:] + extra))
