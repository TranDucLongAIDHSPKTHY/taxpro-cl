"""Re-run of tools/analysis/b1_factorial_ndcg.py restricted to Yelp2018,
pointed at the temperature_user=0.15 corrected V0-V3 checkpoints
(A2-V{0-3}-tempuser0.15), mirroring
factorial_direction_bootstrap_yelp_tempuser_fix.py's patch for the base
Recall@20 script. The original b1_factorial_ndcg.py run imported
DATASET_DIRS directly from factorial_direction_bootstrap (unpatched), so its
Yelp2018 NDCG cells used the superseded temperature_user=0.2 checkpoints,
inconsistent with Table S16's Yelp2018 rows (which use the corrected ones).
This script fixes that for the NDCG companion specifically.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.analysis.b1_factorial_ndcg as base

base.DATASET_DIRS["yelp2018"] = {
    "V0": "log/p0/taxprocl/yelp2018/A2-V0-tempuser0.15",
    "V1": "log/p0/taxprocl/yelp2018/A2-V1-tempuser0.15",
    "V2": "log/p0/taxprocl/yelp2018/A2-V2-tempuser0.15",
    "V3": "log/p0/taxprocl/yelp2018/A2-V3-tempuser0.15",
}

if __name__ == "__main__":
    raise SystemExit(base.main([
        "--datasets", "yelp2018",
        "--output", str(ROOT / "results" / "b1_factorial_ndcg_yelp_tempuser0.15.json"),
    ]))
