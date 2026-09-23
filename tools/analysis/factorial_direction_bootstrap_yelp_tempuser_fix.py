"""Re-run of tools/analysis/factorial_direction_bootstrap.py restricted to
Yelp2018, pointed at the temperature_user=0.15 corrected V0-V3 checkpoints
(A2-V{0-3}-tempuser0.15) instead of the original A2-V0..V3 runs.

Background: a cross-check
found that the original Yelp2018 A2-V0..V3 factorial checkpoints were
trained with temperature_user=0.2 (the base-config default), not 0.15
(Yelp2018's tuned main-configuration value, Table 6) -- confirmed via
results_manifest.csv's checkpoint SHA256 and config diff. This script
reproduces the Yelp2018 rows of Tables S13/S17/S19 using the corrected
checkpoints, with the identical bootstrap methodology (5000 resamples,
per-user pooling across seeds before resampling -- see
factorial_direction_bootstrap.py's own pool_diffs_across_seeds fix) so the
new numbers are directly comparable to the other three datasets' published
values, which used this same script.

Usage:
    python -m tools.analysis.factorial_direction_bootstrap_yelp_tempuser_fix
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.analysis.factorial_direction_bootstrap as base

base.DATASET_DIRS["yelp2018"] = {
    "V0": "log/p0/taxprocl/yelp2018/A2-V0-tempuser0.15",
    "V1": "log/p0/taxprocl/yelp2018/A2-V1-tempuser0.15",
    "V2": "log/p0/taxprocl/yelp2018/A2-V2-tempuser0.15",
    "V3": "log/p0/taxprocl/yelp2018/A2-V3-tempuser0.15",
}

if __name__ == "__main__":
    raise SystemExit(base.main([
        "--datasets", "yelp2018",
        "--output", str(ROOT / "results" / "factorial_direction_bootstrap_yelp_tempuser_fix.json"),
    ]))
