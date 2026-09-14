"""CLI: compare TaxPro-CL's Top-K rankings against a non-taxonomy baseline's.

Usage (from the repository root):
    python -m tests.Recommendation_system.run_comparison
    python -m tests.Recommendation_system.run_comparison --datasets amazon-book \
        --baselines LightGCN --k-values 10 20 --num-users 200

With no arguments this runs every dataset in config.DATASETS against every
baseline in config.WITHOUT_TAXONOMY_MODELS at config.K_VALUES, over the full
eligible test-user population -- that is a large, GPU-time-consuming sweep
touching many checkpoints; consider --num-users for a quick pass first.

Results are upserted into tests/Recommendation_system/result/*.csv: rerunning
a subset only replaces the rows for the (dataset, baseline) pairs you asked
for, leaving previously computed pairs intact.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tests.Recommendation_system import config, csv_store, rank_comparison

RESULT_DIR = Path(__file__).resolve().parent / "result"
DETAILS_CSV = RESULT_DIR / "ranking_details.csv"
SUMMARY_CSV = RESULT_DIR / "ranking_summary.csv"
MANIFEST_CSV = RESULT_DIR / "checkpoint_manifest.csv"
MANIFEST_FIELDNAMES = [
    "Model", "Dataset", "Checkpoint_Dir", "Seed",
    "Validation_Score", "Num_Candidates_Considered",
]

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS, choices=config.DATASETS)
    parser.add_argument(
        "--baselines", nargs="+", default=config.WITHOUT_TAXONOMY_MODELS,
        choices=config.WITHOUT_TAXONOMY_MODELS,
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=config.K_VALUES)
    parser.add_argument(
        "--num-users", type=int, default=None,
        help="Sample this many eligible users instead of the full population.",
    )
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = list(itertools.product(args.datasets, args.baselines))
    started = time.time()
    failures = []

    for pair_index, (dataset_name, baseline_name) in enumerate(pairs, start=1):
        logger.info("[%d/%d] %s vs TaxPro-CL on %s", pair_index, len(pairs), baseline_name, dataset_name)
        tmp_path = RESULT_DIR / "_tmp_details_{}_{}.csv".format(dataset_name, baseline_name)
        try:
            summary_rows, baseline_choice, taxpro_choice, n_users = rank_comparison.compare_dataset_baseline(
                dataset_name=dataset_name,
                baseline_name=baseline_name,
                k_values=args.k_values,
                device=device,
                details_tmp_path=tmp_path,
                batch_size=args.batch_size,
                num_users=args.num_users,
                seed=args.seed,
            )

            def matches_pair(row, _dataset=dataset_name, _baseline=baseline_name):
                return row.get("Dataset") == _dataset and row.get("Baseline") == _baseline

            csv_store.upsert_csv(
                DETAILS_CSV, rank_comparison.DETAIL_FIELDNAMES,
                csv_store.stream_rows(tmp_path), matches_pair,
            )
            csv_store.upsert_csv(
                SUMMARY_CSV, rank_comparison.SUMMARY_FIELDNAMES,
                summary_rows, matches_pair,
            )
            touched = {
                (baseline_choice.model_name, baseline_choice.dataset_name): baseline_choice.as_row(),
                (taxpro_choice.model_name, taxpro_choice.dataset_name): taxpro_choice.as_row(),
            }

            def matches_manifest(row, _keys=set(touched.keys())):
                return (row.get("Model"), row.get("Dataset")) in _keys

            csv_store.upsert_csv(MANIFEST_CSV, MANIFEST_FIELDNAMES, list(touched.values()), matches_manifest)
            logger.info("[%d/%d] done: %d users scored", pair_index, len(pairs), n_users)
        except Exception:
            # GPU is shared with independently-running training jobs on this
            # machine -- a transient CUDA error in one pair should not throw
            # away every other pair already computed or still to come.
            logger.exception(
                "[%d/%d] FAILED: %s vs TaxPro-CL on %s -- skipping, continuing with remaining pairs",
                pair_index, len(pairs), baseline_name, dataset_name,
            )
            failures.append((dataset_name, baseline_name))
        finally:
            tmp_path.unlink(missing_ok=True)

    if failures:
        logger.warning("Pairs that failed and were skipped: %s", failures)

    elapsed = time.time() - started
    logger.info("Finished %d (dataset, baseline) pairs in %.1fs", len(pairs), elapsed)
    logger.info("Details:  %s", DETAILS_CSV)
    logger.info("Summary:  %s", SUMMARY_CSV)
    logger.info("Manifest: %s", MANIFEST_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
