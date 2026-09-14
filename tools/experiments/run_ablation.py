"""Prepare or execute TaxPro-CL screening/control matrices."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("no_merge", "merge_t5", "merge_t10", "merge_t15")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=("taxonomy", "random", "epsilon", "temperature", "granularity"),
        default="taxonomy",
    )
    parser.add_argument(
        # A3 (epsilon), A4 (temperature), and A5 (granularity) were all
        # reported on Amazon-Book only (README.md, "Ablations A3-A5");
        # override explicitly if extending a study to other datasets.
        "--datasets", nargs="+", default=["amazon-book"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def configurations(args):
    if args.study == "taxonomy":
        return [
            {"policy": policy, "direction": "taxonomy", "extra": []}
            for policy in POLICIES
        ]
    if args.study == "random":
        return [{"policy": "no_merge", "direction": "random", "extra": []}]
    if args.study == "epsilon":
        # Matches the full reported sweep (README.md "A3: epsilon_max
        # sensitivity"; ESM Table S4): 0.10 is also the Amazon-Book main
        # config value.
        return [
            {
                "policy": "no_merge",
                "direction": "taxonomy",
                "extra": ["--epsilon-max", str(value)],
                "id": "A3-epsilon-{}".format(value),
            }
            for value in (0.01, 0.05, 0.10, 0.20, 0.40, 0.80)
        ]
    if args.study == "temperature":
        # Matches the full reported sweep (README.md "A4: temperature
        # sensitivity"): 0.10 is also the Amazon-Book main config value.
        return [
            {
                "policy": "no_merge",
                "direction": "taxonomy",
                "extra": ["--temperature", str(value)],
                "id": "A4-temperature-{}".format(value),
            }
            for value in (0.05, 0.10, 0.20)
        ]
    return [
        {
            "policy": "no_merge",
            "direction": "taxonomy",
            "extra": ["--taxonomy-granularity", value],
            "id": "A5-granularity-{}".format(value),
        }
        for value in ("leaf", "parent")
    ]


def main(argv=None):
    args = parse_args(argv)
    rows = []
    for dataset in args.datasets:
        if dataset not in {"amazon-book", "yelp2018", "musical-instruments", "arts-crafts-and-sewing"}:
            raise ValueError("Unsupported dataset: {}".format(dataset))
        for config in configurations(args):
            if args.study == "granularity" and dataset != "amazon-book":
                continue
            for seed in args.seeds:
                rows.append((dataset, config, seed))
    for index, (dataset, config, seed) in enumerate(rows, start=1):
        print(
            "{:02d} dataset={} seed={} policy={} direction={} extra={}".format(
                index,
                dataset,
                seed,
                config["policy"],
                config["direction"],
                " ".join(config["extra"]) or "-",
            )
        )
    if args.dry_run:
        print("Dry-run matrix size: {}".format(len(rows)))
        return 0
    failures = 0
    for dataset, config, seed in rows:
        command = [
            sys.executable,
            "-m",
            "tools.experiments.run_taxpro",
            "--dataset",
            dataset,
            "--seeds",
            str(seed),
            "--taxonomy-policy",
            config["policy"],
            "--augmentation-direction",
            config["direction"],
            "--gpu-id",
            str(args.gpu_id),
            "--device",
            args.device,
            "--skip-completed",
        ]
        command.extend(config["extra"])
        if config.get("id"):
            command.extend(["--config-id", config["id"]])
        if args.smoke:
            command.append("--smoke")
        result = subprocess.run(command, cwd=ROOT, check=False)
        failures += int(result.returncode != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
