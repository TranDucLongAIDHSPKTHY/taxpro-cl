"""Sequential runner for docs/extended_evaluation_protocol.md.

Every run inherits the configuration recorded in the manifest of its reference family and
may differ only in the overrides of its package; anything else stops the runner. Runs set
TAXPRO_DEFER_TEST=1, so no test metric is computed here. Test splits are opened by
--open-test, once per run; a search needs its committed selection file first.

Usage:
    python -m tools.experiments.run_extended --package grid-simgcl grid-complete [--dry-run]
    python -m tools.experiments.run_extended --select grid
    python -m tools.experiments.run_extended --open-test grid
    python -m tools.experiments.run_extended --open-test controls-amazon-book
    python -m tools.experiments.run_extended --registry
    python -m tools.experiments.run_extended --snapshot after
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import model_config_file  # noqa: E402
from tools.experiments.run_confirmatory import (  # noqa: E402
    best_validation,
    rule_ov,
    rule_r,
    validation_means,
)
from utility.utility_data.taxonomy import REDRAWN_SPLIT_BASES, REDRAWN_SPLIT_SEEDS  # noqa: E402
from utility.utility_function import tools as config_tools  # noqa: E402

OUTPUT = Path("log/p0/extended")
SMOKE_OUTPUT = OUTPUT / "smoke"
RESULTS = Path("results/extended")
PROTOCOL = Path("docs/extended_evaluation_protocol.md")


def selection_file(name):
    """Selection file of a search or of an added baseline, frozen once."""
    return RESULTS / f"selection_{name.lower()}.json"


SELECTION_GRID = selection_file("grid")
SELECTION_NCL = selection_file("ncl")
REGISTRY = RESULTS / "run_registry.csv"
SEEDS = (42, 0, 1)
NEW_SEEDS = (2, 3, 4)
LOCKED_SPLIT = 42
OFFICE = "office-products"
TAX = "log/p0/taxprocl"
BASE = "log/p0/baseline"
CONF = "log/p0/confirmatory/office-products"

# Reference run families. Every family is from Environment A.
FAMILIES = {
    "amazon-book": {
        "V3": f"{TAX}/amazon-book/taxpro-cl-v15-prototype-leaf-lambda0.5-same_leaf_weight0-user_ssl-"
              "warmstart20-noblend-temp0.1-DONE-overall-2.30pct-BEST-overall-nearcold-longtail-positive",
        "V0": f"{TAX}/amazon-book/A2-V0-mergedt10",
        "V1": f"{TAX}/amazon-book/A2-V1-mergedt10",
        "V2": f"{TAX}/amazon-book/A2-V2-mergedt10",
        "SimGCL-default": f"{BASE}/SimGCL/amazon-book/simgcl-69bb0902f42f",
        "SimGCL-tuned": f"{BASE}/SimGCL/amazon-book/A7-temp0.15-eps0.1",
    },
    "yelp2018": {
        "V3": f"{TAX}/yelp2018/taxpro-cl-FINAL-no_merge-temp0.125-tempuser0.15-gammacold1.5",
        "V0": f"{TAX}/yelp2018/A2-V0-tempuser0.15",
        "V1": f"{TAX}/yelp2018/A2-V1-tempuser0.15",
        "V2": f"{TAX}/yelp2018/A2-V2-tempuser0.15",
        "SimGCL-default": f"{BASE}/SimGCL/yelp2018/simgcl-20c63ca8d16c",
        "SimGCL-tuned": f"{BASE}/SimGCL/yelp2018/A7-temp0.2-eps0.2",
    },
    "musical-instruments": {
        "V3": f"{TAX}/musical-instruments/taxpro-cl-FINAL-no_merge-temp0.15-gammacold5.0",
        "V0": f"{TAX}/musical-instruments/A2-V0",
        "V1": f"{TAX}/musical-instruments/A2-V1",
        "V2": f"{TAX}/musical-instruments/A2-V2",
        "SimGCL-default": f"{BASE}/SimGCL/musical-instruments/simgcl-f263d2bff497",
        "SimGCL-tuned": f"{BASE}/SimGCL/musical-instruments/A7-temp0.2-eps0.2",
    },
    "arts-crafts-and-sewing": {
        "V3": f"{TAX}/arts-crafts-and-sewing/taxpro-cl-FINAL-no_merge-temp0.125-gammacold1.5-sameleaf0",
        "V0": f"{TAX}/arts-crafts-and-sewing/A2-V0",
        "V1": f"{TAX}/arts-crafts-and-sewing/A2-V1",
        "V2": f"{TAX}/arts-crafts-and-sewing/A2-V2",
        "SimGCL-default": f"{BASE}/SimGCL/arts-crafts-and-sewing/simgcl-c8b32441f415",
        "SimGCL-tuned": f"{BASE}/SimGCL/arts-crafts-and-sewing/A7-temp0.2-eps0.2",
    },
    OFFICE: {
        "V3": f"{CONF}/TaxPro-CL/step2-no_merge-temp0.15-gamma1.5",
        "V0": f"{CONF}/TaxPro-CL/step4-V0-no_merge-temp0.15-gamma1.5",
        "V1": f"{CONF}/TaxPro-CL/step4-V1-no_merge-temp0.15-gamma1.5",
        "V2": f"{CONF}/TaxPro-CL/step4-V2-no_merge-temp0.15-gamma1.5",
        "SimGCL-default": f"{CONF}/SimGCL/grid-temp0.2-eps0.05",
        "SimGCL-tuned": f"{CONF}/SimGCL/grid-temp0.2-eps0.1",
        "LightGCN": f"{CONF}/LightGCN/default",
        "NCL-default": f"{CONF}/NCL/default",
    },
}
REFERENCE_MODEL = {"V0": "TaxPro-CL", "V1": "TaxPro-CL", "V2": "TaxPro-CL", "V3": "TaxPro-CL",
                   "SimGCL-default": "SimGCL", "SimGCL-tuned": "SimGCL", "LightGCN": "LightGCN",
                   "NCL-default": "NCL"}
FACTORIAL = ("V0", "V1", "V2", "V3")
SIMGCL = ("SimGCL-default", "SimGCL-tuned")

# Stored by reference manifests, not accepted by the parser; the value is the one they hold.
REFERENCE_ONLY_KEYS = {"taxonomy_granularity": "leaf"}
# Accepted by the parser, absent from some reference manifests; these values are their behavior.
NEUTRAL_DEFAULTS = {
    "asymmetric_view_direction": "false", "prototype_leave_one_out": "false",
    "prototype_weighting": "interaction", "prototype_init_scope": "all_valid",
    "mixture_alpha": "0.5", "taxonomy_perturbation": "none", "taxonomy_perturbation_rate": "1.0",
    "taxonomy_perturbation_seed": "0", "direction_detach": "false", "pool_layer_zero": "true",
}
SHUFFLE_SEED_BASE = 1000
LABEL_SEED_BASE = 2000
CONTROL_ARMS = {"shuffled": ("shuffle",), "detached": ("detach",), "shuffled-detached": ("shuffle", "detach")}
LABEL_ARMS = (("V3", "noise", "noise50"), ("V3", "missing", "missing50"), ("V2", "missing", "missing50"))
GRID_SIMGCL_TEMPS = (0.25, 0.30, 0.40)
GRID_SIMGCL_EPS = (0.05, 0.10, 0.20)
GRID_ITEM_TEMPS = (0.15, 0.20, 0.25)
GRID_USER_TEMPS = (0.20, 0.30, 0.40)
GRID_GAMMA = 1.5
# (item, user) temperatures; (0.15, 0.20) is the locked cell temp0.15-gamma1.5 and is not run again.
GRID_COMPLETE_CELLS = tuple((temp, user) for user in GRID_USER_TEMPS for temp in GRID_ITEM_TEMPS
                            if (temp, user) != (0.15, 0.20))
NCL_TEMPS = (0.05, 0.10, 0.15, 0.20)
NCL_CLUSTERS = (1000, 2000, 3000)
NCL_DEFAULT = (0.05, 2000)
LOCKED_SIMGCL_TEMPS = (0.05, 0.1, 0.15, 0.2)
LOCKED_COMPLETE_TEMPS = (0.1, 0.125, 0.15)
LOCKED_COMPLETE_GAMMAS = (1.5, 5.0)
REGISTRY_COLUMNS = ("run_id", "exp", "dataset", "arm", "split_seed", "train_seed", "config_sha256",
                    "label_file_sha256", "env", "start_utc", "end_utc", "status", "best_epoch",
                    "val_overall_r20", "val_lt_r20", "test_eval_utc", "checkpoint_sha256", "git_commit")
KINDS = ("smoke", "controls", "label-quality", "seeds", "splits", "grid", "pretrain", "comparator")
REFERENCE_KIND = "reference"
PRETRAINING_STEP = "pretrain-2layers"
PRETRAINING_OVERRIDES = {"GCN_layer": 2, "pool_layer_zero": "false"}


@dataclass(frozen=True)
class Job:
    package: str
    arm: str
    model: str
    dataset: str
    base_dataset: str
    split: int
    cell: str
    seed: int
    config: tuple

    @property
    def run_dir(self):
        root = SMOKE_OUTPUT if self.package == "smoke" else OUTPUT
        return root / self.dataset / self.model / self.cell / f"seed{self.seed}"

    @property
    def run_id(self):
        kind = next(name for name in KINDS if self.package.startswith(name))
        return f"{kind}_{self.base_dataset}_{self.arm}_s{self.split}_r{self.seed}"


def read_manifest(run_dir):
    return json.loads((ROOT / run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def inherited_config(family, model, overrides):
    """Reference configuration plus overrides, on the keys the parser accepts."""
    config = {k: str(v) for k, v in read_manifest(f"{family}/seed42")["configuration"].items()}
    config.pop("resume_checkpoint", None)
    defaults = config_tools.read_configuration(str(model_config_file(model)), model)
    for key in sorted(set(config) - set(defaults)):
        if REFERENCE_ONLY_KEYS.get(key) != config[key]:
            raise ValueError(f"{family}: unexpected key {key}={config[key]}")
        del config[key]
    for key in sorted(set(defaults) - set(config) - {"resume_checkpoint"}):
        if NEUTRAL_DEFAULTS.get(key) != str(defaults[key]):
            raise ValueError(f"{family}: {key}={defaults[key]} is not a neutral default")
    config.update({k: str(v) for k, v in overrides.items()})
    return config


def parameters(model, config):
    """Parameter part of a run family name, as in log/p0/confirmatory."""
    if model == "TaxPro-CL":
        return f"{config['taxonomy_policy']}-temp{config['temperature']}-gamma{config['gamma_cold']}"
    if model == "SimGCL":
        return f"temp{config['temperature']}-eps{config['epsilon']}"
    if model == "NCL":
        return f"temp{config['temperature']}-k{config['k']}"
    return None


def job(package, dataset, reference, step, seed, overrides=None, split=LOCKED_SPLIT, arm=None):
    """One run in <dataset>/<Model>/<step>-<parameters>/seed<seed>."""
    model = REFERENCE_MODEL[reference]
    extra = dict(overrides or {})
    data_id = dataset if split == LOCKED_SPLIT else f"{dataset}-split{split}"
    if split != LOCKED_SPLIT:
        extra["dataset"] = data_id
    config = inherited_config(FAMILIES[dataset][reference], model, extra)
    suffix = parameters(model, config)
    cell = f"{step}-{suffix}" if suffix else step
    return Job(package, arm or reference, model, data_id, dataset, split, cell, seed,
               tuple(sorted(config.items())))


def step_name(prefix, reference):
    """V0-V3 keep their names; SimGCL and LightGCN are named by their selection."""
    return f"{prefix}-{reference.split('-')[-1].replace('LightGCN', 'default')}"


def label_overrides(mode, rate, seed):
    base = SHUFFLE_SEED_BASE if mode == "shuffle" else LABEL_SEED_BASE
    return {"taxonomy_perturbation": mode, "taxonomy_perturbation_rate": rate,
            "taxonomy_perturbation_seed": base + seed}


def control_overrides(arm, seed):
    overrides = label_overrides("shuffle", 1.0, seed) if "shuffle" in CONTROL_ARMS[arm] else {}
    if "detach" in CONTROL_ARMS[arm]:
        overrides["direction_detach"] = "true"
    return overrides


def plan_controls(package, dataset):
    return [job(package, dataset, "V3", f"control-V3-{arm}", seed, control_overrides(arm, seed), arm=f"V3-{arm}")
            for arm in CONTROL_ARMS for seed in SEEDS]


def plan_label_quality(package, dataset):
    return [job(package, dataset, reference, f"label-{reference}-{name}", seed,
                label_overrides(mode, 0.5, seed), arm=f"{reference}-{name}")
            for seed in SEEDS for reference, mode, name in LABEL_ARMS]


def plan_seeds(package, dataset, references):
    return [job(package, dataset, reference, step_name("repeat", reference), seed)
            for reference in references for seed in NEW_SEEDS]


def plan_splits(package, dataset, references):
    return [job(package, dataset, reference, step_name("split", reference), seed, split=split)
            for split in REDRAWN_SPLIT_SEEDS for reference in references for seed in SEEDS]


def plan_grid_simgcl():
    return [job("grid-simgcl", OFFICE, "SimGCL-tuned", "grid", seed, {"temperature": temp, "epsilon": eps},
                arm=f"SimGCL-temp{temp}-eps{eps}")
            for temp in GRID_SIMGCL_TEMPS for eps in GRID_SIMGCL_EPS for seed in SEEDS]


def plan_grid_complete():
    return [job("grid-complete", OFFICE, "V3", f"grid-tempuser{user}", seed,
                {"temperature": temp, "temperature_user": user, "gamma_cold": GRID_GAMMA},
                arm=f"complete-temp{temp}-tempuser{user}")
            for temp, user in GRID_COMPLETE_CELLS for seed in SEEDS]


def plan_grid_ncl():
    """The NCL grid of the development datasets; the default cell is the reference family."""
    return [job("grid-ncl", OFFICE, "NCL-default", "grid", seed, {"temperature": temp, "k": clusters},
                arm=f"NCL-temp{temp}-k{clusters}")
            for temp in NCL_TEMPS for clusters in NCL_CLUSTERS if (temp, clusters) != NCL_DEFAULT
            for seed in SEEDS]


def plan_pretraining(package):
    """LightGCN with the depth and pooling of GALORE's encoder, the start of GALORE. Its test stays closed."""
    return [job(package, OFFICE, "LightGCN", PRETRAINING_STEP, seed, PRETRAINING_OVERRIDES, arm="LightGCN-2layers")
            for seed in SEEDS]


def plan_smoke():
    """Each new kind of run, briefly: the warm start plus two epochs. Never used in results."""
    data = "musical-instruments"
    short = {"training_epochs": "22"}
    jobs = [job("smoke", data, "V3", f"control-V3-{arm}", 42, {**control_overrides(arm, 42), **short},
                arm=f"V3-{arm}") for arm in CONTROL_ARMS]
    jobs += [job("smoke", data, reference, f"label-{reference}-{name}", 42,
                 {**label_overrides(mode, 0.5, 42), **short}, arm=f"{reference}-{name}")
             for reference, mode, name in LABEL_ARMS]
    redrawn = REDRAWN_SPLIT_SEEDS[0]
    jobs.append(job("smoke", data, "V3", "split-V3", 42, short, split=redrawn))
    jobs.append(job("smoke", data, "SimGCL-default", "split-default", 42, {"training_epochs": "2"}, split=redrawn))
    jobs.append(job("smoke", OFFICE, "LightGCN", "split-default", 42, {"training_epochs": "10"}, split=redrawn))
    jobs.append(job("smoke", OFFICE, "LightGCN", PRETRAINING_STEP, 42, {**PRETRAINING_OVERRIDES, "training_epochs": "10"},
                    arm="LightGCN-2layers"))
    return jobs


def plan_of(function, name, *arguments):
    return lambda: function(name, *arguments)


PLANS = {
    "smoke": plan_smoke,
    **{f"controls-{dataset}": plan_of(plan_controls, f"controls-{dataset}", dataset) for dataset in FAMILIES},
    **{f"label-quality-{dataset}": plan_of(plan_label_quality, f"label-quality-{dataset}", dataset)
       for dataset in ("musical-instruments", OFFICE)},
    "seeds-amazon-book-factorial": plan_of(plan_seeds, "seeds-amazon-book-factorial", "amazon-book", FACTORIAL),
    "seeds-amazon-book-simgcl": plan_of(plan_seeds, "seeds-amazon-book-simgcl", "amazon-book", SIMGCL),
    "seeds-yelp2018": plan_of(plan_seeds, "seeds-yelp2018", "yelp2018", FACTORIAL + SIMGCL),
    **{f"splits-{dataset}": plan_of(plan_splits, f"splits-{dataset}", dataset, FACTORIAL + SIMGCL)
       for dataset in REDRAWN_SPLIT_BASES},
    "splits-office-products-lightgcn": plan_of(plan_splits, "splits-office-products-lightgcn", OFFICE, ("LightGCN",)),
    "grid-simgcl": plan_grid_simgcl,
    "grid-complete": plan_grid_complete,
    "grid-ncl": plan_grid_ncl,
    "pretrain-lightgcn": plan_of(plan_pretraining, "pretrain-lightgcn"),
}
EXPECTED_RUNS = {
    "smoke": 10, **{f"controls-{dataset}": 9 for dataset in FAMILIES},
    "label-quality-musical-instruments": 9, f"label-quality-{OFFICE}": 9,
    "seeds-amazon-book-factorial": 12, "seeds-amazon-book-simgcl": 6, "seeds-yelp2018": 18,
    **{f"splits-{dataset}": 54 for dataset in REDRAWN_SPLIT_BASES},
    "splits-office-products-lightgcn": 9, "grid-simgcl": 27, "grid-complete": 24, "grid-ncl": 33,
    "pretrain-lightgcn": 3,
}


CLOSED_PACKAGES = {"smoke", "grid-simgcl", "grid-complete", "grid-ncl", "pretrain-lightgcn"}


def planned(package):
    jobs = PLANS[package]()
    if len(jobs) != EXPECTED_RUNS[package]:
        raise ValueError(f"{package}: {len(jobs)} runs planned, {EXPECTED_RUNS[package]} expected")
    if len({j.run_dir for j in jobs}) != len(jobs) or len({j.run_id for j in jobs}) != len(jobs):
        raise ValueError(f"{package}: two runs share a directory or an identifier")
    return jobs


def run_status(run_dir):
    try:
        return json.loads((ROOT / run_dir / "run_manifest.json").read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None


def log(message, package=None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    path = ROOT / (SMOKE_OUTPUT if package == "smoke" else OUTPUT) / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_clean_code():
    """environment.json records HEAD, so training code must be committed before a package runs."""
    tracked = ("models", "utility", "configure", "main.py", "Parser.py", "tools/experiments/run_extended.py")
    changed = subprocess.run(["git", "status", "--porcelain", "--", *tracked], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    if changed:
        raise SystemExit(f"commit these files before running:\n{changed}")


def require_frozen_protocol():
    recorded = (ROOT / PROTOCOL).with_suffix(".sha256")
    if not recorded.exists():
        raise SystemExit(f"{PROTOCOL} is not frozen: {recorded.name} is missing")
    if sha256_path(ROOT / PROTOCOL) != recorded.read_text(encoding="utf-8").split()[0]:
        raise SystemExit(f"{PROTOCOL} does not match its recorded SHA-256")


def history_is_complete(run_dir):
    """Selection reads this history, so every validated epoch must be present."""
    interval = int(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
                   ["configuration"].get("interval", 1))
    rows = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    epochs = [int(row["epoch"]) for row in rows]
    return bool(epochs) and epochs == list(range(interval, epochs[-1] + 1, interval))


def registry_path(package):
    return ROOT / (SMOKE_OUTPUT / "run_registry.csv" if package == "smoke" else REGISTRY)


def environment_label(run_dir, environment):
    """A or B when the GPU, the platform and, where the log names it, the path style of the
    log all agree; unknown otherwise."""
    gpu, platform = str(environment.get("gpu_name")), str(environment.get("platform"))
    separator = None
    log_file = run_dir / "training.log"
    if log_file.exists():
        with log_file.open(encoding="utf-8", errors="replace") as stream:
            for _, line in zip(range(20), stream):
                if "Training log: " in line:
                    path = line.split("Training log: ", 1)[1]
                    separator = "\\" if "\\" in path else "/"
                    break
    if "4060" in gpu and platform.startswith("Windows") and separator in (None, "\\"):
        return "A"
    if "3090" in gpu and platform.startswith("Linux") and separator in (None, "/"):
        return "B"
    return "unknown"


def registry_row(item, relative=None):
    relative = relative or item.run_dir
    run_dir = ROOT / relative
    manifest = read_manifest(relative)
    environment = manifest.get("environment") or {}
    metadata = manifest.get("model_metadata") or {}
    row = dict.fromkeys(REGISTRY_COLUMNS, "")
    row.update(run_id="" if item.package == REFERENCE_KIND else item.run_id, exp=item.package, dataset=item.dataset, arm=item.arm, split_seed=item.split,
               train_seed=item.seed, status=manifest.get("status"), start_utc=manifest.get("started_at_utc"),
               end_utc=manifest.get("completed_at_utc", ""), git_commit=environment.get("git_commit", ""),
               env=environment_label(run_dir, environment),
               label_file_sha256=metadata.get("taxonomy_label_sha256") or "",
               config_sha256=hashlib.sha256(json.dumps(manifest["configuration"], sort_keys=True,
                                                       separators=(",", ":")).encode("utf-8")).hexdigest())
    if manifest.get("status") == "completed":
        history = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
        best = max(history, key=lambda entry: (entry["primary_value"], -int(entry["epoch"])))
        group = best_validation(run_dir)
        row.update(best_epoch=best["epoch"], val_overall_r20=group["overall"], val_lt_r20=group["long_tail"],
                   checkpoint_sha256=sha256_path(run_dir / "best_validation_model.pt"))
    stamp = run_dir / "sealed_test_evaluation.json"
    if stamp.exists():
        row["test_eval_utc"] = json.loads(stamp.read_text(encoding="utf-8"))["evaluated_at_utc"]
    return row


def reference_rows():
    """One row per reference run, with the environment it ran on. Their test results predate
    this protocol, so test_eval_utc stays empty."""
    rows = []
    for dataset, families in FAMILIES.items():
        for arm, family in families.items():
            for seed in SEEDS:
                item = Job(REFERENCE_KIND, arm, REFERENCE_MODEL[arm], dataset, dataset, LOCKED_SPLIT,
                           family.rsplit("/", 1)[-1], seed, ())
                row = registry_row(item, f"{family}/seed{seed}")
                row.update(run_id=f"{REFERENCE_KIND}_{dataset}_{arm}_s{LOCKED_SPLIT}_r{seed}", exp=REFERENCE_KIND,
                           test_eval_utc="")
                rows.append(row)
    return rows


def update_registry(jobs, extra_rows=()):
    if not jobs and not extra_rows:
        return
    path = registry_path(jobs[0].package) if jobs else ROOT / REGISTRY
    rows = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            rows = {row["run_id"]: row for row in csv.DictReader(stream)}
    rows.update((row["run_id"], row) for row in extra_rows)
    for item in jobs:
        current = rows.get(item.run_id)
        opened = (ROOT / item.run_dir / "sealed_test_evaluation.json").exists()
        if run_status(item.run_dir) and (current is None or current["status"] != "completed"
                                         or (opened and not current["test_eval_utc"])):
            rows[item.run_id] = registry_row(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="\n", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTRY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    os.replace(temporary, path)


def archive_labels(item):
    """Keep one copy of each perturbed label file; runs sharing a seed must share the file."""
    source = ROOT / item.run_dir / "taxonomy_labels.npy"
    if item.package == "smoke" or not source.exists():
        return
    config = dict(item.config)
    name = f"{config['taxonomy_perturbation']}{config['taxonomy_perturbation_rate']}" \
           f"-seed{config['taxonomy_perturbation_seed']}.npy"
    target = ROOT / RESULTS / "labels" / item.dataset / name
    if target.exists():
        if sha256_path(target) != sha256_path(source):
            raise RuntimeError(f"label file differs from {target}: {source}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def execute(item, device):
    run_dir = ROOT / item.run_dir
    if run_status(item.run_dir) == "completed":
        return False
    cmd = [sys.executable, "main.py", "--model", item.model, "--seed", str(item.seed), "--device", device]
    for key, value in item.config:
        cmd += ["--" + key, value]
    last_model = run_dir / "last_model.pt"
    if run_status(item.run_dir) == "running" and last_model.exists():
        cmd += ["--resume_checkpoint", str(last_model.relative_to(ROOT))]
        log(f"RESUME {item.run_id} {item.run_dir}", item.package)
    else:
        log(f"START {item.run_id} {item.run_dir}", item.package)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TAXPRO_DEFER_TEST="1", TAXPROCL_RUN_OUTPUT_DIR=str(item.run_dir), TQDM_DISABLE="1")
    started = time.time()
    for attempt in (1, 2):
        code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        if code == 0 and run_status(item.run_dir) == "completed":
            break
        log(f"FAILED attempt {attempt} (exit {code}) {item.run_id}", item.package)
        if attempt == 1 and last_model.exists() and "--resume_checkpoint" not in cmd:
            cmd += ["--resume_checkpoint", str(last_model.relative_to(ROOT))]
    else:
        update_registry([item])
        raise RuntimeError(f"run failed twice: {item.run_dir}")
    if (run_dir / "final_test_metrics.json").exists():
        raise RuntimeError(f"test metrics were written despite TAXPRO_DEFER_TEST: {run_dir}")
    written = read_manifest(item.run_dir)["configuration"]
    differing = {k: (written.get(k), v) for k, v in item.config
                 if str(written.get(k)) != v and k != "resume_checkpoint"}
    if differing:
        raise RuntimeError(f"configuration differs from the plan in {item.run_dir}: {differing}")
    if not history_is_complete(run_dir):
        raise RuntimeError(f"validation history has gaps: {run_dir}")
    if (run_dir / "best_validation_model.pt").exists() and last_model.exists():
        last_model.unlink()
    archive_labels(item)
    update_registry([item])
    log(f"DONE ({time.time() - started:.0f}s) {item.run_id}", item.package)
    return True


def run_packages(packages, device, dry_run):
    plans = {package: planned(package) for package in packages}
    for package, jobs in plans.items():
        remaining = [j for j in jobs if run_status(j.run_dir) != "completed"]
        print(f"{package}: {len(jobs)} runs planned, {len(remaining)} remaining")
        if dry_run:
            for item in remaining[:3]:
                print("  would run", item.run_id, item.run_dir.as_posix())
    if dry_run:
        return 0
    require_clean_code()
    if set(packages) != {"smoke"}:
        require_frozen_protocol()
    for package, jobs in plans.items():
        update_registry(jobs)
        for item in jobs:
            execute(item, device)
        log(f"PACKAGE COMPLETE {package}; test split still sealed", package)
    return 0


def grid_family(side, cell):
    """Run family of a grid cell, new or from the confirmatory grids."""
    if side == "simgcl":
        new_cells = {f"temp{t}-eps{e}" for t in GRID_SIMGCL_TEMPS for e in GRID_SIMGCL_EPS}
        if cell in new_cells:
            return f"{OUTPUT.as_posix()}/{OFFICE}/SimGCL/grid-{cell}"
        return f"{CONF}/SimGCL/grid-{cell}"
    if "tempuser" in cell:
        temp, user = cell.replace("temp", "", 1).split("-tempuser")
        return f"{OUTPUT.as_posix()}/{OFFICE}/TaxPro-CL/grid-tempuser{user}-no_merge-temp{temp}-gamma{GRID_GAMMA}"
    if side == "ncl":
        if cell == f"temp{NCL_DEFAULT[0]}-k{NCL_DEFAULT[1]}":
            return FAMILIES[OFFICE]["NCL-default"]
        return f"{OUTPUT.as_posix()}/{OFFICE}/NCL/grid-{cell}"
    return f"{CONF}/TaxPro-CL/" + ("step1-no_merge" if cell == "temp0.125-gamma1.5" else f"step2-no_merge-{cell}")


def search_cells():
    """Cells of each search, locked ones first, in the order ties are broken."""
    return {
        "simgcl": [f"temp{t}-eps{e}" for t in LOCKED_SIMGCL_TEMPS + GRID_SIMGCL_TEMPS for e in GRID_SIMGCL_EPS],
        "complete_configuration": [f"temp{t}-gamma{g}" for t in LOCKED_COMPLETE_TEMPS for g in LOCKED_COMPLETE_GAMMAS]
                                  + [f"temp{t}-tempuser{u}" for t, u in GRID_COMPLETE_CELLS],
        "ncl": [f"temp{t}-k{k}" for t in NCL_TEMPS for k in NCL_CLUSTERS],
    }


# Selection file, the packages it waits for, and the searches it selects.
SEARCHES = {
    "grid": (SELECTION_GRID, ("grid-simgcl", "grid-complete"), ("complete_configuration", "simgcl")),
    "ncl": (SELECTION_NCL, ("grid-ncl",), ("ncl",)),
}


def selection_for(name):
    """Rule R on the union of the locked and the new cells, per method; rule Ov is recorded."""
    def means(family):
        return validation_means([ROOT / family / f"seed{seed}" for seed in SEEDS])

    cells_of = search_cells()
    document = {"protocol": PROTOCOL.as_posix(), "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for side in SEARCHES[name][2]:
        cells = [(cell, means(grid_family(side, cell))) for cell in cells_of[side]]
        document[side] = {"rule_R": rule_r(cells), "rule_Ov": rule_ov(cells),
                          "cells": {cell: value["mean"] for cell, value in cells}}
    return document


def freeze_selection(name="grid"):
    selection_file, packages, sides = SEARCHES[name]
    path = ROOT / selection_file
    if path.exists():
        raise SystemExit(f"{path} is frozen already")
    unfinished = [j.run_dir for package in packages for j in planned(package)
                  if run_status(j.run_dir) != "completed"]
    if unfinished:
        raise SystemExit(f"{len(unfinished)} runs are not complete; nothing is selected")
    selection = selection_for(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(selection, indent=2) + "\n").encode("utf-8"))
    digest = sha256_path(path)
    path.with_suffix(".sha256").write_bytes(f"{digest}  {path.name}\n".encode())
    log(f"{name} selection written sha256={digest}; commit it before opening the test split")
    for side in sides:
        print(side, selection[side]["rule_R"], selection[side]["rule_Ov"])


def committed_selection(relative_path):
    """A selection counts only once it is committed unchanged."""
    path = relative_path.as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True)
    changed = subprocess.run(["git", "status", "--porcelain", "--", path], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
    if tracked.returncode != 0 or changed:
        raise SystemExit(f"{path} is not committed; the test split stays sealed")
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def grid_runs_to_open(selection, sides=("complete_configuration", "simgcl")):
    """Only rule-R cells that are still sealed."""
    runs = []
    for side in sides:
        runs += [f"{grid_family(side, selection[side]['rule_R'])}/seed{seed}" for seed in SEEDS]
    return [run for run in runs if not (ROOT / run / "final_test_metrics.json").exists()]


def open_test(package, device, dry_run=False):
    """Evaluate each sealed run once; evaluate_sealed_test refuses a second opening.
    With dry_run, list the runs that would be evaluated and evaluate none."""
    if package in SEARCHES:
        selection_file, packages, sides = SEARCHES[package]
        selection = committed_selection(selection_file)
        jobs = [j for name in packages for j in planned(name)]
        runs = [str(ROOT / run) for run in grid_runs_to_open(selection, sides)]
    else:
        jobs = planned(package)
        missing = [j.run_dir for j in jobs if run_status(j.run_dir) != "completed"]
        if missing:
            raise SystemExit(f"{len(missing)} runs are not complete; the test split stays sealed")
        if not dry_run:
            update_registry(jobs)
        runs = [str(ROOT / j.run_dir) for j in jobs if not (ROOT / j.run_dir / "final_test_metrics.json").exists()]
    if dry_run:
        print(f"{package}: {len(runs)} runs would be evaluated on test")
        for run in runs:
            print(" ", Path(run).relative_to(ROOT).as_posix())
        return 0
    log(f"OPEN TEST {package}: {len(runs)} runs")
    if not runs:
        return 0
    code = subprocess.run([sys.executable, "-m", "tools.experiments.evaluate_sealed_test",
                           "--device", device, "--runs", *runs], cwd=ROOT).returncode
    update_registry(jobs)
    return code


def snapshot_paths():
    paths = [ROOT / "docs" / "confirmatory_protocol.md",
             ROOT / "results" / "confirmatory" / "confirmatory_selection.json",
             ROOT / CONF / "confirmatory_selection.json"]
    for dataset in FAMILIES:
        policy = read_manifest(f"{FAMILIES[dataset]['V3']}/seed42")["configuration"]["taxonomy_policy"]
        paths += [ROOT / "dataset_verify" / dataset / name
                  for name in ("train.txt", "validation.txt", "test.txt", "split_manifest.json")]
        paths.append(ROOT / "preprocessed" / "evaluation_protocol" / "split_seed_42" / dataset / "manifest.json")
        variant = ROOT / "metadata" / "taxonomy_variants" / dataset / policy
        paths += [variant / "manifest.json", variant / "item_to_leaf_id.npy"]
    return paths


def snapshot(stage):
    digests = {path.relative_to(ROOT).as_posix(): sha256_path(path) for path in snapshot_paths()}
    target = ROOT / RESULTS / "integrity_before.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if stage == "before":
        if target.exists():
            raise SystemExit(f"{target} exists")
        target.write_bytes((json.dumps(digests, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        print(f"recorded {len(digests)} files")
        return 0
    recorded = json.loads(target.read_text(encoding="utf-8"))
    changed = sorted(k for k in recorded if recorded[k] != digests.get(k))
    print(f"compared {len(recorded)} files, {len(changed)} changed")
    for name in changed:
        print("  CHANGED", name)
    return 1 if changed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--package", nargs="+", choices=sorted(PLANS))
    action.add_argument("--select", choices=sorted(SEARCHES))
    action.add_argument("--open-test", choices=sorted(set(PLANS) - CLOSED_PACKAGES | set(SEARCHES)))
    action.add_argument("--registry", action="store_true")
    action.add_argument("--snapshot", choices=("before", "after"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.package:
        return run_packages(args.package, args.device, args.dry_run)
    if args.select:
        return freeze_selection(args.select) or 0
    if args.open_test:
        return open_test(args.open_test, args.device, args.dry_run)
    if args.registry:
        update_registry([], reference_rows())
        for package in sorted(set(PLANS) - {"smoke"}):
            update_registry(planned(package))
        return 0
    return snapshot(args.snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
