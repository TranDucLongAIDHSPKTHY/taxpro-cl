"""Portable path contract for the standalone TaxPro-CL experiment codebase."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURE_DIR = PROJECT_ROOT / "configure"
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_VERIFY_DIR = PROJECT_ROOT / "dataset_verify"
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
METADATA_DIR = PROJECT_ROOT / "metadata"
TAXONOMY_VARIANT_DIR = METADATA_DIR / "taxonomy_variants"
# Training/evaluation outputs go to <repo>/log unless TAXPRO_OUTPUT_ROOT points
# elsewhere. Reproduction runs should use a fresh, empty root so that the
# analysis stage only sees the run families the documented commands create.
MODEL_OUTPUT_DIR = Path(os.environ.get("TAXPRO_OUTPUT_ROOT") or (PROJECT_ROOT / "log")).resolve()
P0_OUTPUT_DIR = MODEL_OUTPUT_DIR / "p0"
P0_BASELINE_OUTPUT_DIR = P0_OUTPUT_DIR / "baseline"
P0_TAXPROCL_OUTPUT_DIR = P0_OUTPUT_DIR / "taxprocl"
RESULT_DIR = PROJECT_ROOT / "results"
PROTOTYPE_DIR = PROJECT_ROOT / "prototypes"
FIGURE_DIR = PROJECT_ROOT / "figures"
MANIFEST_DIR = PROJECT_ROOT / "manifests"
CACHE_DIR = DATASET_VERIFY_DIR
PREPROCESSING_LOG_DIR = PROJECT_ROOT / "preprocessing_logs"

TRAIN_FILE_NAME = "train.txt"
VALIDATION_FILE_NAME = "validation.txt"
TEST_FILE_NAME = "test.txt"
TRAINING_LOG_FILE_NAME = "training.log"
VALIDATION_METRICS_FILE_NAME = "validation_metrics.json"
FINAL_TEST_METRICS_FILE_NAME = "final_test_metrics.json"
FINAL_TEST_GROUP_METRICS_FILE_NAME = "final_test_group_metrics.json"
LAST_MODEL_FILE_NAME = "last_model.pt"
BEST_VALIDATION_MODEL_FILE_NAME = "best_validation_model.pt"
ATOMIC_TEMPORARY_SUFFIX = ".tmp"
DOWNLOAD_TEMPORARY_SUFFIX = ".part"

# Optional raw-data/metadata bootstrap contract.  These are deliberately kept
# separate from dataset_verify/: model runs consume only the locked verified
# splits and protocol artifacts.
AMAZON_DATASET_NAME = "amazon-book"
MUSICAL_INSTRUMENTS_DATASET_NAME = "musical-instruments"
ARTS_CRAFTS_AND_SEWING_DATASET_NAME = "arts-crafts-and-sewing"
CDS_AND_VINYL_DATASET_NAME = "cds-and-vinyl"
YELP_DATASET_NAME = "yelp2018"
README_FILE_NAME = "README.md"
ITEM_LIST_FILE_NAME = "item_list.txt"
USER_LIST_FILE_NAME = "user_list.txt"
DATASET_FILES = (README_FILE_NAME, TRAIN_FILE_NAME, TEST_FILE_NAME, ITEM_LIST_FILE_NAME, USER_LIST_FILE_NAME)
DATASET_REMOTE_FILES = {
    AMAZON_DATASET_NAME: DATASET_FILES,
    YELP_DATASET_NAME: (TRAIN_FILE_NAME, TEST_FILE_NAME, ITEM_LIST_FILE_NAME, USER_LIST_FILE_NAME),
}
LIGHTGCN_RAW_URL = "https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data"
LIGHTGCN_YELP_SOURCE_URL = "https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data/yelp2018"

# Amazon category-metadata sources: three Amazon sub-datasets this repo
# evaluates x three vintages (category naming verified directly against each
# year's directory listing: 2014 https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/,
# 2018 .../amazon_v2/metaFiles2/, 2023 .../amazon_2023/raw/meta_categories/).
# tools/data/build_musical_instruments_metadata.py and
# tools/data/build_arts_crafts_and_sewing_metadata.py are the "additive
# companion" scripts (see their own docstrings) that consume the latter two;
# AMAZON_METADATA_URLS/AMAZON_METADATA_SOURCES below are the Books-only
# aliases the pre-existing download/prepare_metadata call sites import.
AMAZON_CATEGORY_NAMES = {
    AMAZON_DATASET_NAME: "Books",
    MUSICAL_INSTRUMENTS_DATASET_NAME: "Musical_Instruments",
    ARTS_CRAFTS_AND_SEWING_DATASET_NAME: "Arts_Crafts_and_Sewing",
    CDS_AND_VINYL_DATASET_NAME: "CDs_and_Vinyl",
}
_AMAZON_METADATA_URL_TEMPLATES = {
    "2014": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_{}.json.gz",
    "2018": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_{}.json.gz",
    "2023": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{}.jsonl.gz",
}
AMAZON_CATEGORY_METADATA_URLS = {
    dataset: {year: template.format(category) for year, template in _AMAZON_METADATA_URL_TEMPLATES.items()}
    for dataset, category in AMAZON_CATEGORY_NAMES.items()
}
AMAZON_CATEGORY_METADATA_SOURCES = {
    dataset: {
        "2014": METADATA_DIR / "amazon" / "2014" / ("meta_" + category + ".json"),
        "2018": METADATA_DIR / "amazon" / "2018" / ("meta_" + category + ".json"),
        "2023": METADATA_DIR / "amazon" / "2023" / ("meta_" + category + ".jsonl"),
    }
    for dataset, category in AMAZON_CATEGORY_NAMES.items()
}
AMAZON_METADATA_URLS = AMAZON_CATEGORY_METADATA_URLS[AMAZON_DATASET_NAME]
AMAZON_METADATA_SOURCES = AMAZON_CATEGORY_METADATA_SOURCES[AMAZON_DATASET_NAME]

YELP_METADATA_URLS = {
    "2018": "https://zenodo.org/records/10998102/files/Yelp2018.zip?download=1",
    "2021": "https://zenodo.org/records/10998102/files/Yelp2021.zip?download=1",
    "2022": "https://zenodo.org/records/10998102/files/Yelp2022.zip?download=1",
}
YELP_METADATA_SOURCES = {year: METADATA_DIR / "yelp" / year / ("Yelp" + year + ".item") for year in ("2018", "2021", "2022")}
MERGE_OUTPUT_NAMES = {
    "amazon": {"metadata": "merged_metadata_amazon.json", "categories": "item2category_amazon.json", "decisions": "merge_decisions_amazon.json", "summary": "merge_summary_amazon.json"},
    "yelp": {"metadata": "merged_metadata_yelp.json", "categories": "item2category_yelp.json", "decisions": "merge_decisions_yelp.json", "summary": "merge_summary_yelp.json"},
}
PREPROCESSING_LOG_FILE_NAME = "preprocessing.log"
PREPARE_METADATA_LOG_FILE_NAME = "prepare_metadata.log"
DOWNLOAD_DATA_LOG_FILE_NAME = "download_data.log"


def model_config_file(model_name):
    return CONFIGURE_DIR / (str(model_name) + ".txt")


def verified_dataset_dir(dataset_name):
    return DATASET_VERIFY_DIR / str(dataset_name)


def dataset_split_file(dataset_directory, split):
    names = {
        "train": TRAIN_FILE_NAME,
        "validation": VALIDATION_FILE_NAME,
        "test": TEST_FILE_NAME,
    }
    return Path(dataset_directory) / names[split]


def evaluation_protocol_dir(dataset_name, split_seed=42):
    return (
        PREPROCESSED_DIR
        / "evaluation_protocol"
        / ("split_seed_" + str(int(split_seed)))
        / str(dataset_name)
    )


def taxonomy_variant_dir(dataset_name, policy):
    return TAXONOMY_VARIANT_DIR / str(dataset_name) / str(policy)


def model_result_dir(model_name, dataset_name, seed=None):
    model_name = str(model_name)
    family_root = (
        P0_TAXPROCL_OUTPUT_DIR
        if model_name == "TaxPro-CL"
        else P0_BASELINE_OUTPUT_DIR / model_name
    )
    path = family_root / str(dataset_name)
    return path if seed is None else path / ("seed" + str(seed))


def p0_run_dir(model_name, dataset_name, config_id, seed):
    """Return the canonical artifact directory for one execution run."""
    return (
        model_result_dir(model_name, dataset_name)
        / str(config_id)
        / ("seed" + str(seed))
    )


def training_log_file(model_name, dataset_name, seed=None):
    directory = model_result_dir(model_name, dataset_name, seed)
    return directory / TRAINING_LOG_FILE_NAME


def checkpoint_file(output_dir, checkpoint_kind):
    names = {
        "last": LAST_MODEL_FILE_NAME,
        "best_validation": BEST_VALIDATION_MODEL_FILE_NAME,
    }
    return Path(output_dir) / names[checkpoint_kind]


def result_file(output_dir, result_kind):
    names = {
        "validation": VALIDATION_METRICS_FILE_NAME,
        "final_test": FINAL_TEST_METRICS_FILE_NAME,
        "final_test_group": FINAL_TEST_GROUP_METRICS_FILE_NAME,
    }
    return Path(output_dir) / names[result_kind]


def prototype_bank_file(dataset_name, policy, config_id, seed):
    return (
        PROTOTYPE_DIR
        / str(dataset_name)
        / str(policy)
        / str(config_id)
        / ("seed" + str(seed))
        / "prototype_bank.pt"
    )


def run_manifest_file(output_dir):
    return Path(output_dir) / "run_manifest.json"


def group_metrics_file(output_dir):
    return result_file(output_dir, "final_test_group")


def temporary_file(target, temporary_kind="atomic"):
    suffixes = {
        "atomic": ATOMIC_TEMPORARY_SUFFIX,
        "download": DOWNLOAD_TEMPORARY_SUFFIX,
    }
    if temporary_kind not in suffixes:
        raise ValueError("Unknown temporary-file kind: {}".format(temporary_kind))
    target = Path(target)
    return target.with_name(target.name + suffixes[temporary_kind])


# Bootstrap helpers are used only by tools.  Training and evaluation
# always consume dataset_verify/ and the locked protocol artifacts.
def configured_for_root(root, configured_path, default_relative):
    root = Path(root).expanduser().resolve()
    if root == PROJECT_ROOT.resolve():
        return Path(configured_path)
    return root / Path(default_relative)


def dataset_root_for(root):
    return configured_for_root(root, DATASET_DIR, "dataset")


def metadata_root_for(root):
    return configured_for_root(root, METADATA_DIR, "metadata")


def preprocessing_log_root_for(root):
    return configured_for_root(root, PREPROCESSING_LOG_DIR, "preprocessing_logs")


def dataset_directory(root, dataset_name):
    return dataset_root_for(root) / str(dataset_name)


def amazon_metadata_year_dir(root, year):
    return metadata_root_for(root) / "amazon" / str(year)


def yelp_metadata_root(root):
    return metadata_root_for(root) / "yelp"


def yelp_metadata_year_dir(root, year):
    return yelp_metadata_root(root) / str(year)


def yelp_metadata_archive(root, year):
    return yelp_metadata_root(root) / "Yelp{}.zip".format(year)


def preprocessing_log_path(root, log_kind):
    names = {
        "pipeline": PREPROCESSING_LOG_FILE_NAME,
        "prepare_metadata": PREPARE_METADATA_LOG_FILE_NAME,
        "download": DOWNLOAD_DATA_LOG_FILE_NAME,
    }
    return preprocessing_log_root_for(root) / names[log_kind]


def item_mapping_file(root, domain):
    dataset_name = AMAZON_DATASET_NAME if domain == "amazon" else YELP_DATASET_NAME
    return dataset_directory(root, dataset_name) / ITEM_LIST_FILE_NAME


def amazon_metadata_source_file(root, year):
    return amazon_category_metadata_source_file(root, AMAZON_DATASET_NAME, year)


def amazon_category_metadata_source_file(root, dataset_name, year):
    path = AMAZON_CATEGORY_METADATA_SOURCES[str(dataset_name)][str(year)]
    return configured_for_root(root, path, Path("metadata") / "amazon" / str(year) / path.name)


def yelp_metadata_source_file(root, year):
    path = YELP_METADATA_SOURCES[str(year)]
    return configured_for_root(root, path, Path("metadata") / "yelp" / str(year) / path.name)


def adjacency_cache_file(dataset_directory, cache_kind, alpha=None, beta=None, suffix=True):
    names = {
        "with_self": "pre_A_with_self.npz",
        "adjacency": "pre_A.npz",
        "interaction": "pre_R.npz",
    }
    if cache_kind == "lightgcn_pp":
        name = "pre_A_{}_{}.npz".format(alpha, beta)
    else:
        name = names[cache_kind]
    path = Path(dataset_directory) / name
    return path if suffix else path.with_suffix("")


def relative_to_project(path):
    """Project-relative POSIX path; an absolute path for locations outside the
    repository (e.g. an isolated TAXPRO_OUTPUT_ROOT)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
