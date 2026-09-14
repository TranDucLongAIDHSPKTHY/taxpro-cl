import argparse
import sys


CLI_ALIASES = {
    "training_epochs": ("--epochs",),
}

DEFAULT_TRAINING_SEEDS = (42, 0, 1, 2, 3, 4)


def parse_bool(value):
    """Parse CLI booleans without Python's surprising bool('False') behavior."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def _base_parser(add_help=True):
    parser = argparse.ArgumentParser(description="ID-GRec", add_help=add_help)

    parser.add_argument(
        "--seed_flag", type=parse_bool, default=True, help="Fix random seed or not"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAINING_SEEDS),
        help="random seeds to run sequentially (default: 42 0 1 2 3 4)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="run a single seed instead of --seeds (backward-compatible option)",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="execution device: auto selects CUDA when available, otherwise CPU",
    )

    parser.add_argument(
        "--cuda",
        type=parse_bool,
        default=None,
        help="deprecated compatibility option; prefer --device",
    )

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")

    parser.add_argument("--model", type=str, default="unknown", help="model name")

    return parser


def parse_model_args(argv=None):
    """Read the common options first so the matching model config can be loaded."""
    parser = _base_parser(add_help=False)
    args, _ = parser.parse_known_args(argv)
    actual_argv = sys.argv[1:] if argv is None else argv
    if args.model == "unknown" and any(value in ("-h", "--help") for value in actual_argv):
        _base_parser().parse_args(actual_argv)
    return args


def parse_args(config=None, argv=None):
    parser = _base_parser()

    for key in sorted(config or {}):
        option_strings = ["--" + key]
        option_strings.extend(CLI_ALIASES.get(key, ()))
        parser.add_argument(
            *option_strings,
            dest="config_" + key,
            default=None,
            metavar="VALUE",
            help="Override model configuration: {}".format(key),
        )

    return parser.parse_args(argv)


def apply_config_overrides(config, args):
    """Apply only explicitly supplied CLI values to a model configuration."""
    overridden = dict(config)
    for key in config:
        value = getattr(args, "config_" + key, None)
        if value is not None:
            overridden[key] = value
    return overridden
