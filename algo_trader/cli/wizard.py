from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from functools import lru_cache
import random
from typing import Callable, Iterable, Literal, Sequence, cast

import numpy as np
import pandas as pd

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.features import default_registry as feature_registry
from algo_trader.pipeline.stages.features.protocols import FeatureGroup, FeatureInputs
from algo_trader.pipeline.stages.features.breakout import (
    SUPPORTED_FEATURES as BREAKOUT_FEATURES,
)
from algo_trader.pipeline.stages.features.cross_sectional import (
    SUPPORTED_FEATURES as CROSS_SECTIONAL_FEATURES,
)
from algo_trader.pipeline.stages.features.mean_reversion import (
    SUPPORTED_FEATURES as MEAN_REVERSION_FEATURES,
)
from algo_trader.pipeline.stages.features.momentum import (
    SUPPORTED_FEATURES as MOMENTUM_FEATURES,
)
from algo_trader.pipeline.stages.features.seasonal import (
    SUPPORTED_FEATURES as SEASONAL_FEATURES,
)
from algo_trader.pipeline.stages.features.volatility import (
    SUPPORTED_FEATURES as VOLATILITY_FEATURES,
)
from algo_trader.pipeline.stages.features.regime import (
    SUPPORTED_FEATURES as REGIME_FEATURES,
)
from algo_trader.preprocessing import default_registry as preprocessor_registry


@dataclass(frozen=True)
class WizardCommand:
    commands: list[list[str]]

    def render(self) -> str:
        return "\n".join(
            f"uv run {' '.join(args)}" for args in self.commands
        )


PromptKind = Literal["optional", "choice"]
FeatureSelectionMode = Literal["all", "manual", "random_count"]
SelectedFeaturesByGroup = dict[str, Sequence[str] | None]
_SAMPLE_ASSETS: tuple[str, ...] = ("SYN_A", "SYN_B", "SYN_C")


@dataclass(frozen=True)
class PromptSpec:
    key: str
    label: str
    kind: PromptKind
    choices: tuple[str, ...] | None = None
    default: str | None = None


@dataclass(frozen=True)
class KeyFeatureOption:
    group: str
    feature_key: str
    output_names: tuple[str, ...]


def run() -> int:
    print("Algo Trader Wizard")
    workflow = _prompt_menu(
        "Select a workflow:",
        [
            ("historical", "historical"),
            ("data_cleaning", "data_cleaning"),
            ("feat. engineering", "feature_engineering"),
            ("data_processing", "data_processing"),
            ("simulation", "simulation"),
        ],
    )
    command = _build_workflow_command(workflow)

    print("\nGenerated command:")
    print(command.render())
    if workflow == "simulation":
        print(
            "\nReminder: simulation behavior is controlled by "
            "config/simulation.yml."
        )
    return 0


def _historical_command() -> WizardCommand:
    args: list[str] = ["algotrader", "historical"]
    config_path = _prompt_optional("Config path (blank for default)")
    if config_path:
        args.extend(["--config", config_path])
    return WizardCommand(commands=[args])


def _data_cleaning_command() -> WizardCommand:
    args: list[str] = ["algotrader", "data_cleaning"]
    start = _prompt_optional("Start month YYYY-MM")
    while not start:
        print("Start month is required.")
        start = _prompt_optional("Start month YYYY-MM")
    args.extend(["--start", start])
    end = _prompt_optional("End month YYYY-MM (blank for full range)")
    if end:
        args.extend(["--end", end])
    return_type = _prompt_choice(
        "return type", ["simple", "log"], default="simple"
    )
    args.extend(["--return-type", return_type])
    assets = _prompt_optional("Assets (comma-separated, blank for config)")
    if assets:
        args.extend(["--assets", assets])
    return WizardCommand(commands=[args])


def _data_processing_command() -> WizardCommand:
    args: list[str] = ["algotrader", "data_processing"]
    preprocessor = _prompt_preprocessor()
    if preprocessor:
        args.extend(["--preprocessor", preprocessor])
    args.extend(_prompt_preprocessor_args(preprocessor))
    return WizardCommand(commands=[args])


def _feature_engineering_command() -> WizardCommand:
    groups = _prompt_feature_groups()
    mode = _prompt_feature_selection_mode(groups)
    if mode == "all" and (not groups or groups == ["all"]):
        return WizardCommand(
            commands=[["algotrader", "feature_engineering", "--group", "all"]]
        )
    selected_groups = _resolve_feature_groups(groups)
    features_by_group = _select_features_by_group(selected_groups, mode)
    commands = _build_feature_commands(selected_groups, features_by_group)
    return WizardCommand(commands=commands)


def _simulation_command() -> WizardCommand:
    args: list[str] = ["algotrader", "simulation"]
    config_path = _prompt_optional(
        "Simulation config path (blank for config/simulation.yml)"
    )
    if config_path:
        args.extend(["--simulation-config", config_path])
    resume = _prompt_choice(
        "Resume latest interrupted Ray Tune experiment?",
        ["no", "yes"],
        default="no",
    )
    if resume == "yes":
        args.append("--resume")
    return WizardCommand(commands=[args])


def _prompt_preprocessor() -> str:
    registry = preprocessor_registry()
    names = registry.list_names()
    return _prompt_choice("preprocessor", names, default="identity")


def _prompt_preprocessor_args(preprocessor: str) -> list[str]:
    builder = _preprocessor_arg_builders().get(
        preprocessor, _generic_preprocessor_args
    )
    return builder()


def _build_workflow_command(workflow: str) -> WizardCommand:
    builders = _workflow_builders()
    builder = builders.get(workflow)
    if builder is None:
        raise ConfigError(f"Unknown workflow '{workflow}'")
    return builder()


def _workflow_builders() -> dict[str, Callable[[], WizardCommand]]:
    return {
        "historical": _historical_command,
        "data_cleaning": _data_cleaning_command,
        "data_processing": _data_processing_command,
        "feature_engineering": _feature_engineering_command,
        "simulation": _simulation_command,
    }


def _preprocessor_arg_builders() -> dict[str, Callable[[], list[str]]]:
    return {
        "identity": _identity_preprocessor_args,
        "pca": _pca_preprocessor_args,
        "zscore": _zscore_preprocessor_args,
    }


def _identity_preprocessor_args() -> list[str]:
    return _build_preprocessor_args(
        [
            PromptSpec(
                key="copy",
                label="copy (true/false, blank for 'false')",
                kind="optional",
            ),
            PromptSpec(
                key="pipeline",
                label="pipeline (blank for default debug)",
                kind="optional",
            ),
        ]
    )


def _zscore_preprocessor_args() -> list[str]:
    return _build_preprocessor_args(
        [
            PromptSpec(
                key="start_date",
                label="start_date YYYY-MM-DD (blank for full range)",
                kind="optional",
            ),
            PromptSpec(
                key="end_date",
                label="end_date YYYY-MM-DD (blank for full range)",
                kind="optional",
            ),
            PromptSpec(
                key="missing",
                label="missing",
                kind="choice",
                choices=("zero", "drop"),
                default="zero",
            ),
            PromptSpec(
                key="pipeline",
                label="pipeline (blank for default debug)",
                kind="optional",
            ),
        ]
    )


def _pca_preprocessor_args() -> list[str]:
    return _build_preprocessor_args(
        [
            PromptSpec(
                key="k",
                label="k (number of factors, blank if using variance)",
                kind="optional",
            ),
            PromptSpec(
                key="variance",
                label="variance target (0-1, blank if using k)",
                kind="optional",
            ),
            PromptSpec(
                key="start_date",
                label="start_date YYYY-MM-DD (blank for full range)",
                kind="optional",
            ),
            PromptSpec(
                key="end_date",
                label="end_date YYYY-MM-DD (blank for full range)",
                kind="optional",
            ),
            PromptSpec(
                key="missing",
                label="missing",
                kind="choice",
                choices=("zero", "drop"),
                default="zero",
            ),
            PromptSpec(
                key="pipeline",
                label="pipeline (blank for default debug)",
                kind="optional",
            ),
        ]
    )


def _generic_preprocessor_args() -> list[str]:
    return _build_preprocessor_args(
        [
            PromptSpec(
                key="pipeline",
                label="pipeline (blank for default debug)",
                kind="optional",
            )
        ]
    )


def _build_preprocessor_args(specs: Iterable[PromptSpec]) -> list[str]:
    args: list[str] = []
    for spec in specs:
        if spec.kind == "optional":
            value = _prompt_optional(spec.label)
            if value:
                args.append(f"--preprocessor-arg {spec.key}={value}")
            continue
        if spec.kind == "choice":
            if spec.choices is None:
                raise ConfigError(f"{spec.key} requires choices")
            value = _prompt_choice(
                spec.label, list(spec.choices), default=spec.default
            )
            args.append(f"--preprocessor-arg {spec.key}={value}")
            continue
        raise ConfigError(f"Unknown prompt kind '{spec.kind}'")
    return args


def _prompt_choice(
    label: str, options: Iterable[str], default: str | None = None
) -> str:
    choices = list(options)
    if not choices:
        raise ConfigError(f"No options available for {label}")
    numbered = ", ".join(
        f"{index}={choice}" for index, choice in enumerate(choices, start=1)
    )
    default_text = f" [{default}]" if default else ""
    while True:
        raw = input(f"{label} ({numbered}){default_text}: ").strip()
        if not raw and default:
            return default
        if raw.isdigit():
            selected = int(raw)
            if 1 <= selected <= len(choices):
                return choices[selected - 1]
        if raw in choices:
            return raw
        print(f"Invalid {label}. Options: {numbered}")


def _prompt_optional(label: str) -> str:
    return input(f"{label}: ").strip()


def _prompt_menu(
    title: str, options: Iterable[tuple[str, str]]
) -> str:
    menu = list(options)
    if not menu:
        raise ConfigError("No options available")
    print(title)
    for index, (label, _) in enumerate(menu, start=1):
        print(f"{index}) {label}")
    while True:
        raw = input("Select number: ").strip()
        if raw.isdigit():
            selected = int(raw)
            if 1 <= selected <= len(menu):
                return menu[selected - 1][1]
        print("Invalid selection. Enter a number from the menu.")


def _feature_group_choices() -> list[str]:
    registry = feature_registry()
    return registry.list_names()


def _prompt_feature_groups() -> list[str]:
    available = [*_feature_group_choices(), "all"]
    label = (
        "groups (comma-separated, blank for all/parallel). Options: "
        + ", ".join(available)
    )
    while True:
        raw = _prompt_optional(label)
        if not raw:
            return []
        selected = [item.strip() for item in raw.split(",") if item.strip()]
        if "all" in selected:
            if len(selected) > 1:
                print("Invalid groups: all cannot be combined with others")
                continue
            return ["all"]
        unknown = sorted(set(selected).difference(available))
        if not unknown:
            return selected
        print(f"Invalid groups: {', '.join(unknown)}")


def _prompt_feature_keys(group: str) -> list[str]:
    supported = _feature_keys_by_group().get(group)
    if supported is None:
        raise ConfigError(f"Unknown feature group '{group}'")
    label = (
        f"features for {group} (comma-separated, blank for all). "
        f"Options: {', '.join(supported)}"
    )
    while True:
        raw = _prompt_optional(label)
        if not raw:
            return []
        selected = [item.strip() for item in raw.split(",") if item.strip()]
        unknown = sorted(set(selected).difference(supported))
        if not unknown:
            return selected
        print(f"Invalid features: {', '.join(unknown)}")


def _prompt_feature_selection_mode(groups: Sequence[str]) -> FeatureSelectionMode:
    default = "all" if not groups or groups == ["all"] else "manual"
    mode = _prompt_choice(
        "feature selection mode",
        ["all", "manual", "random_count"],
        default=default,
    )
    return cast(FeatureSelectionMode, mode)


def _resolve_feature_groups(groups: Sequence[str]) -> list[str]:
    if not groups or groups == ["all"]:
        return _feature_group_choices()
    return list(groups)


def _select_features_by_group(
    groups: Sequence[str], mode: FeatureSelectionMode
) -> SelectedFeaturesByGroup:
    if mode == "all":
        return {group: None for group in groups}
    if mode == "manual":
        return _select_features_manually(groups)
    return _select_features_randomly(groups)


def _select_features_manually(groups: Sequence[str]) -> SelectedFeaturesByGroup:
    selections: SelectedFeaturesByGroup = {}
    first_group = True
    for group in groups:
        if not first_group:
            print()
        first_group = False
        selected = _prompt_feature_keys(group)
        selections[group] = selected if selected else None
    return selections


def _select_features_randomly(groups: Sequence[str]) -> SelectedFeaturesByGroup:
    options = _key_feature_pool(groups)
    feasible_counts = sorted(_feasible_output_counts(options))
    positive_counts = [value for value in feasible_counts if value > 0]
    if not positive_counts:
        raise ConfigError(
            "No feasible random output-feature counts found",
            context={"groups": ",".join(groups)},
        )
    count = _prompt_required_int(_random_count_prompt(positive_counts))
    if count not in positive_counts:
        raise ConfigError(
            "Requested random output-feature count is not feasible",
            context={
                "count": str(count),
                "allowed": _format_count_ranges(positive_counts),
            },
        )
    seed = _prompt_optional_int("Random seed (blank for non-deterministic)")
    selected_keys = _sample_feature_keys_exact_count(
        options=options,
        count=count,
        seed=seed,
    )
    grouped = _group_selected_keys(
        groups=groups, selected_keys=selected_keys
    )
    _print_random_selection(
        selected_keys=selected_keys,
        seed=seed,
        target_count=count,
    )
    return grouped


def _keys_for_group(group: str) -> list[str]:
    by_group = _feature_keys_by_group()
    supported = by_group.get(group)
    if supported is None:
        raise ConfigError(f"Unknown feature group '{group}'")
    return supported


def _key_feature_pool(groups: Sequence[str]) -> list[KeyFeatureOption]:
    registry = feature_registry()
    inputs = _sample_feature_inputs()
    options: list[KeyFeatureOption] = []
    seen_output_names: dict[str, str] = {}
    for group in groups:
        keys = _keys_for_group(group)
        group_runner = registry.get(group)
        for key in keys:
            output_names = tuple(
                _expanded_outputs_for_key(
                    group_runner=group_runner,
                    inputs=inputs,
                    group=group,
                    key=key,
                )
            )
            output_names = tuple(sorted(set(output_names)))
            option_id = f"{group}:{key}"
            for output_name in output_names:
                previous = seen_output_names.get(output_name)
                if previous is not None:
                    raise ConfigError(
                        "Expanded output feature overlap detected for wizard",
                        context={
                            "output": output_name,
                            "first": previous,
                            "second": option_id,
                        },
                    )
                seen_output_names[output_name] = option_id
            options.append(
                KeyFeatureOption(
                    group=group,
                    feature_key=key,
                    output_names=output_names,
                )
            )
    return options


def _random_count_prompt(positive_counts: Sequence[int]) -> str:
    minimum = positive_counts[0]
    maximum = positive_counts[-1]
    if _is_contiguous(positive_counts):
        return f"Random output-feature count (exact, {minimum}..{maximum})"
    allowed = _format_count_ranges(positive_counts)
    return f"Random output-feature count (exact; allowed: {allowed})"


def _is_contiguous(values: Sequence[int]) -> bool:
    if not values:
        return False
    return values[-1] - values[0] + 1 == len(values)


def _format_count_ranges(values: Sequence[int]) -> str:
    if not values:
        return ""
    ranges: list[str] = []
    start = values[0]
    end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append(_format_single_range(start, end))
        start = value
        end = value
    ranges.append(_format_single_range(start, end))
    return ",".join(ranges)


def _format_single_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _feasible_output_counts(options: Sequence[KeyFeatureOption]) -> set[int]:
    possible = {0}
    for option in options:
        weight = len(option.output_names)
        additions = {weight + value for value in possible}
        possible.update(additions)
    return possible


def _sample_feature_keys_exact_count(
    *,
    options: Sequence[KeyFeatureOption],
    count: int,
    seed: int | None,
) -> list[KeyFeatureOption]:
    if count <= 0:
        raise ConfigError(
            "random feature count must be positive",
            context={"count": str(count)},
        )
    shuffled = list(options)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    suffix_possible = _suffix_possible_counts(shuffled)
    if count not in suffix_possible[0]:
        raise ConfigError(
            "Requested random output-feature count is not feasible",
            context={
                "count": str(count),
                "allowed": _format_count_ranges(sorted(suffix_possible[0])),
            },
        )
    remaining = count
    selected: list[KeyFeatureOption] = []
    for index, option in enumerate(shuffled):
        weight = len(option.output_names)
        include_is_possible = (
            weight <= remaining and (remaining - weight) in suffix_possible[index + 1]
        )
        skip_is_possible = remaining in suffix_possible[index + 1]
        include = False
        if include_is_possible and skip_is_possible:
            include = bool(rng.getrandbits(1))
        elif include_is_possible:
            include = True
        if include:
            selected.append(option)
            remaining -= weight
    if remaining != 0:
        raise ConfigError(
            "Failed to select an exact random output-feature count",
            context={"remaining": str(remaining), "target": str(count)},
        )
    return selected


def _suffix_possible_counts(options: Sequence[KeyFeatureOption]) -> list[set[int]]:
    possible_by_index: list[set[int]] = [set() for _ in range(len(options) + 1)]
    possible_by_index[len(options)] = {0}
    for index in range(len(options) - 1, -1, -1):
        tail = possible_by_index[index + 1]
        weight = len(options[index].output_names)
        combined = set(tail)
        combined.update({weight + value for value in tail})
        possible_by_index[index] = combined
    return possible_by_index


def _group_selected_keys(
    *,
    groups: Sequence[str],
    selected_keys: Sequence[KeyFeatureOption],
) -> SelectedFeaturesByGroup:
    grouped: SelectedFeaturesByGroup = {}
    for group in groups:
        grouped[group] = []
    for selected in selected_keys:
        current = grouped.get(selected.group)
        if current is None:
            current = []
        grouped[selected.group] = [*current, selected.feature_key]
    for group in groups:
        current = grouped[group]
        if current is None:
            continue
        grouped[group] = sorted(set(current))
    return grouped


def _print_random_selection(
    *,
    selected_keys: Sequence[KeyFeatureOption],
    seed: int | None,
    target_count: int,
) -> None:
    selected_outputs: set[str] = set()
    grouped_outputs: dict[str, list[str]] = {}
    for selected in selected_keys:
        outputs = grouped_outputs.setdefault(selected.group, [])
        outputs.extend(selected.output_names)
        selected_outputs.update(selected.output_names)
    if len(selected_outputs) != target_count:
        raise ConfigError(
            "Random selection did not match requested output-feature count",
            context={
                "target": str(target_count),
                "selected": str(len(selected_outputs)),
            },
        )
    seed_text = "none" if seed is None else str(seed)
    print(
        f"\nRandom output-feature selection (seed={seed_text}, "
        f"count={len(selected_outputs)}):"
    )
    for group in sorted(grouped_outputs):
        unique_outputs = sorted(set(grouped_outputs[group]))
        print(f"- {group}: {', '.join(unique_outputs)}")
    print(
        "Generated commands use the minimum feature keys that reproduce this "
        "exact output-feature count."
    )

def _expanded_outputs_for_key(
    *,
    group_runner: object,
    inputs: FeatureInputs,
    group: str,
    key: str,
) -> list[str]:
    selected_group = cast(FeatureGroup, _clone_group_with_feature(group_runner, key))
    try:
        output = selected_group.compute(inputs)
    except Exception as exc:
        raise ConfigError(
            "Failed to resolve expanded output features for wizard",
            context={"group": group, "feature": key},
        ) from exc
    feature_names = list(output.feature_names)
    if not feature_names:
        raise ConfigError(
            "No expanded output features found for wizard",
            context={"group": group, "feature": key},
        )
    return feature_names


def _clone_group_with_feature(group_runner: object, key: str) -> FeatureGroup:
    config = getattr(group_runner, "_config", None)
    if config is None:
        raise ConfigError(
            "Feature group config unavailable for wizard",
            context={"group_type": type(group_runner).__name__},
        )
    try:
        selected_config = replace(config, features=[key])
    except TypeError as exc:
        raise ConfigError(
            "Feature group config is not replaceable for wizard",
            context={"group_type": type(group_runner).__name__},
        ) from exc
    selected_group = copy.copy(group_runner)
    setattr(selected_group, "_config", selected_config)
    return cast(FeatureGroup, selected_group)


@lru_cache(maxsize=1)
def _sample_feature_inputs() -> FeatureInputs:
    return _build_sample_feature_inputs()


def _build_sample_feature_inputs() -> FeatureInputs:
    weekly_index = pd.date_range("2018-01-05", periods=260, freq="W-FRI")
    daily_index = pd.bdate_range("2018-01-01", periods=1500)
    weekly_ohlc = _build_sample_ohlc(weekly_index)
    daily_ohlc = _build_sample_ohlc(daily_index)
    return FeatureInputs(
        frames={"weekly_ohlc": weekly_ohlc, "daily_ohlc": daily_ohlc},
        frequency="weekly",
    )


def _build_sample_ohlc(index: pd.DatetimeIndex) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [_SAMPLE_ASSETS, ("Open", "High", "Low", "Close")]
    )
    values = np.zeros((len(index), len(columns)), dtype=float)
    base_line = np.linspace(0.0, 20.0, len(index))
    cycle = np.sin(np.linspace(0.0, 18.0, len(index)))
    for offset, _asset in enumerate(_SAMPLE_ASSETS):
        close = 100.0 + (offset * 5.0) + base_line + (cycle * (1.0 + offset * 0.1))
        open_price = close * 0.999
        high = close * 1.005
        low = close * 0.995
        start = offset * 4
        values[:, start : start + 4] = np.column_stack(
            [open_price, high, low, close]
        )
    return pd.DataFrame(values, index=index, columns=columns)


def _build_feature_commands(
    groups: Sequence[str], features_by_group: SelectedFeaturesByGroup
) -> list[list[str]]:
    commands: list[list[str]] = []
    for group in groups:
        selected = features_by_group.get(group)
        if selected is not None and len(selected) == 0:
            continue
        args = ["algotrader", "feature_engineering", "--group", group]
        if selected:
            for feature in selected:
                args.extend(["--feature", feature])
        commands.append(args)
    if not commands:
        raise ConfigError("No features selected")
    return commands


def _prompt_required_int(label: str) -> int:
    raw = _prompt_optional(label)
    if not raw:
        raise ConfigError(f"{label} is required")
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{label} must be an integer") from exc


def _prompt_optional_int(label: str) -> int | None:
    raw = _prompt_optional(label)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{label} must be an integer") from exc


def _feature_keys_by_group() -> dict[str, list[str]]:
    return {
        "momentum": sorted(MOMENTUM_FEATURES),
        "mean_reversion": sorted(MEAN_REVERSION_FEATURES),
        "breakout": sorted(BREAKOUT_FEATURES),
        "cross_sectional": sorted(CROSS_SECTIONAL_FEATURES),
        "volatility": sorted(VOLATILITY_FEATURES),
        "seasonal": sorted(SEASONAL_FEATURES),
        "regime": sorted(REGIME_FEATURES),
    }
