from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from algo_trader.domain import ConfigError
from algo_trader.pipeline.stages.features import default_registry as feature_registry
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


@dataclass(frozen=True)
class PromptSpec:
    key: str
    label: str
    kind: PromptKind
    choices: tuple[str, ...] | None = None
    default: str | None = None


def run() -> int:
    print("Algo Trader Wizard")
    workflow = _prompt_menu(
        "Select a workflow:",
        [
            ("historical", "historical"),
            ("data_cleaning", "data_cleaning"),
            ("feat. engineering", "feature_engineering"),
            ("data_processing", "data_processing"),
        ],
    )
    command = _build_workflow_command(workflow)

    print("\nGenerated command:")
    print(command.render())
    return 0


def _historical_command() -> WizardCommand:
    args: list[str] = ["algotrader", "historical"]
    config_path = _prompt_optional("Config path (blank for default)")
    if config_path:
        args.extend(["--config", config_path])
    return WizardCommand(commands=[args])


def _data_cleaning_command() -> WizardCommand:
    args: list[str] = ["algotrader", "data_cleaning"]
    start = _prompt_optional("Start month YYYY-MM (blank for full range)")
    if start:
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
    if not groups:
        groups = _feature_group_choices()
    commands: list[list[str]] = []
    first_group = True
    for group in groups:
        if not first_group:
            print()
        first_group = False
        features = _prompt_feature_keys(group)
        args = ["algotrader", "feature_engineering"]
        args.extend(["--group", group])
        for feature in features:
            args.extend(["--feature", feature])
        commands.append(args)
    return WizardCommand(commands=commands)


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
    available = _feature_group_choices()
    label = (
        "groups (comma-separated, blank for all). Options: "
        + ", ".join(available)
    )
    while True:
        raw = _prompt_optional(label)
        if not raw:
            return []
        selected = [item.strip() for item in raw.split(",") if item.strip()]
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
