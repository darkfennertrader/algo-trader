import pytest

from pytest import MonkeyPatch

from algo_trader.cli import wizard
from algo_trader.domain import ConfigError


def _option(
    group: str, key: str, *outputs: str
) -> wizard.KeyFeatureOption:
    return wizard.KeyFeatureOption(
        group=group,
        feature_key=key,
        output_names=tuple(outputs),
    )


def test_sample_feature_keys_exact_count_is_seeded() -> None:
    options = [
        _option("momentum", "vol_scaled_momentum", "z_mom_4w"),
        _option("breakout", "brk_up", "brk_up_4w"),
        _option("breakout", "brk_dn", "brk_dn_4w", "brk_dn_26w"),
    ]
    first = wizard._sample_feature_keys_exact_count(
        options=options, count=2, seed=13
    )
    second = wizard._sample_feature_keys_exact_count(
        options=options, count=2, seed=13
    )
    assert first == second
    assert sum(len(item.output_names) for item in first) == 2


def test_sample_feature_keys_exact_count_fails_when_not_feasible() -> None:
    options = [_option("cross_sectional", "cs_rank", "a", "b", "c", "d", "e", "f")]
    with pytest.raises(
        ConfigError, match="Requested random output-feature count is not feasible"
    ):
        wizard._sample_feature_keys_exact_count(
            options=options, count=5, seed=7
        )


def test_build_feature_commands_skips_empty_random_groups() -> None:
    groups = ["momentum", "breakout"]
    features_by_group = {
        "momentum": ["vol_scaled_momentum"],
        "breakout": [],
    }
    commands = wizard._build_feature_commands(groups, features_by_group)
    assert commands == [
        [
            "algotrader",
            "feature_engineering",
            "--group",
            "momentum",
            "--feature",
            "vol_scaled_momentum",
        ]
    ]


def test_feature_engineering_command_random_exact(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        wizard, "_prompt_feature_groups", lambda: ["momentum", "breakout"]
    )
    monkeypatch.setattr(
        wizard, "_prompt_feature_selection_mode", lambda groups: "random_count"
    )
    monkeypatch.setattr(wizard, "_prompt_required_int", lambda label: 3)
    monkeypatch.setattr(wizard, "_prompt_optional_int", lambda label: 13)
    monkeypatch.setattr(
        wizard,
        "_key_feature_pool",
        lambda groups: [
            _option("momentum", "vol_scaled_momentum", "z_mom_4w"),
            _option("breakout", "brk_dn", "brk_dn_4w", "brk_dn_26w"),
        ],
    )
    monkeypatch.setattr(
        wizard,
        "_print_random_selection",
        lambda selected_keys, seed, target_count: None,
    )

    command = wizard._feature_engineering_command()

    assert command.commands == [
        [
            "algotrader",
            "feature_engineering",
            "--group",
            "momentum",
            "--feature",
            "vol_scaled_momentum",
        ],
        [
            "algotrader",
            "feature_engineering",
            "--group",
            "breakout",
            "--feature",
            "brk_dn",
        ],
    ]


def test_feature_engineering_command_all_shortcut(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(wizard, "_prompt_feature_groups", lambda: ["all"])
    monkeypatch.setattr(
        wizard, "_prompt_feature_selection_mode", lambda groups: "all"
    )

    command = wizard._feature_engineering_command()

    assert command.commands == [["algotrader", "feature_engineering", "--group", "all"]]


def test_key_feature_pool_breakout_includes_all_outputs() -> None:
    pool = wizard._key_feature_pool(["breakout"])
    assert sorted((item.group, item.feature_key) for item in pool) == [
        ("breakout", "brk_dn"),
        ("breakout", "brk_up"),
    ]
    names = sorted(
        output
        for item in pool
        for output in item.output_names
    )
    assert names == [
        "brk_dn_26w",
        "brk_dn_4w",
        "brk_up_26w",
        "brk_up_4w",
    ]
