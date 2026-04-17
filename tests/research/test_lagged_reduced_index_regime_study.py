from __future__ import annotations

from pathlib import Path

from algo_trader.application.research.lagged_reduced_index_regime_study import (
    run_lagged_reduced_index_regime_study,
    write_lagged_reduced_index_regime_outputs,
    write_lagged_reduced_index_regime_plots,
)
from tests.research.regime_study_test_support import (
    assert_regime_outputs_exist,
    sample_regime_observations,
)


def test_lagged_reduced_index_regime_study_writes_expected_outputs(
    tmp_path: Path,
) -> None:
    result = run_lagged_reduced_index_regime_study(sample_regime_observations())

    assert "all_weeks" in result.summary["regime_label"].tolist()
    assert "insufficient_history" in result.summary["regime_label"].tolist()
    assert result.definitions["regime_label"].nunique() >= 2

    write_lagged_reduced_index_regime_outputs(result=result, output_dir=tmp_path)
    write_lagged_reduced_index_regime_plots(result=result, output_dir=tmp_path)

    assert_regime_outputs_exist(tmp_path)
