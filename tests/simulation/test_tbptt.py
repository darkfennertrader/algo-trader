import torch
import pyro

from algo_trader.application.simulation import hooks
from algo_trader.pipeline.stages import modeling


def test_build_tbptt_batches_masks_burn_in_and_invalid_targets() -> None:
    X_train = torch.zeros((5, 2, 1))
    y_train = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [float("nan"), 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ]
    )
    params = hooks._TrainingParams(  # pylint: disable=protected-access
        method="tbptt",
        svi=hooks._SVIParams(  # pylint: disable=protected-access
            steps=1,
            learning_rate=1e-3,
            num_elbo_particles=1,
            log_every=None,
            grad_accum_steps=1,
        ),
        tbptt=hooks._TBPTTParams(  # pylint: disable=protected-access
            window_len=3,
            burn_in_len=1,
        ),
        online_filtering=hooks._OnlineFilteringParams(  # pylint: disable=protected-access
            steps_per_observation=1
        ),
        log_prob_scaling=False,
        target_normalization=False,
    )

    batches = hooks._build_tbptt_batches(  # pylint: disable=protected-access
        X_train, y_train, params
    )

    assert len(batches) == 2
    assert batches[0].M is not None
    assert batches[1].M is not None
    assert batches[0].M.tolist() == [
        [False, False],
        [True, True],
        [False, True],
    ]
    assert batches[1].M.tolist() == [[False, False], [True, True]]
    assert torch.isfinite(batches[0].y).all()


def test_build_online_filtering_batches_preserves_time_order() -> None:
    X_train = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)
    y_train = torch.tensor(
        [
            [1.0, 1.0],
            [float("nan"), 2.0],
            [float("nan"), float("nan")],
            [3.0, 4.0],
        ]
    )
    params = hooks._TrainingParams(  # pylint: disable=protected-access
        method="online_filtering",
        svi=hooks._SVIParams(  # pylint: disable=protected-access
            steps=1,
            learning_rate=1e-3,
            num_elbo_particles=1,
            log_every=None,
            grad_accum_steps=1,
        ),
        tbptt=hooks._TBPTTParams(  # pylint: disable=protected-access
            window_len=2,
            burn_in_len=0,
        ),
        online_filtering=hooks._OnlineFilteringParams(  # pylint: disable=protected-access
            steps_per_observation=2
        ),
        log_prob_scaling=True,
        target_normalization=False,
    )

    batches = hooks._build_online_filtering_batches(  # pylint: disable=protected-access
        X_train, y_train, params
    )

    assert len(batches) == 3
    assert [int(batch.X.shape[0]) for batch in batches] == [1, 1, 1]
    assert [batch.X[0, 0, 0].item() for batch in batches] == [0.0, 2.0, 6.0]
    assert [batch.M.tolist() for batch in batches] == [
        [[True, True]],
        [[False, True]],
        [[True, True]],
    ]
    assert batches[0].obs_scale == 0.5
    assert batches[1].obs_scale == 1.0


def test_param_store_state_round_trip() -> None:
    pyro.clear_param_store()
    pyro.param("alpha_loc", torch.tensor(1.25))
    state = hooks._snapshot_param_store_state()  # pylint: disable=protected-access

    pyro.clear_param_store()
    hooks._restore_param_store_state(  # pylint: disable=protected-access
        {"param_store_state": state}
    )

    restored = pyro.get_param_store().get_param("alpha_loc")
    assert torch.allclose(restored, torch.tensor(1.25))


def test_build_training_state_includes_filtering_state() -> None:
    filtering_state = {
        "h_loc": torch.tensor(0.5),
        "h_scale": torch.tensor(0.2),
        "steps_seen": 7,
    }

    state = hooks._build_training_state(  # pylint: disable=protected-access
        "model_name",
        "guide_name",
        hooks._TrainingArtifacts(  # pylint: disable=protected-access
            norm_state=None,
            posterior_summary=None,
            training_diagnostics=None,
            filtering_state=filtering_state,
        ),
    )

    assert state["filtering_state"]["steps_seen"] == 7
    assert torch.allclose(state["filtering_state"]["h_loc"], torch.tensor(0.5))


def test_build_prediction_batch_preserves_filtering_state() -> None:
    filtering_state = {
        "h_loc": torch.tensor(-0.3),
        "h_scale": torch.tensor(0.4),
        "steps_seen": 9,
    }

    batch = hooks._build_prediction_batch(  # pylint: disable=protected-access
        torch.zeros((1, 2, 1)),
        torch.zeros((1, 1)),
        filtering_state=filtering_state,
    )

    assert batch.filtering_state is filtering_state


def test_online_filtering_steps_return_updated_filtering_state() -> None:
    params = hooks._TrainingParams(  # pylint: disable=protected-access
        method="online_filtering",
        svi=hooks._SVIParams(  # pylint: disable=protected-access
            steps=1,
            learning_rate=1e-3,
            num_elbo_particles=1,
            log_every=None,
            grad_accum_steps=1,
        ),
        tbptt=hooks._TBPTTParams(  # pylint: disable=protected-access
            window_len=2,
            burn_in_len=0,
        ),
        online_filtering=hooks._OnlineFilteringParams(  # pylint: disable=protected-access
            steps_per_observation=1
        ),
        log_prob_scaling=False,
        target_normalization=False,
    )
    model = modeling.default_model_registry().get(
        "factor_model_l10_online_filtering"
    )
    guide = modeling.default_guide_registry().get(
        "factor_guide_l10_online_filtering"
    )
    batches = [
        modeling.ModelBatch(
            X_asset=torch.zeros((1, 2, 1)),
            X_global=torch.zeros((1, 1)),
            y=torch.zeros((1, 2)),
        ),
        modeling.ModelBatch(
            X_asset=torch.ones((1, 2, 1)),
            X_global=torch.ones((1, 1)),
            y=torch.full((1, 2), 0.1),
        ),
    ]

    pyro.clear_param_store()
    loss_history, filtering_state = hooks._run_online_filtering_steps(  # pylint: disable=protected-access
        svi=hooks._build_svi(  # pylint: disable=protected-access
            model=model, guide=guide, params=params
        ),
        batches=batches,
        params=params,
        context={},
        initial_filtering_state={
            "h_loc": torch.tensor(0.25),
            "h_scale": torch.tensor(0.15),
            "steps_seen": 5,
        },
    )

    assert len(loss_history) == 2
    assert filtering_state is not None
    assert int(getattr(filtering_state, "steps_seen")) == 7
