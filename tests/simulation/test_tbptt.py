import torch
import pyro

from algo_trader.application.simulation import hooks


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
