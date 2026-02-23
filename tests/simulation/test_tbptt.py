import torch

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
