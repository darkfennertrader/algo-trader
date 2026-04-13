# Family 15, Version v15_l1: Multi-Output Index-Relative Model

This family keeps the trusted raw-return probabilistic backbone, but adds a
second supervised output head for the index block in relative coordinates.

The design goal is to stop asking one output object to solve two different
tasks:

- raw return forecasting for the whole mixed universe
- index-relative discrimination inside the equity-index slice

## Core Idea

The model uses one shared latent backbone and two observation heads.

Head 1: raw-return head
- predicts next-period raw returns for all assets
- keeps the same posterior object needed by the rest of the pipeline

Head 2: index-relative head
- uses only the index block
- projects index returns into relative coordinates
- supervises those coordinates with a weighted auxiliary likelihood

The raw-return head remains primary.
The index-relative head is secondary and acts as a structured auxiliary task.

## Shared Latent Backbone

The latent structure is the same trusted online-filtering backbone:

- latent factor structure
- time-varying regime path
- index-t copula mixture for heavy-tailed dependence

So the family change is not a new latent-state design.
It is a new multi-output observation design.

## Head 1: Raw Returns

For each time t, the model defines the usual raw-return distribution:

- y_t ~ p_raw(y_t | latent_state_t)

This is the same object that later produces:

- posterior predictive samples
- posterior mean
- posterior covariance

## Head 2: Index-Relative Coordinates

Let y_index_t be the vector of index returns at time t.

Build a basis B whose columns define:

- index level
- US relative to equal-weight index basket
- Europe relative to equal-weight index basket
- US minus Europe spread
- residual index directions completing the basis

Then the relative coordinates are:

- z_t = transpose(y_index_t) * B

The model also projects the raw-head predictive moments into this basis:

- mu_z_t = transpose(mu_index_t) * B
- Sigma_z_t = transpose(B) * Sigma_index_t * B

Coordinates are then standardized robustly with:

- coordinate median as center
- median absolute deviation as scale

so the auxiliary supervision acts on comparable relative units.

## Likelihood Structure

The training objective contains:

- one raw-return likelihood for all assets
- one auxiliary Student-t likelihood on the standardized index-relative groups

In plain text:

- total objective = raw-return term + weighted index-relative term

This is why the family is multi-output:

- same latent backbone
- two observation heads

## Why This Family Exists

Family 13 and Family 14 showed:

- index calibration and ordering can improve
- but monetizable index signal remains weak

That suggests the index slice may need explicit supervision as its own target,
not only a side regularizer or a measurement replacement.

`v15_l1` is the first version that makes that idea explicit.
