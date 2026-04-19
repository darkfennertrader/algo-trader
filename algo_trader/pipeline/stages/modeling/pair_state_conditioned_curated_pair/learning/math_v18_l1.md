**Family 18: `pair_state_conditioned_curated_pair`**

`v18_l1` is the first model opened directly from the pair-state study on the
reduced index universe. The study result was narrow:

- the most durable pair was `IBCH20_minus_IBDE40`
- the most durable ex-ante state for that pair was `range`

So `v18_l1` does not try to supervise the whole reduced index block, and it
does not supervise all curated pairs. It keeps the shared raw-return backbone
and adds one sharply conditioned auxiliary head.

## Structure

The model still produces the usual raw-return predictive distribution for all
assets:

- raw head: full-asset weekly returns

The auxiliary head is restricted to one curated pair spread:

- curated pair head: `IBCH20_minus_IBDE40`

The auxiliary observation is then activated only on weeks classified ex ante as
`range` from the lagged realized pair spread history.

## Lagged pair-state definition

Let `s_t` be the realized weekly spread for the curated pair at week `t`.

For week `t`, the model computes a lagged state from information only through
`t-1`:

- lagged history = `{s_1, ..., s_(t-1)}`
- trend signal at `t` = mean of the most recent `state_window` lagged spreads
- threshold at `t` = expanding median of the absolute trend signal history

The week is labeled `range` when:

- `abs(trend_t) <= threshold_t`

or when the most recent lagged spread is exactly zero.

The first week has insufficient history and is excluded from the curated pair
auxiliary head.

## Likelihood design

The full raw-return head stays primary.

The auxiliary pair head is a standardized Student-t observation on the curated
pair coordinate, but only on `range` weeks:

- raw head: always active
- curated pair range head: active only when the lagged pair state is `range`
- residual auxiliary head: active on all weeks

So the practical objective is:

- keep broad posterior coherence from the raw head
- teach the model about the only pair/state object that survived the research

## Why this version exists

`v17_l1` showed that a curated-pair head by itself was too broad. The pair-state
study then showed that the learnable object is not merely a pair, but a pair
under a specific lagged state. `v18_l1` is the smallest model that encodes that
result directly.
