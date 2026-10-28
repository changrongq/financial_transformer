# Inverse RL Market Simulation Experiment Plan (2026-05-29)

## Goal

Use a small cross-stock basket to determine whether the current 3-agent MVP captures reusable market-formation structure, or only fits stock `300124`.

## Guiding Rule

Do not change the objective or add structural role preferences until the current setup is tested on clearly different stock styles.

## Basket

### Control

- `300124`: current reference case, medium volatility, medium turnover, already non-collapsed.

### Contrast Set

- `000001`: low-vol financial, low turnover, should stress whether the current setup still avoids degeneracy in calmer names.
- `300344`: high-vol, high-turnover extreme name, should reveal whether herd and panic components dominate naturally.
- `601857`: low-turnover large-cap cyclical, should test whether slow capital becomes relatively more important.
- `000538`: defensive low-vol name, reserve case if one of the above becomes uninformative.

## Execution Order

### Stage 1: Scout Runs

Run `40` trials per contrast stock in this order:

1. `000001`
2. `300344`
3. `601857`

Promote only informative names to Stage 2.

### Stage 2: Full Runs

Run full `80` trial calibrations for the two most informative contrast names.

### Stage 3: Decision

Choose the next model change only after comparing all reports across the basket.

## What Counts As Informative

- best loss clearly below both baselines
- low-vol penalty remains controlled
- zero-drawdown penalty remains zero or near-zero after fixes
- agent roles differ across stock styles rather than collapsing into the same pattern
- top-k trend drift helps explain whether trend is structurally useful or only conditionally present

## Decision Rules

- If low-vol names still collapse while active names do not, the next change should target state dynamics for calm regimes.
- If high-vol names fit only through herd and trend stays fragile everywhere, the next change should target trend state design, not trend share constraints.
- If roles separate differently across stock styles, keep the 3-agent structure and extend validation before expanding model size.