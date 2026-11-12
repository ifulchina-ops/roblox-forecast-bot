# Roblox Up-and-Coming top-10 forecaster

Entry for make.fun bounty `roblox-up-and-coming-forecast-01` — predict Roblox's
Up-and-Coming top 10 for every hour of the 72 hours after entries close.

## What it does

`main.py` (Python 3.13, standard library only, no network, deterministic) reads
`/input/history.ndjson` + `/input/task.json` and writes `/output/forecast.json`:
for each of places 1–10, a probability for each named game being at that place,
plus the guessed top 10 in order. Score is the mean log score over the 72
hourly charts, so the forecast is the time average of the per-hour place
marginals — the log-score-optimal single forecast for a window scored as the
mean over its reads.

## Method

Latent-score diffusion Monte Carlo (3,000 runs, fixed seed), with every
parameter measured on the public chart archive (34 reads, 2026-10-30–11-07):

- **Scores:** each game has a latent score; the chart order is the order of
  scores. Scores start at −rank (last read), drift by recent rank velocity
  (decay-weighted, half-life 18 h, shrunk ×0.30 — raw velocity overfits) plus
  a small rank-band prior drift, and diffuse with rank-dependent volatility
  (σ ≈ 1.2 ranks/√h at the top, ≈ 3.7 at the bottom) estimated from the
  input history and shrunk toward the archive priors.
- **Dropout:** games leave the chart with a rank-dependent hourly hazard
  (≈ 0.05%/h top 10, ≈ 1%/h below rank 60).
- **Newcomers:** phantom games enter at the measured rate (~1/hour), climb
  at rates drawn from the observed climber distribution (lognormal, median
  1.0 ranks/h), and compete in the ranking. Their place mass is left unnamed
  (their ids are unknown, so no forecast can name them) — the scorer spreads
  unnamed mass over the games on each read's chart not named at that place.
- **Naming:** for each place, the named set maximises the expected log score
  under the model (every prefix size is tried and the best kept).
- **Top 10:** games ranked by total top-10 probability, ordered by expected
  place.

## Run

```bash
python3 main.py   # reads /input, writes /output
```

With env overrides for local testing:

```bash
MAKEFUN_INPUT_DIR=./input MAKEFUN_OUTPUT_DIR=./output python3 main.py
```

Runtime ≈ 24 s on the practice history (limit 300 s, 1 CPU, 1024 MB).
If the Monte Carlo fails for any reason, the program falls back to the
persistence baseline (a valid forecast) rather than writing nothing.

## Backtest (honest numbers)

Scored with a Python port of the kit's `grader/forecast/ranked.ts`, verified
to reproduce the baseline score exactly (−43.512039):

| Test | Model | Baseline | Delta |
| --- | --- | --- | --- |
| Practice window (10 reads, 72 h) | −34.66 | −43.51 | +8.85 |
| History slices, mean of 5 windows | — | — | +10.53 |

The house baseline (chart does not move) is beaten by ~9–11 nats in every
test window. The house AI model's score is not public; the practice
leaderboard's genuine (non-oracle) entries cluster around −31 to −35, so this
entry is competitive with that pack but not a clear favourite. The top
practice scores (−13 to −24) are fits to the published practice reads and
will not transfer to the final window.

## Limitations

- The final window's chart is genuinely unpredictable at the tail: ~17% of
  top-10 slots at 48–72 h are games that were not on the chart at the close,
  and no forecast can name them. The model's phantom calibration overshoots
  slightly at late hours (simulated newcomer share 0.25 vs measured 0.17 at
  49–72 h) and puts 0.28 of place 1 on unnamed newcomers — the main
  calibration risk, accepted because it won on 4 of 5 backtest windows.
- The 28-day age-out rule could not be validated: games stay on the chart
  days past `created + 28d` in the archive, so no explicit age-out is
  modelled (dropout is captured by the empirical hazard instead).
- Signals that did not survive backtesting and were dropped: playing-count
  volatility adjustment, Top Playing Now drift bonus, player-count trends
  (rank is already the growth ranking; the counts add noise).
- The practice history is sparse (34 reads over 9 days, gaps of 4–11 h), so
  the volatility/drift estimates are noisy; the final history (hourly house
  reads) will estimate them better.

## License

MIT — see `LICENSE`. Copyright (c) 2026 Eleenskie.
