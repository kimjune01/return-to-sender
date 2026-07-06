# return-to-sender

Synthetic validation of nodewise e-values under graph interference with FDR
control: the abuse filter specified in
[Return to Sender](https://june.kim/return-to-sender).

The five-step composition (exposure-integrated estimand, Horvitz-Thompson /
AIPW nodewise scores, per-node betting supermartingales, terminal e-values,
e-BH) runs end to end on a simulated agent-email platform: 400 agents, 20
abusive relays, 20 degree-matched honest subcontractors, independent
Bernoulli throttling per decision window.

## Run

```bash
uv sync
uv run pytest          # unbiasedness, supermartingale validity, e-BH
uv run python run_experiment.py   # main study + ablations, writes results.json
```

## Results (200 replications per cell, q = 0.10)

Designed randomization, AIPW scores: FDR 0.000 at every horizon, power 0.909
at 100 windows and 0.977 at 200, zero subcontractor false flags anywhere.
Remove the randomization (reactive throttling, analyst still assumes the
design): FDR 0.691 at tau = 0.25 where the matched control holds 0.000.
Rare friction (p = 0.05): e-values stay valid and detect nothing at any
tested horizon.

## License

AGPL-3.0. The companion paper is CC BY-SA 4.0.
