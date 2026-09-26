<h1 align="center">PowerMarketJax</h1>

<p align="center"><strong>A JAX benchmark suite for multi-agent reinforcement learning in electricity markets.</strong></p>

<p align="center">
  <img alt="Python 3.10–3.12" src="https://img.shields.io/badge/python-3.10%E2%80%933.12-3776AB?logo=python&logoColor=white">
  <img alt="JAX" src="https://img.shields.io/badge/JAX-0.4.30%2B-FF7F50">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-3DA639">
  <img alt="5 markets" src="https://img.shields.io/badge/markets-5-informational">
</p>

---

## Abstract

Power markets are a natural testbed for multi-agent reinforcement learning (MARL): many self-interested participants repeatedly submit bids, and a market-clearing mechanism then determines dispatch and prices subject to grid constraints and settlement rules. Existing MARL environments typically cover a single market, simplify the clearing mechanism, or call a CPU optimisation solver on every step, which slows large-scale training and limits the systematic study of bidding strategies.

**PowerMarketJax** is a benchmark suite for MARL across five power markets: day-ahead wholesale, real-time balancing, ancillary services, peer-to-peer local energy, and local flexibility. Each environment implements its own clearing, pricing and settlement rules behind a common interface. Market simulation and policy training are both written in JAX, so environment rollout, policy forward pass and gradient update run inside one `jit` boundary, with parallelism across environments and across market participants.

## What you get

- **Five market environments** under one functional interface: `reset(key, params)`, `step(key, state, action, params)` and `step_auto_reset`, all pure and compatible with `jit`, `vmap` and a fixed-length `lax.scan`.
- **Clearing inside the computation graph.** Four of the markets clear a linear program with one shared primal-dual interior-point method written for the accelerator. Prices are the duals of that program: the LMP, and the reserve price where the market has one. The peer-to-peer market clears a double auction by sorting and a comparison reduction. No host callback and no CPU solver sit on the step path.
- **Reward from settlement.** The reward is the profit settled on realised awards and prices. Feasibility violations go to a separate `costs` channel and never enter the reward, so each market is a constrained Markov game.
- **Learners**: independent PPO and independent SAC, each with and without parameter sharing, reached only through the tuple a market's constructor returns.
- **Non-learning references**: truthful (cost) bidding, the markup ceiling, a best fixed markup swept on a grid, and a unilateral best-response sweep that measures how much a single agent can still gain by deviating.
- **Public data**: demand, generation, price and smart-meter series from Great Britain, Australia, Belgium, Switzerland and the RTS-GMLC test system, shipped with the repository. No proprietary inputs.

## The five markets

| Market | Constructor | Agents | Step | Clearing | Constraint cost channels |
|---|---|---|---|---|---|
| M1 Day-ahead wholesale | `envs.day_ahead.make_env` | thermal units, markup on the cost curve | one market day | multi-period SCUC in three steps: relax, round, re-solve with the commitment fixed; LMP is the dual of the fixed-commitment dispatch | `shed_energy_mwh`, `min_up_down_violation` |
| M2 Real-time balancing | `envs.real_time.make_env` | the same units, a markup every period | one period | single-period SCED with fixed commitment; deviations from the day-ahead position settle at the real-time LMP | `shed_mwh` |
| M3 Ancillary services | `envs.ancillary.make_ancillary_env` | units, an energy markup plus one reserve offer per product | one period | energy and reserve co-optimised, products ordered by response time; LMP and reserve prices are both duals | `shed_energy_mwh`, `unmet_requirement_mw_0/1` |
| M4 Peer-to-peer local energy | `envs.p2p.make_p2p_env` | prosumers with PV and a battery: a price and a battery command | one period | double auction, price at the midpoint of the compatible interval; residuals settle with the grid at the retail and export tariffs | `clip` |
| M5 Local flexibility | `envs.local_flexibility.make_local_flex_env` | battery aggregators: a price, a deliverable deviation, a scheduled charging power | one period | the distribution operator's procurement LP under linearised DistFlow voltage and line limits; pay-as-bid | `shed_energy_mwh`, `voltage_violation_sum_pu`, `thermal_violation_sum_pu` |

The markets are chained the way real ones are. The commitment and positions fixed by M1 are the input that M2 and M3 settle against. M4 and M5 take the retail and export tariffs as the outside option.

The full model of each market (observation, action, offer mapping, clearing problem and settlement) is given in the paper's appendix.

## How it works

One step of every environment runs the same sequence. The action is mapped to an admissible offer, the market clears, and the primal solution gives the awards while the dual solution gives the prices. Settlement then gives each agent's profit, which is its reward. The sequence is one `step`, so the clearing problem is compiled with `jit`, vectorised with `vmap` over environments and repeated under `lax.scan`.

Two design choices keep the clearing on the accelerator:

| Conventional pattern | PowerMarketJax |
|---|---|
| A CPU LP/MILP solver called once per step | One interior-point method in JAX, shared by the four LP markets; by default it runs a fixed iteration count, so a batch of environments does not wait on its hardest member |
| An exact MILP for unit commitment | Relax → round → re-solve with commitment fixed. Prices, output and settlement come from the third, fixed-commitment solve. The gap to an exact MILP is measured offline (`tools/commitment/milp_reference.py`) |
| Prices reconstructed after clearing | Prices read from the duals of the same solve that produced the awards |

Clearing and settlement are checked against independently written numpy references (`tests/envs/*/reference.py`), the power-flow solvers against their own references (`tests/equivalence/`), and the LP duals against HiGHS.

## Learners and baselines

| Family | Shipped | Where |
|---|---|---|
| Learners | IPPO-PS, IPPO-NoPS, SAC-PS, SAC-NoPS. All are independent learners: each agent sees only its local observation and optimises its own profit. PS shares one network across agents, NoPS gives each agent its own | `powermarketjax/learning/` |
| Non-learning arms | truthful bidding, markup ceiling | `tools/benchmark/run_eval_0*.py` |
| Best fixed markup | one markup held constant, swept on a grid | `tools/benchmark/run_bestfixed_grid.py` |
| Best-response gap | unilateral best-response sweep, per unit | `tools/benchmark/run_withholding.py` |
| Constrained references (M4) | Lagrangian and open-loop references | `tools/p2p_experiment/` |

The shared learner configuration for M1–M3 is `tools/benchmark/hyperparams.py`. Every field is required, and the module records where each value comes from.

## Installation

Requires Python 3.10–3.12.

```bash
conda create -y -n powermarketjax python=3.11
conda activate powermarketjax
pip install -e ".[dev]"
```

Optional extras:

- `cuda`: `pip install -e ".[dev,cuda]"` on Linux with CUDA 12. It adds `jax-cuda12-plugin` alongside the installed `jaxlib` at the same version, leaving `jax` and `jaxlib` untouched.
- `rl`: adds `rlax` and `distrax`. **Market 04's drivers need it.** `tools/p2p_experiment/` imports `distrax` directly and `tools/benchmark/run_rl_04.py` imports it through that package, so without the extra they fail at import time, `--help` included. `optax`, which `powermarketjax/learning/` needs, already comes with `flax`.
- `bench`: pins the packages the appendix speed experiments compare against (PyPSA with linopy and HiGHS, cvxpy with Clarabel, Stable-Baselines3 with PyTorch and Gymnasium) at the versions they were timed with. Only `tools/env_bench/` imports them.
- `figs`: adds `matplotlib`. Only plotting scripts import it; nothing under `powermarketjax/` and none of the drivers named here do, so no command in this README needs it.

Nothing under `powermarketjax/` imports `matplotlib`, `rlax` or `distrax`, so the five markets and both learners work on a `dev` install alone.

Check the install:

```bash
python -c "import powermarketjax, jax; print(powermarketjax.__version__, jax.__version__, jax.devices())"
pytest -q
```

GPU-dependent tests skip rather than fail when no accelerator is present.

## Quick start

Open the day-ahead market on the 29-bus GB case, reset it, and run a three-step rollout under `lax.scan` with every agent bidding at twice its cost. One step is one market day, so this clears three consecutive days of a multi-period security-constrained unit commitment inside `jit`.

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import make_env, load_commitment, load_gb_demand
from powermarketjax.utils.jax_utils import scan_rollout

env, spec = make_env(load_case("29gb"),
                     load_commitment(n_periods=24), load_gb_demand(),
                     kind="markup", markup_max=2.0,
                     cap_scale=0.60, ramp_scale=1.00)
params = env.make_params(episode_len=7)

obs, state = env.reset(jax.random.PRNGKey(0), params)
actions = jnp.full((3, spec["n_agents"]), 2.0)          # (n_steps, n_agents)
final_state, obs_traj, reward_traj, cost_traj, done_traj, info_traj = jax.jit(
    lambda k, s: scan_rollout(env, k, s, params, actions))(jax.random.PRNGKey(1), state)

print(obs_traj.shape, reward_traj.shape, cost_traj.shape)   # (3, 66, 112) (3, 66) (3, 66, 2)
print(info_traj["converged"])                               # [ True  True  True]
print(spec["cost_names"])                                   # ('shed_energy_mwh', 'min_up_down_violation')
```

**The scenario parameters have no defaults.** `cap_scale` and `ramp_scale` scale the line ratings and ramp limits the case ships with, and omitting either raises `ValueError`. At `cap_scale` 1.00 no line of this network reaches its limit, every bus clears at one price, and nothing downstream would reveal that the market had become one without congestion.

**One interface, two return shapes.** M3, M4 and M5 return `(reset, step, step_auto_reset, spec)`. M1 and M2 return `(env, spec)`, where `env` carries the same three functions as attributes plus `make_params`. `powermarketjax.learning.unpack_env` accepts either shape and returns the four-tuple, so a market-agnostic rollout is written once:

```python
from powermarketjax.learning import unpack_env

# Continues the block above; M3-M5 constructors' four-tuples go in the same way.
reset, step, step_auto_reset, spec4 = unpack_env((env, spec))
obs, state = reset(jax.random.PRNGKey(0), params)
obs, state, reward, costs, done, info = step_auto_reset(
    jax.random.PRNGKey(1), state, jnp.full(spec4["n_agents"], 2.0), params)
print(reward.shape, costs.shape, spec4["termination"])   # (66,) (66, 2) truncation
```

`spec` is a plain dictionary of static values. It gives the observation dimension, action shape and box, the names of the `costs` columns (`spec["cost_names"]`), the agent count, the episode-boundary convention (`spec["termination"]`: `"truncation"` for four markets, `"terminal"` for M4) and the scenario parameters the environment was actually built with.

## Reproducing benchmark results

Each market has a learning driver and a non-learning driver under `tools/benchmark/`. They write the same product format. Every scenario parameter is a module constant exposed as a flag that defaults to it, and the driver refuses to run on a commitment or position fixture built at other scale factors.

```bash
# Learning arms: parameter sharing on M1, seed 0; then one network per agent.
python tools/benchmark/run_rl_01.py --algo ippo --seed 0 --iterations 200 \
    --out-dir runs/m1-ippo-ps --curve-out runs/m1-ippo-ps/curve.jsonl
python tools/benchmark/run_rl_01.py --algo ippo --seed 0 --per-agent-params --out-dir runs/m1-ippo-nops

# M2 and M3 settle against a day-ahead position (built by tools/commitment/da_position.py --out <file>).
python tools/benchmark/run_rl_02.py --position <da-position.npz> --seed 0 --out-dir runs/m2-ippo-ps
python tools/benchmark/run_rl_03.py --position <da-position.npz> --seed 0 --out-dir runs/m3-ippo-ps

# M4, 1200 households (needs the rl extra).
python tools/benchmark/run_rl_04.py --agents 1200 --seed 0 --out-dir runs/m4-ippo-ps

# Non-learning arms on the held-out days.
python tools/benchmark/run_eval_01.py --days eval --out-dir results/m1-open-loop

# Best fixed markup.
python tools/benchmark/run_bestfixed_grid.py --market 01 --seed 0 --out-dir results/m1-bestfixed

# Unilateral best-response sweep, then assembly over the shards.
python tools/benchmark/run_withholding.py sweep --market 01 --days eval \
    --fixture <commitment.npz> --out-dir results/m1-br
python tools/benchmark/run_withholding.py assemble --dir results/m1-br --out results/m1-br/summary.json
```

Commitment fixtures are built by `tools/commitment/precommit.py`; the ones the tests use are in `tests/fixtures/`. M5 runs are under `tools/flex_experiment/` (learning baselines in `concentration_baseline.py`, the unilateral sweep in `unilateral_sweep_05.py`), and the M4 constrained references under `tools/p2p_experiment/`. Every driver takes `--help`.

The appendix speed experiments are under `tools/env_bench/` (needs the `bench` extra): `rollout_bench.py` and `train_bench.py` time this package's rollout and PPO training, `sb3_bench.py` the Stable-Baselines3 baseline, and `external/` the PyPSA and cvxpy implementations of market 02's clearing (`record_02.py` records the per-step inputs that `pypsa_time.py` replays).

## Where PowerMarketJax sits

| | Scope | Clearing | Backend |
|---|---|---|---|
| Single-market RL environments for power markets (e.g. OpenGridGym, SustainGym) | one market | market-specific, often simplified pricing | CPU |
| JAX environments for economics and trading (e.g. EconoJax, JAX-LOB) | order-matching markets | rule-based matching, no grid | JAX |
| Accelerator-native RL suites (e.g. Brax, Pgx, JaxMARL) | games, physics, control | none | JAX |
| **PowerMarketJax** | **five chained power markets, transmission and distribution** | **each market's own LP or auction, prices from duals** | **JAX** |

## Repository layout

```text
powermarketjax/
  case/          CaseData, PTDF and graph matrices, 18 built-in network cases (10 transmission, 8 distribution)
  physics/       DC, AC (Newton) and radial BFS power flow, single- and three-phase
  resources/     battery, PV/wind, flexible-load, EV and diesel device models
  data/          manifest-driven loader, 15 dataset manifests, parquet series
  solvers/       the interior-point method the LP markets share
  envs/          the five markets: day_ahead/, real_time/, ancillary/, p2p/, local_flexibility/
  learning/      IPPO and SAC, and unpack_env
  wrappers/      framework adapters, kept out of envs/
  spaces.py, utils/   space definitions and JAX helpers
tools/
  benchmark/     learning and non-learning drivers, evaluation-day splits, shared hyperparameters
  commitment/    day-ahead commitment and position fixtures, MILP optimality-gap reference
  day_ahead/, real_time/, ancillary/, p2p_experiment/, flex_experiment/   per-market diagnostics and references
  data_prep/     acquisition scripts, one per dataset (provenance)
  env_bench/     rollout and training throughput, and the PyPSA and cvxpy implementations the appendix speed experiments compare against
tests/           L0 JAX constraints, L1 domain correctness, L2 numpy equivalence, L3 step-by-step clearing
```

Arrows of dependency run one way: `case`, `physics`, `resources` and `data` → `envs` → `learning`. Nothing under `envs/` imports the learner.

## Public data sources

All series below ship in `powermarketjax/data/parquet/` except the Ausgrid zone-substation series. Each has a manifest under `powermarketjax/data/manifests/` and a sidecar JSON next to the parquet file that records its source and processing. Copies must keep the attributions below.

| Data | Publisher | Licence | Attribution / obligation |
|---|---|---|---|
| GB historic demand (`gb_neso_demand`) | NESO | [NESO Open Data Licence](https://www.neso.energy/data-portal/neso-open-licence) | "Supported by National Energy SO Open Data" |
| GB demand forecast and outturn, generation by type, market index price (`gb_forecast_actual_demand`, `gb_gen_by_type`, `gb_market_mid`) | Elexon | [BMRS data licence](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/) | "Contains BMRS data © Elexon Limited copyright and database right 2026" |
| NEM 5-minute demand, demand forecast, NSW1 dispatch price and rooftop PV (`aemo_5min_demand`, `aemo_forecast`, `aemo_nsw1_dispatch_price`, `aemo_nsw1_rooftop_pv`) | AEMO | [AEMO Copyright Permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions) | attribution to the Australian Energy Market Operator (AEMO) as author |
| NEM network and generator case `813nem` (`powermarketjax/case/raw_cases/egrimod_nem/`) | Xenophon & Hill (2018), egrimod-nem | CC BY 4.0 | cite Xenophon, A. K. & Hill, D. J., *Scientific Data* 5, 180203 (2018); see that directory's `LICENCE.md` |
| Solar Home household series (`ausgrid_solar_home`) | Ausgrid | CC BY 3.0 AU | "Ausgrid, Solar Home Electricity Data (2010–2013)" |
| Digital-meter residential profiles (`fluvius_dm_residential`) | Fluvius | Fluvius open data licence | name Fluvius and the dataset's last-update date (2025-09-29) |
| Swiss day-ahead price (`ch_dayahead_price`) | Fraunhofer ISE Energy-Charts, from Bundesnetzagentur / SMARD.de | CC BY 4.0 | as recorded in the sidecar |
| SwissDN feeder 459_0 load, PV, BESS and topology (`swissdn_459_0_mv_load`) | ETH Zurich, Zapparoli et al., *Scientific Data* 12, 1491 (2025) | CC BY 4.0 | cite the paper and the Zenodo record 10.5281/zenodo.15056134 |
| ElCom tariffs for feeder 459_0 | Swiss Federal Electricity Commission | opendata.swiss "Open use" | as recorded in the sidecar |
| RTS-GMLC time series and 73-bus case (`rts_gmlc_timeseries`, `rts_gmlc_timeseries_30min`) | NREL / GridMod | DOE/NREL/ALLIANCE data use disclaimer | the disclaimer in `powermarketjax/case/raw_cases/rts_gmlc/LICENCE.md` must accompany every copy |

**Not shipped: the Ausgrid zone-substation series** (`ausgrid_zone_substation_fy25_imputed`). The publisher's page states that the data is the sole property of Ausgrid, which retains all copyright, and no result in the paper uses it. Its manifest stays registered. Loading it raises because the parquet file is absent, and a test that fails for that reason is reported as skipped with the reason. No test in this release reads it.

**Acquisition scripts** in `tools/data_prep/` record where each shipped file came from. Each takes `--out-dir` and writes outside the package. How far each reproduces the shipped file, compared column by column:

| Script | Reproduces the shipped file to |
|---|---|
| `gb_neso_demand.py` | 20 of 22 columns bit for bit, `TSD` included |
| `gb_forecast_actual_demand.py` | `DAForecast` bit for bit up to 2025-04-13 |
| `gb_market_mid.py` | time axis bit for bit; prices in part |
| `gb_gen_by_type.py` | time axis bit for bit; values not bit for bit |
| `aemo_5min_demand.py` | bit for bit on the overlapping span; the earliest part is no longer published |
| `aemo_forecast.py` | forecasts bit for bit on the overlapping span; the earliest part is no longer published |
| `egrimod_nem_generators.py` | byte for byte |
| `ausgrid_zone_substation_fy25_imputed.py` | retrieves Ausgrid's raw release (output without `_imputed`); the structure matches, the imputation cannot be reproduced |
| `aemo_nsw1_price_and_rooftop_pv.py`, `ausgrid_solar_home.py`, `ch_dayahead_price.py`, `elcom_swissdn_tariff.py`, `fluvius_dm_quarterly.py`, `rts_gmlc_timeseries.py`, `swissdn_mv_459_0.py` | not compared for this release; these write into `powermarketjax/data/parquet/` directly |

## Limitations and scope

PowerMarketJax is a benchmark suite, not a market simulator for operational use. The networks are public test cases and reduced models, the demand and price series are historical public records, and each market implements one clearing and settlement design out of the many used in practice. Day-ahead commitment is a relax-round-resolve heuristic whose gap to an exact MILP is measured, not zero. A strong result on PowerMarketJax is evidence about learning in these market designs, not a statement about any real market's participants.

## License and anonymity

MIT; see [`LICENSE`](LICENSE). The only collective credit is *PowerMarketJax Contributors*. The vendored foundations (network cases, power flow, device models, data loader) are MIT, *PowerZooJax Contributors*; the same file lists what they cover. Data carry their own terms (above).
