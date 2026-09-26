# PowerMarketJax

**A JAX benchmark suite for multi-agent reinforcement learning in electricity markets.**

Power markets are a natural testbed for multi-agent reinforcement learning (MARL): many self-interested participants repeatedly submit bids, and a market-clearing mechanism then determines dispatch and prices subject to grid constraints and settlement rules. Existing MARL environments typically cover one market, simplify the clearing mechanism, or call a CPU solver on every step.

PowerMarketJax is a suite of five power market environments: day-ahead wholesale (M1), real-time balancing (M2), ancillary services (M3), peer-to-peer energy trading (M4) and local flexibility (M5). Each market keeps its own clearing, pricing and settlement rules behind a common agent–market interface. Market clearing and policy training are both written in JAX, so environment rollout, policy evaluation and policy update run inside one compiled computation graph, vectorised over environments and over agents. No host callback and no CPU solver sit on the step path.

## What you get

- **Five market environments** under one interface: `reset(key, params)`, `step(key, state, action, params)` and `step_auto_reset`, all pure and compatible with `jit`, `vmap` and a fixed-length `lax.scan`. Each market is formulated as a general-sum partially observable stochastic game (POSG).
- **Clearing inside the computation graph.** The four optimisation-based markets clear a linear program with one shared primal–dual interior-point solver; the peer-to-peer market clears a double auction. Prices come from the clearing itself: the duals of the same solve that produced the dispatch, the auction's uniform clearing price, or the accepted bid price under pay-as-bid.
- **Learners**: independent PPO (IPPO) and independent SAC, each with parameter sharing (PS) or agent-specific parameters (NoPS).
- **Non-learning strategies**: truthful bidding, a fixed bid markup swept on a grid, and a unilateral best-response sweep.
- **Public data** from Great Britain, Australia, Belgium, Switzerland and the RTS-GMLC test system, shipped with the repository. Every shipped file is under an open licence.
- **Tests** that check clearing and settlement against separately written NumPy references, and drivers under `tools/lp_bench/` that compare the LP dual prices with HiGHS.

## What the paper finds

Learned bidding behaviour depends strongly on the market design. Independent learners can miss better strategies when the gain requires many agents to change together, when the more profitable strategy lies beyond a region of lower profit, or when the profit disappears as more agents adopt the same strategy.

- **M1, day-ahead wholesale.** All four learners settle near bidding at 1.5 times marginal cost on the 29-bus GB case. Raising every bid together to twice marginal cost would earn more, but raising one generator's bid at a time, with the others at their learned bids, never lifts total profit above the learned level. The joint gain is not visible in each agent's own learning signal.
- **M2, real-time balancing, and M3, ancillary services.** In M2, a generator that bids above cost can lose output to its competitors, so the learned bids move toward marginal cost. In M3, raising the reserve bid of a single generator produces little gain, while all generators raising their reserve bids together would gain much more; the learned policies remain far below that joint gain.
- **M4, peer-to-peer energy trading.** With 1,200 households trading every 15 minutes, a fixed battery-arbitrage schedule pays for one household but becomes unprofitable once a few hundred households adopt it. Simultaneous charging raises the midday clearing price and simultaneous discharging lowers the evening price, which reduces the price differential the arbitrage relies on.
- **M5, local flexibility.** Under the pay-as-bid rule with a price floor, the learners stay at or close to the floor. A single aggregator can earn more by bidding well above the floor, but small moves away from the floor reduce its profit, and the more profitable region lies much farther away. The learners instead raise their profit through planned charging, which increases the flexibility requirement that the distribution system operator (DSO) then procures.

## The five markets

| Market                         | Constructor                                    | Agents and action                                                                    | Step                                       | Clearing and pricing                                                                                                                                                           |
| ------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| M1 Day-ahead wholesale         | `envs.day_ahead.make_env`                    | generator agents; a bid price as a markup on marginal cost                           | one delivery day, 24 hours cleared jointly | unit commitment in three stages: relax, round, re-solve with the rounded commitment fixed; LMPs are the duals of the fixed-commitment dispatch                                 |
| M2 Real-time balancing         | `envs.real_time.make_env`                    | the same generator agents; a bid price every interval                                | one 30-minute interval                     | redispatch of the committed generators with fixed commitment; two-settlement rule, deviations from the day-ahead schedule settled at the real-time LMP                         |
| M3 Ancillary services          | `envs.ancillary.make_ancillary_env`          | generator agents; an energy bid price plus one reserve bid price per reserve product | one 30-minute interval                     | energy and reserves co-optimised, reserve products distinguished by response time; real-time LMPs and reserve prices are both duals                                            |
| M4 Peer-to-peer energy trading | `envs.p2p.make_p2p_env`                      | prosumer agents with PV and a battery; a bid or ask price and a battery command      | one 15-minute interval                     | double auction with a uniform local clearing price at the midpoint of the matched interval; unmatched surplus or deficit settled with the grid at the retail and export prices |
| M5 Local flexibility           | `envs.local_flexibility.make_local_flex_env` | aggregator agents; a price–quantity bid plus a scheduled charging power             | one 1-hour interval                        | the DSO's procurement LP under linearised DistFlow voltage and line-flow constraints; pay-as-bid rule                                                                          |

Constructor paths are relative to `powermarketjax`. The markets are chained: M2 and M3 settle against a day-ahead commitment and schedule produced once by M1 under truthful bidding and frozen as a fixture. M4 takes the retail and export prices as the outside option; M5 takes an exogenous energy price as the cost of charging. M3, M4 and M5 return `(reset, step, step_auto_reset, spec)`; M1 and M2 return `(env, spec)` with the same three functions as attributes. `powermarketjax.learning.unpack_env` accepts either shape. `spec` is a plain dictionary of static values: observation dimension, action box and the scenario parameters the environment was built with.

## Installation

Requires Python 3.10–3.12; the results were produced with Python 3.11 and JAX 0.10.2.

```bash
conda create -y -n powermarketjax python=3.11
conda activate powermarketjax
pip install -e ".[dev]"
```

Optional extras: `cuda` (the CUDA 12 plugin, Linux only), `rl` (`rlax` and `distrax`; `tools/p2p_experiment/` and the M4 driver need it), `bench` (the external packages `tools/env_bench/` imports, pinned). Nothing under `powermarketjax/` imports `rlax` or `distrax`, so the five markets and both learners work on a `dev` install alone.

```bash
python -c "import powermarketjax, jax; print(powermarketjax.__version__, jax.__version__, jax.devices())"
pytest -q     # about two hours on 16 CPU cores; the GPU-only test skips without a GPU
```

## Quick start

Open the day-ahead wholesale market on the 29-bus GB case and run a three-step rollout under `lax.scan` with every generator bidding at twice its marginal cost. One step is one delivery day, so this clears three consecutive days of a multi-period security-constrained unit commitment inside `jit`.

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
final_state, obs_traj, reward_traj, _, done_traj, info_traj = jax.jit(
    lambda k, s: scan_rollout(env, k, s, params, actions))(jax.random.PRNGKey(1), state)

print(obs_traj.shape, reward_traj.shape)   # (3, 66, 112) (3, 66)
print(info_traj["converged"])              # [ True  True  True]
```

The scenario parameters have no defaults: `cap_scale` and `ramp_scale` scale the line ratings and ramp limits the case ships with, and omitting either raises `ValueError`.

The same rollout is written once for all five markets through `unpack_env`:

```python
from powermarketjax.learning import unpack_env

reset, step, step_auto_reset, spec4 = unpack_env((env, spec))   # a four-tuple from M3-M5 goes in the same way
obs, state = reset(jax.random.PRNGKey(0), params)
obs, state, reward, _, done, info = step_auto_reset(
    jax.random.PRNGKey(1), state, jnp.full(spec4["n_agents"], 2.0), params)
print(reward.shape, spec4["termination"])   # (66,) truncation
```

## Reproducing the paper's results

The learning drivers for M1–M4 and the non-learning drivers for M1–M3 are under `tools/benchmark/`; M4's truthful-bidding arm is the same driver with `--arm truthful`, and the M5 runs are under `tools/flex_experiment/`. Every scenario parameter is a module constant exposed as a flag that defaults to it, and each driver records the configuration it ran with in its products. A driver refuses a commitment or schedule fixture built at other scale factors. Every driver takes `--help`.

```bash
# Learning arms on M1: parameter sharing, then agent-specific parameters.
python tools/benchmark/run_rl_01.py --algo ippo --seed 0 --out-dir runs/m1-ippo-ps
python tools/benchmark/run_rl_01.py --algo ippo --seed 0 --per-agent-params --out-dir runs/m1-ippo-nops

# M2 and M3 settle against a day-ahead schedule (tools/commitment/da_position.py builds one).
python tools/benchmark/run_rl_02.py --position tests/fixtures/day_ahead_position_29gb_T24_step1prime_seasons.npz \
    --seed 0 --out-dir runs/m2-ippo-ps
python tools/benchmark/run_rl_03.py --position tests/fixtures/day_ahead_position_29gb_T24_step1prime_seasons.npz \
    --seed 0 --out-dir runs/m3-ippo-ps

# M4, 1,200 households (needs the rl extra).
python tools/benchmark/run_rl_04.py --agents 1200 --seed 0 --out-dir runs/m4-ippo-ps

# Non-learning strategies on the held-out days; best fixed markup; unilateral best-response sweep.
python tools/benchmark/run_eval_01.py --days eval --out-dir results/m1-open-loop
python tools/benchmark/run_bestfixed_grid.py --market 01 --seed 0 --out-dir results/m1-bestfixed
python tools/benchmark/run_withholding.py sweep --market 01 --days eval \
    --fixture tests/fixtures/day_ahead_commitment_29gb_T24_relax.npz --out-dir results/m1-br
```

Commitment fixtures are built by `tools/commitment/precommit.py` (pass `--cap-scale 0.6 --ramp-scale 1.0` to match the drivers' defaults); the ones the tests use are in `tests/fixtures/`. The M4 reference drivers are under `tools/p2p_experiment/`. The shared learner configuration for M1–M3 is `tools/benchmark/hyperparams.py`.

## Repository layout

```text
powermarketjax/
  case/          network cases, PTDF and graph matrices
  physics/       DC, AC and radial BFS power flow
  resources/     battery, PV, flexible-load, EV and diesel device models
  data/          manifest-driven loader and the shipped parquet series
  solvers/       the interior-point solver the optimisation-based markets share
  envs/          the five markets: day_ahead/, real_time/, ancillary/, p2p/, local_flexibility/
  learning/      IPPO and SAC, and unpack_env
tools/           benchmark drivers, commitment fixtures, per-market experiments, HiGHS comparisons, data acquisition scripts
tests/           mirror the package layout; each file is marked with its layer: JAX constraints, domain correctness, NumPy equivalence, step-by-step clearing
```

## Data and licences

All series ship in `powermarketjax/data/parquet/`. Each parquet file has a sidecar JSON next to it that records its source and processing (the by-municipality ElCom table shares the annual table's sidecar), and the series loaded by name have a manifest under `powermarketjax/data/manifests/`. Copies must keep the attributions below.

| Data                                                                                                                                                                       | Publisher                                                        | Licence                                                                                                                            | Attribution / obligation                                                                                                                            |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| GB historic demand (`gb_neso_demand`)                                                                                                                                    | NESO                                                             | [NESO Open Data Licence v1.0](https://www.neso.energy/data-portal/neso-open-licence)                                                | "Supported by National Energy SO Open Data"                                                                                                         |
| GB demand forecast and outturn, generation by type, market index price (`gb_forecast_actual_demand`, `gb_gen_by_type`, `gb_market_mid`)                              | Elexon                                                           | [Licence to use BMRS open data](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/) | "Contains BMRS data © Elexon Limited copyright and database right 2026"                                                                            |
| NEM 5-minute demand, demand forecast, NSW1 dispatch price and rooftop PV (`aemo_5min_demand`, `aemo_forecast`, `aemo_nsw1_dispatch_price`, `aemo_nsw1_rooftop_pv`) | AEMO                                                             | [AEMO Copyright Permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions)                               | attribution to the Australian Energy Market Operator (AEMO) as author                                                                               |
| NEM network and generator case`813nem` (`powermarketjax/case/raw_cases/egrimod_nem/`)                                                                                  | Xenophon & Hill (2018), egrimod-nem                              | CC BY 4.0                                                                                                                          | cite Xenophon, A. K. & Hill, D. J.,*Scientific Data* 5, 180203 (2018); see that directory's `LICENCE.md`                                        |
| Solar Home household series (`ausgrid_solar_home`)                                                                                                                       | Ausgrid                                                          | [CC BY 3.0 AU](http://creativecommons.org/licenses/by/3.0/au/)                                                                      | "Ausgrid, Solar Home Electricity Data (2010–2013)", with the changes recorded in the sidecar                                                       |
| Digital-meter residential profiles (`fluvius_dm_residential`)                                                                                                            | Fluvius                                                          | [Fluvius open data licence](https://opendata.fluvius.be/p/licentieopendatafluvius/)                                                 | name Fluvius and the dataset's last-update date (2025-09-29)                                                                                        |
| Swiss day-ahead price (`ch_dayahead_price`)                                                                                                                              | Fraunhofer ISE Energy-Charts, from Bundesnetzagentur / SMARD.de  | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)                                                                           | as recorded in the sidecar                                                                                                                          |
| SwissDN feeder 459_0 load, PV, BESS and topology (`swissdn_459_0_mv_load`)                                                                                               | ETH Zurich, Zapparoli et al.,*Scientific Data* 12, 1491 (2025) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)                                                                           | cite the paper and the Zenodo record 10.5281/zenodo.15056134                                                                                        |
| ElCom tariffs for feeder 459_0                                                                                                                                             | Swiss Federal Electricity Commission                             | [opendata.swiss &#34;Open use&#34;](https://ld.admin.ch/vocabulary/TermsOfUse/Open-Use)                                             | as recorded in the sidecar                                                                                                                          |
| RTS-GMLC time series and 73-bus case (`rts_gmlc_timeseries`, `rts_gmlc_timeseries_30min`)                                                                              | NREL / GridMod                                                   | [DOE/NREL/ALLIANCE data use disclaimer](https://github.com/GridMod/RTS-GMLC#data-use-disclaimer-agreement)                          | the disclaimer in`powermarketjax/case/raw_cases/rts_gmlc/LICENCE.md` must accompany every copy, and any publication must credit DOE/NREL/ALLIANCE |

The scripts in `tools/data_prep/`, one per upstream source, record how each shipped file was obtained.

## License

MIT; see [`LICENSE`](LICENSE). The only collective credit is *PowerMarketJax Contributors*. Data carry their own terms (above).
