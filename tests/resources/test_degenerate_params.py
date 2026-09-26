"""Degenerate device parameters must fail at setup, not produce NaN at runtime.

Written for this repository on 2026-08-05 -- no upstream counterpart.

The failure mode being guarded: ``update_soc`` divides by capacity and by
``eta_discharge``. A zero in either puts NaN in ``BatteryBundleState.soc``
while ``p_inject`` stays clean, so nothing looks wrong at the grid interface.
The NaN then rides the whole ``lax.scan`` trajectory into obs -> policy ->
gradients and kills training silently.

This matters concretely for the P2P market: heterogeneous prosumers must be
carried in fixed-length struct-of-arrays bundles, so padding is unavoidable,
and ``capacity_mwh=0`` is the obvious way to pad.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import create_case33bw
from powermarketjax.resources import make_battery_bundle, make_vehicle_params


@pytest.fixture(scope="module")
def case33():
    return create_case33bw()


def _no_nan_over_both_directions(bundle):
    """Step the bundle discharging and charging; report whether anything is NaN/Inf."""
    state = bundle.reset(jax.random.PRNGKey(0))
    for a in (0.9, -0.9):
        action = jnp.full((int(bundle.action_dim),), a, jnp.float32)
        out = bundle.step(state, action, None)
        for leaf in jax.tree_util.tree_leaves(out):
            arr = np.asarray(leaf)
            if arr.dtype.kind == "f" and (np.isnan(arr).any() or np.isinf(arr).any()):
                return False
    return True


class TestBatteryBundleRejectsDegenerate:

    @pytest.mark.parametrize("kwargs", [
        pytest.param({"capacity_mwh": 0.0}, id="capacity_zero"),
        pytest.param({"capacity_mwh": -1.0}, id="capacity_negative"),
        pytest.param({"eta_discharge": 0.0}, id="eta_discharge_zero"),
        pytest.param({"eta_charge": 0.0}, id="eta_charge_zero"),
        pytest.param({"eta_charge": 1.5}, id="eta_charge_above_one"),
        pytest.param({"eta_discharge": 1.5}, id="eta_discharge_above_one"),
        pytest.param({"power_mw": -1.0}, id="power_negative"),
        pytest.param({"dt_hours": 0.0}, id="dt_zero"),
        pytest.param({"soc_min": 0.9, "soc_max": 0.1}, id="soc_bounds_inverted"),
    ])
    def test_raises_at_construction(self, case33, kwargs):
        with pytest.raises(ValueError):
            make_battery_bundle(case33, bus_ids=[5], **kwargs)

    def test_padding_a_device_population_the_documented_way(self, case33):
        """The supported way to model "this prosumer has no battery".

        A small positive capacity with zero power gives a device that cannot
        move energy, and produces no NaN. This is what the error message on
        ``capacity_mwh=0`` tells the caller to do, so it must actually work.
        """
        bundle = make_battery_bundle(
            case33, bus_ids=[5, 10, 20],
            capacity_mwh=[4.0, 1e-3, 4.0],
            power_mw=[1.0, 0.0, 1.0],
        )
        assert _no_nan_over_both_directions(bundle)

        state = bundle.reset(jax.random.PRNGKey(0))
        new_state, p_inject = bundle.step(
            state, jnp.array([0.9, 0.9, 0.9], jnp.float32), None,
        )[:2]
        soc = np.asarray(new_state.soc)
        assert not np.isnan(soc).any(), "padded device must not produce NaN SOC"
        np.testing.assert_allclose(
            np.asarray(p_inject)[1], 0.0, atol=1e-6,
            err_msg="a zero-power padded device must not inject",
        )

    def test_healthy_parameters_still_work(self, case33):
        """Guard against the validation being too strict."""
        bundle = make_battery_bundle(case33, bus_ids=[5, 10])
        assert _no_nan_over_both_directions(bundle)

    def test_eta_exactly_one_is_allowed(self, case33):
        """Lossless is a legitimate idealisation; (0, 1] is the valid band."""
        bundle = make_battery_bundle(
            case33, bus_ids=[5], eta_charge=1.0, eta_discharge=1.0,
        )
        assert _no_nan_over_both_directions(bundle)

    def test_zero_power_alone_is_allowed(self, case33):
        """power_mw=0 is valid (an immobile device); only negative is rejected."""
        bundle = make_battery_bundle(case33, bus_ids=[5], power_mw=0.0)
        assert _no_nan_over_both_directions(bundle)

    def test_error_message_names_the_offending_device(self, case33):
        with pytest.raises(ValueError, match=r"device index 1"):
            make_battery_bundle(
                case33, bus_ids=[5, 10, 20], capacity_mwh=[4.0, 0.0, 4.0],
            )


class TestVehicleParamsRejectsDegenerate:

    @pytest.mark.parametrize("kwargs", [
        pytest.param({"E_max_kWh": 0.0}, id="capacity_zero"),
        pytest.param({"E_max_kWh": -10.0}, id="capacity_negative"),
        pytest.param({"eta_discharge": 0.0}, id="eta_discharge_zero"),
        pytest.param({"eta_charge": 0.0}, id="eta_charge_zero"),
        pytest.param({"eta_charge": 1.2}, id="eta_charge_above_one"),
        pytest.param({"delta_t_minutes": 0.0}, id="dt_zero"),
        pytest.param({"soc_min": 0.9, "soc_max": 0.2}, id="soc_bounds_inverted"),
    ])
    def test_raises_at_construction(self, kwargs):
        with pytest.raises(ValueError):
            make_vehicle_params(**kwargs)

    def test_defaults_still_work(self):
        from powermarketjax.resources import VehicleEnv

        params = make_vehicle_params()
        env = VehicleEnv()
        _, state = env.reset(jax.random.PRNGKey(0), params)
        out = env.step(jax.random.PRNGKey(1), state, jnp.array([0.5]), params)
        for leaf in jax.tree_util.tree_leaves(out[:4]):
            arr = np.asarray(leaf)
            if arr.dtype.kind == "f":
                assert not np.isnan(arr).any()


NAN = float("nan")
INF = float("inf")


class TestNonFiniteParameters:
    """Added 2026-08-06: the 2026-08-05 guards compared bounds with ``<=`` /
    ``>``, and NaN is False against every such comparison, so NaN walked
    through and produced exactly the NaN SOC the guards existed to prevent.
    inf leaked the other way -- it satisfies the bounds but freezes SOC while
    power stays unconstrained.

    Measured before the fix (case33bw, one device, action +0.9):
      capacity_mwh=4.0 (healthy) -> p_inject 3.04 MW, soc 0.5 -> 0.1
      capacity_mwh=inf           -> p_inject 18 MW,   soc stays 0.5
      capacity_mwh=nan           -> soc NaN
      p_discharge_max_kW=-5 (vehicle) -> discharge *raised* soc 0.800 -> 0.820
    """

    @pytest.mark.parametrize("kwargs", [
        pytest.param({"capacity_mwh": NAN}, id="capacity_nan"),
        pytest.param({"capacity_mwh": INF}, id="capacity_inf"),
        pytest.param({"eta_discharge": NAN}, id="eta_discharge_nan"),
        pytest.param({"power_mw": NAN}, id="power_nan"),
        pytest.param({"power_mw": INF}, id="power_inf"),
        pytest.param({"dt_hours": NAN}, id="dt_nan"),
    ])
    def test_battery_rejects(self, case33, kwargs):
        with pytest.raises(ValueError):
            make_battery_bundle(case33, bus_ids=[5], **kwargs)

    @pytest.mark.parametrize("kwargs", [
        pytest.param({"E_max_kWh": NAN}, id="capacity_nan"),
        pytest.param({"E_max_kWh": INF}, id="capacity_inf"),
        pytest.param({"delta_t_minutes": NAN}, id="dt_nan"),
        pytest.param({"p_charge_max_kW": NAN}, id="p_charge_nan"),
        pytest.param({"p_discharge_max_kW": -5.0}, id="p_discharge_negative"),
    ])
    def test_vehicle_rejects(self, kwargs):
        with pytest.raises(ValueError):
            make_vehicle_params(**kwargs)

    def test_healthy_parameters_still_work(self, case33):
        """Guard against the finiteness check being too strict."""
        assert _no_nan_over_both_directions(
            make_battery_bundle(case33, bus_ids=[5]))
        make_vehicle_params(p_charge_max_kW=0.0, p_discharge_max_kW=0.0)
