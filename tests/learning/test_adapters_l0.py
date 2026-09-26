"""L0 contract for the harness adaptation layer.

`unpack_env` exists because the five markets return two different shapes and
declare the action space under two different sets of keys, and the harness has to
read one of each.  The tests here are the ones that would go silent if the
adapter drifted, and there are four of those:

* a market returning the four-tuple env interface passes through with its keys intact;
* a market returning `(env, spec)` yields the three callables off the object
  rather than by position, which is the mismatch the module was written for,
  since `built[3]` is `get_obs` there and reading it as the spec fails much later
  and somewhere that does not name the cause;
* the day-ahead nested `action` dictionary is lifted to the top level;
* two independent routes to `baseline_action` are compared rather than one being
  trusted, and a disagreement raises.

The last is the reason this file exists rather than the first three.  That branch
is a guard, it was added on 2026-08-17 together with the module, and until this
test it had never been run against an input that makes it fire, which is the
state in which a guard reads as protection and is not.  The check is built from a
market whose declared `action_low` and whose own `truthful_action` disagree, which
is the only shape the branch can be reached by.

No `x64` fixture: nothing here builds a clearing operator, so the module does not
depend on the global precision either way.
"""
import jax.numpy as jnp
import numpy as np
import pytest

# `powermarketjax.learning` imports the IPPO module, which needs `optax`, and
# `optax` is an `rl` extra rather than a core dependency (see the package
# docstring).  CI installs `dev`, so a bare import here would turn an absent
# optional dependency into a suite failure.  The adapter itself needs neither.
pytest.importorskip("optax", reason="powermarketjax.learning needs the rl extra")

from powermarketjax.learning.adapters import unpack_env


def _fns():
    """Three distinguishable callables, so a positional mix-up is visible."""
    return (lambda: "reset", lambda: "step", lambda: "step_auto_reset")


class _Bundle:
    """A market returning `(env, spec)`, as the day-ahead and real-time ones do."""

    def __init__(self, truthful=None, extra_attrs=True):
        self.reset, self.step, self.step_auto_reset = _fns()
        if extra_attrs:
            # the two fields that make the four-tuple too small to hold this
            # market, and the reason the adapter exists rather than a rewrite
            self.get_obs = lambda: "get_obs"
            self.make_params = lambda n: "params"
        if truthful is not None:
            self.truthful_action = lambda: truthful


def test_the_four_tuple_of_adr_0010_passes_through_unchanged():
    spec = dict(action_shape=(3,), action_low=0.0, action_high=1.0,
                baseline_action=np.zeros(3), n_agents=3)
    reset, step, sar, out = unpack_env((*_fns(), spec))
    assert (reset(), step(), sar()) == ("reset", "step", "step_auto_reset")
    assert out["action_shape"] == (3,) and out["action_low"] == 0.0
    np.testing.assert_array_equal(out["baseline_action"], np.zeros(3))


def test_the_callables_come_off_the_object_and_not_by_position():
    # `built[3]` on this shape is `get_obs`; a positional harness silently binds
    # a callable where the spec belongs, which is the defect being prevented
    built = (_Bundle(), dict(action=dict(shape=(2,), low=1.0, high=2.0),
                             kind="markup"))
    reset, step, sar, out = unpack_env(built)
    assert (reset(), step(), sar()) == ("reset", "step", "step_auto_reset")
    assert "get_obs" not in out


def test_the_nested_action_dictionary_is_lifted_to_the_top_level():
    built = (_Bundle(), dict(action=dict(shape=(66,), low=1.0, high=2.0),
                             kind="markup"))
    _r, _s, _sa, out = unpack_env(built)
    assert out["action_shape"] == (66,)
    assert out["action_low"] == 1.0 and out["action_high"] == 2.0
    # markup one is the truthful offer by construction (§9.3), so the baseline is
    # the lower corner rather than a rule invented by the adapter
    np.testing.assert_allclose(np.asarray(out["baseline_action"]), np.ones(66))


def test_the_market_dictionary_is_not_modified():
    spec = dict(action=dict(shape=(4,), low=1.0, high=3.0), kind="markup")
    unpack_env((_Bundle(), spec))
    # two callers unpacking the same built environment must not see each other's
    # additions, so the copy is part of the contract and not an implementation
    # detail
    assert set(spec) == {"action", "kind"}


def test_a_market_with_no_route_to_a_baseline_leaves_the_key_absent():
    # kind is absent, so the markup derivation does not apply and the object
    # exposes no truthful action; failing on a missing key is the intended
    # outcome, since the alternative is a plausible wrong array
    _r, _s, _sa, out = unpack_env((*_fns(), dict(action_shape=(3,))))
    assert "baseline_action" not in out


def test_the_two_baseline_routes_are_compared_and_a_disagreement_raises():
    # the guard's only reachable shape: `action_low` says the truthful markup is
    # one while the market's own truthful action says otherwise.  Without this
    # case the branch has never executed, and a baseline silently taken from the
    # wrong route would put every arm of the benchmark on the wrong reference
    # while every reported number still looked ordinary.
    built = (_Bundle(truthful=jnp.full((5,), 1.5)),
             dict(action=dict(shape=(5,), low=1.0, high=2.0), kind="markup"))
    with pytest.raises(ValueError, match="truthful_action"):
        unpack_env(built)


def test_agreeing_routes_do_not_raise_and_the_baseline_is_the_agreed_value():
    # the control for the test above: same shape, same code path, only the
    # disagreement removed.  Without it the previous test could pass because the
    # branch raises on any market that exposes a truthful action at all.
    built = (_Bundle(truthful=jnp.ones((5,))),
             dict(action=dict(shape=(5,), low=1.0, high=2.0), kind="markup"))
    _r, _s, _sa, out = unpack_env(built)
    np.testing.assert_allclose(np.asarray(out["baseline_action"]), np.ones(5))


def test_a_shape_the_adapter_cannot_read_is_rejected_by_name():
    with pytest.raises(TypeError, match="reset"):
        unpack_env((object(), dict()))
    with pytest.raises(TypeError, match="four-tuple"):
        unpack_env((lambda: 0, lambda: 0, lambda: 0))
