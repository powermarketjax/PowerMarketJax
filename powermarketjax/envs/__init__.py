"""Market environments.

This package exports nothing of its own: each of the five markets is a
subpackage -- ``day_ahead``, ``real_time``, ``ancillary``, ``p2p``,
``local_flexibility`` -- and is imported from there.  The layer stays
framework-neutral, so adapters for JaxMARL / PureJaxRL live in
``powermarketjax.wrappers``.

The agent index is an array axis, so ``reward`` is ``(N,)`` and ``action`` is
``(N, ...)`` with ``N`` static.  Dict-keyed (PettingZoo / JaxMARL) forms belong
in ``powermarketjax.wrappers``.

Do **not** subclass ``powermarketjax.resources.env_base.Environment``: it is the
resource layer's *single-agent* base class and all five markets are multi-agent.
Two of its conventions are followed here without inheriting from it -- ``step``
returns a 6-tuple keeping ``costs`` separate from ``reward``, and
``step_auto_reset`` applies ``stop_gradient``.
"""
