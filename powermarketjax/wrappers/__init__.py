"""Training-framework adapters.

Any JaxMARL / PureJaxRL coupling belongs here, so it never reaches
``powermarketjax.envs``.  `p2p` holds what market 04 needs to reach
``powermarketjax.learning.ippo`` without either side being bent to fit the
other.
"""
