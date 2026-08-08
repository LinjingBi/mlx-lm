"""Model-side implementation of the local Unix-socket runtime.

Import ``SequentialRuntime`` from ``mlx_lm.local_runtime.engine`` or
``LocalResponseGenerator`` from ``mlx_lm.local_runtime.response_generator``.
Keeping this module lazy lets protocol and daemon tooling load without
initializing MLX.
"""
