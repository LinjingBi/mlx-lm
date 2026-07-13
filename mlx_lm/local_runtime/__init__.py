"""Model-side implementation of the sequential local runtime.

Import ``SequentialRuntime`` from ``mlx_lm.local_runtime.engine``. Keeping this
module lazy lets protocol and daemon tooling load without initializing MLX.
"""
