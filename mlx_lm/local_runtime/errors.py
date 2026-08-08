"""Shared local-runtime exceptions that do not import MLX."""


class RuntimeBusy(RuntimeError):
    """The runtime cannot accept the operation right now (busy or at cap)."""
