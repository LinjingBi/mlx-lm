"""Shared local-runtime exceptions that do not import MLX."""


class RuntimeBusy(RuntimeError):
    """The sequential runtime is already executing a generation."""
