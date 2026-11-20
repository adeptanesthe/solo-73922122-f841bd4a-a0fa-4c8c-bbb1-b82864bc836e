"""Compatibility entry point: ``uvicorn app:app`` from the ``app`` directory.

The real application lives in :mod:`app.main`; this module simply re-exports it
so both ``uvicorn app.main:app`` and ``uvicorn app:app`` work.
"""

from .main import app  # noqa: F401

__all__ = ["app"]
