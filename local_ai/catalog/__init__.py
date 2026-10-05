"""Catalog phone book."""

from local_ai.catalog.app import create_app
from local_ai.catalog.process import bind_settings, main, prepare

__all__ = ["bind_settings", "create_app", "main", "prepare"]
