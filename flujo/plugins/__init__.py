"""
Built-in plugins for flujo.
"""

from flujo.domain.plugins import register_plugin
from .sql_validator import SQLSyntaxValidator

register_plugin(SQLSyntaxValidator)

__all__ = ["SQLSyntaxValidator"]
