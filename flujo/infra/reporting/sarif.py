from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any, Dict, List
import json

from flujo.domain.pipeline_validation import ValidationReport, ValidationFinding

# Best effort import for rules catalog (might not be available in all contexts)
_rules_catalog: ModuleType | None
_get_rule: Callable[[str], object | None]
try:
    from flujo.validation import rules_catalog as _rules_catalog
    from flujo.validation.rules_catalog import get_rule as _get_rule
except ImportError:
    _rules_catalog = None

    def _get_rule(_: str) -> object | None:
        return None


class SarifGenerator:
    """Generates SARIF 2.1.0 reports from Flujo ValidationReports."""

    def __init__(self) -> None:
        self.rules_index: Dict[str, int] = {}
        self.sarif_rules: List[Dict[str, Any]] = []

    def to_sarif(self, report: ValidationReport) -> str:
        """Convert a ValidationReport to a SARIF JSON string."""
        self.rules_index = {}
        self.sarif_rules = []
        sarif_results = []

        # Pre-populate rules from catalog if available
        if _rules_catalog:
            try:
                for _rule in getattr(_rules_catalog, "_CATALOG", {}).values():
                    self._append_rule_def(_rule)
            except Exception:
                pass

        # convert findings
        all_findings = report.errors + report.warnings
        for f in all_findings:
            sarif_results.append(self._convert_finding(f))

        sarif = {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {"driver": {"name": "flujo-validate", "rules": self.sarif_rules}},
                    "results": sarif_results,
                }
            ],
        }
        return json.dumps(sarif)

    def _convert_finding(self, f: ValidationFinding) -> Dict[str, Any]:
        rule_meta = self._ensure_rule(f.rule_id)

        # Determine level
        level = "error" if f.severity == "error" else "warning"

        # Location
        region: Dict[str, Any] = {}
        if f.line:
            region["startLine"] = int(f.line)
        if f.column:
            region["startColumn"] = int(f.column)

        phys: Dict[str, Any] = {}
        if f.file:
            phys["uri"] = str(f.file)

        loc: Dict[str, Any] = {"physicalLocation": {"artifactLocation": phys}}
        if region:
            loc["physicalLocation"]["region"] = region

        return {
            "ruleId": rule_meta["ruleId"],
            "level": level,
            "message": {"text": f"{f.step_name + ': ' if f.step_name else ''}{f.message}"},
            "locations": [loc],
            "properties": {
                "suggestion": f.suggestion,
                "location_path": f.location_path,
            },
        }

    def _ensure_rule(self, rule_id: str) -> Dict[str, Any]:
        rid = (rule_id or "").upper()
        if rid not in self.rules_index:
            info = _get_rule(rid)
            self._append_rule_def(info, rid)
        return {"ruleId": rid}

    def _append_rule_def(self, info: Any, rid: str | None = None) -> None:
        rule_id = (rid or getattr(info, "id", "") or "").upper()
        if not rule_id or rule_id in self.rules_index:
            return

        rule_def = {
            "id": rule_id,
            "name": (getattr(info, "title", None) or rule_id),
            "shortDescription": {"text": (getattr(info, "title", None) or rule_id)},
        }

        desc = getattr(info, "description", None)
        if desc:
            rule_def["fullDescription"] = {"text": desc}

        uri = getattr(info, "help_uri", None)
        if uri:
            rule_def["helpUri"] = uri
        else:
            rule_def["helpUri"] = (
                f"https://aandresalvarez.github.io/flujo/reference/validation_rules/#{rule_id.lower()}"
            )

        self.sarif_rules.append(rule_def)
        self.rules_index[rule_id] = len(self.sarif_rules) - 1
