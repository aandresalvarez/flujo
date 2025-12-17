from __future__ import annotations
import json
import os
import fnmatch
from typing import Any, Dict, List
from dataclasses import dataclass

from flujo.domain.pipeline_validation import ValidationReport, ValidationFinding
from flujo.cli.helpers import validate_pipeline_file


@dataclass
class BaselineDelta:
    applied: bool
    file: str | None
    added_errors: List[ValidationFinding]
    added_warnings: List[ValidationFinding]
    removed_errors: List[ValidationFinding]
    removed_warnings: List[ValidationFinding]


class ValidationService:
    """Domain service for validating pipelines and managing rules/baselines."""

    def validate_file(
        self,
        path: str,
        include_imports: bool = True,
        rules_path: str | None = None,
        profile_mapping: Dict[str, str] | None = None,
    ) -> ValidationReport:
        """Run validation and apply optional rule overrides."""
        # 1. Core validation
        report = validate_pipeline_file(path, include_imports=include_imports)

        # 2. Apply mapping (from profile or file)
        if profile_mapping:
            return self._apply_mapping(report, profile_mapping)
        elif rules_path:
            return self._apply_rules_from_file(report, rules_path)

        return report

    def compute_baseline_delta(
        self,
        current_report: ValidationReport,
        baseline_path: str,
    ) -> tuple[ValidationReport, BaselineDelta]:
        """
        Compare current report against a baseline file.
        Returns (FilteredReport, DeltaInfo).
        The FilteredReport contains only *new* issues (added errors/warnings).
        """
        if not os.path.exists(baseline_path):
            return current_report, BaselineDelta(
                applied=False,
                file=baseline_path,
                added_errors=[],
                added_warnings=[],
                removed_errors=[],
                removed_warnings=[],
            )

        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                prev_raw = json.load(f)
        except Exception:
            return current_report, BaselineDelta(
                applied=False,
                file=baseline_path,
                added_errors=[],
                added_warnings=[],
                removed_errors=[],
                removed_warnings=[],
            )

        def _key_of(d: dict[str, Any]) -> tuple[str, str]:
            return (str(d.get("rule_id", "")).upper(), str(d.get("step_name", "")))

        cur_err = [e.model_dump() for e in current_report.errors]
        cur_warn = [w.model_dump() for w in current_report.warnings]

        prev_err = (
            [x for x in (prev_raw.get("errors") or []) if isinstance(x, dict)]
            if isinstance(prev_raw, dict)
            else []
        )
        prev_warn = (
            [x for x in (prev_raw.get("warnings") or []) if isinstance(x, dict)]
            if isinstance(prev_raw, dict)
            else []
        )

        prev_err_keys = {_key_of(x) for x in prev_err}
        prev_warn_keys = {_key_of(x) for x in prev_warn}
        cur_err_keys = {_key_of(x) for x in cur_err}
        cur_warn_keys = {_key_of(x) for x in cur_warn}

        added_raw_err = [x for x in cur_err if _key_of(x) not in prev_err_keys]
        added_raw_warn = [x for x in cur_warn if _key_of(x) not in prev_warn_keys]
        removed_raw_err = [x for x in prev_err if _key_of(x) not in cur_err_keys]
        removed_raw_warn = [x for x in prev_warn if _key_of(x) not in cur_warn_keys]

        # Reconstruct findings
        def _to_findings(arr: List[Dict[str, Any]]) -> List[ValidationFinding]:
            out = []
            for it in arr:
                try:
                    out.append(ValidationFinding(**it))
                except Exception:
                    continue
            return out

        added_errors = _to_findings(added_raw_err)
        added_warnings = _to_findings(added_raw_warn)
        removed_errors = _to_findings(removed_raw_err)
        removed_warnings = _to_findings(removed_raw_warn)

        # The new report seen by the user/CLI logic should effectively be "just the new stuff"
        # so that exits 0 if everything is in baseline.
        filtered_report = ValidationReport(errors=added_errors, warnings=added_warnings)

        delta = BaselineDelta(
            applied=True,
            file=baseline_path,
            added_errors=added_errors,
            added_warnings=added_warnings,
            removed_errors=removed_errors,
            removed_warnings=removed_warnings,
        )
        return filtered_report, delta

    def load_rules_mapping(self, rules_path: str) -> Dict[str, str] | None:
        """Parse rules file (JSON or TOML) into a simple dict {RULE_ID: SEVERITY}."""
        if not os.path.exists(rules_path):
            return None

        try:
            # Try JSON first
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._normalize_mapping(data)
        except Exception:
            # Fallback to TOML
            try:
                try:
                    import tomllib as _tomllib
                except ImportError:
                    import tomli as _tomllib  # type: ignore

                with open(rules_path, "rb") as f:
                    data = _tomllib.load(f)

                # Support nested structure from our schema
                # validation.rules = { ... }
                vm = data
                if isinstance(data, dict):
                    if "validation" in data and isinstance(data["validation"], dict):
                        if "rules" in data["validation"]:
                            vm = data["validation"]["rules"]
                        else:
                            # Or maybe just under validation directly?
                            pass
                return self._normalize_mapping(vm)
            except Exception:
                return None

    def _normalize_mapping(self, data: Any) -> Dict[str, str] | None:
        if isinstance(data, dict):
            return {str(k).upper(): str(v).lower() for k, v in data.items()}
        return None

    def _apply_rules_from_file(self, report: ValidationReport, rules_path: str) -> ValidationReport:
        mapping = self.load_rules_mapping(rules_path)
        if not mapping:
            return report
        return self._apply_mapping(report, mapping)

    def _apply_mapping(self, report: ValidationReport, mapping: Dict[str, str]) -> ValidationReport:
        """Apply severity overrides based on a mapping."""
        sev_map = {str(k).upper(): str(v).lower() for k, v in mapping.items()}

        def _resolve(rule_id: str) -> str | None:
            rid = rule_id.upper()
            if rid in sev_map:
                return sev_map[rid]
            # Glob support
            for pat, sev in sev_map.items():
                if "*" in pat or "?" in pat or ("[" in pat and "]" in pat):
                    if fnmatch.fnmatch(rid, pat):
                        return sev
            return None

        new_errors = []
        new_warnings = []

        # Process existing errors
        for e in report.errors:
            s = _resolve(e.rule_id)
            if s == "off":
                continue
            elif s == "warning":
                new_warnings.append(e)
            else:
                new_errors.append(e)

        # Process existing warnings
        for w in report.warnings:
            s = _resolve(w.rule_id)
            if s == "off":
                continue
            elif s == "error":
                new_errors.append(w)
            else:
                new_warnings.append(w)

        return ValidationReport(errors=new_errors, warnings=new_warnings)
