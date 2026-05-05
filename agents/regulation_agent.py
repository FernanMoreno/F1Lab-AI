"""Regulation analysis agent.

Analyzes F1 regulations and identifies potential issues.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from reglabsim.regulation.base import Regulation


class RegulationAgent:
    """Agent for regulation analysis.

    Parses regulations, identifies constraints, and suggests improvements.

    Example:
        >>> agent = RegulationAgent()
        >>> analysis = agent.analyze(regulation)
    """

    def __init__(self):
        """Initialize agent."""
        pass

    def analyze(self, regulation: Regulation) -> Dict[str, Any]:
        """Analyze a regulation.

        Args:
            regulation: Regulation to analyze.

        Returns:
            Dict with analysis results.
        """
        # Parse key parameters
        analysis = {
            "regulation_id": regulation.name,
            "version": regulation.version,
            "status": regulation.status,
            "has_active_aero": regulation.has_active_aero,
            "max_ers_energy_mj": regulation.max_ers_energy_mj,
            "max_ers_deployment_kw": regulation.max_ers_deployment_kw,
            "drs_zones": regulation.drs_zones,
            "assumptions": regulation.assumptions,
        }

        # Identify potential issues
        issues = []

        if regulation.max_ers_energy_mj > 6.0:
            issues.append({
                "type": "high_ers",
                "severity": "warning",
                "description": "High ERS energy may lead to battery dominance",
            })

        if regulation.has_active_aero and not analysis.get("assumptions"):
            issues.append({
                "type": "active_aero_assumption",
                "severity": "info",
                "description": "Active aero enabled - assumptions should be documented",
            })

        analysis["issues"] = issues

        return analysis

    def compare(
        self,
        reg_a: Regulation,
        reg_b: Regulation,
    ) -> Dict[str, Any]:
        """Compare two regulations.

        Args:
            reg_a: First regulation.
            reg_b: Second regulation.

        Returns:
            Comparison results.
        """
        diff = reg_a.diff(reg_b)

        return {
            "regulation_a": reg_a.name,
            "regulation_b": reg_b.name,
            "differences": diff,
            "summary": f"Compared {reg_a.name} vs {reg_b.name}",
        }

    def suggest_improvements(
        self,
        regulation: Regulation,
        failure_modes: List[str],
    ) -> List[str]:
        """Suggest regulation improvements.

        Args:
            regulation: Regulation to improve.
            failure_modes: Identified failure modes.

        Returns:
            List of suggestions.
        """
        suggestions = []

        for mode in failure_modes:
            if mode == "battery_dominance":
                suggestions.append(
                    "Consider reducing ERS max deployment power or capacity"
                )
            elif mode == "artificial_overtaking":
                suggestions.append(
                    "Increase overtake mode activation gap or add cooldown"
                )
            elif mode == "dangerous_closing_speeds":
                suggestions.append(
                    "Add closing speed monitoring and limit boost power"
                )
            elif mode == "train_formation":
                suggestions.append(
                    "Consider improving DRS effectiveness or reducing dirty air"
                )

        return suggestions