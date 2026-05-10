"""Minimal deterministic audit report builder for EvidenceBundle outputs."""

from __future__ import annotations

from typing import Any

_AUDIT_SCHEMA = "audit_report.v1"

_LIMITATIONS: list[str] = [
    (
        "This is a deterministic counterfactual stress-test,"
        " not a calibrated regulatory recommendation."
    ),
    "State hash coverage is partial.",
    "Patch effectiveness is scenario-dependent.",
]


def build_audit_report(bundle: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic audit report from an EvidenceBundle dict.

    Reads only — does not mutate the input bundle.
    """
    return {
        "schema_version": _AUDIT_SCHEMA,
        "run": _extract_run_metadata(bundle),
        "summary": _extract_summary(bundle),
        "unsafe_legal_events": _extract_unsafe_legal_events(bundle),
        "counterfactuals": _extract_counterfactuals(bundle),
        "limitations": list(_LIMITATIONS),
    }


def render_audit_report_markdown(report: dict[str, Any]) -> str:
    """Render an audit report dict to deterministic Markdown.

    Uses cautious language throughout — no regulatory truth claims.
    """
    lines: list[str] = []

    lines.append("# F1Lab-AI Counterfactual Audit Report")
    lines.append("")

    # --- Run ---
    lines.append("## Run")
    run = report.get("run") or {}
    lines.append(f"- **Run ID:** {run.get('run_id', '')}")
    lines.append(f"- **World ID:** {run.get('world_id', '')}")
    lines.append(f"- **Seed:** {run.get('seed', '')}")
    lines.append(f"- **Track:** {run.get('track', '')}")
    lines.append(f"- **Regulation:** {run.get('regulation_id', '')}")
    lines.append(f"- **Config hash:** {run.get('config_hash', '')}")
    lines.append("")

    # --- Summary ---
    summary = report.get("summary") or {}
    lines.append("## Summary")
    count = summary.get("unsafe_legal_state_count", 0)
    lines.append(
        f"In this deterministic stress-test run, {count} unsafe legal state(s) were detected."
    )
    lines.append("")
    lines.append(f"- **Unsafe legal states:** {count}")
    lines.append(f"- **Max hazard score:** {summary.get('max_hazard_score', 'N/A')}")
    lines.append(f"- **Mean hazard score:** {summary.get('mean_hazard_score', 'N/A')}")
    segments = summary.get("unsafe_legal_segments") or []
    lines.append(f"- **Affected segments:** {', '.join(segments) if segments else 'none'}")
    status_counts = summary.get("safety_verdict_status_counts") or {}
    counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))
    lines.append(f"- **Safety verdict status counts:** {counts_str or 'none'}")
    lines.append("")

    # --- Unsafe legal events table ---
    lines.append("## Unsafe Legal Events")
    events = report.get("unsafe_legal_events") or []
    if not events:
        lines.append("No unsafe legal events detected in this run.")
    else:
        lines.append(
            "| Event ref | Lap | Segment | Car | Legal | Safety | Hazard |"
            " Reaction margin (s) | Closing speed (kph) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for ev in events:
            lines.append(
                f"| {ev.get('event_ref', '')} "
                f"| {ev.get('lap', '')} "
                f"| {ev.get('segment_id', '')} "
                f"| {ev.get('car_id', '')} "
                f"| {ev.get('legal_status', '')} "
                f"| {ev.get('safety_status', '')} "
                f"| {ev.get('hazard_score', '')} "
                f"| {ev.get('reaction_margin_s', '')} "
                f"| {ev.get('closing_speed_kph', '')} |"
            )
    lines.append("")

    # --- Counterfactual patch results ---
    lines.append("## Counterfactual Patch Results")
    counterfactuals = report.get("counterfactuals") or []
    if not counterfactuals:
        lines.append("No counterfactual patch reruns found.")
    else:
        lines.append(
            "| Patch | Type | Verdict | Mitigation success | Hazard reduced |"
            " Target events | Resolved events |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for cf in counterfactuals:
            lines.append(
                f"| {cf.get('patch_id', '')} "
                f"| {cf.get('patch_type', '')} "
                f"| {cf.get('verdict', '')} "
                f"| {cf.get('mitigation_success', False)} "
                f"| {cf.get('hazard_reduced', False)} "
                f"| {cf.get('target_event_count', 0)} "
                f"| {cf.get('resolved_event_count', 0)} |"
            )
        lines.append("")

        for cf in counterfactuals:
            patch_label = cf.get("patch_id") or "patch"
            lines.append(f"### Delta Metrics — {patch_label}")
            dm = cf.get("delta_metrics") or {}
            lines.append(
                f"- **unsafe_legal_state_count_delta:**"
                f" {dm.get('unsafe_legal_state_count_delta', 'N/A')}"
            )
            lines.append(
                f"- **max_hazard_score_delta:** {dm.get('max_hazard_score_delta', 'N/A')}"
            )
            lines.append(
                f"- **mean_hazard_score_delta:** {dm.get('mean_hazard_score_delta', 'N/A')}"
            )

            verdict = cf.get("verdict", "")
            if verdict == "mitigated":
                lines.append(
                    "> In this deterministic stress-test, the patch eliminated all unsafe legal "
                    "states. This does not constitute proof of safety."
                )
            elif verdict in ("improved", "improved_hazard"):
                lines.append(
                    "> The patch reduced hazard or event count but did not eliminate all unsafe "
                    "legal states. This is not a calibrated regulatory recommendation."
                )
            elif verdict == "worse":
                lines.append("> The patch increased hazard or event count in this scenario.")
            else:
                lines.append(
                    "> No significant change observed in this deterministic stress-test run."
                )
            lines.append("")

            repro = cf.get("reproducibility") or {}
            if repro:
                lines.append(f"#### Reproducibility — {patch_label}")
                lines.append(f"- **Same seed:** {repro.get('same_seed', False)}")
                lines.append(f"- **Same world:** {repro.get('same_world_id', False)}")
                lines.append(
                    f"- **Baseline config hash:** {repro.get('baseline_config_hash', 'N/A')}"
                )
                lines.append(
                    f"- **Patched config hash:** {repro.get('patched_config_hash', 'N/A')}"
                )
                lines.append(
                    f"- **State hash coverage:** {repro.get('state_hash_coverage', 'partial')}"
                )
                lines.append("")

    # --- Limitations ---
    lines.append("## Limitations")
    limitations = report.get("limitations") or _LIMITATIONS
    for lim in limitations:
        lines.append(f"- {lim}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_run_metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    world_manifest = bundle.get("world_manifest") or {}
    return {
        "run_id": str(bundle.get("run_id") or ""),
        "world_id": str(
            bundle.get("world_id") or world_manifest.get("world_id") or ""
        ),
        "seed": int(bundle.get("seed") or world_manifest.get("seed") or 0),
        "track": str(bundle.get("track") or world_manifest.get("track") or ""),
        "regulation_id": str(
            bundle.get("regulation_id") or world_manifest.get("regulation_id") or ""
        ),
        "config_hash": str(
            bundle.get("config_hash") or world_manifest.get("config_hash") or ""
        ),
    }


def _extract_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    metrics = bundle.get("metrics") or {}
    return {
        "unsafe_legal_state_count": int(metrics.get("unsafe_legal_state_count") or 0),
        "has_unsafe_legal_state": bool(metrics.get("has_unsafe_legal_state", False)),
        "max_hazard_score": metrics.get("max_hazard_score"),
        "mean_hazard_score": metrics.get("mean_hazard_score"),
        "unsafe_legal_segments": sorted(metrics.get("unsafe_legal_segments") or []),
        "safety_verdict_status_counts": dict(
            metrics.get("safety_verdict_status_counts") or {}
        ),
    }


def _extract_unsafe_legal_events(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Build compact event summaries from unsafe_legal_states.

    Avoids dumping full raw payloads — uses refs and compact verdict fields only.
    """
    raw_events: list[Any] = bundle.get("unsafe_legal_states") or []
    metrics = bundle.get("metrics") or {}
    metric_refs: list[str] = list(metrics.get("unsafe_legal_event_refs") or [])

    summaries: list[dict[str, Any]] = []
    for idx, event in enumerate(raw_events):
        if not isinstance(event, dict):
            continue
        event_ref = _resolve_event_ref(event, idx, metric_refs)
        details = _get_details(event)
        safety_verdict = _get_safety_verdict(details, event)

        summaries.append({
            "event_ref": event_ref,
            "lap": int(event.get("lap") or 0),
            "segment_id": str(event.get("segment_id") or details.get("segment_id") or ""),
            "car_id": str(event.get("car_id") or details.get("car_id") or ""),
            "legal_status": _pick_str(
                details, event, "legal_status", "attacker_legal_status"
            ),
            "safety_status": _pick_safety_status(details, safety_verdict),
            "hazard_score": _pick_float(safety_verdict, details, "hazard_score"),
            "reaction_margin_s": _pick_float(
                safety_verdict, details, "reaction_margin_s"
            ),
            "delta_speed_kph": _pick_float(
                safety_verdict, details, "delta_speed_kph"
            ),
            "closing_speed_kph": _pick_float(
                details, safety_verdict, "closing_speed_kph"
            ),
            "amplifiers": list(safety_verdict.get("amplifiers") or []),
            "regulatory_causes": list(safety_verdict.get("regulatory_causes") or []),
            "reason_codes": list(safety_verdict.get("reason_codes") or []),
            "safety_verdict": _compact_safety_verdict(safety_verdict),
        })

    return summaries


def _extract_counterfactuals(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    patch_reruns: list[Any] = bundle.get("patch_reruns") or []
    result: list[dict[str, Any]] = []
    for entry in patch_reruns:
        if not isinstance(entry, dict):
            continue
        cf = entry.get("counterfactual_report")
        if isinstance(cf, dict):
            result.append(_counterfactual_from_cf_report(entry, cf))
        else:
            result.append(_counterfactual_from_rerun_entry(entry))
    return result


def _counterfactual_from_cf_report(
    entry: dict[str, Any], cf: dict[str, Any]
) -> dict[str, Any]:
    delta_summary = cf.get("delta_summary") or {}
    baseline_summary = cf.get("baseline_summary") or {}
    patched_summary = cf.get("patched_summary") or {}
    baseline_metrics: dict[str, Any] = dict(entry.get("baseline_metrics") or {})
    patched_metrics: dict[str, Any] = dict(entry.get("patched_metrics") or {})
    delta_metrics: dict[str, Any] = dict(entry.get("delta_metrics") or {})

    verdict = str(delta_summary.get("verdict") or entry.get("verdict") or "unknown")
    mitigation_success = bool(delta_summary.get("mitigation_success", False))
    hazard_reduced = bool(delta_metrics.get("hazard_reduced", False))

    target_count = int(
        baseline_summary.get("unsafe_legal_state_count")
        or baseline_metrics.get("unsafe_legal_state_count")
        or 0
    )
    resolved_count = int(
        patched_summary.get("unsafe_legal_state_count")
        or patched_metrics.get("unsafe_legal_state_count")
        or 0
    )
    target_refs: list[str] = list(
        cf.get("target_event_refs") or entry.get("target_event_refs") or []
    )
    resolved_refs: list[str] = list(
        cf.get("resolved_event_refs") or entry.get("resolved_event_refs") or []
    )

    return {
        "patch_id": str(cf.get("patch_id") or entry.get("patch_id") or ""),
        "patch_type": str(cf.get("patch_type") or entry.get("patch_type") or ""),
        "verdict": verdict,
        "mitigation_success": mitigation_success,
        "hazard_reduced": hazard_reduced,
        "target_event_count": target_count,
        "resolved_event_count": resolved_count,
        "baseline_metrics": baseline_metrics,
        "patched_metrics": patched_metrics,
        "delta_metrics": delta_metrics or dict(delta_summary),
        "target_event_refs": target_refs,
        "resolved_event_refs": resolved_refs,
        "reproducibility": dict(entry.get("reproducibility") or {}),
    }


def _counterfactual_from_rerun_entry(entry: dict[str, Any]) -> dict[str, Any]:
    baseline_metrics: dict[str, Any] = dict(entry.get("baseline_metrics") or {})
    patched_metrics: dict[str, Any] = dict(entry.get("patched_metrics") or {})
    delta_metrics: dict[str, Any] = dict(entry.get("delta_metrics") or {})

    verdict = str(delta_metrics.get("verdict") or entry.get("verdict") or "unknown")
    mitigation_success = bool(delta_metrics.get("mitigation_success", False))
    hazard_reduced = bool(delta_metrics.get("hazard_reduced", False))

    return {
        "patch_id": str(entry.get("patch_id") or ""),
        "patch_type": str(entry.get("patch_type") or ""),
        "verdict": verdict,
        "mitigation_success": mitigation_success,
        "hazard_reduced": hazard_reduced,
        "target_event_count": int(
            baseline_metrics.get("unsafe_legal_state_count") or 0
        ),
        "resolved_event_count": int(
            patched_metrics.get("unsafe_legal_state_count") or 0
        ),
        "baseline_metrics": baseline_metrics,
        "patched_metrics": patched_metrics,
        "delta_metrics": delta_metrics,
        "target_event_refs": list(entry.get("target_event_refs") or []),
        "resolved_event_refs": list(entry.get("resolved_event_refs") or []),
        "reproducibility": dict(entry.get("reproducibility") or {}),
    }


# ---------------------------------------------------------------------------
# Low-level field extraction
# ---------------------------------------------------------------------------


def _get_details(event: dict[str, Any]) -> dict[str, Any]:
    """Extract details sub-dict supporting Shape A, Shape B, and flat events."""
    details = event.get("details")
    if isinstance(details, dict):
        return details
    payload = event.get("payload")
    if isinstance(payload, dict):
        inner = payload.get("details")
        if isinstance(inner, dict):
            return inner
    if "hazard_score" in event:
        return event
    return {}


def _get_safety_verdict(
    details: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any]:
    sv = details.get("safety_verdict")
    if isinstance(sv, dict):
        return sv
    sv = event.get("safety_verdict")
    if isinstance(sv, dict):
        return sv
    return {}


def _resolve_event_ref(
    event: dict[str, Any], idx: int, metric_refs: list[str]
) -> str:
    ref = event.get("event_ref")
    if isinstance(ref, str) and ref:
        return ref
    if idx < len(metric_refs):
        return str(metric_refs[idx])
    ev_type = str(event.get("event_type", "unsafe_legal_state"))
    lap = int(event.get("lap", 0))
    seg = str(event.get("segment_id", "unknown"))
    car = str(event.get("car_id", "unknown"))
    return f"{ev_type}:{lap}:{seg}:{car}:{idx:04d}"


def _pick_str(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    *keys: str,
) -> str | None:
    for key in keys:
        val = primary.get(key)
        if isinstance(val, str) and val:
            return val
    for key in keys:
        val = fallback.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _pick_safety_status(
    details: dict[str, Any], safety_verdict: dict[str, Any]
) -> str | None:
    status = safety_verdict.get("status")
    if isinstance(status, str):
        return status
    status = details.get("safety_status")
    if isinstance(status, str):
        return status
    return None


def _pick_float(
    primary: dict[str, Any], fallback: dict[str, Any], key: str
) -> float | None:
    val = primary.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    val = fallback.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _compact_safety_verdict(sv: dict[str, Any]) -> dict[str, Any] | None:
    if not sv:
        return None
    return {
        "schema_version": sv.get("schema_version"),
        "status": sv.get("status"),
        "hazard_score": sv.get("hazard_score"),
        "reaction_margin_s": sv.get("reaction_margin_s"),
        "delta_speed_kph": sv.get("delta_speed_kph"),
        "confidence": sv.get("confidence"),
    }
