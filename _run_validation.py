from reglabsim.validation.public_race import run_public_race_target_pack

report = run_public_race_target_pack()
s = report["summary"]
print(f"status:                     {s['status']}")
print(f"mean_overall_score:         {s['mean_overall_score']}")
print(f"mean_baseline_plausibility: {s.get('mean_baseline_plausibility_score', 'N/A')}")
print(f"mean_lap_mape_pct:          {s['mean_lap_mape_pct']}")
print(f"credible_proxy_count:       {s['credible_proxy_count']}")
print(f"plausible_2026_count:       {s.get('plausible_2026_count', 'N/A')}")
print()
for case in report["cases"]:
    sc = case["public_validation"]["scorecard"]
    em = case["public_validation"]["error_metrics"]
    m  = case["metrics"]
    plaus = sc.get("baseline_plausibility_score", "N/A")
    plaus_str = f"{plaus:.3f}" if isinstance(plaus, float) else plaus
    rc = sc.get("race_control_activity_score", "N/A")
    rc_str = f"{rc:.3f}" if isinstance(rc, float) else rc
    print(
        f"{case['query']['track_id']:12s}"
        f" | overall={sc['overall_score']:.3f}"
        f" | plaus={plaus_str}"
        f" | lap={sc['lap_score']:.3f}"
        f" | safety={sc['safety_score']:.3f}"
        f" | rc_act={rc_str}"
        f" | mape={em['avg_lap_time_mape_pct']:.2f}%"
        f" | nm={m.get('near_miss_count',0)} warn={m.get('warning_count',0)}"
        f" | min={m.get('minor_contact_count',0)} maj={m.get('major_contact_count',0)}"
        f" | ret={m['retirements']}"
    )
