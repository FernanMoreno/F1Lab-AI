"""Streamlit dashboard for F1Lab-AI.

Provides visualization and analysis interface for simulation results.
"""

from __future__ import annotations

import streamlit as st  # type: ignore
from pathlib import Path


def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="F1Lab-AI Dashboard",
        page_icon="🏎️",
        layout="wide",
    )

    st.title("🏎️ F1Lab-AI: Regulation Stress-Testing Laboratory")
    st.markdown("""
    Advanced simulation platform for testing how F1 regulatory changes
    affect overtaking, battery dependency, DRS trains, dominance, robustness,
    and race strategies.
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Experiments", "Regulations", "Metrics", "Analysis"],
    )

    if page == "Home":
        show_home()
    elif page == "Experiments":
        show_experiments()
    elif page == "Regulations":
        show_regulations()
    elif page == "Metrics":
        show_metrics()
    elif page == "Analysis":
        show_analysis()


def show_home():
    """Show home page."""
    st.header("Welcome to F1Lab-AI")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Regulations", "4")
        st.caption("2025, 2026 initial, 2026 refined, experimental")

    with col2:
        st.metric("Car Families", "8")
        st.caption("Synthetic archetypes for testing")

    with col3:
        st.metric("Circuits", "4")
        st.caption("Monza, Monaco, Baku, Barcelona")

    st.divider()
    st.subheader("Quick Start")

    st.code("""
    from reglabsim import create_facade

    facade = create_facade()

    # Run a battle experiment
    result = facade.run_battle_experiment(
        "configs/experiments/baku_closing_speed.yaml"
    )

    # Compute metrics
    metrics = facade.compute_metrics(result)
    print(metrics)
    """, language="python")

    st.divider()
    st.subheader("Latest Experiments")

    st.info("No experiment results yet. Run experiments to see results here.")


def show_experiments():
    """Show experiments page."""
    st.header("Experiments")

    experiments_dir = Path("configs/experiments")
    if not experiments_dir.exists():
        st.warning("Experiments directory not found")
        return

    experiment_files = list(experiments_dir.glob("*.yaml"))

    for exp_file in experiment_files:
        with st.expander(exp_file.stem):
            import yaml

            with open(exp_file) as f:
                config = yaml.safe_load(f)

            st.json(config)


def show_regulations():
    """Show regulations page."""
    st.header("Regulations")

    reg_dir = Path("configs/regulations")
    if not reg_dir.exists():
        st.warning("Regulations directory not found")
        return

    reg_files = list(reg_dir.glob("*.yaml"))

    selected_reg = st.selectbox(
        "Select Regulation",
        [f.stem for f in reg_files],
    )

    if selected_reg:
        import yaml

        reg_path = reg_dir / f"{selected_reg}.yaml"
        with open(reg_path) as f:
            config = yaml.safe_load(f)

        st.subheader(f"{config.get('name', selected_reg)}")
        st.caption(f"Version: {config.get('version', 'N/A')} | Status: {config.get('status', 'N/A')}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Power Unit")
            pu = config.get("power_unit", {})
            for key, value in pu.items():
                st.text(f"{key}: {value}")

        with col2:
            st.subheader("Active Aero")
            aa = config.get("active_aero", {})
            for key, value in aa.items():
                st.text(f"{key}: {value}")


def show_metrics():
    """Show metrics page."""
    st.header("Regulation Health Metrics")

    st.markdown("""
    These metrics measure the health of a regulation across different
    dimensions of racing quality.
    """)

    metric_descriptions = {
        "battery_dependency_index": "Measures how much race performance depends on electrical energy",
        "artificial_pass_index": "Measures overtakes caused by energy advantage vs natural pace",
        "dangerous_closing_speed_index": "Measures frequency of unsafe closing speeds",
        "train_formation_index": "Measures frequency of untrainable car formations",
        "dominant_architecture_risk": "Measures if one car type dominates all others",
        "regulation_robustness_score": "Overall regulation health score",
    }

    for metric, description in metric_descriptions.items():
        with st.expander(metric):
            st.text(description)


def show_analysis():
    """Show analysis page."""
    st.header("Analysis")

    st.info("Run experiments and view results to see analysis here.")

    if st.button("Run Sample Analysis"):
        st.info("Analysis functionality requires completed experiments")


if __name__ == "__main__":
    main()