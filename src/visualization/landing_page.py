(
    tab_project,
    tab_overview,
    tab_jobs,
    tab_servers,
    tab_energy,
    tab_maintenance,
    tab_compare,
    tab_library,
) = st.tabs(
    [
        "Project Overview",
        "Overview",
        "Jobs & Demand",
        "Server Utilization",
        "Energy & PUE",
        "Maintenance",
        "Scenario Comparison",
        "Scenario Library",
    ]
)

with tab_project:

    st.title("Workload Prediction and Scheduling Optimization for Energy-Efficient, Reliable, and Service-Aware Data Centers")

    st.header("Executive Summary")

    st.markdown("""
    HERE GOES THE EXECUTIVE SUMMARY
    BLABLABLABLABLABLABLA
    """)

    st.header("Architecture Diagram")
    st.image("DOCS/architecture_diagram.png")

    st.header("Forecasting Module")
    st.markdown("""
    HERE GOES THE FORECASTING SUMMARY
    BLABLABLABLABLABLABLA
    """)

    st.header("Optimization Module")
    st.markdown("""
    The MILP model is the decision-making base of the project. It focuses on scheduling jobs across physical servers over 
    96 time slots of 15 minutes each (one full day), and it handles five operational dimensions simultaneously:

    The model simultaneously optimizes five operational dimensions:

    - **Energy Efficiency:** Minimizes operational costs and Power Usage
      Effectiveness (PUE) by reducing energy waste from cooling and
      infrastructure overhead.

    - **Reliability and Maintenance:** Schedules preventive and corrective
      maintenance activities while enforcing redundancy requirements for
      critical workloads.

    - **Thermal Management:** Maintains server inlet temperatures within
      ASHRAE A1 operating limits to ensure safe and efficient operation.

    - **Workload Prioritization:** Differentiates between batch and
      interactive jobs, reserving capacity for latency-sensitive workloads.

    - **Server State Management:** Minimizes unnecessary server power
      transitions and supports affinity and anti-affinity scheduling
      requirements.

    The model produces an optimized daily schedule that specifies workload
    placement, server activation states, and maintenance windows while
    satisfying capacity, reliability, and operational constraints.
    """)

    st.link_button(
    "💻 View GitHub Repository",
    "https://github.com/GasparFG/DAMO_class_23"
    )

    with open("docs/final_report.pdf", "rb") as pdf:
        st.download_button(
            label="📄 Download Final Report",
            data=pdf,
            file_name="final_report.pdf",
            mime="application/pdf"
        )

    st.header("Team Members")

    st.markdown("""
    ### Ana Carolina Quiroz Vazquez
    ana.quiroz2010@myunfc.ca

    ### Gaspar Franco Garcia
    gaspar.franco6692@myunf.ca

    ### Maria Jose Agualimpia Martinez
    maria.agualimpia7916@myunfc.ca

    ### Monica Giselle Mendoza Mendoza
    monica.mendoza5170@myunfc.ca
    """)