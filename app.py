import streamlit as st
import mlflow
import pandas as pd
from evaluations.evaluator import evaluate_model
from evaluations.report import generate_html_report

mlflow.set_tracking_uri("http://127.0.0.1:5000")

st.set_page_config(page_title="Model Hub", layout="wide")
st.title("🤖 Model Evaluation & Lifecycle Platform")

# --- Sidebar: Model Registry ---
st.sidebar.header("Registered Models")
client = mlflow.tracking.MlflowClient()
registered_models = client.search_registered_models()
model_names = [rm.name for rm in registered_models]
selected_model = st.sidebar.selectbox("Select a model", model_names)

if selected_model:
    versions = client.search_model_versions(f"name='{selected_model}'")
    st.sidebar.subheader("Versions")
    for v in versions:
        st.sidebar.write(f"v{v.version} – {v.current_stage}")

    # --- Main area: Evaluate button ---
    if st.button("Run Evaluation on Hidden Test Set"):
        model_uri = f"models:/{selected_model}/latest"
        with st.spinner("Evaluating..."):
            results = evaluate_model(model_uri)
        st.success("Evaluation complete!")

        # Generate report
        report_html = generate_html_report(results, selected_model, versions[-1].version)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{results['accuracy']:.3f}")
        col2.metric("ROC AUC", f"{results['roc_auc']:.3f}")
        col3.metric("Fairness Δ", f"{results['fairness_delta']:.3f}")

        # Decision
        if results['go_no_go'] == "GO":
            st.balloons()
            st.header("✅ GO – Model is production ready!")
            # Simulate webhook notification
            st.info("🔔 Slack notification sent to #model-deployments")
            st.info("📥 Auto‑PR created: Update production config")
        else:
            st.header("❌ NO‑GO – Model needs improvement")

        # Show report
        with st.expander("View Full Report"):
            st.components.v1.html(report_html, height=800, scrolling=True)