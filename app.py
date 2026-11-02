from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Firecracker VM Predictor")
st.title("Firecracker VM Boot-Time Predictor")

model_file = Path("firecracker_boot_regression.pkl")
if not model_file.exists():
    st.error("Model not found. Run train_model.py first.")
    st.stop()

artifact = joblib.load(model_file)
vcpu = st.number_input("vCPU count", min_value=1, max_value=64, value=1)
ram = st.number_input("RAM (MiB)", min_value=64, max_value=65536, value=256, step=64)
workload = st.number_input("Workload intensity", min_value=0.0, max_value=1.0, value=0.625, step=0.025)

if st.button("Predict boot time"):
    X = pd.DataFrame([{
        "vcpu_count": vcpu,
        "memory_mib": ram,
        "workload_intensity": workload
    }])[artifact["features"]]
    yhat = float(artifact["model"].predict(X)[0])
    st.metric("Estimated boot latency", f"{yhat:.2f} seconds")
    for col, value in X.iloc[0].items():
        lo, hi = artifact["training_feature_ranges"][col]
        if value < lo or value > hi:
            st.warning(f"{col}={value} is outside the training range [{lo}, {hi}]. This is extrapolation.")
    st.caption("Validate estimates against new Firecracker measurements before relying on them.")
