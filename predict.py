import argparse
import joblib
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--model", default="firecracker_boot_regression.pkl")
p.add_argument("--vcpu", type=int, required=True)
p.add_argument("--ram", type=float, required=True, help="RAM in MiB")
p.add_argument("--workload", type=float, required=True)
a = p.parse_args()

artifact = joblib.load(a.model)
X = pd.DataFrame([{
    "vcpu_count": a.vcpu,
    "memory_mib": a.ram,
    "workload_intensity": a.workload
}])[artifact["features"]]

for col, value in X.iloc[0].items():
    lo, hi = artifact["training_feature_ranges"][col]
    if value < lo or value > hi:
        print(f"WARNING: {col}={value} is outside training range [{lo}, {hi}].")

print(f"Predicted boot latency: {artifact['model'].predict(X)[0]:.2f} seconds")
