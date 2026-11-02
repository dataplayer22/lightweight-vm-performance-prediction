import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["vcpu_count", "memory_mib", "workload_intensity"]
TARGET = "boot_latency_seconds"

if len(sys.argv) < 2:
    raise SystemExit("Usage: python train_model.py <dataset.xlsx-or-csv>")

path = Path(sys.argv[1])
if not path.exists():
    raise FileNotFoundError(path)

df = pd.read_excel(path) if path.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(path)
required = ["experiment", *FEATURES, TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

boot = df.loc[df["experiment"] == "boot_latency", required].copy()
for col in FEATURES + [TARGET]:
    boot[col] = pd.to_numeric(boot[col], errors="coerce")
boot = boot.dropna(subset=FEATURES + [TARGET])

if len(boot) < 2:
    raise ValueError("Need at least two complete boot-latency rows.")

print("Boot rows:", len(boot))
print("Distinct configurations:", boot[FEATURES].drop_duplicates().shape[0])
if boot[FEATURES].drop_duplicates().shape[0] < 2:
    print("WARNING: Same input configuration in every row. This model cannot learn configuration effects yet.")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(boot[FEATURES], boot[TARGET])

artifact = {
    "model": model,
    "features": FEATURES,
    "target": TARGET,
    "training_rows": len(boot),
    "training_feature_ranges": {
        c: [float(boot[c].min()), float(boot[c].max())] for c in FEATURES
    }
}
joblib.dump(artifact, "firecracker_boot_regression.pkl")
print("Saved firecracker_boot_regression.pkl")
print("Training fit is not independent test accuracy.")
