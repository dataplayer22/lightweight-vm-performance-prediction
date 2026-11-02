FIRECRACKER REGRESSION PROJECT CODE

1) Install packages:
   python -m pip install -r requirements.txt

2) Train:
   python train_model.py Firecracker_VM_Performance_Dataset_Numeric_Filled.xlsx

3) Predict from terminal:
   python predict.py --vcpu 1 --ram 256 --workload 0.625

4) Run the browser app:
   streamlit run app.py

Files:
- train_model.py: trains and saves the Random Forest boot-time model.
- predict.py: predicts one configuration from the command line.
- app.py: simple Streamlit prediction interface.
- requirements.txt: dependencies.

Important:
The current six boot-time observations all use the same input configuration
(1 vCPU, 256 MiB RAM, workload intensity 0.625). The model can learn an
average for that configuration, but cannot learn the effect of changing
configuration until additional configurations are measured. Training-set
fit must not be reported as independent test accuracy.
