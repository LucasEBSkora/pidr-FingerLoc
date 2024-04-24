import pandas as pd

def format(path):
  fingerprints = pd.read_csv(f"data/{path}.csv", index_col=0).drop(columns=["ID"])
  data = fingerprints.groupby(["x","y"], sort=False).mean().reset_index()
  std_dev = fingerprints.groupby(["x","y"], sort=False).std().reset_index()
  for i in range(1, 6):
    data[f"rssi{i}_std_dev"] = std_dev[f"rssi{i}"]
  data.to_csv(f"data/{path}_mean_values.csv")

format("fingerprints")
format("samples")