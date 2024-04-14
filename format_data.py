import pandas as pd
import numpy as np

def format(path):
  fingerprints = pd.read_csv(f"data/{path}.csv", index_col=0)
  data = fingerprints.drop(columns=["ID"]).groupby(["x","y"], sort=False).mean().reset_index()
  data.to_csv(f"data/{path}_mean_values.csv")

format("fingerprints")
format("samples")