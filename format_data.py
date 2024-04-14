import pandas as pd
import numpy as np

fingerprints = pd.read_csv("data/fingerprints.csv", index_col=0)
data = fingerprints.drop(columns=["ID"]).groupby(["x","y"], sort=False).mean().reset_index()
data.to_csv("data/fingerprint_mean_values.csv")