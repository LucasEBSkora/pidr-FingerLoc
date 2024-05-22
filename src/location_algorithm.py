import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

fingerprintsAVG = pd.read_csv("../data/fingerprints_mean_values.csv", index_col=0)

def formatFingerprints(fingerprints):
  X = fingerprints[["rssi1", "rssi2", "rssi3", "rssi4", "rssi5"]].to_numpy()
  Y = (fingerprints[["x", "y"]]).to_numpy()
  return (X, Y)

X,Y = formatFingerprints(fingerprintsAVG)

alg = KNeighborsRegressor(n_neighbors=8)
alg = alg.fit(X,Y)

def locate(rssiSamples):
  return alg.predict(rssiSamples)