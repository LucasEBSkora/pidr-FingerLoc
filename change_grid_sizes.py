import pandas as pd

fingerprints = pd.read_csv("data/fingerprints.csv", index_col=0)
fingerprintsAVG = pd.read_csv("./data/fingerprints_mean_values.csv", index_col=0)

def linesInGrid(column, size):
  size = int(10*size)
  return (column * 10 % size) == 0

def changeGrid(df, size):
  df = df[linesInGrid(df.x, size)]
  df = df[linesInGrid(df.y, size)]
  return df

for size in [1.2, 1.8]:
  fingerprintsGrid = changeGrid(fingerprints, size)
  fingerprintsGrid.to_csv(f"data/fingerprints_grid{size}.csv")
  fingerprintsAVGGrid = changeGrid(fingerprintsAVG, size)
  fingerprintsAVGGrid.to_csv(f"data/fingerprints_mean_values_grid{size}.csv")