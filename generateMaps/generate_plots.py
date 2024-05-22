import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import Divider, Size

fingerprints = pd.read_csv("../data/fingerprints.csv", index_col=0)
samples = pd.read_csv("../data/samples.csv", index_col=0)

# plt.style.use('_mpl-gallery')

def generate_map(fingerprints, samples):
  # gets only the unique location values of the fingerprints and samples
  fingerprints_xy = fingerprints[["x", "y"]].drop_duplicates()
  samples_xy = samples[["x", "y"]].drop_duplicates()
  data = pd.concat([fingerprints_xy, samples_xy])

  fig, ax = plt.subplots(figsize=(7.8/2, 4.2/2))

  # Colors fingerprints blue and samples green
  colors = [(0,0,1)]*len(fingerprints_xy) + [(0,1,0)]*len(samples_xy)

  ax.scatter(data["y"], data["x"], vmin=0, vmax=8, color=colors)

  #makes the graph with a tick on each meter, 
  #with a border of 60cm on every direction
  ax.set(xlim=(-0.6, 7.2), xticks=np.arange(0, 8), xticklabels=np.arange(0,8),
        ylim=(-0.6, 3.6), yticks=np.arange(0, 4))

  plt.savefig("plots/map.png")

def generate_heatmap(fingerprints):
  data = fingerprints.drop(columns=["ID"]).groupby(["x","y"], sort=False).mean().reset_index()
  print(data)
  x = data["x"]
  y = data["y"]
  for i in range(1,6):
    z = data[f"rssi{i}"]
    fig = plt.figure(figsize=(3.8/2+2,7.4/2+2))
    
    ax = fig.add_axes([0.1,0.1,0.8,0.8])

    cntr = ax.tricontourf(x, y, z, cmap="RdBu_r", algorithm="serial")
    fig.colorbar(cntr)
    ax.plot(x, y, 'ko', ms=3)

    plt.savefig(f"plots/beacon{i}heatmap.png")

generate_map(fingerprints, samples)
generate_heatmap(fingerprints)

