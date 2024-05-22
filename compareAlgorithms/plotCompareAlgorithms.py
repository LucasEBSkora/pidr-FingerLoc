import pandas as pd
import matplotlib.pyplot as plt

def loadDataframe(gridSize, sampleGrouping, algorithm):
  if gridSize == 0.6:
    gridSize = ""
  else:
     gridSize = f"grid_{gridSize}"
  if sampleGrouping == "independent":
    sampleGrouping = ""
  else:
    sampleGrouping = "Avg"
  return pd.read_csv(f"results/{algorithm}{sampleGrouping}Results{gridSize}.csv")

def addData(allData, gridSize, sampleGrouping, algorithm):
  df = loadDataframe(gridSize, sampleGrouping, algorithm)
  allData[gridSize][sampleGrouping]["R^2 Score"][algorithm] = df["accuracyScore"]
  allData[gridSize][sampleGrouping]["time to fit model (ms)"][algorithm] = df["timeToFitModel"]
  allData[gridSize][sampleGrouping]["time to predict position (ms)"][algorithm] = df["timeToProcessSamples"]
  return allData

allData = {}
for gridSize in [0.6, 1.2, 1.8]:
    allData[gridSize] = {}
    for sampleGrouping in ["independent", "average"]:
      allData[gridSize][sampleGrouping] = {"R^2 Score": {}, "time to fit model (ms)": {}, "time to predict position (ms)": {}}
      for algorithm in ["KNN", "WKNN"]:
        allData = addData(allData, gridSize, sampleGrouping, algorithm)

allData = addData(allData, 0.6, "average", "PKNN")

print(allData.keys())
for gridSize in allData.keys():
   print(gridSize)
   for sampleGrouping in allData[gridSize].keys():
      print(f"\t{sampleGrouping}")
      for metric in allData[gridSize][sampleGrouping].keys():
        print(f"\t\t{metric}")
        file_name = f"plots/{metric}_{sampleGrouping}_{gridSize}.png"
        plot_name = f"{metric} with {sampleGrouping} samples and {gridSize} grid"
        print(plot_name)
        data = pd.DataFrame(allData[gridSize][sampleGrouping][metric])
        data.plot(xlabel='K', ylabel=metric)
        data.plot(title=plot_name)
        plt.savefig(file_name)