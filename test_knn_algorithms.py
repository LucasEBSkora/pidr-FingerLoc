import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from time import process_time_ns

fingerprints = pd.read_csv("data/fingerprints.csv", index_col=0)
fingerprintsAVG = pd.read_csv("./data/fingerprints_mean_values.csv", index_col=0)
samples = pd.read_csv("data/samples.csv", index_col=0)
samplesAVG = pd.read_csv("data/samples_mean_values.csv", index_col=0)

def formatFingerprints(fingerprints):
  X = fingerprints[["rssi1", "rssi2", "rssi3", "rssi4", "rssi5"]].to_numpy()
  Y = (fingerprints[["x", "y"]]).to_numpy()
  return (X, Y)

Xsamples, Ysamples = formatFingerprints(samples)
XsamplesAvg, YsamplesAvg = formatFingerprints(samplesAVG)

for grid in [(pd.read_csv("data/fingerprints.csv", index_col=0), pd.read_csv("./data/fingerprints_mean_values.csv", index_col=0), ""),
             (pd.read_csv("data/fingerprints_grid1.2.csv", index_col=0), pd.read_csv("./data/fingerprints_mean_values_grid1.2.csv", index_col=0), "grid_1.2"),
             (pd.read_csv("data/fingerprints_grid1.8.csv", index_col=0), pd.read_csv("./data/fingerprints_mean_values_grid1.8.csv", index_col=0), "grid_1.8")]:

  X,Y = formatFingerprints(grid[0])
  Xavg,Yavg = formatFingerprints(grid[1])

  def makeBasicModel(X, Y, k, weight):
    alg = KNeighborsRegressor(n_neighbors=k, weights=weight)
    return alg.fit(X, Y)

  def makeKNNModel(k):
    return makeBasicModel(X,Y,k, 'uniform')

  def makeWeightedKNNModel(k):
    return makeBasicModel(X,Y,k, 'distance')

  def makeKNNModelAvg(k):
    return makeBasicModel(Xavg, Yavg, k, 'uniform')

  def makeWeightedKNNModelAvg(k):
    return makeBasicModel(Xavg, Yavg, k, 'distance')

  def testScoreAndAvgTime(model, X, Y):
    t0 = process_time_ns()
    score = model.score(X, Y)
    dt = process_time_ns() - t0
    return (score, dt/(len(X)))

  def plotColumns(dataframe, columnNames, plotname):
    dataframe[columnNames].plot()
    plt.savefig(plotname)

  def runTests(testName, makeModelFunction, maxk):
    timeToFitModel = []
    timeToProcessSamples = []
    accuracyScore = []
    timeToProcessSamplesAvg = []
    accuracyScoreAvg = []

    for k in range(1, maxk+1):

      t0 = process_time_ns()
      alg = makeModelFunction(k)
      dt = process_time_ns() - t0
      timeToFitModel.append(dt/1e6)

      score, time = testScoreAndAvgTime(alg, XsamplesAvg, YsamplesAvg)
      timeToProcessSamplesAvg.append(time)
      accuracyScoreAvg.append(score)

      score, time = testScoreAndAvgTime(alg, Xsamples, Ysamples)
      timeToProcessSamples.append(time)
      accuracyScore.append(score)

    testResults = pd.DataFrame({'timeToFitModel' : timeToFitModel,
                      'timeToProcessSamples' : timeToProcessSamples,
                      'accuracyScore' : accuracyScore,
                      'timeToProcessSamplesAvg' : timeToProcessSamplesAvg,
                      'accuracyScoreAvg' : accuracyScoreAvg},
                      index=list(range(1,1+maxk)))

    testResults.to_csv(f"./results/{testName}Results{grid[2]}.csv")
    plotColumns(testResults, ['accuracyScore', 'accuracyScoreAvg'], f"./results/{testName}ScoreResults{grid[2]}.png")
    plotColumns(testResults, ['timeToFitModel', 'timeToProcessSamples', 'timeToProcessSamplesAvg'],f"./results/{testName}timeResults{grid[2]}.png")
    testResults['scoreByTime'] = testResults['accuracyScore']/testResults['timeToProcessSamples']
    testResults['scoreByTimeAvg'] = testResults['accuracyScoreAvg']/testResults['timeToProcessSamplesAvg']
    plotColumns(testResults, ['scoreByTime', 'scoreByTimeAvg'], f"./results/{testName}scoreByTime{grid[2]}.png")

  runTests("KNN", makeKNNModel, len(Xavg))
  runTests("KNNAvg", makeKNNModelAvg, len(Xavg))
  runTests("WKNN", makeWeightedKNNModel, len(Xavg))
  runTests("WKNNAvg", makeWeightedKNNModelAvg, len(Xavg))