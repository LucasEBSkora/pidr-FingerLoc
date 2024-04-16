import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from time import process_time_ns
import numpy as np

fingerprints = pd.read_csv("data/fingerprints.csv", index_col=0)
fingerprintsAVG = pd.read_csv("data/fingerprints_mean_values.csv", index_col=0)
samples = pd.read_csv("data/samples.csv", index_col=0)
samplesAVG = pd.read_csv("data/samples_mean_values.csv", index_col=0)

def SpaceToLabel(x, y):
  return np.int8((x*20+y)/0.6)

def formatFingerprints(fingerprints):
  X = fingerprints[["rssi1", "rssi2", "rssi3", "rssi4", "rssi5"]].to_numpy()
  Y = fingerprints[["x", "y"]].to_numpy()
  Yflat =  SpaceToLabel(fingerprints["x"], fingerprints["y"])
  return (X, Y, Yflat)

def formatSamples(fingerprints):
  X = fingerprints[["rssi1", "rssi2", "rssi3", "rssi4", "rssi5"]].to_numpy()
  Y = fingerprints[["x", "y"]].to_numpy()
  return (X, Y)

XAVG,YAVG, YflatAVG = formatFingerprints(fingerprintsAVG)

X,Y, Yflat = formatFingerprints(fingerprints)
Yavg = fingerprints[["x", "y"]].groupby(["x","y"], sort=False).mean().reset_index()
Xsamples, Ysamples = formatSamples(samples)
XsamplesAvg, YsamplesAvg = formatSamples(samplesAVG)

def makeNaiveBayesModel():
  alg = GaussianNB()
  return alg.fit(X, Yflat)

def testScoreAndAvgTime(model, samples, expected):
  t0 = process_time_ns()
  probs = model.predict_proba(samples)
  print(model.classes_)
  Yavg["label"] = SpaceToLabel(Yavg['x'], Yavg['y'])
  Yavg.set_index("label", inplace=True)
  Yavg.sort_index()
  print(SpaceToLabel(Yavg['x'], Yavg['y']))
  print(Yavg)
  Ypred = (probs @ Yavg)
  print(Ypred)
  dt = process_time_ns() - t0
  return (score, dt/(len(samples)))

def plotColumns(dataframe, columnNames, plotname):
  dataframe[columnNames].plot()
  plt.savefig(plotname)

def runTests(testName, makeModelFunction):

  t0 = process_time_ns()
  alg = makeModelFunction()
  dt = process_time_ns() - t0
  timeToFitModel = dt/1e6

  score, time = testScoreAndAvgTime(alg, XsamplesAvg, YsamplesAvg)
  timeToProcessSamplesAvg = time
  accuracyScoreAvg = score

  # score, time = testScoreAndAvgTime(alg, Xsamples, Ysamples)
  # timeToProcessSamples = time
  # accuracyScore = score

  testResults = pd.Series({'timeToFitModel' : timeToFitModel,
                    # 'timeToProcessSamples' : timeToProcessSamples,
                    # 'accuracyScore' : accuracyScore,
                    'timeToProcessSamplesAvg' : timeToProcessSamplesAvg,
                    'accuracyScoreAvg' : accuracyScoreAvg}
                    )
  
  print(YsamplesAvg)
  print(alg.predict(XsamplesAvg))

  testResults.to_csv(f"./results/{testName}Results.csv")

runTests("NaiveBayes", makeNaiveBayesModel)
