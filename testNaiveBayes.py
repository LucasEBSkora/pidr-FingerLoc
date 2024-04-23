import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from time import process_time_ns
import numpy as np
from math import sqrt

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

model = makeNaiveBayesModel()

def makeNaiveBayesAvgModel():
  alg = GaussianNB()
  return alg.fit(XAVG, YflatAVG)

def LabelToSpace(labels):
  space = np.zeros((len(labels), 2))
  space[:, 0] = (labels // 20)*0.6
  space[:, 1] = (labels % 20)*0.6
  return space

def calcScore(true, pred):
  u = ((true - pred)** 2).sum()
  v = ((true - true.mean()) ** 2).sum()
  return 1 - (u/v)

def testScoreAndAvgTimeClosestClass(samples, expected):
  t0 = process_time_ns()
  labels = model.predict(samples)
  Ypred = LabelToSpace(labels)
  score = calcScore(Ypred, expected)
  dt = process_time_ns() - t0
  return (score, dt/(len(samples)))

def testScoreAndAvgTimeProb(samples, expected):
  t0 = process_time_ns()
  probs = model.predict_proba(samples)
  Yavg["label"] = SpaceToLabel(Yavg['x'], Yavg['y'])
  Yavg.set_index("label", inplace=True)
  Yavg.sort_index()
  Ypred = (probs @ Yavg).to_numpy()
  score = calcScore(Ypred, expected)
  dt = process_time_ns() - t0
  return (score, dt/(len(samples)))

def plotColumns(dataframe, columnNames, plotname):
  dataframe[columnNames].plot()
  plt.savefig(plotname)

def runTests(testName, testScoreAndTime):

  t0 = process_time_ns()
  dt = process_time_ns() - t0
  timeToFitModel = dt/1e6

  score, time = testScoreAndTime(XsamplesAvg, YsamplesAvg)
  timeToProcessSamplesAvg = time
  accuracyScoreAvg = score

  # score, time = testScoreAndTime(Xsamples, Ysamples)
  timeToProcessSamples = time
  accuracyScore = score

  testResults = pd.Series({'timeToFitModel' : timeToFitModel,
                    'timeToProcessSamples' : timeToProcessSamples,
                    'accuracyScore' : accuracyScore,
                    'timeToProcessSamplesAvg' : timeToProcessSamplesAvg,
                    'accuracyScoreAvg' : accuracyScoreAvg}
                    )
  
  testResults.to_csv(f"./results/{testName}Results.csv")

runTests("NaiveBayesClosest", testScoreAndAvgTimeClosestClass)
runTests("NaiveBayesProb", testScoreAndAvgTimeProb)
model = makeNaiveBayesAvgModel()
runTests("NaiveBayesAvgClosest", testScoreAndAvgTimeClosestClass)
runTests("NaiveBayesAvgProb", testScoreAndAvgTimeProb)