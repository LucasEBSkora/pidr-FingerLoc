import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from time import process_time_ns
import numpy as np

fingerprints = pd.read_csv("../data/fingerprints.csv", index_col=0)
fingerprintsAVG = pd.read_csv("../data/fingerprints_mean_values.csv", index_col=0)
samples = pd.read_csv("../data/samples.csv", index_col=0)
samplesAVG = pd.read_csv("../data/samples_mean_values.csv", index_col=0)

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

t0 = process_time_ns()
model = makeNaiveBayesModel()
dt = process_time_ns() - t0
timeToFitModel = dt/1e6

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

def calcAverageError(predicted, expected):
  error = predicted - expected
  error = error * error
  error = error[:, 0] + error[:, 1]
  error = np.sum(np.sqrt(error)) / len(predicted)
  return error

def testScoreAndAvgTimeClosestClass(samples, expected):
  t0 = process_time_ns()
  labels = model.predict(samples)
  Ypred = LabelToSpace(labels)
  score = calcScore(Ypred, expected)
  dt = process_time_ns() - t0
  error = calcAverageError(Ypred, expected)
  return (score, dt/(len(samples)*1e6), error)

def testScoreAndAvgTimeProb(samples, expected):
  t0 = process_time_ns()
  probs = model.predict_proba(samples)
  Yavg["label"] = SpaceToLabel(Yavg['x'], Yavg['y'])
  Yavg.set_index("label", inplace=True)
  Yavg.sort_index()
  Ypred = (probs @ Yavg).to_numpy()
  score = calcScore(Ypred, expected)
  dt = process_time_ns() - t0
  error = calcAverageError(Ypred, expected)
  return (score, dt/(len(samples)*1e6), error)

def plotColumns(dataframe, columnNames, plotname):
  dataframe[columnNames].plot()
  plt.savefig(plotname)

def runTests(testName, testScoreAndTime):

  accuracyScoreAvg, timeToProcessSamplesAvg, errorAvg  = testScoreAndTime(XsamplesAvg, YsamplesAvg)

  score, time, error = testScoreAndTime(Xsamples, Ysamples)

  testResults = pd.Series({'timeToFitModel' : timeToFitModel,
                    'timeToProcessSamples' : time,
                    'accuracyScore' : score,
                    'error': error,
                    'timeToProcessSamplesAvg' : timeToProcessSamplesAvg,
                    'accuracyScoreAvg' : accuracyScoreAvg,
                    'errorAvg': errorAvg}
                    )
  
  testResults.to_csv(f"./results/{testName}Results.csv")

runTests("NaiveBayesClosest", testScoreAndAvgTimeClosestClass)
runTests("NaiveBayesProb", testScoreAndAvgTimeProb)
t0 = process_time_ns()
model = makeNaiveBayesAvgModel()
dt = process_time_ns() - t0
timeToFitModel = dt/1e6

runTests("NaiveBayesAvgClosest", testScoreAndAvgTimeClosestClass)
runTests("NaiveBayesAvgProb", testScoreAndAvgTimeProb)