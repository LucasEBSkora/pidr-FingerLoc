import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from time import process_time_ns
from math import pi, exp
from numpy import concatenate, zeros, sqrt, sum

fingerprintsAVG = pd.read_csv("../data/fingerprints_mean_values.csv", index_col=0)
samples = pd.read_csv("../data/samples.csv", index_col=0)
samplesAVG = pd.read_csv("../data/samples_mean_values.csv", index_col=0)

def formatSamples(samples):
  X = samples[["rssi1", "rssi2", "rssi3", "rssi4", "rssi5"]].to_numpy()
  X = concatenate([X, zeros((len(X), 5), float)], axis=1)
  Y = (samples[["x", "y"]]).to_numpy()
  return (X, Y)

def formatFingerprints(fingerprints):
  X = fingerprints.drop(columns=['x', 'y']).to_numpy()
  Y = (fingerprints[["x", "y"]]).to_numpy()
  return (X, Y)

Xprob,Yavg = formatFingerprints(fingerprintsAVG)
Xsamples, Ysamples = formatSamples(samples)
XsamplesAvg, YsamplesAvg = formatSamples(samplesAVG)

def normalPDF(mean, stddev, sample):
  return exp(-(((sample-mean)/stddev)**2)/2)/(stddev*sqrt(2*pi))

def probabilisticDistance(v1, v2):
  distance = 5
  for i in range(0, 5):
    mean = v2[i]
    stddev = v2[i+5]
    distance -= normalPDF(mean, stddev, v1[i])
  return distance

def makeProbabilisticKNNModel(k):
  alg = KNeighborsRegressor(n_neighbors=k, metric=probabilisticDistance)
  alg.fit(Xprob, Yavg)
  return alg

def testScoreAndAvgTime(model, X, Y):
  t0 = process_time_ns()
  score = model.score(X, Y)
  dt = process_time_ns() - t0
  return (score, dt/(len(X)*1e6))

def calcAverageError(model, X, Y):
  Ypred = model.predict(X)
  error = Y - Ypred
  error = error * error
  error = error[:, 0] + error[:, 1]
  error = sum(sqrt(error)) / len(X)
  return error

def plotColumns(dataframe, columnNames, plotname):
  dataframe[columnNames].plot()
  plt.savefig(plotname)

def runTests(testName, makeModelFunction, maxk):
  timeToFitModel = []
  timeToProcessSamples = []
  accuracyScore = []
  errorAvg = []
  timeToProcessSamplesAvg = []
  accuracyScoreAvg = []
  error = []

  for k in range(1, maxk+1):

    t0 = process_time_ns()
    alg = makeModelFunction(k)
    dt = process_time_ns() - t0
    timeToFitModel.append(dt/1e6)

    score, time = testScoreAndAvgTime(alg, XsamplesAvg, YsamplesAvg)
    timeToProcessSamplesAvg.append(time)
    accuracyScoreAvg.append(score)
    errorAvg.append(calcAverageError(alg, XsamplesAvg, YsamplesAvg))

    score, time = testScoreAndAvgTime(alg, Xsamples, Ysamples)
    timeToProcessSamples.append(time)
    accuracyScore.append(score)
    error.append(calcAverageError(alg, Xsamples, Ysamples))

  testResults = pd.DataFrame({'timeToFitModel' : timeToFitModel,
                    'timeToProcessSamples' : timeToProcessSamples,
                    'accuracyScore' : accuracyScore,
                    'error': error,
                    'timeToProcessSamplesAvg' : timeToProcessSamplesAvg,
                    'accuracyScoreAvg' : accuracyScoreAvg,
                    'errorAvg': errorAvg},
                    index=list(range(1,47)))

  testResults.to_csv(f"./results/{testName}AvgResults.csv")
  plotColumns(testResults, ['accuracyScore', 'accuracyScoreAvg'], f"./results/{testName}ScoreResults.png")
  plotColumns(testResults, ['timeToFitModel', 'timeToProcessSamples', 'timeToProcessSamplesAvg'],f"./results/{testName}timeResults.png")
  testResults['scoreByTime'] = testResults['accuracyScore']/testResults['timeToProcessSamples']
  testResults['scoreByTimeAvg'] = testResults['accuracyScoreAvg']/testResults['timeToProcessSamplesAvg']
  plotColumns(testResults, ['scoreByTime', 'scoreByTimeAvg'], f"./results/{testName}scoreByTime.png")

runTests("PKNN", makeProbabilisticKNNModel, 46)