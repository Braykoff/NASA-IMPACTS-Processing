from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from multiprocessing import Process, Manager
import sys
import os
from pathlib import Path

# Attempts to create a folder
def attemptCreateFolder(path):
  try:
    os.mkdir(path)
  except:
    pass

# Gets a list of all csv files in a directory (and no subdirectories)
def getChildCSVFiles(path):
  path = Path(path)
  csvFiles = []

  for f in path.iterdir():
    if f.is_file() and f.suffix.lower() == ".csv":
      csvFiles.append(f)

  return csvFiles

# Computes RMSE
def computeRmse(y_true, y_pred):
  return np.sqrt(mean_squared_error(y_true, y_pred))

# Computes Bias
def computeBias(y_true, y_pred):
  return np.mean(y_pred - y_true)

# Bootstraps data and returns rmse, bias, mae, and r^2
def getBootstrapMetrics(yTrue: pd.Series, yPred: pd.Series, n=10_000):
  rmseScores = []
  biasScores = []
  maeScores = []
  r2Scores = []

  for _ in range(n):
    # Bootstrap sample
    indices = resample(np.arange(len(yTrue)), replace=True)
    yTrueResampled = yTrue.iloc[indices]
    yPredResampled = yPred.iloc[indices]

    # Compute metrics
    rmseScores.append(computeRmse(yTrueResampled, yPredResampled))
    biasScores.append(computeBias(yTrueResampled, yPredResampled))
    maeScores.append(mean_absolute_error(yTrueResampled, yPredResampled))
    r2Scores.append(r2_score(yTrueResampled, yPredResampled))

  return rmseScores, biasScores, maeScores, r2Scores

# Returns mean, lower 95%, upper 95%
def getConfidenceIntervals(scores):
  mean = np.mean(scores)
  lower = np.percentile(scores, 2.5)
  upper = np.percentile(scores, 97.5)
  return mean, mean - lower, upper - mean

# Runs bootstrap for a single station (run in separate thread)
def runBootstrapForStation(station, source, sites, threshold, metricResults, dataAccumulated, sourcesAccumulated):
  # Get data for this station
  data = pd.DataFrame(columns=[station])
  numFiles = 0

  for site in sites:
    if station in site.columns:
      data = site[[station]] if len(data) == 0 else pd.concat([data, site[[station]]])
      data = data[~data.index.duplicated(keep="first")]
      numFiles += 1

  if numFiles == 0:
    print(f"Could not find any other file to match site {station}!")
    return
    
  print(f"Used {numFiles} files for site {station}, {len(data)} rows")

  # Remove missing data and data below threshold
  dataFiltered = data[(data[station] != -9999) & (data[station] >= threshold)]
  sourceFiltered = source[(source[station] != -9999) & (source[station] >= threshold)]

  # Drop duplicates
  dataFiltered = dataFiltered[~dataFiltered.index.duplicated(keep="first")]
  sourceFiltered = sourceFiltered[~sourceFiltered.index.duplicated(keep="first")]

  # Find matching days
  common = dataFiltered.index.intersection(sourceFiltered.index)
  dataFiltered = dataFiltered.loc[common]
  sourceFiltered = sourceFiltered.loc[common]

  dataFiltered = dataFiltered[station]
  sourceFiltered = sourceFiltered[station]

  # Accumulate this data for overall plot
  dataAccumulated.extend(dataFiltered)
  sourcesAccumulated.extend(sourceFiltered)

  # Get bootstrapped metrics for this site
  # Prediction = StageIV/MRMS, True = ASOS
  rmseScores, biasScores, crmseScores, correlationScores = getBootstrapMetrics(dataFiltered, sourceFiltered)

  metricResults.append((station, [
    getConfidenceIntervals(rmseScores),
    getConfidenceIntervals(biasScores),
    getConfidenceIntervals(crmseScores),
    getConfidenceIntervals(correlationScores)
  ]))

def graph(sourceFilename, sourceName, siteFilenames, frequency, threshold):
  # Open files
  source = pd.read_csv(sourceFilename, index_col="Time", parse_dates=True)

  print(f"Read source file {sourceFilename} ({sourceName}) with {len(source.columns)} sites")

  sites = []
  for name in siteFilenames:
    site = pd.read_csv(name, index_col="Time", parse_dates=True)
    sites.append(site)
    print(f"Read {name} with {len(site.columns)} sites")

  # Init output
  attemptCreateFolder("bootstrap")
  print("Attempted to create output folders")

  # Init graph
  _, axs = plt.subplots(2, 2)
  axs = axs.flatten()

  for i, title in enumerate(["RMSE", "Bias", "MAE", "R^2"]):
    axs[i].set_title(title)
    axs[i].grid()
    axs[i].tick_params(axis="x", rotation=90)

  # Give bias a horizontal line
  axs[1].axhline(y=0.0, color="black", linestyle="--")

  # Create plots for each station in multiple threads
  with Manager() as manager:
    metricResults = manager.list()
    dataAccumulated = manager.list()
    sourcesAccumulated = manager.list()
    processes = []

    # Start each process
    for station in source.columns:
      p = Process(target=runBootstrapForStation, args=(station, source, sites, threshold, metricResults, dataAccumulated, sourcesAccumulated))
      p.start()
      processes.append(p)

    # Wait for each process to finish
    for p in processes:
      p.join()
    
    # Graph results
    for result in metricResults:
      station, intervals = result
      for i in range(4):
        #axs[i].plot([station], [intervals[i][0]], marker="o")
        yErr = [[intervals[i][1]], [intervals[i][2]]]
        axs[i].errorbar([station], [intervals[i][0]], yerr=yErr, fmt="o", capsize=5, color="blue")
    
    print("All threads finished")

    # Graph overall results
    print("Graphing overall...")
    overallMetrics = getBootstrapMetrics(pd.Series(list(dataAccumulated)), pd.Series(list(sourcesAccumulated)))

    for i in range(4):
      confidence = getConfidenceIntervals(overallMetrics[i])
      yErr = [[confidence[1]], [confidence[2]]]
      axs[i].errorbar(["NEUS"], [confidence[0]], yerr=yErr, fmt="o", capsize=5, color="green")

  # Format and save graph
  plt.suptitle(f"Sites vs {sourceName} ({frequency}hr frequency)\n{threshold}mm Threshold")

  plt.gcf().set_size_inches(16, 10)
  plt.tight_layout()

  saveName = f"bootstrap/SitesVs{sourceName}_BootstrappedMetrics_{frequency}hr.jpg"
  plt.savefig(saveName, dpi=300)
  print(f"Saved as {saveName}")
  plt.close()

# Run
if __name__ == "__main__":
  ASOS1Hr = getChildCSVFiles("Processed")
  ASOS6Hr = getChildCSVFiles("Processed/aggregated/6hr")
  ASOS24Hr = getChildCSVFiles("Processed/aggregated/24hr")

  #graph(
  #  sys.argv[1], # Processed/MRMS.csv
  #  "MRMS_Radar",
  #  sys.argv[2:], # Processed/ASOS...
  #  1, # Frequency (hours)
  #  0.25 # Minimum precipitation threshold (mm)
  #)

  # Just run everything:
  graph("Processed/MRMS/MRMS_1hr_Radar.csv", "MRMS_Radar", ASOS1Hr, 1, 0.25)
  graph("Processed/MRMS/MRMS_6hr_Radar.csv", "MRMS_Radar", ASOS6Hr, 6, 0.25)
  graph("Processed/MRMS/MRMS_24hr_Radar.csv", "MRMS_Radar", ASOS24Hr, 24, 0.25)
  graph("Processed/MRMS/MRMS_1hr_Multi.csv", "MRMS_Multi", ASOS1Hr, 1, 0.25)
  graph("Processed/MRMS/MRMS_6hr_Multi.csv", "MRMS_Multi", ASOS6Hr, 6, 0.25)
  graph("Processed/MRMS/MRMS_24hr_Multi.csv", "MRMS_Multi", ASOS24Hr, 24, 0.25)

  graph("Processed/StageIV/StageIV_1hr.csv", "StageIV", ASOS1Hr, 1, 0.25)
  graph("Processed/StageIV/StageIV_6hr.csv", "StageIV", ASOS6Hr, 6, 0.25)
  graph("Processed/StageIV/StageIV_24hr.csv", "StageIV", ASOS24Hr, 24, 0.25)

  # python3 Scripts/bootstrapping/GraphBootstrappedErrorMetrics.py Processed/MRMS/MRMS_1hr_Radar.csv Processed/CombinedASOS_FixedRanges.csv Processed/CombinedSBU_v2.csv Processed/CombinedSBU_BNL_v2.csv Processed/CombinedUCONN_2122_v2.csv Processed/CombinedUCONN_2223_v2_truncated.csv Processed/CombinedUCONN_2324_v2_truncated.csv 
