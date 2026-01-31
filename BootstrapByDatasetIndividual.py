from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from multiprocessing import Process, Manager
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

# Inits bootstrapping plot, returns figure and flattened axes
def getBootstrapPlot(stationName, threshold):
  fig, axs = plt.subplots(2, 2)
  axs = axs.flatten()

  for i, title in enumerate(["RMSE", "Bias", "MAE", "R^2"]):
    axs[i].set_title(title)
    axs[i].grid()
    axs[i].tick_params(axis="x", rotation=90)

  # Give bias a horizontal line
  axs[1].axhline(y=0.0, color="black", linestyle="--")

  # Remove bottom ticks from first and second graphs
  axs[0].tick_params(labelbottom=False)
  axs[1].tick_params(labelbottom=False)

  fig.suptitle(f"Weather Models vs {stationName} Measurements\n{threshold}mm Threshold")
  fig.supxlabel("Model")

  fig.set_size_inches(14, 12)

  return fig, axs

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

def graphSingleSite(datasets, allSitesByFreq, station, threshold):
  freqList = [1, 6, 24]

  # Init graph
  fig, axs = getBootstrapPlot(station, threshold)

  for dsName, dsList in datasets.items():
    for idx, dataset in enumerate(dsList):
      # Get data for this station and frequency
      siteData = pd.DataFrame(columns=[station])
      numFiles = 0

      for site in allSitesByFreq[idx]:
        if station in site.columns:
          siteData = site[[station]] if len(siteData) == 0 else pd.concat([siteData, site[[station]]])
          siteData = siteData[~siteData.index.duplicated(keep="first")]
          numFiles += 1

      if numFiles == 0:
        raise Exception(f"No files found that contain {station}")

      # Remove data below threshold
      datasetFiltered = dataset[(dataset[station] != -9999) & (dataset[station] >= threshold)]
      siteData = siteData[(siteData[station] != -9999) & (siteData[station] >= threshold)]

      # Drop duplicates
      datasetFiltered = datasetFiltered[~datasetFiltered.index.duplicated(keep="first")]
      siteData = siteData[~siteData.index.duplicated(keep="first")]

      # Find matching entries
      common = datasetFiltered.index.intersection(siteData.index)
      datasetFiltered = datasetFiltered.loc[common]
      siteData = siteData.loc[common]

      # Bootstrap this data
      # Prediction = StageIV/MRMS, True = ASOS
      rmseScores, biasScores, crmseScores, correlationScores = getBootstrapMetrics(datasetFiltered[station], siteData[station])
      errBars = [
        getConfidenceIntervals(rmseScores),
        getConfidenceIntervals(biasScores),
        getConfidenceIntervals(crmseScores),
        getConfidenceIntervals(correlationScores)
      ]

      for i, err in enumerate(errBars):
        yErr = [[err[1]], [err[2]]]
        axs[i].errorbar([f"{dsName}_{freqList[idx]}hr"], [err[0]], yerr=yErr, fmt="o", capsize=5, color="blue")
  
  # Save
  fig.tight_layout()
  saveName = f"bootstrap/sites/{station}_vs_Models_Bootstrapped.jpg"
  fig.savefig(saveName, dpi=300)
  print(f"Saved graph as {saveName}")
  plt.close(fig)

# datasets should be a dictionary, with the dataset name as a key and a list
# containing the 1 hour, 6 hour, and 24 hour frequency csv file names (in that
# order) as the corresponding value.
# freq(n)HrSites should be the path to a directory with all the csv files in it.
# Subdirectories will not be checked.
def main(datasets, freq1HrSites, freq6HrSites, freq24HrSites, threshold):
  # Open each dataset and get list of stations
  stations = None

  for dsName, dsList in datasets.items():
    for idx, ds in enumerate(dsList):
      ds = pd.read_csv(ds, index_col="Time", parse_dates=True)
      datasets[dsName][idx] = ds

      if stations == None:
        stations = set(ds.columns)
      else:
        stations &= set(ds.columns)

  print(f"Found {len(stations)} stations in datasets")

  # Get csv files from each site and each frequency
  print("Reading site csv files")
  allSites = [[], [], []] # list of list of pd DataFrames for each site files

  for i in range(3):
    childCSVFiles = getChildCSVFiles([freq1HrSites, freq6HrSites, freq24HrSites][i])

    for csv in childCSVFiles:
      allSites[i].append(pd.read_csv(csv, index_col="Time", parse_dates=True))
      print(f"Read {csv}")

  # Init output
  attemptCreateFolder("bootstrap")
  attemptCreateFolder("bootstrap/sites")

  # Run each station in a separate thread
  with Manager():
    processes = []

    # Start each process
    for station in stations:
      p = Process(
        target=graphSingleSite,
        args=(datasets, allSites, station, threshold)
      )
      processes.append(p)
      p.start()

    # Wait for each process to finish
    for p in processes:
      p.join()

  # All processes have ended
  print("All processes finished")

# Run
if __name__ == "__main__":
  main(
    # Dataset names and 1hr, 6hr, 24hr csv filenames:
    {
      "StageIV": [
        "Processed/StageIV/StageIV_1hr.csv",
        "Processed/StageIV/StageIV_6hr.csv",
        "Processed/StageIV/StageIV_24hr.csv"
      ],
      "MRMS_Multi": [
        "Processed/MRMS/MRMS_1hr_Multi.csv",
        "Processed/MRMS/MRMS_6hr_Multi.csv",
        "Processed/MRMS/MRMS_24hr_Multi.csv"
      ],
      "MRMS_Radar": [
        "Processed/MRMS/MRMS_1hr_Radar.csv",
        "Processed/MRMS/MRMS_6hr_Radar.csv",
        "Processed/MRMS/MRMS_24hr_Radar.csv"
      ]
    },
    # Directories for 1hr, 6hr, and 24hr csv files:
    "Processed",
    "Processed/aggregated/6hr",
    "Processed/aggregated/24hr",
    0.25 # mm threshold
  )
