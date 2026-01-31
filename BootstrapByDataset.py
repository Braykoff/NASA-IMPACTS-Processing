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

# Computes RMSE
def computeRmse(y_true, y_pred):
  return np.sqrt(mean_squared_error(y_true, y_pred))

# Computes Bias
def computeBias(y_true, y_pred):
  return np.mean(y_pred - y_true)

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

def getErrorBarsForDataset(datasetName, datasetFilename, sites, threshold, errBarsResults):
  # Open files
  dataset = pd.read_csv(datasetFilename, index_col="Time", parse_dates=True)
  print(f"Read dataset {datasetName} ({datasetFilename})")

  # Get data from each site
  datasetAccumulated = []
  siteDataAccumulated = []

  for siteName in dataset.columns:
    # Get data for this station
    siteData = pd.DataFrame(columns=[siteName])
    numFiles = 0

    for site in sites:
      if siteName in site.columns:
        siteData = site[[siteName]] if len(siteData) == 0 else pd.concat([siteData, site[[siteName]]])
        siteData = siteData[~siteData.index.duplicated(keep="first")]
        numFiles += 1

    if numFiles == 0:
      raise Exception(f"Could not find any file with {siteName}!")
    
    # Remove missing data and data below threshold
    datasetFiltered = dataset[(dataset[siteName] != -9999) & (dataset[siteName] >= threshold)]
    siteData = siteData[(siteData[siteName] != -9999) & (siteData[siteName] >= threshold)]

    # Drop duplicates
    datasetFiltered = datasetFiltered[~datasetFiltered.index.duplicated(keep="first")]
    siteData = siteData[~siteData.index.duplicated(keep="first")]

    # Find matching entries
    common = datasetFiltered.index.intersection(siteData.index)
    datasetFiltered = datasetFiltered.loc[common]
    siteData = siteData.loc[common]

    # Accumulate this data
    datasetAccumulated.extend(datasetFiltered[siteName])
    siteDataAccumulated.extend(siteData[siteName])

  # Bootstrap all data combined
  # Prediction = StageIV/MRMS, True = ASOS
  rmseScores, biasScores, crmseScores, correlationScores = getBootstrapMetrics(pd.Series(datasetAccumulated), pd.Series(siteDataAccumulated))
  
  errBarsResults[datasetName] = [
    getConfidenceIntervals(rmseScores),
    getConfidenceIntervals(biasScores),
    getConfidenceIntervals(crmseScores),
    getConfidenceIntervals(correlationScores)
  ]
  
  print(f"Finished {datasetName} (used {len(datasetAccumulated)} entries)")

# datasets should be a dictionary, with the dataset name as a key and a list
# containing the 1 hour, 6 hour, and 24 hour frequency csv file names (in that
# order) as the corresponding value.
# freq(n)HrSites should be the path to a directory with all the csv files in it.
# Subdirectories will not be checked.
def main(datasets, freq1HrSites, freq6HrSites, freq24HrSites, threshold):
  freqList = ["1hr", "6hr", "24hr"]

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

  # Init 2 graphs: one grouped by dataset, one by frequency
  figByDS, axsByDS = getBootstrapPlot("Station", threshold)
  figByFreq, axsByFreq = getBootstrapPlot("Station", threshold)

  # Run each dataset in a separate thread
  with Manager() as manager:
    errBarsResults = manager.dict()
    processes = []

    # Start each process
    for idx, freq in enumerate(freqList):
      for name, filenames in datasets.items():
        p = Process(
          target=getErrorBarsForDataset, 
          args=(f"{name}_{freq}", filenames[idx], allSites[idx], threshold, errBarsResults),
        )
        p.start()
        processes.append(p)

    # Wait for each process to finish
    for p in processes:
      p.join()

    print("All threads finished")

    # Add each dataset to the graph (by dataset)
    for name in datasets.keys():
      for freq in freqList:
        datasetName = f"{name}_{freq}"
        datasetResults = errBarsResults[datasetName]

        for i in range(4):
          yErr = [[datasetResults[i][1]], [datasetResults[i][2]]]
          axsByDS[i].errorbar([datasetName], [datasetResults[i][0]], yerr=yErr, fmt="o", capsize=5, color="blue")

    # Add each dataset to the graph (by frequency)
    for freq in freqList:
      for name in datasets.keys():
        datasetName = f"{name}_{freq}"
        datasetResults = errBarsResults[datasetName]

        for i in range(4):
          yErr = [[datasetResults[i][1]], [datasetResults[i][2]]]
          axsByFreq[i].errorbar([datasetName], [datasetResults[i][0]], yerr=yErr, fmt="o", capsize=5, color="blue")
    
  # Format and save graph
  for idx, fig in enumerate([figByDS, figByFreq]):
    fig.tight_layout()

    saveName = f"bootstrap/ModelsVsStation_Bootstrapped_By{"Dataset" if idx == 0 else "Freq"}.jpg"
    fig.savefig(saveName, dpi=300)
    print(f"Saved graph as {saveName}")
    plt.close(fig)

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