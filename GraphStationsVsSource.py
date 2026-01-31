from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os

binSize = 2.5 # mm

# Attempts to create a folder
def attemptCreateFolder(path):
  try:
    os.mkdir(path)
  except:
    pass

# Returns bin colors for scatter plot
def createHistogram(a, b, bins=90):
  histogram, xBins, yBins = np.histogram2d(a, b, bins=bins)
  xIdx = np.clip(np.digitize(a, xBins), 0, histogram.shape[0]-1)
  yIdx = np.clip(np.digitize(b, yBins), 0, histogram.shape[1]-1)
  return histogram[xIdx, yIdx]

# Gets at index using iloc for Pandas objects and regular indexing for others
def safeGetAtIndex(x, index):
  if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
    return x.iloc[index]
  else:
    return x[index]

# Returns bias and stddev of bias
def getBias(x, y):
  totalDiff = 0.0
  biases = []

  for i in range(0, len(x)):
    bias = safeGetAtIndex(y, i) - safeGetAtIndex(x, i)
    totalDiff += bias
    biases.append(bias)

  return totalDiff / len(x), np.std(biases)

# Returns coefficient (m), intercept (b), and r^2 value
def getLinearRegression(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    model = LinearRegression().fit(x, y)
    r2 = model.score(x, y)

    return model.coef_[0], model.intercept_, r2

# Return RMSE
def getRMSE(x, y):
  return np.sqrt(np.mean((y - x) ** 2))

# Returns mean absolute error
def getMAE(x, y):
  return np.mean(np.abs(y - x))

# Returns mean absolute percentage error
def getMAPE(x, y):
  return np.mean(np.abs((y - x) / y)) * 100

# Returns mean bias percentage error
def getMBPE(x, y):
  return np.mean((y - x) / x) * 100

# Makes a graph with a scatter plot and a histogram
# Returns r2, bias, rmse, mae, mape, mbpe stats
def graphSiteVsSource(siteData, sourceName, sourceData, siteName, frequency, threshold, outputName):
  dataMax = max(max(siteData), max(sourceData))
  plt.figure()

  # Create histogram
  plt.subplot(1, 2, 1)
  plt.grid()
  plt.gca().set_axisbelow(True)
  plt.gca().set_aspect("equal")

  plt.xlim(0, dataMax)
  plt.ylim(0, dataMax)

  colors = createHistogram(siteData, sourceData)
  plt.scatter(siteData, sourceData, c=colors, alpha=0.5, s=1.5, cmap="jet")

  plt.gca().set_xlabel(f"{siteName} Precipitation (mm)")
  plt.gca().set_ylabel(f"{sourceName} Precipitation (mm)")

  # y = x line
  plt.axline(
    [0, 0],
    [dataMax, dataMax],
    color="black",
    alpha=0.5,
    linewidth=0.5
  )

  # Create histogram
  plt.subplot(1, 2, 2)
  plt.grid()
  plt.gca().set_axisbelow(True)

  bins = np.arange(0, dataMax + binSize, binSize)

  plt.hist(
    siteData, 
    alpha=0.5, 
    label="Sites", 
    bins=bins,
    weights=np.ones(len(siteData)) / len(siteData) # make percentage
  )
  plt.hist(
    sourceData, 
    alpha=0.5, 
    label=sourceName, 
    bins=bins,
    weights=np.ones(len(sourceData)) / len(sourceData)
  )

  plt.gca().set_xlabel("Precipitation (mm)")
  plt.gca().set_ylabel("Frequency (%)")
  plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
  plt.ylim(0, 1)
  plt.legend(loc="upper right")

  # Get stats
  # Site is always observation
  m, b, r2 = getLinearRegression(siteData, sourceData)
  bias, _ = getBias(siteData, sourceData)
  rmse = getRMSE(siteData, sourceData)
  mae = getMAE(siteData, sourceData)
  mape = getMAPE(siteData, sourceData)
  mbpe = getMBPE(siteData, sourceData)

  # Figure title
  title = [
    f"{frequency} Hour Total {siteName} vs {sourceName}",
    f"{len(siteData)} Observations ({threshold}mm threshold)",
    f"y ~ {round(m, 3)}x + {round(b, 3)} (R^2: {round(r2, 3)})",
    f"RMSE: {round(rmse, 3)}",
    f"Mean Bias: {round(bias, 3)}, Mean Bias Percent Error: {round(mbpe, 3)}%",
    f"Mean Absolute Error: {round(mae, 3)}, Mean Absolute Percent Error: {round(mape, 3)}%"
  ]
  plt.suptitle("\n".join(title))

  # Format
  plt.gcf().set_size_inches(13, 7)
  plt.tight_layout()
  plt.savefig(outputName, dpi=300, bbox_inches="tight")
  print(f"Saved as {outputName}")
  plt.close()

  return r2, bias, rmse, mae, mape, mbpe

# Source = StageIV or MRMS data
# Sites = ASOS, UCONN, SBU, etc
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
  attemptCreateFolder("graphs")
  attemptCreateFolder(f"graphs/SiteVs{sourceName}_{frequency}hr")
  print("Attempted to create output folders")

  # Save error metrics
  metrics = {
    "site": [],
    "r2": [],
    "bias": [],
    "rmse": [],
    "mae": [],
    "mape": [],
    "mbpe": []
  }

  # Accumulate data for overall graph
  allData = []
  allSource = []

  # Create plots for each station
  for station in source.columns:
    # Find what other file this data is in
    data = pd.DataFrame(columns=[station])
    numFiles = 0
    
    for site in sites:
      if station in site.columns:
        data = site[[station]] if len(data) == 0 else pd.concat([data, site[[station]]])
        data = data[~data.index.duplicated(keep="first")]
        numFiles += 1

    if numFiles == 0:
      print(f"Could not find any other file to match site {station}!")
      continue
      
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

    # Accumulate data
    allData.extend(dataFiltered)
    allSource.extend(sourceFiltered)

    # Create single station graph and get metrics
    r2, bias, rmse, mae, mape, mbpe = graphSiteVsSource(
      dataFiltered,
      sourceName,
      sourceFiltered,
      station,
      frequency,
      threshold,
      f"graphs/SiteVs{sourceName}_{frequency}hr/{station}_vs_{sourceName}.jpg"
    )

    # Save metrics
    metrics["site"].append(station)
    metrics["r2"].append(r2)
    metrics["bias"].append(bias)
    metrics["rmse"].append(rmse)
    metrics["mae"].append(mae)
    metrics["mape"].append(mape)
    metrics["mbpe"].append(mbpe)

  # Create all site graph
  allData = np.array(allData)
  allSource = np.array(allSource)

  graphSiteVsSource(
    allData, 
    sourceName,
    allSource,
    "Sites",
    frequency,
    threshold,
    f"graphs/AllSites_vs_{sourceName}_{frequency}hr.jpg"
  )

  # Save metrics
  metrics = pd.DataFrame(metrics)
  metrics.set_index("site", inplace=True)
  metricsFilename = f"Metrics_SitesVs{sourceName}_{frequency}hr.csv"
  metrics.to_csv(metricsFilename)
  print(f"Saved as {metricsFilename}")

# Run
if __name__ == "__main__":
  graph(
    sys.argv[1], # Processed/StageIV_1hr.csv or Processed/MRMS_1hr.csv or ...
    "MRMS_Multi", # MRMS or StageIV
    sys.argv[2:], # Processed/CombinedASOS.csv, Processed/CombinedSBU.csv...
    1, # hour frequency (1, 6, or 24). Used in graph title
    0.25, # minimum mm of precipitation to include in graph
  )

  # Execute from command line:
  #python3 Scripts/graphing/GraphStationsVsSource.py Processed/MRMS/MRMS_1hr_Multi.csv Processed/CombinedASOS_FixedRanges.csv Processed/CombinedSBU_v2.csv Processed/CombinedSBU_BNL_v2.csv Processed/CombinedUCONN_2122_v2.csv Processed/CombinedUCONN_2223_v2_truncated.csv Processed/CombinedUCONN_2324_v2_truncated.csv 
  #python3 Scripts/graphing/GraphStationsVsSource.py Processed/MRMS/MRMS_6hr_Multi.csv Processed/aggregated/ASOS_6h.csv Processed/aggregated/SBU_6h.csv Processed/aggregated/SBU_BNL_6h.csv Processed/aggregated/UCONN_2122_6h.csv Processed/aggregated/UCONN_2223_6h.csv Processed/aggregated/UCONN_2324_6h.csv
  #python3 Scripts/graphing/GraphStationsVsSource.py Processed/MRMS/MRMS_24hr_Multi.csv Processed/aggregated/ASOS_24h.csv Processed/aggregated/SBU_24h.csv Processed/aggregated/SBU_BNL_24h.csv Processed/aggregated/UCONN_2122_24h.csv Processed/aggregated/UCONN_2223_24h.csv Processed/aggregated/UCONN_2324_24h.csv