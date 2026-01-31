import pandas as pd
import sys
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Returns coefficient (m), intercept (b), and r^2 value
def getLinearRegression(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    model = LinearRegression().fit(x, y)
    r2 = model.score(x, y)

    return model.coef_[0], model.intercept_, r2

# Creates a 2x3 graph of all six error metrics vs some x axis
def graphAllMetrics(allData, xAxisLabel, xAxis, markers, graphTitle, saveName):
  plt.figure()

  for i, data in enumerate(allData):
    title, data = data
    plt.subplot(2, 3, i+1)
    plt.grid()
    plt.gca().set_axisbelow(True)

    if title == "R^2":
      # Force y axis of r^2 graph to be 0 - 1
      plt.gca().set_ylim(0, 1)

    # Plot each point individually, because they have different markers
    for pt in range(len(data)):
      plt.scatter(
        xAxis[pt],
        data.iloc[pt],
        alpha=1,
        s=9,
        marker=markers[pt],
        color="C0"
      )

    plt.gca().set_xlabel(xAxisLabel)
    xRange = (-0.1, 1.1)
    plt.xlim(xRange)
    plt.gca().set_ylabel(title)

    # Add trend line
    m, b, r2 = getLinearRegression(xAxis, data)

    plt.axline(
      [xRange[0], m * xRange[0] + b],
      [xRange[1], m * xRange[1] + b],
      color="black",
      alpha=0.75,
      linewidth=1
    )
    plt.title(f"y ~ {round(m, 5)}x + {round(b, 3)}\nR^2: {round(r2, 3)}")

  plt.suptitle(graphTitle)
  plt.gcf().set_size_inches(13, 6.5)
  plt.tight_layout()
  plt.savefig(saveName, dpi=300, bbox_inches="tight")
  print(f"Saved as {saveName}")
  plt.close()

def graph(metricsFilename, rqiFilename, saveName):
  # Open files
  metrics = pd.read_csv(metricsFilename, index_col="site")
  print(f"Read {metricsFilename}")

  rqi = pd.read_csv(rqiFilename, index_col="Time")
  print(f"Read {rqiFilename}")

  # Get locations
  markers = []
  meanRQI = []

  for row in metrics.itertuples():
    meanRQI.append(rqi[row.Index].mean(skipna=True))

    # Different markers for different sites
    if row.Index == "SBU" or row.Index == "SBU_BNL":
      markers.append("s") # square for SBU
    elif row.Index == "GAIL" or row.Index == "D3R":
      markers.append("^") # triangle for UCONN
    else:
      markers.append("o") # circle for ASOS

  # Combine data into single, iterable list
  allData = [
    ("R^2", metrics["r2"]), 
    ("Bias", metrics["bias"]), 
    ("RMSE", metrics["rmse"]),
    ("MAE", metrics["mae"]),
    ("MAPE", metrics["mape"]),
    ("MBPE", metrics["mbpe"])
  ]

  # MARK: Metric vs RQI
  graphAllMetrics(
    allData,
    "RQI",
    meanRQI,
    markers,
    f"Sites/{saveName} Error Metrics vs RQI",
    f"graphs/Sites_vs_{saveName}_RQIErrorMetrics.jpg"
  )

# Run
if __name__ == "__main__":
  graph(
    sys.argv[1], # Processed/metrics.csv
    sys.argv[2], # Processed/MRMS/MRMS_1hr_RQI.csv
    "MRMS_Radar_6hr" # Save name
  )

  # python3 Scripts/graphing/GraphErrorMetricsVsRQI.py Processed/MRMSComparisons/Metrics_SitesVsMRMS_Radar_6hr.csv Processed/MRMS/MRMS_1hr_RQI.csv
  