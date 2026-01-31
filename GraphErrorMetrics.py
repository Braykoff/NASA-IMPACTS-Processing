import pandas as pd
import sys
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from pyproj import Geod

# Returns coefficient (m), intercept (b), and r^2 value
def getLinearRegression(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    model = LinearRegression().fit(x, y)
    r2 = model.score(x, y)

    return model.coef_[0], model.intercept_, r2


# Init lat/lon to cartesian coordinate converter
geod = Geod(ellps="WGS84")

def getDistanceM(lat1, lon1, lat2, lon2):
  _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
  return dist

def getNearestNexradDist(lat, long, nexrad):
  distances = []

  for site in nexrad.itertuples():
    distances.append(getDistanceM(lat, long, site.LAT, site.LON)/1000) # meters to km
    # TODO ok to ignore elevation?
  
  return min(distances)

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
    xRange = (min(min(xAxis), 0), max(xAxis)+20)
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

def graph(siteInfoFilename, metricsFilename, nexradFilename, mapFile, saveName):
  # Open files
  sites = pd.read_csv(siteInfoFilename, index_col="Site")
  metrics = pd.read_csv(metricsFilename, index_col="site")
  print(f"Read {siteInfoFilename} and {metricsFilename}")

  nexrad = pd.read_csv(nexradFilename, index_col="ICAO")
  print(f"Read {nexradFilename}")

  states = gpd.read_file(mapFile)
  print(f"Read {mapFile}")

  # Get locations
  elevations = []
  lats = []
  longs = []
  markers = []
  nexradDistances = []

  for row in metrics.itertuples():
    info = sites.loc[row.Index]
    elevations.append(info.Elevation)
    lats.append(info.Lat)
    longs.append(info.Lon)
    nexradDistances.append(getNearestNexradDist(info.Lat, info.Lon, nexrad))

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

  # MARK: Metric vs Elevation
  # Create error metric elevation graph
  graphAllMetrics(
    allData,
    "Elevation (m)",
    elevations,
    markers,
    f"Sites/{saveName} Error Metrics vs Elevation",
    f"graphs/Sites_vs_{saveName}_ErrorMetrics.jpg"
  )

  # MARK: Metric vs Radar
  # Create error metrics vs radar distance graphs
  graphAllMetrics(
    allData,
    "Distance from NEXRAD (km)",
    nexradDistances,
    markers,
    f"Sites/{saveName} Error Metrics vs Distance from NEXRAD Radar",
    f"graphs/Site_vs_{saveName}_ErrorMetricsFromRadar.jpg"
  )

  # MARK: Spatial
  # Create error metric spatial graphs
  plt.figure()

  #graphMaxLats = (40, 46) # S, N
  #graphMaxLongs = (-75, -69) # W, E
  padding = 0.5
  graphMaxLats = (min(lats)-padding, max(lats)+padding)
  graphMaxLongs = (min(longs)-padding, max(longs)+padding)

  # Enforce square graphs
  #halfMaxDiff = max(graphMaxLats[1] - graphMaxLats[0], graphMaxLongs[1] - graphMaxLongs[0]) / 2.0
  #avg = (np.average(graphMaxLats), np.average(graphMaxLongs))

  #graphMaxLats = (avg[0]-halfMaxDiff, avg[0]+halfMaxDiff)
  #graphMaxLongs = (avg[1]-halfMaxDiff, avg[1]+halfMaxDiff)

  for i, data in enumerate(allData):
    title, data = data

    ax = plt.subplot(2, 3, i+1, projection=ccrs.PlateCarree())
    ax.set_xlim(graphMaxLongs)
    ax.set_ylim(graphMaxLats)

    states.plot(ax=ax, color="None", edgecolor="black", alpha=0.7)

    normalRange = [data.min(), data.max()]
    extreme = np.abs(normalRange).max()

    # Plot each point individually, because they have different markers
    for pt in range(len(data)):
      # Diverging color bar for Bias and MBPE plots, normal for others
      diverging = (i == 1 or i == 5)

      plt.scatter(
        longs[pt],
        lats[pt],
        c=data.iloc[pt],
        marker=markers[pt],
        alpha=1,
        s=20,
        cmap = "seismic" if diverging else "viridis",
        vmin = -extreme if diverging else normalRange[0],
        vmax = extreme if diverging else normalRange[1]
      )

    plt.colorbar(ax=ax, shrink=.5, extend="both")
    ax.set_title(title)
  
  plt.suptitle(f"Sites/{saveName} Error Metrics")
  filename = f"graphs/Sites_vs_{saveName}_ErrorMetricsSpatial.jpg"
  plt.gcf().set_size_inches(16, 9) # 15, 11 for square graphs
  plt.tight_layout()
  plt.savefig(filename, dpi=300, bbox_inches="tight")
  print(f"Saved as {filename}")
  plt.close()

# Run
if __name__ == "__main__":
  graph(
    sys.argv[1], # Data/Site_Info.csv
    sys.argv[2], # Processed/metrics.csv
    sys.argv[3], # Data/NEXRAD_Locations.csv
    sys.argv[4], # Maps/US_States_Maps/s_08mr23.shp
    "MRMS_Multi_1hr" # Save name
  )

  # python3 Scripts/graphing/GraphErrorMetrics.py Data/Site_Info.csv Processed/MRMSComparisons/Metrics_SitesVsMRMS_Multi_1hr.csv Data/NEXRAD_Locations.csv Maps/US_States/s_08mr23.shp