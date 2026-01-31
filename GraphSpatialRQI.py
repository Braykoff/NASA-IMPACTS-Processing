import pandas as pd
import sys
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib import pyplot as plt

def graph(siteInfoFilename, rqiFilename, mapFilename):
  # Open files
  sites = pd.read_csv(siteInfoFilename, index_col="Site")
  rqi = pd.read_csv(rqiFilename, index_col="Time")

  print(f"Read {siteInfoFilename} and {rqiFilename}")

  states = gpd.read_file(mapFilename)
  print(f"Read {mapFilename}")

  # Get average RQI
  lats = []
  longs = []
  markers = []
  meanRQI = []

  for row in sites.itertuples():
    lats.append(row.Lat)
    longs.append(row.Lon)
    meanRQI.append(rqi[row.Index].mean(skipna=True))

    # Different markers for different sites
    if row.Index == "SBU" or row.Index == "SBU_BNL":
      markers.append("s") # square for SBU
    elif row.Index == "GAIL" or row.Index == "D3R":
      markers.append("^") # triangle for UCONN
    else:
      markers.append("o") # circle for ASOS

  # Determine graph range
  padding = 0.5
  graphMaxLats = (min(lats)-padding, max(lats)+padding)
  graphMaxLongs = (min(longs)-padding, max(longs)+padding)

  # Init graph
  plt.figure()
  ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
  ax.set_xlim(graphMaxLongs)
  ax.set_ylim(graphMaxLats)

  states.plot(ax=ax, color="None", edgecolor="black", alpha=0.7)

  # Plot each point individually (because of different markers)
  for pt in range(len(meanRQI)):
    plt.scatter(
      longs[pt],
      lats[pt],
      c=meanRQI[pt],
      marker=markers[pt],
      alpha=1,
      s=20,
      cmap = "viridis",
      vmin = 0.0, # Range of 0 - 1 for RQI
      vmax = 1.0
    )

  # Format graph
  plt.colorbar(ax=ax, shrink=.5)
  ax.set_title("MRMS RQI")
  plt.gcf().set_size_inches(8, 6)
  plt.tight_layout()

  # Save 
  filename = "RQI_Spatial.jpg"
  plt.savefig(filename, dpi=300, bbox_inches="tight")
  print(f"Saved as {filename}")
  plt.close()

# Run
if __name__ == "__main__":
  graph(
    sys.argv[1], # Data/Site_Info.csv
    sys.argv[2], # Data/MRMS/RQI.csv
    sys.argv[3] # Maps/US_States/s_08mr23.shp
  )