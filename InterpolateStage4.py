import xarray as xa
import pandas as pd
import numpy as np
import sys
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree
from pathlib import Path
import re
from pyproj import Transformer
import time

# Init lat/lon to cartesian coordinate converter
transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

def latLonToXYZ(lat, lon):
  if isinstance(lat, np.ndarray):
    return transformer.transform(lon, lat, np.zeros(len(lat)))
  else:
    return transformer.transform(lon, lat, 0)

def main(siteInfoFile, stage4Dir, freq):
  start = time.time()

  # Open site info file
  sites = pd.read_csv(siteInfoFile, index_col="Site")
  print(f"Found {len(sites)} sites in {siteInfoFile}")

  # Init output
  output = {"Time": []}

  for site in sites.itertuples():
    output[site.Index] = []
    #output[f"{site.Index}_stddev"] = []

  # RegEx pattern that matches file names. Group 1 is the date: year, month, day, hour
  pattern = r"st4_conus.(\d{10})." + str(freq).zfill(2) + "h.grb2$"

  # Iterate over grb2 files
  stage4Dir = Path(stage4Dir)

  for root, _, files in stage4Dir.walk():
    for idx, file in enumerate(files):
      # Check that this is a grb2 file
      match = re.match(pattern, file)

      if not match:
        # This is not a grb2 file
        print(f"({idx+1}/{len(files)}) Skipped {file}")
        continue

      # Open file
      data = xa.open_dataset(root.joinpath(file), engine="cfgrib")
      date = pd.to_datetime(f"{match.group(1)}:00:00", format="%Y%m%d%H:%M:%S")
      print(f"({idx+1}/{len(files)}) Read {file} ({date})")

      output["Time"].append(date)

      # Create KDTree
      dataCoords = np.column_stack([data.latitude.values.ravel(), data.longitude.values.ravel()])
      dataXYZ = np.column_stack(latLonToXYZ(data.latitude.values.ravel(), data.longitude.values.ravel()))
      dataTp = data.tp.values.ravel()

      tree = KDTree(dataXYZ)

      # Filter out all points not within 16 km of a site
      filteredIndexes = set()

      for site in sites.itertuples():
        targetCoords = np.array([site.Lat % 360, site.Lon % 360])
        targetXYZ = latLonToXYZ(targetCoords[0], targetCoords[1])

        filteredIndexes = filteredIndexes.union(tree.query_ball_point(targetXYZ, 16000)) # 16 km

      filteredIndexes = list(filteredIndexes)

      # Create interpolator
      interpolator = LinearNDInterpolator(
        dataCoords[filteredIndexes],
        dataTp[filteredIndexes]
      )

      # Get data for each site
      for site in sites.itertuples():
        # Interpolate
        targetCoords = np.array([site.Lat % 360, site.Lon % 360])
        interpolated = interpolator(targetCoords)[0]
        interpolated = round(interpolated, 3)
        
        # Save to output
        output[site.Index].append(interpolated)
        
    print(f"Finished all sub files of {root}")

  # Save
  out = pd.DataFrame(output)
  out.set_index("Time", inplace=True)
  out.sort_index(inplace=True)
  out.to_csv(f"output_{freq}hr.csv")
  print(f"Saved as output_{freq}hr.csv")

  print(f"Completed in {time.time() - start} seconds")

if __name__ == "__main__":
  freq = 1 if len(sys.argv) != 4 else sys.argv[3]
  print(f"Using frequency of {freq} hour")
  
  main(
    sys.argv[1], # Data/Site_Info.csv
    sys.argv[2], # Data/Stage_IV_...
    freq # 1, 6, or 24
  )
