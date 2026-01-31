import xarray as xa
import pandas as pd
import numpy as np
import sys
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import re
import time

def main(siteInfoFile, mrmsDir, pattern, saveName="output.csv"):
  start = time.time()

  # Open site info file
  sites = pd.read_csv(siteInfoFile, index_col="Site")
  print(f"Found {len(sites)} sites in {siteInfoFile}")

  # Init output
  output = {"Time": []}

  for site in sites.itertuples():
    output[site.Index] = []

  # Iterate over grb2 files
  mrmsDir = Path(mrmsDir)

  for root, _, files in mrmsDir.walk():
    for idx, file in enumerate(files):
      # Check that this is a grb2 file
      match = re.match(pattern, file)

      if not match:
        # This is not a grb2 file
        print(f"({idx+1}/{len(files)}) Skipped {file}")
        continue
      
      # Open file
      data = xa.open_dataset(root.joinpath(file), engine="cfgrib", decode_timedelta=True)
      date = pd.to_datetime(f"{match.group(1)} {match.group(2)}", format="%Y%m%d %H%M%S")
      print(f"{idx+1}/{len(files)} Read {file} ({date})")

      output["Time"].append(date)

      # Create interpolator
      interpolator = RegularGridInterpolator(
        (data.latitude.values, data.longitude.values),
        data.unknown.values,
        fill_value=-9999
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
  out.to_csv(saveName)
  print(f"Saved to {saveName}")

  print(f"Completed in {time.time() - start} seconds")

# Run
if __name__ == "__main__":
  fileNamePatterns = [
    ("1hr_Multi", r"MRMS_MultiSensor_QPE_01H_Pass2_00.00_(\d{8})-(\d{6}).grib2$"),
    ("6hr_Multi", r"MRMS_MultiSensor_QPE_06H_Pass2_00.00_(\d{8})-(\d{6}).grib2$"),
    ("24hr_Multi", r"MRMS_MultiSensor_QPE_24H_Pass2_00.00_(\d{8})-(\d{6}).grib2$"),
    ("1hr_Radar", r"MRMS_RadarOnly_QPE_01H_00.00_(\d{8})-(\d{6}).grib2$"),
    ("6hr_Radar", r"MRMS_RadarOnly_QPE_06H_00.00_(\d{8})-(\d{6}).grib2$"),
    ("24hr_Radar", r"MRMS_RadarOnly_QPE_24H_00.00_(\d{8})-(\d{6}).grib2$")
  ]

  for group in fileNamePatterns:
    main(
      sys.argv[1], # Data/Site_Info.csv
      sys.argv[2], # Data/MRMS/...
      # Regex to match grb2 file names:
      # Group 1 should be date in yyyyMMdd
      # Group 2 should be time in hhmmss
      group[1],
      saveName=f"MRMS_{group[0]}.csv"
    )
