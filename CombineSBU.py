import pandas as pd
from pathlib import Path
import sys
import re
import numpy as np

def main(sbuDir, suffix):
  # Init output
  outColumn = "SBU" + suffix

  output = {
    "Time": [],
    outColumn: []
  }

  # Iterate over SBU files
  root = Path(sbuDir)
  files = next(root.walk())[2]

  pattern = r"IMPACTS_SBU_pluvio_(\d{8})" + suffix + r"\.csv$"

  for file in files:
    # Check if this file is valid
    match = re.match(pattern, file)

    if not match:
      print(f"Skipped {file} because its name did not match the pattern")
      continue

    # Load file
    data = pd.read_csv(root.joinpath(file))
    date = match.group(1) # Regex group 1
    print(f"Read {file}")

    # Combine by hour
    data["hour"] = np.floor(data["hour"])
    data.drop(columns=["hhmnss"], inplace=True)

    data = data.groupby(data["hour"]).sum()

    # Format date column
    data["date"] = pd.to_datetime(
      date + " " + data.index.astype(int).astype(str) + ":00:00",
      format="%Y%m%d %H:%M:%S"
    )

    # Output
    output["Time"].extend(data["date"])
    output[outColumn].extend(data["accum_mm"])

  # Fill in missing dates
  out = pd.DataFrame(output)
  out.sort_values(by="Time", inplace=True)
  out.set_index("Time", inplace=True)

  fullRange = pd.to_datetime(pd.date_range(
    start=out.index.min(),
    end=out.index.max(),
    freq="h"
  ))
  out = out.reindex(fullRange)
  out.fillna(-9999.0, inplace=True) # missing entires with -9999
  out.index.name = "Time"

  # Save
  out.to_csv("combined.csv")
  print("Saved as combined.csv")

if __name__ == "__main__":
  main(
    sys.argv[1], # Data/SBU_Pluvio,
    "_BNL" # suffix between the date and file extension, such as "_BNL"
  )
