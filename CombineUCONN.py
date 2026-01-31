import pandas as pd
import sys
from pathlib import Path
import re

def main(root, pattern):
  root = Path(root)

  # Init output
  output = pd.DataFrame({
    "Time": []
  })
  output.set_index("Time", inplace=True)

  # Iterate over files
  files = next(root.walk())[2]
  
  for file in files:
    match = re.match(pattern, file)

    # Check if valid file
    if not match:
      print(f"Skipped {file} because it did not match the pattern")
      continue

    data = pd.read_csv(root.joinpath(file), header=None, sep="\\s+")
    site = match.group(1)
    print(f"Read {file} ({site})")

    # Add datetime column
    data["date"] = pd.to_datetime(
      data[0].astype(str) + " " + data[1].astype(str) + " " + data[2].astype(str) + ":00:00",
      format="%Y %j %H:%M:%S"
    )

    # Sum all columns with same datetime
    data = data.groupby(data["date"]).sum()

    # Combine with output
    fileData = pd.DataFrame({
      "Time": data.index,
      site: data[5]
    })
    fileData.set_index("Time", inplace=True)

    #output = output.add(fileData, fill_value=0.0)
    output = output.combine_first(fileData)

  # Fill in missing data 
  fullRange = pd.to_datetime(pd.date_range(
    start=output.index.min(),
    end=output.index.max(),
    freq="h"
  ))

  output = output.reindex(fullRange)
  output.fillna(-9999.0, inplace=True) # missing entires with -9999
  output.index.name = "Time"

  # Sort
  output.sort_index(inplace=True)
  
  # Save
  output.to_csv("combined.csv")
  print("Saved to combined.csv")
  
if __name__ == "__main__":
  main(
    sys.argv[1], # ex Data/UCONN_Pluvio/21_22
    # Regex that matches the file name, with group 1 being the site:
    #r"(apu\d{2})_pluvio\d{3}_\d{6}_precip.impact2022$"
    #r"(apu\d{2})_pluvio\d{3}_\d{6}_precip.impacts2023$"
    r"(apu\d{2})_pluvio\d{3}_\d{6}_precip.uconn2024$"
  )
