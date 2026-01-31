import pandas as pd
import sys
from pathlib import Path
import numpy as np

def main(infoCSV, rootPath):
  rootPath = Path(rootPath)

  # Init output
  output = pd.DataFrame({
    "Time": []
  })
  output.set_index("Time", inplace=True)

  # Read info csv
  info = pd.read_csv(infoCSV, index_col="Station")
  print(f"Read {infoCSV}")

  # Iterate over stations
  stateCSV = None
  lastState = None

  for station in info.index:
    station = info.loc[station]

    # Read corresponding ASOS csv, if not already read
    if station.State != lastState:
      stateCSVPath = rootPath.joinpath(f"asos_{station.State}.csv")
      stateCSV = pd.read_csv(stateCSVPath, parse_dates=["valid"])
      print(f"Read {stateCSVPath}")
      lastState = station.State

      if not "p01m" in stateCSV.columns:
        # Convert inches to mm
        stateCSV["p01m"] = np.round(stateCSV["p01i"] * 25.4, decimals=4)

    # Only get this station data
    filtered = stateCSV.drop(stateCSV[stateCSV.station != station.name].index)
    
    # Only get rows that are the correct time
    filtered.drop(filtered[filtered.valid.dt.minute != station.ResetTime].index, inplace=True)

    # Add missing hours
    fullRange = pd.to_datetime(pd.date_range(
      start=filtered["valid"].min(),
      end=filtered["valid"].max(),
      freq="h"
    ))

    filtered = filtered.set_index("valid").reindex(fullRange)
    filtered.reset_index(inplace=True)
    filtered.rename(columns={"index": "valid"}, inplace=True)
    filtered.fillna({"p01m": np.nan}, inplace=True)

    # Create range rounded UP to the nearest hour
    fullRange = pd.to_datetime(pd.date_range(
      start=filtered["valid"].min().ceil("h"),
      end=filtered["valid"].max().ceil("h"),
      freq="h"
    ))

    # Combine with output
    stationData = pd.DataFrame({
      "Time": fullRange,
      str(station.name): filtered["p01m"]
    })
    stationData.set_index("Time", inplace=True)

    output = output.join(stationData, how="outer")
  
  # Save
  output.fillna(-9999, inplace=True)
  output.sort_values(by="Time", inplace=True)
  output.to_csv("combined.csv")
  print("Saved as combined.csv")

if __name__ == "__main__":
  main(
    sys.argv[1], # ASOS_Info.csv
    sys.argv[2] #  Data/ASOS_AWPAG
  )
