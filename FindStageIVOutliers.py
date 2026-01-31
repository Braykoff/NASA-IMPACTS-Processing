import pandas as pd
import sys

def findOutliers(stageIVFilename, otherSourceFilenames, minPrecip, minDifference):
  # Open files
  stageIV = pd.read_csv(stageIVFilename, index_col="Time", parse_dates=True)

  print(f"Read StageIV file {stageIVFilename} with {len(stageIV.columns)} sites")

  sources = []
  for name in otherSourceFilenames:
    source = pd.read_csv(name, index_col="Time", parse_dates=True)
    sources.append(source)
    print(f"Read {name} with {len(source.columns)} sites")

  # Init output
  output = {
    "Time": [],
    "Site": [],
    "SitePrecip": [],
    "StageIVPrecip": [],
  }

  total = 0

  # Find outliers
  for station in stageIV.columns:
    # Find what other file this data is in
    data = pd.DataFrame(columns=[station])
    numFiles = 0
    
    for source in sources:
      if station in source.columns:
        data = source[[station]] if len(data) == 0 else pd.concat([data, source[[station]]])
        data = data[~data.index.duplicated(keep="first")]
        numFiles += 1

    if numFiles == 0:
      print(f"Could not find any other file to match site {station}!")
      continue
      
    print(f"Used {numFiles} files for site {station}, {len(data)} rows")

    # Remove missing data
    dataFiltered = data[data[station] >= 0]
    stageIVFiltered = stageIV[stageIV[station] >= 0]

    common = dataFiltered.index.intersection(stageIVFiltered.index)
    dataFiltered = dataFiltered.loc[common]
    stageIVFiltered = stageIVFiltered.loc[common]

    # Remove data below threshold
    filter = (dataFiltered[station] >= minPrecip) | (stageIVFiltered[station] >= minPrecip)
    dataFiltered = dataFiltered[filter]
    stageIVFiltered = stageIVFiltered[filter]

    # Find outliers
    diffMask = (dataFiltered[station] - stageIVFiltered[station]).abs() >= minDifference

    dataFiltered = dataFiltered[diffMask]
    stageIVFiltered = stageIVFiltered[diffMask]

    output["Time"].extend(dataFiltered.index)
    output["Site"].extend([station] * len(dataFiltered))
    output["SitePrecip"].extend(dataFiltered[station])
    output["StageIVPrecip"].extend(stageIVFiltered[station])

    print(f"Found {len(dataFiltered)} outliers in {station}")
    total += len(dataFiltered)

  print(f"Found {total} total outliers")

  # Save
  out = pd.DataFrame(output)
  out.sort_values(by="Time", inplace=True)
  out.reset_index(inplace=True, drop=True)
  out.to_csv("outliers.csv")
  print("Saved as outliers.csv")

# Run
if __name__ == "__main__":
  findOutliers(
    sys.argv[1], # Processed/StageIV_1hr.csv
    sys.argv[2:], # Processed/Combined...
    10, # min amount (mm) of precip to compare
    5 # min difference (mm) of precip to be considered an outlier
  )

  #python3 Scripts/FindStageIVOutliers.py Processed/StageIV_1hr.csv Processed/CombinedASOS_FixedRanges.csv Processed/CombinedSBU_v2.csv Processed/CombinedSBU_BNL_v2.csv Processed/CombinedUCONN_2122_v2.csv Processed/CombinedUCONN_2223_v2_truncated.csv Processed/CombinedUCONN_2324_v2_truncated.csv 