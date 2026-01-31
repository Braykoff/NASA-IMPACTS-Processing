import pandas as pd
import numpy as np
import sys
import datetime

# Removes all the dates before start and after end, respective of the year
def removeStartEnd(path, start, end):
  # Open
  data = pd.read_csv(path, parse_dates=True, index_col="Time")
  print(f"Read {path}")

  # Truncate
  original = len(data)
  data = data.loc[start:end]
  print(f"Removed {original - len(data)} rows")

  # Save
  data.to_csv("truncated.csv")
  print("Saved as truncated.csv")

# Keeps only the dates between start and end, irrespective of the year
def keepOnlyDateRanges(path, start, end):
  # Open
  data = pd.read_csv(path, parse_dates=True, index_col="Time")
  print(f"Read {path}")

  # Remove
  original = len(data)

  if start.month <= end.month:
    # Range is within same year
    data = data[
      ((data.index.month == start.month) & (data.index.day >= start.day)) |
      ((data.index.month > start.month) & (data.index.month < end.month)) |
      ((data.index.month == end.month) & (data.index.day <= end.day))
    ]
  else:
    # Range goes to the next year
    data = data[
      ((data.index.month == start.month) & (data.index.day >= start.day)) |
      (data.index.month > start.month) | 
      (data.index.month < end.month) |
      ((data.index.month == end.month) & (data.index.day <= end.day))
    ]

  print(f"Removed {original - len(data)} rows")

  data.to_csv("truncated.csv")
  print("Saved as truncated.csv")

if __name__ == "__main__":
  #removeStartEnd(
  #  sys.argv[1], # file path
  #  np.datetime64("2023-11-01"),
  #  np.datetime64("2024-05-01")
  #)

  keepOnlyDateRanges(
    sys.argv[1], # file path
    datetime.date(1970, 11, 1), # year is irrelevant here
    datetime.date(1970, 5, 1)
  )
