import pandas as pd
import sys

def aggregate(filename):
  # Open file
  data = pd.read_csv(filename, index_col="Time", parse_dates=True)
  print(f"Read {filename}")

  # Replace -9999
  data.replace(-9999, 0, inplace=True)

  # Aggregate
  intervals = [6, 24]

  for interval in intervals:
    # Get first date to use
    start = min(data.index)

    if interval == 6:
      start = start.replace(hour=1, minute=0, second=0)
    elif interval == 24:
      if start.hour < 13:
        start = start - pd.Timedelta(days=1)
      
      start = start.replace(hour=13, minute=0, second=0)

    # Create bins and sort
    bins = pd.date_range(start=start, end=max(data.index) + pd.Timedelta(hours=interval), freq=f"{interval}h")
    data["interval"] = pd.cut(data.index, bins=bins, right=False)

    grouped = data.groupby("interval", observed=False).sum()
    grouped["Time"] = grouped.index.categories.right[grouped.index.codes]

    # Subtract 1 from each timestamp's hour (the right bound isn't actually included in the date range)
    #grouped["Time"] -= pd.Timedelta(minutes=1)
    grouped["Time"] -= pd.Timedelta(hours=1)
    
    # Fix indexes
    grouped.set_index("Time", inplace=True)

    # Save
    name = f"Aggregated{interval}h.csv"
    grouped.to_csv(name)
    print(f"Saved to {name}")

# Run
if __name__ == "__main__":
  aggregate(sys.argv[1])
