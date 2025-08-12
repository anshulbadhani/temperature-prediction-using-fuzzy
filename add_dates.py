import pandas as pd

# Step 1: Read your CSV without date column
df = pd.read_csv("data.csv")

# Step 2: Create date range
start_date = "2025-01-01"
end_date = "2025-07-31"
date_range = pd.date_range(start=start_date, end=end_date, freq="D")

# Step 3: Check if lengths match
if len(date_range) != len(df):
    raise ValueError(f"Number of rows in CSV ({len(df)}) does not match days in range ({len(date_range)})")

# Step 4: Add date column
df.insert(0, "date", date_range)

# Step 5: Save new CSV
df.to_csv("weather_data_with_dates.csv", index=False)

print("✅ Date column added successfully. Saved as 'weather_data_with_dates.csv'")
