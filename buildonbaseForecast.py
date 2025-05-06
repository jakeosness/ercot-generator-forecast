import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# === Load base and forecast files
base_df = pd.read_csv("prediction_input_base.csv")  # Already contains 72 rows per generator
forecast_df = pd.read_csv("forecastdata.csv")

# === Get forecast start date from forecast file
forecast_df['DeliveryDate'] = pd.to_datetime(forecast_df['DeliveryDate'])
start_forecast_date = forecast_df['DeliveryDate'].min().date()

# === Determine how many generators
num_generators = base_df['Resource Name'].nunique()
rows_per_generator = 72

# === Generate timestamp rows (same 72 repeated per generator)
timestamp_rows = []
for h in range(rows_per_generator):
    forecast_dt = datetime.combine(start_forecast_date, datetime.min.time()) + timedelta(hours=h)
    sced_dt = datetime(2024, 12, 2, 0, 0) + timedelta(hours=h)
    timestamp_rows.append({
        "Date": forecast_dt.date(),
        "SCED Time Stamp": forecast_dt.strftime("%H:%M"),
        "SCED_DateTime": sced_dt
    })

# Repeat timestamps for all generators
timestamp_df = pd.DataFrame(timestamp_rows * num_generators)

# === Combine timestamp columns with base_df
base_df = pd.concat([timestamp_df, base_df.reset_index(drop=True)], axis=1)

# === Fix HourEnding values: convert '24:00' to '00:00' and to time object
forecast_df['HourEnding'] = forecast_df['HourEnding'].replace("24:00", "00:00")
forecast_df['HourEnding'] = pd.to_datetime(forecast_df['HourEnding'], format="%H:%M").dt.time

# === Filter for 'Y' in InUseFlag
forecast_df = forecast_df[forecast_df['InUseFlag'] == 'Y'].copy()

# === Rename relevant columns to match model expectations
forecast_rename_map = {
    'SystemTotal': 'ForecastedLoadTotal',
    'Coast': 'ForecastLoadCoast',
    'East': 'ForecastLoadEast',
    'FarWest': 'ForecastLoadFarWest',
    'North': 'ForecastLoadNorth',
    'NorthCentral': 'ForecastLoadNorthCentral',
    'SouthCentral': 'ForecastLoadSouthCentral',
    'Southern': 'ForecastLoadSouthern',
    'West': 'ForecastLoadWest'
}
forecast_df.rename(columns=forecast_rename_map, inplace=True)

# === Create 'Date' column from DeliveryDate and drop unneeded columns
forecast_df['Date'] = pd.to_datetime(forecast_df['DeliveryDate']).dt.date
forecast_df.drop(columns=['DeliveryDate', 'Model', 'DSTFlag', 'InUseFlag'], inplace=True)

# === Sort and backfill first-hour load values per day
forecast_df = forecast_df.sort_values(['Date', 'HourEnding'])
forecast_df = forecast_df.groupby('Date').apply(lambda g: g.ffill()).reset_index(drop=True)

# === Format base file's Date and SCED Time Stamp
base_df['Date'] = pd.to_datetime(base_df['Date']).dt.date
base_df['SCED Time Stamp'] = pd.to_datetime(base_df['SCED Time Stamp'], format="%H:%M").dt.time

# === Merge forecast data into base input on Date and Hour
merged_df = base_df.merge(
    forecast_df,
    left_on=['Date', 'SCED Time Stamp'],
    right_on=['Date', 'HourEnding'],
    how='left'
).drop(columns=['HourEnding'])

# === Re-format SCED Time Stamp to remove seconds
merged_df['SCED Time Stamp'] = merged_df['SCED Time Stamp'].apply(lambda t: t.strftime('%H:%M'))

# === Add natural gas price (same value for all rows)
merged_df["Natural Gas Price per Million BTU"] = 2.96

# === Add missing TPO MW and Price columns as NaN
target_cols = [f"Submitted TPO-MW{i}" for i in range(1, 11)] + \
              [f"Submitted TPO-Price{i}" for i in range(1, 11)]

for col in target_cols:
    if col not in merged_df.columns:
        merged_df[col] = np.nan

# === Save final prediction-ready file
merged_df.to_csv("prediction_input.csv", index=False)
print("✅ Saved: prediction_input.csv (with forecast, gas, and target columns)")
