# buildbaseonForecast.py

import pandas as pd 
import numpy as np
from datetime import datetime, timedelta

def generate_prediction_file(gas_price: float = 2.96):
    # Load base and forecast input files
    base_df = pd.read_csv("prediction_input_base.csv")  # Contains static generator data (72 rows per generator)
    forecast_df = pd.read_csv("forecastdata.csv")       # Contains hourly system load forecasts by region

    # Convert forecast date column and get earliest forecast date
    forecast_df['DeliveryDate'] = pd.to_datetime(forecast_df['DeliveryDate'])
    start_forecast_date = forecast_df['DeliveryDate'].min().date()

    # Determine number of generators and hours per generator
    num_generators = base_df['Resource Name'].nunique()
    rows_per_generator = 72  # Assumes 72 hours of forecast (3 days)

    # Create timestamp metadata for each hour, repeated for each generator
    timestamp_rows = []
    for h in range(rows_per_generator):
        forecast_dt = datetime.combine(start_forecast_date, datetime.min.time()) + timedelta(hours=h)
        sced_dt = datetime(2024, 12, 2, 0, 0) + timedelta(hours=h)  # Placeholder SCED datetime
        timestamp_rows.append({
            "Date": forecast_dt.date(),
            "SCED Time Stamp": forecast_dt.strftime("%H:%M"),
            "SCED_DateTime": sced_dt
        })

    # Expand timestamps for each generator and attach to base_df
    timestamp_df = pd.DataFrame(timestamp_rows * num_generators)
    base_df = pd.concat([timestamp_df, base_df.reset_index(drop=True)], axis=1)

    # Normalize HourEnding column and filter forecast rows in use
    forecast_df['HourEnding'] = forecast_df['HourEnding'].replace("24:00", "00:00")
    forecast_df['HourEnding'] = pd.to_datetime(forecast_df['HourEnding'], format="%H:%M").dt.time
    forecast_df = forecast_df[forecast_df['InUseFlag'] == 'Y'].copy()

    # Rename columns to match model expectations
    forecast_df.rename(columns={
        'SystemTotal': 'ForecastedLoadTotal',
        'Coast': 'ForecastLoadCoast',
        'East': 'ForecastLoadEast',
        'FarWest': 'ForecastLoadFarWest',
        'North': 'ForecastLoadNorth',
        'NorthCentral': 'ForecastLoadNorthCentral',
        'SouthCentral': 'ForecastLoadSouthCentral',
        'Southern': 'ForecastLoadSouthern',
        'West': 'ForecastLoadWest'
    }, inplace=True)

    # Create separate date column and drop unused columns
    forecast_df['Date'] = pd.to_datetime(forecast_df['DeliveryDate']).dt.date
    forecast_df.drop(columns=['DeliveryDate', 'Model', 'DSTFlag', 'InUseFlag'], inplace=True)

    # Forward fill missing hourly forecast values within each day
    forecast_df = forecast_df.sort_values(['Date', 'HourEnding'])
    forecast_df = forecast_df.groupby('Date').apply(lambda g: g.ffill()).reset_index(drop=True)

    # Normalize base_df timestamp formats for joining
    base_df['Date'] = pd.to_datetime(base_df['Date']).dt.date
    base_df['SCED Time Stamp'] = pd.to_datetime(base_df['SCED Time Stamp'], format="%H:%M").dt.time

    # Merge forecast data into base_df on Date + Hour
    merged_df = base_df.merge(
        forecast_df,
        left_on=['Date', 'SCED Time Stamp'],
        right_on=['Date', 'HourEnding'],
        how='left'
    ).drop(columns=['HourEnding'])

    # Reformat time column and add natural gas price to every row
    merged_df['SCED Time Stamp'] = merged_df['SCED Time Stamp'].apply(lambda t: t.strftime('%H:%M'))
    merged_df["Natural Gas Price per Million BTU"] = gas_price

    # Add empty placeholder columns for TPO MW and Price (model targets) if missing
    target_cols = [f"Submitted TPO-MW{i}" for i in range(1, 11)] + \
                  [f"Submitted TPO-Price{i}" for i in range(1, 11)]
    for col in target_cols:
        if col not in merged_df.columns:
            merged_df[col] = np.nan

    # Export the final prediction input file for downstream use
    merged_df.to_csv("prediction_input.csv", index=False)
    print("✅ Saved: prediction_input.csv")
