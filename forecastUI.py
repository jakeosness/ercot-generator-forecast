import streamlit as st 
st.set_page_config(layout="wide")  # Set Streamlit to use full browser width

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import base64
from io import BytesIO
import streamlit.components.v1 as components

# === Import function to generate prediction input file ===
from buildonbaseForecast import generate_prediction_file

# === Configuration Paths ===
DATA_DIR = "Model_Info2/prediction_csvs/"                      # Directory containing forecasted output CSVs
PRICE_RANGE_CSV = "Generator_TPO_Price_Range.csv"              # File with min/max TPO price per generator
FORECAST_CSV = "forecastdata.csv"                              # Forecast file to determine start date

# === Load forecast start date for use in plots ===
forecast_df = pd.read_csv(FORECAST_CSV)
forecast_df['DeliveryDate'] = pd.to_datetime(forecast_df['DeliveryDate'])
forecast_start_date = forecast_df['DeliveryDate'].iloc[0].date()

# === Sidebar: Title and Prediction File Generator ===
st.sidebar.title("View Options")

# --- User inputs gas price and triggers file generation ---
st.sidebar.markdown("### Generate Prediction File")
gas_price_input = st.sidebar.text_input("Enter Natural Gas Price ($/MMBtu)", value="2.96")

if st.sidebar.button("Generate Prediction File"):
    try:
        gas_price = float(gas_price_input)
        generate_prediction_file(gas_price=gas_price)
        st.sidebar.success(f"✅ prediction_input.csv generated with gas price ${gas_price}")
    except ValueError:
        st.sidebar.error("❌ Please enter a valid numeric gas price.")

# === Sidebar: Visualization Mode + Generator Selection ===
view_mode = st.sidebar.radio("Select display mode", ["Grid View", "Scrollable Row View"])

# List of all generator forecast files
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_forecast_block.csv")]
selected_file = st.sidebar.selectbox("Select a Generator", sorted(all_files))

# === Load price range per generator (used to annotate plots) ===
price_range_df = pd.read_csv(PRICE_RANGE_CSV)

# === Function to combine all forecasts into a downloadable Excel workbook ===
def generate_combined_excel(data_dir):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    combined_df = []

    for file in sorted(os.listdir(data_dir)):
        if file.endswith("_forecast_block.csv"):
            gen_name = file.replace("_forecast_block.csv", "")
            df = pd.read_csv(os.path.join(data_dir, file))
            df.insert(0, "Generator", gen_name)
            combined_df.append(df)

    all_combined = pd.concat(combined_df, ignore_index=True)
    all_combined.to_excel(writer, sheet_name="All_Generators", index=False)

    for df in combined_df:
        gen_name = df["Generator"].iloc[0]
        df.to_excel(writer, sheet_name=gen_name[:31], index=False)  # Sheet name limited to 31 chars

    writer.close()
    output.seek(0)
    return output

# === Sidebar: Excel Download Button ===
excel_data = generate_combined_excel(DATA_DIR)
st.sidebar.download_button(
    label="📥 Download All Forecasts (Excel)",
    data=excel_data,
    file_name="All_Generator_Forecasts.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# === Main App Title ===
st.title("ERCOT Forecasted Offer Curves")

# === Load and Plot Forecast Data for Selected Generator ===
if selected_file:
    df = pd.read_csv(os.path.join(DATA_DIR, selected_file))
    gen_name = selected_file.replace("_forecast_block.csv", "")
    st.subheader(f"Forecasted Curves: {gen_name}")

    # Load min/max price lines for visual annotation
    price_range = price_range_df[price_range_df["Resource Name"] == gen_name]
    price_min = price_range["Min_NonZero_TPO_Price"].values[0] if not price_range.empty else None
    price_max = price_range["Max_TPO_Price"].values[0] if not price_range.empty else None

    # Forecast covers multiple days (each with 24 hours of forecast)
    num_days = len(df) // 24
    start_date = pd.Timestamp(forecast_start_date)

    for day in range(num_days):
        display_date = start_date + pd.Timedelta(days=day)
        st.markdown(f"### Day {day + 1} ({display_date.strftime('%B %d, %Y')})")
        st.markdown(f"**Hours {day*24}–{(day+1)*24 - 1}**")

        # === Grid View: 4x6 = 24 plots side by side ===
        if view_mode == "Grid View":
            fig = plt.figure(figsize=(28, 18))
            gs = gridspec.GridSpec(4, 6, figure=fig)

            for i, h in enumerate(range(day * 24, (day + 1) * 24)):
                mw = df.loc[h, [f"Pred_TPO_MW{j+1}" for j in range(8)]].values
                price = df.loc[h, [f"Pred_TPO_Price{j+1}" for j in range(8)]].values

                ax = fig.add_subplot(gs[i // 6, i % 6])
                ax.plot(mw, price, marker='o', color='blue')
                ax.set_title(f"Hour {h % 24:02d}", fontsize=10)
                ax.set_xlabel("MW")
                ax.set_ylabel("Price")
                ax.tick_params(labelsize=8)

                # Add min/max price guide lines
                if price_min is not None:
                    ax.axhline(price_min, color='green', linestyle='--', linewidth=1)
                if price_max is not None:
                    ax.axhline(price_max, color='red', linestyle='--', linewidth=1)

            plt.tight_layout()
            st.pyplot(fig)

        # === Scrollable Row View: 24 plots in a horizontal scroll ===
        else:
            fig, axes = plt.subplots(1, 24, figsize=(48, 4), constrained_layout=True)

            for i, h in enumerate(range(day * 24, (day + 1) * 24)):
                mw = df.loc[h, [f"Pred_TPO_MW{j+1}" for j in range(8)]].values
                price = df.loc[h, [f"Pred_TPO_Price{j+1}" for j in range(8)]].values

                ax = axes[i]
                ax.plot(mw, price, marker='o', color='blue')
                ax.set_title(f"H{h % 24}", fontsize=8)
                ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
                ax.tick_params(axis='both', labelsize=6)

                if i % 6 == 0:
                    ax.set_xlabel("MW", fontsize=6)
                    ax.set_ylabel("Price", fontsize=6)

                # Add min/max price guide lines
                if price_min is not None:
                    ax.axhline(price_min, color='green', linestyle='--', linewidth=1)
                if price_max is not None:
                    ax.axhline(price_max, color='red', linestyle='--', linewidth=1)

            # Convert to scrollable image for wide horizontal layout
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_base64 = base6.b64encode(buf.read()).decode("utf-8")
            buf.close()

            components.html(
                f"""<div style='overflow-x:auto; width:100%;'>
                        <img src='data:image/png;base64,{img_base64}' style='height:450px; width:10000px;'/>
                    </div>""",
                height=500,
                scrolling=True
            )
