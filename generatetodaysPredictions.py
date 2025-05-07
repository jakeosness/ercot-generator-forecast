# generate_predictions.py

import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model

def run_model_prediction():
    # === CONFIGURATION ===
    INPUT_CSV = "prediction_input.csv"
    FEATURE_SCALER_PATH = "Model_Info2/feature_scaler.pkl"
    TARGET_SCALER_PATH = "Model_Info2/target_scaler.pkl"
    ENCODER_DIR = "Model_Info2"
    MODEL_PATH = "Model_Info2/sequence_transformer_model.keras"
    OUTPUT_NPY = "Model_Info2/y_pred_today.npy"
    SEQUENCE_LENGTH = 24

    # === Load and preprocess data
    df = pd.read_csv(INPUT_CSV)
    df["GasLoadRatio"] = df["Natural Gas Price per Million BTU"] / (df["ForecastedLoadTotal"] + 1e-6)

    df["SCED_DateTime"] = pd.to_datetime(df["SCED_DateTime"], format="%Y-%m-%d %H:%M:%S")
    df["Hour"] = df["SCED_DateTime"].dt.hour
    df["DayOfWeek"] = df["SCED_DateTime"].dt.dayofweek

    original_names = df["Resource Name"].copy()

    # === Prepare features
    embedding_col = "Resource Name"
    onehot_cols = ["Resource Type", "QSE", "DME", "Resource Tech"]
    target_cols = [f"Submitted TPO-MW{i}" for i in range(1, 9)] + [f"Submitted TPO-Price{i}" for i in range(1, 9)]
    excluded_cols = ["Date", "SCED_DateTime", "SCED Time Stamp"] + target_cols + [embedding_col] + onehot_cols
    numeric_cols = [col for col in df.columns if col not in excluded_cols]

    # === Encode categorical features
    embed_le = joblib.load(os.path.join(ENCODER_DIR, "label_encoder_Resource Name.pkl"))
    df[embedding_col] = df[embedding_col].astype(str).apply(lambda x: x if x in embed_le.classes_ else "Unknown")
    if "Unknown" not in embed_le.classes_:
        embed_le.classes_ = np.append(embed_le.classes_, "Unknown")
    df[embedding_col] = embed_le.transform(df[embedding_col])

    onehot_enc = joblib.load(os.path.join(ENCODER_DIR, "onehot_encoder.pkl"))
    df_onehot = pd.DataFrame(onehot_enc.transform(df[onehot_cols]).astype(float))

    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    numeric_ordered = feature_scaler.feature_names_in_
    df_numeric = pd.DataFrame(feature_scaler.transform(df[numeric_ordered]), columns=numeric_ordered)

    X_all = pd.concat([
        df[[embedding_col]].reset_index(drop=True),
        df_onehot.reset_index(drop=True),
        df_numeric.reset_index(drop=True)
    ], axis=1)

    # === Sequence creation
    df["Date"] = df["Date"].astype(str)
    X_all["group"] = df[embedding_col].astype(str) + "_" + df["Date"]
    original_names_seq = original_names.values
    X_seq, resource_names = [], []

    for group in X_all["group"].unique():
        block = X_all[X_all["group"] == group].drop(columns=["group"])
        if len(block) == SEQUENCE_LENGTH:
            X_seq.append(block.values)
            resource_names.append(original_names_seq[X_all[X_all["group"] == group].index[0]])

    X_seq = np.array(X_seq)
    X_embed = X_seq[:, :, 0].astype(int)
    X_values = X_seq.copy()

    # === Load model and predict
    model = load_model(MODEL_PATH)
    y_pred_scaled = model.predict([X_values, X_embed])
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 16)).reshape(-1, 24, 16)

    np.save(OUTPUT_NPY, y_pred)

    # === Save individual generator forecasts
    os.makedirs("Model_Info2/prediction_csvs", exist_ok=True)
    combined = {}
    for i, gen_name in enumerate(resource_names):
        if gen_name not in combined:
            combined[gen_name] = []
        combined[gen_name].append(y_pred[i])

    for gen_name, blocks in combined.items():
        full_pred = np.concatenate(blocks, axis=0)
        df_mw = pd.DataFrame(full_pred[:, :8], columns=[f"Pred_TPO_MW{j+1}" for j in range(8)])
        df_price = pd.DataFrame(full_pred[:, 8:], columns=[f"Pred_TPO_Price{j+1}" for j in range(8)])
        df_out = pd.concat([df_mw, df_price], axis=1)
        df_out.to_csv(f"Model_Info2/prediction_csvs/{gen_name}_forecast_block.csv", index=False)

    print(f"✅ Saved predictions for {len(combined)} generators.")
